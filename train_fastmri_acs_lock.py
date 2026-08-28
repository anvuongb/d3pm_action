"""Stage A2: warm-start ACS-locked remainder policy (10 epochs).

Never writes models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth.
Loads parent image-Z `both` weights; does not train from scratch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from itw.configs import FastMRIConfig, default_device
from itw.data.fastmri import FastMRIDataset
from itw.eval import (
    acs_lock_topk_from_logits,
    baseline_row_mask_batch,
    evaluate_fastmri_baselines_loader,
    plot_fastmri_grid,
    save_eval_report,
)
from itw.masks import (
    acs_bounds,
    apply_kspace_row_mask,
    gumbel_row_mask_acs_locked,
)
from itw.train import (
    _row_mask_ste,
    build_dataloader,
    build_mask_model,
    load_d3pm_coarse,
    mask_generator_cond,
    train_fastmri_nested,
)

PARENT = Path("models_mask_gen_fastmri_nested")
PARENT_FINAL = PARENT / "mask_gen_fastmri_final.pth"
COARSE_SURVIVAL_PT = PARENT / "coarse_survival_table.pt"
SAVE_DIR = PARENT / "acs_lock_10ep"
A2_S = (0.10, 0.125, 0.25, 0.4)
A2_BASELINES = ("acs_random", "acs_vd_gaussian", "acs_equispaced")
PLOT_METHODS = ("learned", "learned_acs_lock", "acs_random", "acs_vd_gaussian")
A1_POSTHOC_S025 = {
    "nmse": 0.0065,
    "ssim": 0.752,
    "psnr": 32.12,
    "h_coarse": 0.042,
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_train_log(path: Path) -> dict:
    epochs: list[dict] = []
    collapsed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("COLLAPSED"):
            collapsed = True
            continue
        if not line.startswith("epoch:"):
            continue
        parts: dict[str, float] = {}
        for chunk in line.split(","):
            if ":" not in chunk:
                continue
            k, v = chunk.split(":", 1)
            parts[k.strip()] = float(v.strip())
        epochs.append(parts)
    last = epochs[-1] if epochs else {}
    return {
        "collapsed": collapsed,
        "epochs_completed": len(epochs),
        "final_loss_ema": last.get("loss"),
        "final_density_ema": last.get("density"),
        "final_h_coarse_mean": last.get("H_coarse"),
        "final_nmse_mean": last.get("nmse"),
        "epochs": epochs,
        "from_train_log": True,
    }


def _load_train_status(status_path: Path, log_path: Path) -> dict:
    if status_path.is_file():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and (
                data.get("epochs_completed") is not None or data.get("epochs")
            ):
                return data
        except json.JSONDecodeError:
            pass
    if log_path.is_file() and log_path.stat().st_size > 0:
        parsed = _parse_train_log(log_path)
        if parsed["epochs_completed"] or parsed["collapsed"]:
            return parsed
    raise SystemExit(
        f"missing usable train status ({status_path}) and train.log ({log_path})"
    )


def _completed_train_status(
    final: Path, status_path: Path, n_epochs: int
) -> dict | None:
    """Only treat a run as done if the ckpt exists and status is complete."""
    if not final.is_file() or not status_path.is_file():
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(status, dict):
        return None
    if status.get("collapsed"):
        return status
    completed = status.get("epochs_completed")
    if completed is None:
        return None
    if int(completed) < int(n_epochs):
        return None
    return status


def _assert_safe_save_dir(save_dir: Path) -> None:
    resolved = save_dir.resolve()
    if resolved == PARENT.resolve():
        raise SystemExit(f"refusing to train into parent {PARENT}")
    collapsed = Path("models_mask_gen_fastmri")
    if resolved == collapsed.resolve() or save_dir.name == "models_mask_gen_fastmri":
        raise SystemExit("refusing collapsed save_dir models_mask_gen_fastmri/")
    if (save_dir / "mask_gen_fastmri_final.pth").resolve() == PARENT_FINAL.resolve():
        raise SystemExit("parent checkpoint path collided with A2 save_dir")


def _make_cfg(*, save_dir: str) -> FastMRIConfig:
    return FastMRIConfig(
        device=default_device(),
        mask_objective="nested_d3pm",
        n_epochs=10,
        batch_size=8,
        num_workers=4,
        save_every=1,
        lr=1e-4,
        sparsity_loss_weight=50.0,
        sparsity_min=0.1,
        sparsity_max=0.4,
        entropy_alpha=1.0,
        entropy_beta=0.0,
        recon_loss_weight=1.0,
        policy_input="scout_image",
        acs_lock=True,
        save_dir=save_dir,
    )


def _confirm_acs_lock_ste(cfg: FastMRIConfig) -> None:
    """Fail fast if train STE is not the ACS-locked variant."""
    if not bool(cfg.acs_lock):
        raise SystemExit("A2 requires cfg.acs_lock=True")
    h = int(cfg.image_size)
    acs_width = int(cfg.scout_size)
    logits = torch.zeros(1, 1, h, 2, device=cfg.device)
    logits[..., 0] = 8.0
    logits[..., 1] = -8.0
    sparsity = torch.tensor([0.25], device=cfg.device)
    _soft, hard = _row_mask_ste(cfg, logits, sparsity, temperature=0.5)
    lo, hi = acs_bounds(h, acs_width)
    if not torch.all(hard[0, 0, lo:hi, 0] == 1):
        raise SystemExit("ACS-lock STE hook not applied (ACS region not hard-1)")
    print(
        f"ACS-lock STE confirmed: gumbel_row_mask_ste_acs_locked "
        f"ACS[{lo}:{hi}] hard-1, dens={float(hard.mean()):.4f}"
    )


def _load_parent_weights(cfg: FastMRIConfig) -> torch.nn.Module:
    if not PARENT_FINAL.is_file():
        raise FileNotFoundError(PARENT_FINAL)
    model = build_mask_model(cfg)
    ckpt = torch.load(PARENT_FINAL, map_location=cfg.device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["mask_generator"], strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"warm-start loaded {PARENT_FINAL} "
        f"md5={_md5(PARENT_FINAL)[:8]}… params={n_params} "
        f"missing={list(missing)} unexpected={list(unexpected)}"
    )
    return model


def _load_mask_net(cfg: FastMRIConfig) -> torch.nn.Module:
    ckpt_path = Path(cfg.save_dir) / "mask_gen_fastmri_final.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    if ckpt_path.resolve() == PARENT_FINAL.resolve():
        raise SystemExit("refusing to load parent nested checkpoint as A2 ckpt")
    model = build_mask_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["mask_generator"])
    model.eval()
    return model


def _plot_grids(cfg: FastMRIConfig, model, out: Path) -> None:
    viz_ds = FastMRIDataset(cfg.data_root)
    viz_dl = DataLoader(viz_ds, batch_size=4, shuffle=False, num_workers=0)
    z, c, kspace = next(iter(viz_dl))
    sparsity = torch.full((z.shape[0],), 0.25, device=cfg.device)
    acs_width = int(cfg.scout_size)
    model.eval()
    with torch.no_grad():
        z_d = z.to(cfg.device)
        k_d = kspace.to(cfg.device)
        cond = mask_generator_cond(cfg, z_d, k_d)
        row_logits = model(cond, sparsity)
        named_masks = {
            "learned": gumbel_row_mask_acs_locked(
                row_logits, sparsity, acs_width=acs_width, temperature=0.5, hard=True
            ),
            "learned_acs_lock": acs_lock_topk_from_logits(
                row_logits, sparsity, acs_width
            ),
        }
        for name in ("acs_random", "acs_vd_gaussian"):
            named_masks[name] = baseline_row_mask_batch(
                name,
                z.shape[0],
                cfg.image_size,
                sparsity,
                cfg.device,
                acs_width=acs_width,
            )
        titles = {
            "learned": "learned Gumbel ACS-lock s=0.25 (A2)",
            "learned_acs_lock": "learned ACS+top-k remainder s=0.25 (A2)",
            "acs_random": "acs_random s=0.25 (A2)",
            "acs_vd_gaussian": "acs_vd_gaussian s=0.25 (A2)",
        }
        for name in PLOT_METHODS:
            rows = named_masks[name]
            y = apply_kspace_row_mask(k_d, rows)
            dest = out / f"{name}_s025.png"
            plot_fastmri_grid(
                z,
                c,
                rows.cpu(),
                y.cpu(),
                title=titles[name],
                save_path=dest,
                kspace=kspace,
            )
    plt.close("all")
    print("wrote plots", [str(out / f"{n}_s025.png") for n in PLOT_METHODS])


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A2 FastMRI ACS-lock warm-start")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate existing acs_lock_10ep checkpoint",
    )
    args = parser.parse_args()

    _assert_safe_save_dir(SAVE_DIR)
    if not PARENT_FINAL.is_file():
        raise SystemExit(f"missing parent checkpoint {PARENT_FINAL}")
    if not COARSE_SURVIVAL_PT.is_file():
        raise SystemExit(f"missing {COARSE_SURVIVAL_PT}; pass it in, do not rebuild")

    parent_md5_before = _md5(PARENT_FINAL)
    parent_mtime_before = PARENT_FINAL.stat().st_mtime

    cfg = _make_cfg(save_dir=str(SAVE_DIR))
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    dest_survival = SAVE_DIR / "coarse_survival_table.pt"
    if dest_survival.resolve() != COARSE_SURVIVAL_PT.resolve():
        shutil.copy2(COARSE_SURVIVAL_PT, dest_survival)

    _confirm_acs_lock_ste(cfg)
    print(
        f"device={cfg.device} policy_input={cfg.policy_input} "
        f"acs_lock={cfg.acs_lock} n_epochs={cfg.n_epochs} "
        f"save_dir={cfg.save_dir} data_root={cfg.data_root} "
        "init=warm_start_parent"
    )

    dataloader = build_dataloader(cfg)
    d3pm_coarse = load_d3pm_coarse(cfg)
    coarse_survival = torch.load(
        dest_survival, map_location=cfg.device, weights_only=False
    )
    fine_survival = torch.linspace(1.0, 0.01, cfg.n_t)

    final = SAVE_DIR / "mask_gen_fastmri_final.pth"
    status_path = SAVE_DIR / "train_status.json"
    log_path = SAVE_DIR / cfg.log_file
    if args.eval_only:
        if not final.is_file():
            raise SystemExit(f"--eval-only but missing {final}")
        print(f"skip train (--eval-only): {final}")
    else:
        existing = _completed_train_status(final, status_path, cfg.n_epochs)
        if existing is not None:
            if existing.get("collapsed"):
                print("existing A2 run collapsed; not retraining")
            else:
                print(
                    f"skip train: {final} exists with "
                    f"epochs_completed={existing.get('epochs_completed')}"
                )
        else:
            if final.is_file() or status_path.is_file():
                print(
                    "incomplete A2 artifacts (ckpt/status missing or stale); "
                    "training from parent warm-start"
                )
            model = _load_parent_weights(cfg)
            print(
                "training ACS-lock remainder from parent warm-start "
                f"({cfg.n_epochs} epochs, save_every={cfg.save_every})"
            )
            train_fastmri_nested(
                cfg,
                model=model,
                dataloader=dataloader,
                d3pm_coarse=d3pm_coarse,
                d3pm_fine=None,
                fine_survival=fine_survival,
                coarse_survival=coarse_survival,
            )

    parent_md5_after = _md5(PARENT_FINAL)
    parent_mtime_after = PARENT_FINAL.stat().st_mtime
    if parent_md5_after != parent_md5_before or parent_mtime_after != parent_mtime_before:
        raise SystemExit("parent checkpoint was modified during A2; abort")
    if PARENT_FINAL.resolve() == (SAVE_DIR / "mask_gen_fastmri_final.pth").resolve():
        raise SystemExit("parent checkpoint path collided with A2 save_dir")

    status = _load_train_status(status_path, log_path)
    print("train_status", {k: status[k] for k in status if k != "epochs"})
    if status.get("collapsed"):
        print("COLLAPSED: density near 0; stopping before eval")
        return

    model = _load_mask_net(cfg)
    out = SAVE_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    print("eval A2 ACS-lock Gumbel + top-k vs ACS remainder heuristics", A2_S)
    if not bool(cfg.acs_lock):
        raise SystemExit("A2 eval requires acs_lock=True for train-consistent Gumbel")
    report = evaluate_fastmri_baselines_loader(
        model,
        None,
        d3pm_coarse,
        cfg,
        dataloader,
        fine_survival,
        coarse_survival,
        sparsities=A2_S,
        max_batches=10,
        baselines=A2_BASELINES,
        include_learned_acs_lock=True,
    )
    combined = {
        **report,
        "train": {k: status[k] for k in status if k != "epochs"},
        "train_epochs": status.get("epochs", []),
        "init": "warm_start_parent",
        "parent_ckpt": str(PARENT_FINAL),
        "parent_md5": parent_md5_after,
        "acs_lock": True,
        "a1_posthoc_s025": A1_POSTHOC_S025,
    }
    save_eval_report(combined, out / "eval_a2.json")
    print("eval_a2", report)
    _plot_grids(cfg, model, out)
    print("STAGE A2 DONE wrote", out)


if __name__ == "__main__":
    main()
