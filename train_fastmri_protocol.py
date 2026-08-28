"""Stage 6: 30-epoch image-Z `both` protocol (from scratch).

Never writes models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth.
Never loads models_mask_gen_fastmri/ or acs_kspace_input/ as train init.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from itw.configs import FastMRIConfig, default_device
from itw.data.fastmri import FastMRIDataset
from itw.eval import (
    baseline_row_mask_batch,
    evaluate_fastmri_baselines_loader,
    plot_fastmri_grid,
    save_eval_report,
    sparsity_key,
)
from itw.masks import apply_kspace_row_mask, gumbel_row_mask
from itw.train import (
    build_dataloader,
    build_mask_model,
    load_d3pm_coarse,
    mask_generator_cond,
    train_fastmri_nested,
)

PARENT = Path("models_mask_gen_fastmri_nested")
PARENT_FINAL = PARENT / "mask_gen_fastmri_final.pth"
COARSE_SURVIVAL_PT = PARENT / "coarse_survival_table.pt"
SAVE_DIR = PARENT / "protocol_30ep"
STAGE6_S = (0.1, 0.125, 0.25, 0.4)
STAGE6_BASELINES = ("random", "equispaced", "vd_gaussian", "acs_random")
PLOT_METHODS = ("learned",) + STAGE6_BASELINES
CURVE_METHODS = ("learned", "acs_random", "vd_gaussian", "random")

# Stage 3 image-Z `both` (10 batches, 10-epoch ckpt). No s=0.125.
STAGE3_LEARNED = {
    0.10: {"nmse": 0.191, "ssim": 0.446, "psnr": 19.46, "h_coarse": 0.156},
    0.25: {"nmse": 0.086, "ssim": 0.587, "psnr": 24.13, "h_coarse": 0.107},
    0.40: {"nmse": 0.029, "ssim": 0.743, "psnr": 29.86, "h_coarse": 0.107},
}


def _assert_safe_save_dir(save_dir: Path) -> None:
    resolved = save_dir.resolve()
    if resolved == PARENT.resolve():
        raise SystemExit(f"refusing to train into parent {PARENT}")
    collapsed = Path("models_mask_gen_fastmri")
    if resolved == collapsed.resolve() or save_dir.name == "models_mask_gen_fastmri":
        raise SystemExit("refusing collapsed save_dir models_mask_gen_fastmri/")
    acs = PARENT / "acs_kspace_input"
    if resolved == acs.resolve():
        raise SystemExit("refusing ACS-kspace save_dir as protocol default")


def _make_cfg(*, save_dir: str) -> FastMRIConfig:
    return FastMRIConfig(
        device=default_device(),
        mask_objective="nested_d3pm",
        n_epochs=30,
        batch_size=8,
        num_workers=4,
        save_every=5,
        lr=1e-4,
        sparsity_loss_weight=50.0,
        sparsity_min=0.1,
        sparsity_max=0.4,
        entropy_alpha=1.0,
        entropy_beta=0.0,
        recon_loss_weight=1.0,
        policy_input="scout_image",
        save_dir=save_dir,
    )


def _load_mask_net(cfg: FastMRIConfig) -> torch.nn.Module:
    ckpt_path = Path(cfg.save_dir) / "mask_gen_fastmri_final.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    if ckpt_path.resolve() == PARENT_FINAL.resolve():
        raise SystemExit("refusing to load parent nested checkpoint as protocol ckpt")
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
    model.eval()
    with torch.no_grad():
        z_d = z.to(cfg.device)
        k_d = kspace.to(cfg.device)
        cond = mask_generator_cond(cfg, z_d, k_d)
        row_logits = model(cond, sparsity)
        named_masks = {
            "learned": gumbel_row_mask(row_logits, temperature=0.5, hard=True)
        }
        for name in STAGE6_BASELINES:
            named_masks[name] = baseline_row_mask_batch(
                name,
                z.shape[0],
                cfg.image_size,
                sparsity,
                cfg.device,
                acs_width=int(cfg.scout_size),
            )
        for name in PLOT_METHODS:
            rows = named_masks[name]
            y = apply_kspace_row_mask(k_d, rows)
            dest = out / f"{name}_s025.png"
            plot_fastmri_grid(
                z,
                c,
                rows.cpu(),
                y.cpu(),
                title=f"{name} row mask s=0.25 (30-epoch image-Z)",
                save_path=dest,
                kspace=kspace,
            )
    plt.close("all")


def _s_from_key(key: str) -> float:
    return float(key.split("=", 1)[1])


def _plot_curves(report: dict, out: Path) -> None:
    s_vals = [_s_from_key(k) for k in report if k.startswith("s=")]
    s_vals.sort()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    style = {
        "learned": {"label": "learned_30ep", "marker": "o", "color": "C0"},
        "acs_random": {"label": "ACS+random", "marker": "s", "color": "C1"},
        "vd_gaussian": {"label": "vd_gaussian", "marker": "^", "color": "C2"},
        "random": {"label": "random", "marker": "x", "color": "C3"},
    }
    for metric, ax in zip(("nmse", "ssim"), axes):
        for name in CURVE_METHODS:
            ys = [report[sparsity_key(s)][name][metric] for s in s_vals]
            ax.plot(s_vals, ys, **style[name])
        s3 = sorted(STAGE3_LEARNED)
        ax.plot(
            s3,
            [STAGE3_LEARNED[s][metric] for s in s3],
            linestyle="--",
            marker="D",
            color="C0",
            alpha=0.7,
            label="learned_10ep (Stage 3)",
        )
        ax.set_xlabel("sparsity s")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(s_vals)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Stage 6: 30-epoch image-Z vs baselines")
    fig.tight_layout()
    dest = out / "nmse_ssim_vs_s.png"
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print("wrote", dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 FastMRI 30-epoch protocol")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate existing protocol_30ep checkpoint",
    )
    args = parser.parse_args()

    _assert_safe_save_dir(SAVE_DIR)
    if PARENT_FINAL.resolve() == (SAVE_DIR / "mask_gen_fastmri_final.pth").resolve():
        raise SystemExit("parent checkpoint path collided with protocol save_dir")
    if not COARSE_SURVIVAL_PT.is_file():
        raise SystemExit(f"missing {COARSE_SURVIVAL_PT}; pass it in, do not rebuild")

    cfg = _make_cfg(save_dir=str(SAVE_DIR))
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"device={cfg.device} policy_input={cfg.policy_input} "
        f"n_epochs={cfg.n_epochs} save_dir={cfg.save_dir} data_root={cfg.data_root} "
        "init=from_scratch"
    )

    dataloader = build_dataloader(cfg)
    d3pm_coarse = load_d3pm_coarse(cfg)
    coarse_survival = torch.load(
        COARSE_SURVIVAL_PT, map_location=cfg.device, weights_only=False
    )
    fine_survival = torch.linspace(1.0, 0.01, cfg.n_t)

    final = SAVE_DIR / "mask_gen_fastmri_final.pth"
    status_path = SAVE_DIR / "train_status.json"
    if args.eval_only:
        if not final.is_file():
            raise SystemExit(f"--eval-only but missing {final}")
        print(f"skip train (--eval-only): {final}")
    elif final.is_file() and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("collapsed"):
            print("existing protocol run collapsed; not retraining")
        else:
            print(f"skip train: {final} exists")
    else:
        print(
            "training image-Z both recipe from scratch "
            f"({cfg.n_epochs} epochs, save_every={cfg.save_every})"
        )
        train_fastmri_nested(
            cfg,
            dataloader=dataloader,
            d3pm_coarse=d3pm_coarse,
            d3pm_fine=None,
            fine_survival=fine_survival,
            coarse_survival=coarse_survival,
        )

    if PARENT_FINAL.resolve() == (SAVE_DIR / "mask_gen_fastmri_final.pth").resolve():
        raise SystemExit("parent checkpoint path collided with protocol save_dir")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    print("train_status", {k: status[k] for k in status if k != "epochs"})
    if status.get("collapsed"):
        print("COLLAPSED: density near 0; stopping before eval")
        return

    model = _load_mask_net(cfg)
    out = SAVE_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    print("eval 30-epoch learned vs MRI row-mask baselines", STAGE6_S)
    report = evaluate_fastmri_baselines_loader(
        model,
        None,
        d3pm_coarse,
        cfg,
        dataloader,
        fine_survival,
        coarse_survival,
        sparsities=STAGE6_S,
        max_batches=10,
        baselines=STAGE6_BASELINES,
    )
    combined = {
        **report,
        "train": {k: status[k] for k in status if k != "epochs"},
        "train_epochs": status.get("epochs", []),
        "init": "from_scratch",
        "stage3_learned_overlay": STAGE3_LEARNED,
    }
    save_eval_report(combined, out / "eval_baselines.json")
    print("eval_baselines", report)
    _plot_grids(cfg, model, out)
    _plot_curves(report, out)
    print("STAGE 6 DONE wrote", out)


if __name__ == "__main__":
    main()
