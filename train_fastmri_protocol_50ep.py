"""Stage P3: 50-epoch from-scratch ACS-lock policy (A4 recipe, new 100ep coarse).

Never writes models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth.
Never writes into acs_lock_10ep/, models_d3pm_fastmri_coarse_kspace/, or
models_d3pm_fastmri_fine/. Does not load acs_lock_10ep/ or the parent nested
ckpt. Rebuilds coarse survival from the 100ep k-space coarse (does not copy
the old 50ep coarse_survival_table.pt). Fine survival is dummy linspace (β=0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from itw.configs import FastMRIConfig, default_device
from itw.eval import evaluate_fastmri_baselines_loader, save_eval_report
from itw.masks import acs_bounds
from itw.schedule import build_row_survival_table
from itw.train import (
    _row_mask_ste,
    build_dataloader,
    build_mask_model,
    load_d3pm_coarse,
    train_fastmri_nested,
)

PARENT = Path("models_mask_gen_fastmri_nested")
PARENT_FINAL = PARENT / "mask_gen_fastmri_final.pth"
OLD_COARSE_SURVIVAL = PARENT / "coarse_survival_table.pt"
ACS_LOCK_10EP = PARENT / "acs_lock_10ep"
OLD_COARSE_DIR = Path("models_d3pm_fastmri_coarse_kspace")
OLD_FINE_DIR = Path("models_d3pm_fastmri_fine")
NEW_COARSE_CKPT = Path(
    "models_d3pm_fastmri_coarse_kspace_100ep/model_absorb_cosine_final.pth"
)
SAVE_DIR = PARENT / "protocol_50ep_acs_lock"
PROTECTED = (
    PARENT_FINAL,
    ACS_LOCK_10EP / "mask_gen_fastmri_final.pth",
    OLD_COARSE_DIR / "model_absorb_cosine_final.pth",
    OLD_FINE_DIR / "model_absorb_cosine_final.pth",
    OLD_COARSE_SURVIVAL,
    ACS_LOCK_10EP / "coarse_survival_table.pt",
    ACS_LOCK_10EP / "train_status.json",
)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(path: Path) -> tuple[str, float, int]:
    st = path.stat()
    return _md5(path), st.st_mtime, st.st_size


def _snapshot_protected() -> dict[str, tuple[str, float, int]]:
    snap: dict[str, tuple[str, float, int]] = {}
    for path in PROTECTED:
        if not path.is_file():
            raise SystemExit(f"missing protected artifact {path}")
        snap[str(path)] = _fingerprint(path)
    return snap


def _assert_protected_unchanged(before: dict[str, tuple[str, float, int]]) -> None:
    after = _snapshot_protected()
    for key, prev in before.items():
        if after[key] != prev:
            raise SystemExit(f"protected artifact changed: {key}")


def _assert_safe_save_dir(save_dir: Path) -> None:
    resolved = save_dir.resolve()
    if resolved == PARENT.resolve():
        raise SystemExit(f"refusing to train into parent {PARENT}")
    if resolved == ACS_LOCK_10EP.resolve():
        raise SystemExit(f"refusing to train into {ACS_LOCK_10EP}")
    if resolved == OLD_COARSE_DIR.resolve() or resolved == OLD_FINE_DIR.resolve():
        raise SystemExit(f"refusing to train into D3PM dir {resolved}")
    collapsed = Path("models_mask_gen_fastmri")
    if resolved == collapsed.resolve() or save_dir.name == "models_mask_gen_fastmri":
        raise SystemExit("refusing collapsed save_dir models_mask_gen_fastmri/")
    if (save_dir / "mask_gen_fastmri_final.pth").resolve() == PARENT_FINAL.resolve():
        raise SystemExit("parent checkpoint path collided with P3 save_dir")


def _make_cfg(*, save_dir: str) -> FastMRIConfig:
    return FastMRIConfig(
        device=default_device(),
        mask_objective="nested_d3pm",
        n_epochs=50,
        batch_size=8,
        num_workers=4,
        save_every=5,
        lr=1e-4,
        sparsity_min=0.1,
        sparsity_max=0.75,
        sparsity_loss_weight=50.0,
        entropy_alpha=1.0,
        entropy_beta=0.0,
        recon_loss_weight=1.0,
        policy_input="scout_image",
        acs_lock=True,
        d3pm_coarse_checkpoint=str(NEW_COARSE_CKPT),
        save_dir=save_dir,
    )


def _confirm_acs_lock_ste(cfg: FastMRIConfig) -> None:
    if not bool(cfg.acs_lock):
        raise SystemExit("P3 requires cfg.acs_lock=True")
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


def _completed_train_status(
    final: Path, status_path: Path, n_epochs: int
) -> dict | None:
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
    raise SystemExit(f"missing usable train status ({status_path}) log={log_path}")


def _load_mask_net(cfg: FastMRIConfig) -> torch.nn.Module:
    ckpt_path = Path(cfg.save_dir) / "mask_gen_fastmri_final.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    if ckpt_path.resolve() == PARENT_FINAL.resolve():
        raise SystemExit("refusing to load parent nested checkpoint as P3 ckpt")
    if ckpt_path.resolve() == (ACS_LOCK_10EP / "mask_gen_fastmri_final.pth").resolve():
        raise SystemExit("refusing to load acs_lock_10ep as P3 ckpt")
    model = build_mask_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["mask_generator"])
    model.eval()
    return model


def _log_key_epochs(status: dict) -> None:
    wanted = {0, 9, 24, 49}
    epochs = status.get("epochs") or []
    print("P3 train trajectory (epoch 0/9/24/49):")
    for rec in epochs:
        ep = int(rec.get("epoch", -1))
        if ep not in wanted:
            continue
        print(
            f"  epoch {ep:02d} density={rec.get('density_ema'):.4f} "
            f"H_c={rec.get('h_coarse_mean'):.4f} "
            f"nmse={rec.get('nmse_mean'):.4f} "
            f"loss={rec.get('loss_ema'):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage P3 FastMRI 50-epoch ACS-lock from scratch"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; 5-batch s=0.25 sanity on existing P3 ckpt",
    )
    args = parser.parse_args()

    _assert_safe_save_dir(SAVE_DIR)
    if not NEW_COARSE_CKPT.is_file():
        raise SystemExit(f"missing 100ep coarse ckpt {NEW_COARSE_CKPT}")
    if ACS_LOCK_10EP.resolve() == SAVE_DIR.resolve():
        raise SystemExit("P3 save_dir collided with acs_lock_10ep")

    protected_before = _snapshot_protected()
    print("protected snapshot")
    for path, (md5, mtime, size) in protected_before.items():
        print(f"  {path} md5={md5} mtime={mtime} size={size}")

    cfg = _make_cfg(save_dir=str(SAVE_DIR))
    if abs(float(cfg.entropy_beta) - 0.0) > 1e-12:
        raise SystemExit(f"P3 requires entropy_beta=0, got {cfg.entropy_beta}")
    if Path(cfg.d3pm_coarse_checkpoint).resolve() != NEW_COARSE_CKPT.resolve():
        raise SystemExit("P3 must use 100ep coarse ckpt, not the old 50ep coarse")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    _confirm_acs_lock_ste(cfg)
    print(
        f"device={cfg.device} policy_input={cfg.policy_input} "
        f"acs_lock={cfg.acs_lock} n_epochs={cfg.n_epochs} "
        f"sparsity=[{cfg.sparsity_min},{cfg.sparsity_max}] "
        f"alpha={cfg.entropy_alpha} beta={cfg.entropy_beta} "
        f"recon={cfg.recon_loss_weight} "
        f"coarse={cfg.d3pm_coarse_checkpoint} "
        f"save_dir={cfg.save_dir} data_root={cfg.data_root} "
        "init=from_scratch"
    )

    dataloader = build_dataloader(cfg)
    d3pm_coarse = load_d3pm_coarse(cfg)

    final = SAVE_DIR / "mask_gen_fastmri_final.pth"
    status_path = SAVE_DIR / "train_status.json"
    log_path = SAVE_DIR / cfg.log_file
    coarse_surv_path = SAVE_DIR / "coarse_survival_table.pt"
    fine_surv_path = SAVE_DIR / "fine_survival_table.pt"

    if args.eval_only:
        if not final.is_file():
            raise SystemExit(f"--eval-only but missing {final}")
        print(f"skip train (--eval-only): {final}")
        coarse_survival = torch.load(
            coarse_surv_path, map_location=cfg.device, weights_only=False
        )
        fine_survival = torch.load(
            fine_surv_path, map_location=cfg.device, weights_only=False
        )
    else:
        existing = _completed_train_status(final, status_path, cfg.n_epochs)
        if existing is not None:
            if existing.get("collapsed"):
                print("existing P3 run collapsed; not retraining")
            else:
                print(
                    f"skip train: {final} exists with "
                    f"epochs_completed={existing.get('epochs_completed')}"
                )
            coarse_survival = torch.load(
                coarse_surv_path, map_location=cfg.device, weights_only=False
            )
            fine_survival = torch.load(
                fine_surv_path, map_location=cfg.device, weights_only=False
            )
        else:
            print(
                "rebuilding coarse_survival with build_row_survival_table "
                f"from {NEW_COARSE_CKPT} (not copying {OLD_COARSE_SURVIVAL})"
            )
            coarse_survival = build_row_survival_table(
                d3pm_coarse,
                dataloader,
                device=cfg.device,
                n_bins=cfg.num_classes,
                max_batches=cfg.schedule_calibration_batches,
            )
            fine_survival = torch.linspace(1.0, 0.01, cfg.n_t)
            print(
                "training ACS-lock from scratch "
                f"({cfg.n_epochs} epochs, save_every={cfg.save_every}, "
                "no warm-start)"
            )
            train_fastmri_nested(
                cfg,
                dataloader=dataloader,
                d3pm_coarse=d3pm_coarse,
                d3pm_fine=None,
                fine_survival=fine_survival,
                coarse_survival=coarse_survival,
            )

    _assert_protected_unchanged(protected_before)
    if PARENT_FINAL.resolve() == (SAVE_DIR / "mask_gen_fastmri_final.pth").resolve():
        raise SystemExit("parent checkpoint path collided with P3 save_dir")
    if not final.is_file():
        raise SystemExit(f"P3 final ckpt missing after train: {final}")

    status = _load_train_status(status_path, log_path)
    print("train_status", {k: status[k] for k in status if k != "epochs"})
    _log_key_epochs(status)
    if status.get("collapsed"):
        print("COLLAPSED: density near 0; stopping before sanity eval")
        return

    model = _load_mask_net(cfg)
    out = SAVE_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    print("P3 sanity: 5 batches at s=0.25 (not the P4 10/25/50/75 grid)")
    report = evaluate_fastmri_baselines_loader(
        model,
        None,
        d3pm_coarse,
        cfg,
        dataloader,
        fine_survival,
        coarse_survival,
        sparsities=(0.25,),
        max_batches=5,
        baselines=("acs_random",),
        include_learned_acs_lock=False,
    )
    combined = {
        **report,
        "train": {k: status[k] for k in status if k != "epochs"},
        "train_epochs": status.get("epochs", []),
        "init": "from_scratch",
        "acs_lock": True,
        "entropy_beta": cfg.entropy_beta,
        "d3pm_coarse_checkpoint": cfg.d3pm_coarse_checkpoint,
        "coarse_survival_rebuilt_from": str(NEW_COARSE_CKPT),
        "note": "P3 sanity only; full 10/25/50/75 grid is P4",
    }
    save_eval_report(combined, out / "sanity_s025.json")
    print("sanity_s025", report)
    _assert_protected_unchanged(protected_before)
    print("STAGE P3 DONE wrote", final)


if __name__ == "__main__":
    main()
