"""Stage 4: retrain nested policy with ACS k-space conditioning.

Never writes models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth.
"""

from __future__ import annotations

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
SAVE_DIR = PARENT / "acs_kspace_input"
STAGE4_S = (0.1, 0.25, 0.4)
STAGE4_BASELINES = ("random", "equispaced", "vd_gaussian", "acs_random")
PLOT_METHODS = ("learned",) + STAGE4_BASELINES


def _assert_safe_save_dir(save_dir: Path) -> None:
    resolved = save_dir.resolve()
    if resolved == PARENT.resolve():
        raise SystemExit(f"refusing to train into parent {PARENT}")
    collapsed = Path("models_mask_gen_fastmri")
    if resolved == collapsed.resolve() or save_dir.name == "models_mask_gen_fastmri":
        raise SystemExit("refusing collapsed save_dir models_mask_gen_fastmri/")


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
        policy_input="acs_kspace",
        save_dir=save_dir,
    )


def _load_mask_net(cfg: FastMRIConfig) -> torch.nn.Module:
    ckpt_path = Path(cfg.save_dir) / "mask_gen_fastmri_final.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
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
        for name in STAGE4_BASELINES:
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
                title=f"{name} row mask s=0.25 (ACS k-space policy)",
                save_path=dest,
                kspace=kspace,
            )
    plt.close("all")


def main() -> None:
    _assert_safe_save_dir(SAVE_DIR)
    if not PARENT_FINAL.is_file():
        raise SystemExit(f"missing parent checkpoint {PARENT_FINAL}")
    if not COARSE_SURVIVAL_PT.is_file():
        raise SystemExit(f"missing {COARSE_SURVIVAL_PT}; pass it in, do not rebuild")

    cfg = _make_cfg(save_dir=str(SAVE_DIR))
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"device={cfg.device} policy_input={cfg.policy_input} "
        f"save_dir={cfg.save_dir} data_root={cfg.data_root}"
    )

    dataloader = build_dataloader(cfg)
    d3pm_coarse = load_d3pm_coarse(cfg)
    coarse_survival = torch.load(
        COARSE_SURVIVAL_PT, map_location=cfg.device, weights_only=False
    )
    fine_survival = torch.linspace(1.0, 0.01, cfg.n_t)

    final = SAVE_DIR / "mask_gen_fastmri_final.pth"
    status_path = SAVE_DIR / "train_status.json"
    if final.is_file() and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("collapsed"):
            print("existing ACS run collapsed; not retraining")
        else:
            print(f"skip train: {final} exists")
    else:
        print("training ACS-kspace policy (both recipe, 10 epochs)")
        train_fastmri_nested(
            cfg,
            dataloader=dataloader,
            d3pm_coarse=d3pm_coarse,
            d3pm_fine=None,
            fine_survival=fine_survival,
            coarse_survival=coarse_survival,
        )

    if PARENT_FINAL.resolve() == (SAVE_DIR / "mask_gen_fastmri_final.pth").resolve():
        raise SystemExit("parent checkpoint path collided with ACS save_dir")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    print("train_status", {k: status[k] for k in status if k != "epochs"})
    if status.get("collapsed"):
        print("COLLAPSED: density near 0; stopping before eval")
        return

    model = _load_mask_net(cfg)
    out = SAVE_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    print("eval ACS-input learned vs MRI row-mask baselines")
    report = evaluate_fastmri_baselines_loader(
        model,
        None,
        d3pm_coarse,
        cfg,
        dataloader,
        fine_survival,
        coarse_survival,
        sparsities=STAGE4_S,
        max_batches=10,
        baselines=STAGE4_BASELINES,
    )
    save_eval_report(report, out / "eval_baselines.json")
    print("eval_baselines", report)
    _plot_grids(cfg, model, out)
    print("STAGE 4 DONE wrote", out)


if __name__ == "__main__":
    main()
