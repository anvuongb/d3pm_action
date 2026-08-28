"""Load nested row-mask policy (never models_mask_gen_fastmri/), then eval/plot."""

from __future__ import annotations

import argparse
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
from itw.masks import apply_kspace_row_mask, gumbel_row_mask
from itw.schedule import build_row_survival_table
from itw.train import (
    build_dataloader,
    build_mask_model,
    load_d3pm_coarse,
    load_d3pm_fine,
    mask_generator_cond,
    train_fastmri,
)

STAGE1_SPARSITIES = (0.1, 0.25, 0.4)
STAGE1_BASELINES = ("random", "equispaced", "vd_gaussian", "acs_random")
PLOT_METHODS = ("learned",) + STAGE1_BASELINES

A1_SPARSITIES = (0.10, 0.125, 0.25, 0.4)
A1_BASELINES = (
    "random",
    "equispaced",
    "vd_gaussian",
    "acs_random",
    "acs_vd_gaussian",
    "acs_equispaced",
)
A1_PLOT_METHODS = ("learned", "learned_acs_lock", "acs_random", "acs_vd_gaussian")


def _load_or_train(cfg: FastMRIConfig, *, allow_train: bool = True):
    save = Path(cfg.save_dir)
    final = save / "mask_gen_fastmri_final.pth"
    if final.is_file():
        print(f"loading nested checkpoint {final}")
        model = build_mask_model(cfg)
        ckpt = torch.load(final, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["mask_generator"])
        model.eval()
        fine_pt = save / "fine_survival_table.pt"
        coarse_pt = save / "coarse_survival_table.pt"
        fine_s = (
            torch.load(fine_pt, map_location="cpu", weights_only=False)
            if fine_pt.is_file()
            else None
        )
        coarse_s = (
            torch.load(coarse_pt, map_location="cpu", weights_only=False)
            if coarse_pt.is_file()
            else None
        )
        return model, {"fine_survival": fine_s, "coarse_survival": coarse_s}
    if not allow_train:
        raise FileNotFoundError(
            f"Missing {final}; pass without --eval-only to train, "
            "or point save_dir at models_mask_gen_fastmri_nested/"
        )
    print("no valid nested checkpoint; training mask policy")
    return train_fastmri(cfg)


def _ensure_survival_tables(
    cfg: FastMRIConfig,
    tables: dict,
    dataloader,
    d3pm_coarse,
) -> dict[str, torch.Tensor]:
    save = Path(cfg.save_dir)
    fine_s = tables.get("fine_survival")
    coarse_s = tables.get("coarse_survival")
    if fine_s is None:
        print("no fine_survival_table.pt; dummy linspace (entropy_beta=0)")
        fine_s = torch.linspace(1.0, 0.01, cfg.n_t)
    if coarse_s is None:
        print("no coarse_survival_table.pt; rebuilding with build_row_survival_table")
        coarse_s = build_row_survival_table(
            d3pm_coarse,
            dataloader,
            device=cfg.device,
            n_bins=cfg.num_classes,
            max_batches=cfg.schedule_calibration_batches,
        )
        torch.save(coarse_s.detach().cpu(), save / "coarse_survival_table.pt")
    return {
        "fine_survival": fine_s.to(cfg.device),
        "coarse_survival": coarse_s.to(cfg.device),
    }


def _plot_mask_grid(
    z: torch.Tensor,
    c: torch.Tensor,
    kspace: torch.Tensor,
    rows: torch.Tensor,
    title: str,
    save_path: Path,
) -> None:
    y = apply_kspace_row_mask(kspace.to(rows.device), rows)
    plot_fastmri_grid(
        z,
        c,
        rows.cpu(),
        y.cpu(),
        title=title,
        save_path=save_path,
        kspace=kspace,
    )


def _plot_stage1_grids(cfg: FastMRIConfig, model, out: Path) -> None:
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
        for name in STAGE1_BASELINES:
            named_masks[name] = baseline_row_mask_batch(
                name,
                z.shape[0],
                cfg.image_size,
                sparsity,
                cfg.device,
                acs_width=int(cfg.scout_size),
            )
        for name in PLOT_METHODS:
            _plot_mask_grid(
                z,
                c,
                k_d,
                named_masks[name],
                title=f"{name} row mask s=0.25",
                save_path=out / f"{name}_s025.png",
            )
    plt.close("all")


def _plot_a1_grids(cfg: FastMRIConfig, model, out: Path) -> None:
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
            "learned": gumbel_row_mask(row_logits, temperature=0.5, hard=True),
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
        for name in A1_PLOT_METHODS:
            _plot_mask_grid(
                z,
                c,
                k_d,
                named_masks[name],
                title=f"{name} row mask s=0.25",
                save_path=out / f"{name}_s025.png",
            )
    plt.close("all")


def _plot_a1_grids(cfg: FastMRIConfig, model, out: Path) -> None:
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
            "learned": gumbel_row_mask(row_logits, temperature=0.5, hard=True),
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
        for name in A1_PLOT_METHODS:
            _plot_mask_grid(
                z,
                c,
                k_d,
                named_masks[name],
                title=f"{name} row mask s=0.25",
                save_path=out / f"{name}_s025.png",
            )
    plt.close("all")


ABLATION_POLICIES = (
    ("nmse_only", Path("models_mask_gen_fastmri_nested/ablation_nmse_only")),
    ("hcoarse_only", Path("models_mask_gen_fastmri_nested/ablation_hcoarse_only")),
)


def _load_mask_from_dir(cfg: FastMRIConfig, save_dir: Path) -> torch.nn.Module:
    ckpt_path = save_dir / "mask_gen_fastmri_final.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    model = build_mask_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["mask_generator"])
    model.eval()
    return model


def _eval_and_plot_ablations(
    cfg: FastMRIConfig,
    dataloader,
    d3pm_coarse,
    fine_s: torch.Tensor,
    coarse_s: torch.Tensor,
    out: Path,
) -> dict[str, dict]:
    """Load existing ablation ckpts (no train); eval learned at s=0.25; honest plots."""
    viz_ds = FastMRIDataset(cfg.data_root)
    viz_dl = DataLoader(viz_ds, batch_size=4, shuffle=False, num_workers=0)
    z, c, kspace = next(iter(viz_dl))
    ablations: dict[str, dict] = {}
    for name, save_dir in ABLATION_POLICIES:
        ckpt = save_dir / "mask_gen_fastmri_final.pth"
        if not ckpt.is_file():
            print(f"skip ablation {name}: missing {ckpt}")
            continue
        print(f"Stage 3 ablation eval {name} from {ckpt}")
        model = _load_mask_from_dir(cfg, save_dir)
        report = evaluate_fastmri_baselines_loader(
            model,
            None,
            d3pm_coarse,
            cfg,
            dataloader,
            fine_s,
            coarse_s,
            sparsities=(0.25,),
            max_batches=10,
            baselines=(),
        )
        ablations[name] = report["s=0.25"]["learned"]
        dests = [
            out / f"learned_{name}_s025.png",
            save_dir / "eval" / "learned_s025.png",
        ]
        sparsity = torch.full((z.shape[0],), 0.25, device=cfg.device)
        with torch.no_grad():
            row_logits = model(
                mask_generator_cond(cfg, z.to(cfg.device), kspace.to(cfg.device)),
                sparsity,
            )
            rows = gumbel_row_mask(row_logits, temperature=0.5, hard=True)
            for dest in dests:
                dest.parent.mkdir(parents=True, exist_ok=True)
                _plot_mask_grid(
                    z,
                    c,
                    kspace,
                    rows,
                    title=f"{name} learned row mask s=0.25",
                    save_path=dest,
                )
        plt.close("all")
    return ablations


def main() -> None:
    parser = argparse.ArgumentParser(description="FastMRI nested policy eval")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load checkpoint only; never start a training run",
    )
    parser.add_argument(
        "--acs-lock-diag",
        action="store_true",
        help="Stage A1 load-only ACS-lock diagnostic; never train",
    )
    args = parser.parse_args()

    cfg = FastMRIConfig(
        device=default_device(),
        mask_objective="nested_d3pm",
        n_epochs=10,
        batch_size=8,
        num_workers=4,
        save_every=1,
        policy_input="scout_image",
        acs_lock=False,
    )
    if args.acs_lock_diag:
        out = Path(cfg.save_dir) / "eval_acs_lock"
    else:
        out = Path(cfg.save_dir) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={cfg.device} save_dir={cfg.save_dir} data_root={cfg.data_root}")
    if str(cfg.save_dir).rstrip("/") == "models_mask_gen_fastmri":
        raise SystemExit("refusing collapsed save_dir models_mask_gen_fastmri/")

    allow_train = (not args.eval_only) and (not args.acs_lock_diag)
    model, tables = _load_or_train(cfg, allow_train=allow_train)
    d3pm_fine = load_d3pm_fine(cfg) if cfg.entropy_beta > 0 else None
    d3pm_coarse = load_d3pm_coarse(cfg)
    dataloader = build_dataloader(cfg)
    tables = _ensure_survival_tables(cfg, tables, dataloader, d3pm_coarse)
    fine_s = tables["fine_survival"]
    coarse_s = tables["coarse_survival"]
    print(
        "survival fine",
        float(fine_s.min()),
        float(fine_s.max()),
        "coarse",
        float(coarse_s.min()),
        float(coarse_s.max()),
    )

    if args.acs_lock_diag:
        print("Stage A1: ACS-lock diagnostic (load-only, no train)")
        if bool(cfg.acs_lock):
            raise SystemExit("A1 parent eval requires acs_lock=False")
        report = evaluate_fastmri_baselines_loader(
            model,
            d3pm_fine,
            d3pm_coarse,
            cfg,
            dataloader,
            fine_s,
            coarse_s,
            sparsities=A1_SPARSITIES,
            max_batches=10,
            baselines=A1_BASELINES,
            include_learned_acs_lock=True,
        )
        save_eval_report(report, out / "eval_a1.json")
        print("eval_a1", report)
        _plot_a1_grids(cfg, model, out)
        print("wrote", out)
        return

    print("Stage 3: learned vs MRI row-mask baselines (NMSE/SSIM/PSNR)")
    report = evaluate_fastmri_baselines_loader(
        model,
        d3pm_fine,
        d3pm_coarse,
        cfg,
        dataloader,
        fine_s,
        coarse_s,
        sparsities=STAGE1_SPARSITIES,
        max_batches=10,
        baselines=STAGE1_BASELINES,
    )
    ablations = _eval_and_plot_ablations(
        cfg, dataloader, d3pm_coarse, fine_s, coarse_s, out
    )
    if ablations:
        report["ablations"] = {"s=0.25": ablations}
    save_eval_report(report, out / "eval_baselines.json")
    print("eval_baselines", report)

    _plot_stage1_grids(cfg, model, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
