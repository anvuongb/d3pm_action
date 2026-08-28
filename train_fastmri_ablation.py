"""Stage 2 FastMRI loss ablations: train nmse_only / hcoarse_only, eval all three.

Does not retrain `both` (existing nested checkpoint) and never writes
`models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth`.
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
STAGE2_S = (0.25,)
STAGE2_BASELINES = ("random", "equispaced", "vd_gaussian", "acs_random")

ABLATIONS = (
    {
        "name": "nmse_only",
        "entropy_alpha": 0.0,
        "entropy_beta": 0.0,
        "recon_loss_weight": 1.0,
        "save_dir": str(PARENT / "ablation_nmse_only"),
    },
    {
        "name": "hcoarse_only",
        "entropy_alpha": 1.0,
        "entropy_beta": 0.0,
        "recon_loss_weight": 0.0,
        "save_dir": str(PARENT / "ablation_hcoarse_only"),
    },
)

POLICIES = (
    {"name": "both", "save_dir": str(PARENT), "train": False},
    *({**run, "train": True} for run in ABLATIONS),
)


def _assert_safe_save_dir(save_dir: Path) -> None:
    resolved = save_dir.resolve()
    if resolved == PARENT.resolve():
        raise SystemExit(f"refusing to train into parent {PARENT} (would overwrite both)")
    collapsed = Path("models_mask_gen_fastmri")
    if resolved == collapsed.resolve() or save_dir.name == "models_mask_gen_fastmri":
        raise SystemExit("refusing collapsed save_dir models_mask_gen_fastmri/")


def _load_mask_net(cfg: FastMRIConfig) -> torch.nn.Module:
    ckpt_path = Path(cfg.save_dir) / "mask_gen_fastmri_final.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    model = build_mask_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["mask_generator"])
    model.eval()
    return model


def _parse_both_train_log(path: Path) -> dict:
    epochs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("epoch:"):
            continue
        parts = {}
        for chunk in line.split(","):
            if ":" not in chunk:
                continue
            k, v = chunk.split(":", 1)
            parts[k.strip()] = float(v.strip())
        epochs.append(parts)
    last = epochs[-1] if epochs else {}
    return {
        "collapsed": False,
        "retrained": False,
        "epochs_completed": len(epochs),
        "final_loss_ema": last.get("loss"),
        "final_density_ema": last.get("density"),
        "final_h_coarse_mean": last.get("H_coarse"),
        "final_nmse_mean": last.get("nmse"),
        "epochs": epochs,
    }


def _make_cfg(*, save_dir: str, alpha: float, beta: float, recon: float) -> FastMRIConfig:
    return FastMRIConfig(
        device=default_device(),
        mask_objective="nested_d3pm",
        n_epochs=10,
        batch_size=8,
        num_workers=4,
        save_every=1,
        sparsity_loss_weight=50.0,
        sparsity_min=0.1,
        sparsity_max=0.4,
        entropy_alpha=alpha,
        entropy_beta=beta,
        recon_loss_weight=recon,
        save_dir=save_dir,
    )


def _train_ablation(
    run: dict,
    *,
    dataloader,
    d3pm_coarse,
    coarse_survival: torch.Tensor,
) -> dict:
    save_dir = Path(run["save_dir"])
    _assert_safe_save_dir(save_dir)
    if save_dir.resolve() == PARENT.resolve():
        raise SystemExit("refusing to overwrite parent nested checkpoint")
    final = save_dir / "mask_gen_fastmri_final.pth"
    cfg = _make_cfg(
        save_dir=str(save_dir),
        alpha=run["entropy_alpha"],
        beta=run["entropy_beta"],
        recon=run["recon_loss_weight"],
    )
    if final.is_file():
        print(f"skip train {run['name']}: {final} exists")
        status_path = save_dir / "train_status.json"
        if status_path.is_file():
            return json.loads(status_path.read_text(encoding="utf-8"))
        return {"collapsed": False, "skipped": True, "epochs_completed": None}

    print(
        f"training {run['name']} alpha={cfg.entropy_alpha} beta={cfg.entropy_beta} "
        f"recon={cfg.recon_loss_weight} save_dir={cfg.save_dir}"
    )
    fine_survival = torch.linspace(1.0, 0.01, cfg.n_t)
    train_fastmri_nested(
        cfg,
        dataloader=dataloader,
        d3pm_coarse=d3pm_coarse,
        d3pm_fine=None,
        fine_survival=fine_survival,
        coarse_survival=coarse_survival,
    )
    status_path = save_dir / "train_status.json"
    if status_path.is_file():
        return json.loads(status_path.read_text(encoding="utf-8"))
    return {"collapsed": False, "epochs_completed": cfg.n_epochs}


def _eval_policy(
    name: str,
    save_dir: str,
    *,
    dataloader,
    d3pm_coarse,
    fine_survival: torch.Tensor,
    coarse_survival: torch.Tensor,
) -> dict:
    cfg = FastMRIConfig(
        device=default_device(),
        mask_objective="nested_d3pm",
        batch_size=8,
        num_workers=4,
        entropy_beta=0.0,
        save_dir=save_dir,
    )
    model = _load_mask_net(cfg)
    print(f"eval {name} from {Path(save_dir) / 'mask_gen_fastmri_final.pth'}")
    report = evaluate_fastmri_baselines_loader(
        model,
        None,
        d3pm_coarse,
        cfg,
        dataloader,
        fine_survival,
        coarse_survival,
        sparsities=STAGE2_S,
        max_batches=10,
        baselines=STAGE2_BASELINES,
    )
    return report


def _plot_learned(
    name: str,
    save_dir: str,
    z: torch.Tensor,
    c: torch.Tensor,
    kspace: torch.Tensor,
    device: str,
    out_paths: list[Path],
) -> None:
    cfg = FastMRIConfig(device=device, save_dir=save_dir)
    model = _load_mask_net(cfg)
    sparsity = torch.full((z.shape[0],), 0.25, device=device)
    with torch.no_grad():
        row_logits = model(
            mask_generator_cond(cfg, z.to(device), kspace.to(device)),
            sparsity,
        )
        rows = gumbel_row_mask(row_logits, temperature=0.5, hard=True)
        y = apply_kspace_row_mask(kspace.to(device), rows)
    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        plot_fastmri_grid(
            z,
            c,
            rows.cpu(),
            y.cpu(),
            title=f"{name} learned row mask s=0.25",
            save_path=path,
            kspace=kspace,
        )
    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 FastMRI loss ablation")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate existing ablation checkpoints",
    )
    args = parser.parse_args()

    if not PARENT_FINAL.is_file():
        raise SystemExit(f"missing both checkpoint {PARENT_FINAL}")
    if not COARSE_SURVIVAL_PT.is_file():
        raise SystemExit(f"missing {COARSE_SURVIVAL_PT}; do not rebuild for Stage 2")

    device = default_device()
    print(f"device={device}")

    parent_cfg = FastMRIConfig(
        device=device,
        mask_objective="nested_d3pm",
        n_epochs=10,
        batch_size=8,
        num_workers=4,
        save_dir=str(PARENT),
    )
    dataloader = build_dataloader(parent_cfg)
    d3pm_coarse = load_d3pm_coarse(parent_cfg)
    coarse_survival = torch.load(
        COARSE_SURVIVAL_PT, map_location=device, weights_only=False
    )
    fine_survival = torch.linspace(1.0, 0.01, parent_cfg.n_t)

    train_summaries: dict[str, dict] = {
        "both": _parse_both_train_log(PARENT / "train.log"),
    }

    if not args.eval_only:
        for run in ABLATIONS:
            train_summaries[run["name"]] = _train_ablation(
                run,
                dataloader=dataloader,
                d3pm_coarse=d3pm_coarse,
                coarse_survival=coarse_survival,
            )
    else:
        for run in ABLATIONS:
            status_path = Path(run["save_dir"]) / "train_status.json"
            if status_path.is_file():
                train_summaries[run["name"]] = json.loads(
                    status_path.read_text(encoding="utf-8")
                )
            else:
                train_summaries[run["name"]] = {"missing_status": True}

    eval_reports: dict[str, dict] = {}
    for pol in POLICIES:
        report = _eval_policy(
            pol["name"],
            pol["save_dir"],
            dataloader=dataloader,
            d3pm_coarse=d3pm_coarse,
            fine_survival=fine_survival,
            coarse_survival=coarse_survival,
        )
        eval_reports[pol["name"]] = report
        out = Path(pol["save_dir"]) / "eval"
        if pol["name"] != "both":
            save_eval_report(report, out / "eval_baselines.json")

    s_key = "s=0.25"
    combined_s = {}
    for pol_name, report in eval_reports.items():
        combined_s[pol_name] = report[s_key]["learned"]
    for bname in STAGE2_BASELINES:
        combined_s[bname] = eval_reports["both"][s_key][bname]
    combined = {
        "s=0.25": combined_s,
        "train": train_summaries,
        "per_policy_eval": eval_reports,
    }
    combined_path = PARENT / "eval" / "eval_ablation_stage2.json"
    save_eval_report(combined, combined_path)
    print("wrote", combined_path)

    viz_ds = FastMRIDataset(parent_cfg.data_root)
    viz_dl = DataLoader(viz_ds, batch_size=4, shuffle=False, num_workers=0)
    z, c, kspace = next(iter(viz_dl))
    parent_eval = PARENT / "eval"
    _plot_learned(
        "both",
        str(PARENT),
        z,
        c,
        kspace,
        device,
        [parent_eval / "learned_both_s025.png"],
    )
    _plot_learned(
        "nmse_only",
        str(PARENT / "ablation_nmse_only"),
        z,
        c,
        kspace,
        device,
        [
            PARENT / "ablation_nmse_only" / "eval" / "learned_s025.png",
            parent_eval / "learned_nmse_only_s025.png",
        ],
    )
    _plot_learned(
        "hcoarse_only",
        str(PARENT / "ablation_hcoarse_only"),
        z,
        c,
        kspace,
        device,
        [
            PARENT / "ablation_hcoarse_only" / "eval" / "learned_s025.png",
            parent_eval / "learned_hcoarse_only_s025.png",
        ],
    )
    print("STAGE 2 DONE")


if __name__ == "__main__":
    main()
