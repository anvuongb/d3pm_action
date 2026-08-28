"""Eval + visualize FastMRI D3PM priors (fine 96x96 and coarse row profile)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from d3pm_runner_fastmri import FastMRIDiscDataset
from itw.configs import FastMRIConfig, default_device
from itw.data.fastmri import FastMRIDataset
from itw.discrete import magnitude_to_fine_disc, magnitude_to_row_disc
from itw.eval import plot_fastmri_grid, random_row_mask_batch, save_eval_report
from itw.masks import apply_kspace_row_mask
from itw.train import load_d3pm_coarse, load_d3pm_fine


def _bins_to_unit(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    return x.float() / max(n_bins - 1, 1)


@torch.no_grad()
def ce_at_t(d3pm, x: torch.Tensor, t_val: int, cond: torch.Tensor) -> float:
    t = torch.full((x.shape[0],), t_val, device=x.device, dtype=torch.long)
    noise = torch.rand((*x.shape, d3pm.num_classses), device=x.device)
    xt = d3pm.q_sample(x, t, noise)
    logits = d3pm.model_predict(xt, t, cond)
    return float(
        F.cross_entropy(
            logits.flatten(start_dim=0, end_dim=-2),
            x.flatten(start_dim=0, end_dim=-1),
        )
    )


@torch.no_grad()
def predict_x0(d3pm, x: torch.Tensor, t_val: int, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.full((x.shape[0],), t_val, device=x.device, dtype=torch.long)
    noise = torch.rand((*x.shape, d3pm.num_classses), device=x.device)
    xt = d3pm.q_sample(x, t, noise)
    logits = d3pm.model_predict(xt, t, cond)
    return xt, logits.argmax(dim=-1)


def plot_fine_denoise(
    c300: torch.Tensor,
    x0: torch.Tensor,
    xts: dict[int, torch.Tensor],
    preds: dict[int, torch.Tensor],
    samples: torch.Tensor | None,
    n_bins: int,
    save_path: Path,
) -> None:
    n = min(4, x0.shape[0])
    extra = 1 if samples is not None else 0
    cols = 3 + 2 * len(xts) + extra
    fig, axes = plt.subplots(n, cols, figsize=(3 * cols, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    titles = ["C 300", "C_fine 96"]
    for t in xts:
        titles += [f"x_t t={t}", f"x0-hat t={t}"]
    if samples is not None:
        titles.append("uncond sample")
    for i in range(n):
        axes[i, 0].imshow(c300[i, 0].cpu(), cmap="gray")
        axes[i, 1].imshow(_bins_to_unit(x0[i, 0], n_bins).cpu(), cmap="gray", vmin=0, vmax=1)
        col = 2
        for t in xts:
            axes[i, col].imshow(_bins_to_unit(xts[t][i, 0], n_bins).cpu(), cmap="gray", vmin=0, vmax=1)
            axes[i, col + 1].imshow(
                _bins_to_unit(preds[t][i, 0], n_bins).cpu(), cmap="gray", vmin=0, vmax=1
            )
            col += 2
        if samples is not None:
            axes[i, col].imshow(
                _bins_to_unit(samples[i, 0], n_bins).cpu(), cmap="gray", vmin=0, vmax=1
            )
        for j in range(cols):
            axes[i, j].axis("off")
    for j, title in enumerate(titles):
        axes[0, j].set_title(title)
    fig.suptitle("Fine D3PM: real magnitude, noised input, x0 prediction")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_coarse_profiles(
    true_row: torch.Tensor,
    xt_row: torch.Tensor,
    pred_row: torch.Tensor,
    samples: torch.Tensor | None,
    n_bins: int,
    t_val: int,
    save_path: Path,
) -> None:
    n = min(4, true_row.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    h = true_row.shape[-2]
    xs = range(h)
    for i, ax in enumerate(axes):
        ax.plot(xs, _bins_to_unit(true_row[i, 0, :, 0], n_bins).cpu(), label="true C_row", lw=1.5)
        ax.plot(xs, _bins_to_unit(xt_row[i, 0, :, 0], n_bins).cpu(), label=f"x_t t={t_val}", alpha=0.6)
        ax.plot(xs, _bins_to_unit(pred_row[i, 0, :, 0], n_bins).cpu(), label="x0-hat", lw=1.2)
        if samples is not None:
            ax.plot(
                xs,
                _bins_to_unit(samples[i, 0, :, 0], n_bins).cpu(),
                label="uncond sample",
                alpha=0.7,
                ls="--",
            )
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"s{i}")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("PE row")
    fig.suptitle("Coarse D3PM row-profile prior")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_ce_curves(fine_ce: dict[int, float], coarse_ce: dict[int, float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(fine_ce), list(fine_ce.values()), marker="o", label="fine 96x96")
    ax.plot(list(coarse_ce), list(coarse_ce.values()), marker="s", label="coarse row")
    ax.axhline(float(torch.log(torch.tensor(8.0))), color="gray", ls=":", label="chance log(8)")
    ax.set_xlabel("t")
    ax.set_ylabel("x0 CE (nats)")
    ax.set_title("D3PM prior CE vs diffusion timestep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = FastMRIConfig(device=default_device(), batch_size=4, num_workers=0)
    out = Path("models_d3pm_fastmri_fine/eval")
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={cfg.device} out={out}")

    d3pm_fine = load_d3pm_fine(cfg)
    d3pm_coarse = load_d3pm_coarse(cfg)
    d3pm_fine.eval()
    d3pm_coarse.eval()

    ds = FastMRIDataset(cfg.data_root)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    z, c, kspace = next(iter(dl))
    z, c, kspace = z.to(cfg.device), c.to(cfg.device), kspace.to(cfg.device)
    x_fine = magnitude_to_fine_disc(c, size=cfg.fine_size, n_bins=cfg.num_classes)
    x_row = magnitude_to_row_disc(c, n_bins=cfg.num_classes)
    cond = torch.zeros(c.shape[0], dtype=torch.long, device=cfg.device)

    timesteps = (50, 200, 400, 700, 900)
    fine_ce: dict[int, float] = {}
    coarse_ce: dict[int, float] = {}
    # CE on a few sequential batches so it is not one-batch noise
    disc_dl = DataLoader(
        FastMRIDiscDataset(cfg.data_root, mode="fine", fine_size=cfg.fine_size, n_bins=cfg.num_classes),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    coarse_dl = DataLoader(
        FastMRIDiscDataset(cfg.data_root, mode="coarse", n_bins=cfg.num_classes),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    fine_batches = [batch for i, batch in enumerate(disc_dl) if i < 4]
    coarse_batches = [batch for i, batch in enumerate(coarse_dl) if i < 4]
    for t in timesteps:
        fvals = [ce_at_t(d3pm_fine, x.to(cfg.device), t, cd.to(cfg.device)) for x, cd in fine_batches]
        cvals = [ce_at_t(d3pm_coarse, x.to(cfg.device), t, cd.to(cfg.device)) for x, cd in coarse_batches]
        fine_ce[t] = sum(fvals) / len(fvals)
        coarse_ce[t] = sum(cvals) / len(cvals)
        print(f"t={t:4d}  CE_fine={fine_ce[t]:.4f}  CE_coarse={coarse_ce[t]:.4f}")

    denoise_ts = (200, 500)
    xts, preds = {}, {}
    for t in denoise_ts:
        xts[t], preds[t] = predict_x0(d3pm_fine, x_fine, t, cond)
    xt_row, pred_row = predict_x0(d3pm_coarse, x_row, 400, cond)

    print("sampling fine prior (4 images, 999 reverse steps)...")
    x_init = torch.zeros(c.shape[0], 1, cfg.fine_size, cfg.fine_size, dtype=torch.long, device=cfg.device)
    samples_fine = d3pm_fine.sample(x_init, cond)

    print("sampling coarse prior...")
    row_init = torch.zeros(c.shape[0], 1, cfg.image_size, 1, dtype=torch.long, device=cfg.device)
    samples_row = d3pm_coarse.sample(row_init, cond)

    plot_fine_denoise(
        c.cpu(), x_fine.cpu(), {t: v.cpu() for t, v in xts.items()},
        {t: v.cpu() for t, v in preds.items()}, samples_fine.cpu(),
        cfg.num_classes, out / "fine_denoise_samples.png",
    )
    plot_coarse_profiles(
        x_row.cpu(), xt_row.cpu(), pred_row.cpu(), samples_row.cpu(),
        cfg.num_classes, 400, out / "coarse_profiles.png",
    )
    plot_ce_curves(fine_ce, coarse_ce, out / "ce_vs_t.png")

    sparsity = torch.full((z.shape[0],), 0.25, device=cfg.device)
    random_rows = random_row_mask_batch(z.shape[0], cfg.image_size, sparsity, cfg.device)
    y_random = apply_kspace_row_mask(kspace, random_rows)
    plot_fastmri_grid(
        z.cpu(), c.cpu(), random_rows.cpu(), y_random.cpu(),
        title="random row mask s=0.25 (no learned policy yet)",
        save_path=out / "acquisition_random_s025.png",
    )
    plt.close("all")

    hist = torch.bincount(x_fine.flatten().cpu(), minlength=cfg.num_classes).float()
    hist = hist / hist.sum()
    report = {
        "device": cfg.device,
        "fine_checkpoint": cfg.d3pm_fine_checkpoint,
        "coarse_checkpoint": cfg.d3pm_coarse_checkpoint,
        "ce_fine": {str(k): v for k, v in fine_ce.items()},
        "ce_coarse": {str(k): v for k, v in coarse_ce.items()},
        "fine_bin_fraction": hist.tolist(),
        "note": (
            "Learned-vs-random nested mask eval needs train_fastmri first. "
            "models_mask_gen_fastmri/ is pre-fix and invalid."
        ),
    }
    save_eval_report(report, out / "prior_eval.json")
    print(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
