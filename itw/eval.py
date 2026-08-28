"""Evaluation utilities for ITW mask generators."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .discrete import (
    apply_row_absorbing_observation,
    magnitude_to_fine_disc,
    magnitude_to_row_disc,
)
from .entropy import (
    coarse_cond_entropy_loss,
    d3pm_cond_entropy_loss,
    fine_cond_entropy_loss,
    nested_cond_entropy_loss,
)
from .infonce import InfoNCELoss
from .masks import (
    apply_kspace_row_mask,
    apply_masked_observation,
    expand_row_mask,
    gumbel_mask,
    gumbel_row_mask,
)
from .schedule import sparsity_to_timestep
from .train import discretize, forward_mask_logits


def random_mask_batch(
    batch_size: int,
    shape: tuple[int, int, int],
    sparsity: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Bernoulli mask with per-sample target density."""
    c, h, w = shape
    masks = []
    for i in range(batch_size):
        flat = torch.rand(h * w, device=device)
        k = int(round(sparsity[i].item() * h * w))
        k = max(0, min(h * w, k))
        if k == 0:
            m = torch.zeros(h, w, device=device)
        else:
            _, idx = torch.topk(flat, k)
            m = torch.zeros(h * w, device=device)
            m[idx] = 1.0
            m = m.view(h, w)
        masks.append(m)
    mask = torch.stack(masks, dim=0).unsqueeze(1)
    if c > 1:
        return mask
    return mask


def random_row_mask_batch(
    batch_size: int,
    n_rows: int,
    sparsity: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    """Select exactly floor(s * H) random rows. Returns [B, 1, H, 1]."""
    masks = []
    for i in range(batch_size):
        k = int(sparsity[i].item() * n_rows)
        k = max(0, min(n_rows, k))
        m = torch.zeros(n_rows, device=device)
        if k > 0:
            idx = torch.randperm(n_rows, device=device)[:k]
            m[idx] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0).view(batch_size, 1, n_rows, 1)


@torch.no_grad()
def evaluate_batch(
    d3pm,
    model,
    cfg,
    x: torch.Tensor,
    cond: torch.Tensor,
    sparsity: torch.Tensor,
    survival_table: torch.Tensor,
) -> dict[str, float]:
    x_disc = discretize(x.to(cfg.device), cfg.num_classes)
    cond = cond.to(cfg.device)
    sparsity = sparsity.to(cfg.device)
    t = sparsity_to_timestep(sparsity, survival_table, cfg.n_t)

    mask_logits = forward_mask_logits(model, cfg, x_disc, cond, sparsity)
    learned_mask = gumbel_mask(mask_logits, temperature=0.5, hard=True)
    random_m = random_mask_batch(
        x.shape[0],
        (cfg.image_channels, cfg.image_size, cfg.image_size),
        sparsity,
        cfg.device,
    )

    y_learned = apply_masked_observation(
        x_disc, learned_mask, cfg.num_classes, multichannel=cfg.multichannel
    )
    y_random = apply_masked_observation(
        x_disc, random_m, cfg.num_classes, multichannel=cfg.multichannel
    )

    h_learned = d3pm_cond_entropy_loss(d3pm, y_learned, t, cond, learned_mask).item()
    h_random = d3pm_cond_entropy_loss(d3pm, y_random, t, cond, random_m).item()

    learned_density = learned_mask.mean(dim=(1, 2, 3))
    random_density = random_m.mean(dim=(1, 2, 3))
    sparsity_err_learned = F.l1_loss(learned_density, sparsity).item()
    sparsity_err_random = F.l1_loss(random_density, sparsity).item()

    return {
        "h_learned": h_learned,
        "h_random": h_random,
        "delta_h": h_random - h_learned,
        "sparsity_err_learned": sparsity_err_learned,
        "sparsity_err_random": sparsity_err_random,
        "mean_sparsity_learned": learned_density.mean().item(),
        "mean_sparsity_target": sparsity.mean().item(),
    }


@torch.no_grad()
def evaluate_fastmri_batch(
    model,
    proj_head,
    cfg,
    z: torch.Tensor,
    c: torch.Tensor,
    kspace: torch.Tensor,
    sparsity: torch.Tensor,
) -> dict[str, float]:
    """Compare learned vs random-row masks via InfoNCE and row density."""
    z = z.to(cfg.device)
    c = c.to(cfg.device)
    kspace = kspace.to(cfg.device)
    sparsity = sparsity.to(cfg.device)

    row_logits = model(z, sparsity)
    learned_rows = gumbel_row_mask(row_logits, temperature=0.5, hard=True)
    random_rows = random_row_mask_batch(
        z.shape[0], cfg.image_size, sparsity, cfg.device
    )

    y_learned = apply_kspace_row_mask(kspace, learned_rows)
    y_random = apply_kspace_row_mask(kspace, random_rows)

    info_nce = InfoNCELoss(temperature=cfg.infonce_temperature)
    mi_learned = info_nce(proj_head(y_learned), proj_head(c)).item()
    mi_random = info_nce(proj_head(y_random), proj_head(c)).item()

    learned_density = learned_rows.mean(dim=(1, 2, 3))
    random_density = random_rows.mean(dim=(1, 2, 3))

    return {
        "mi_learned": mi_learned,
        "mi_random": mi_random,
        "delta_mi": mi_random - mi_learned,
        "sparsity_err_learned": F.l1_loss(learned_density, sparsity).item(),
        "sparsity_err_random": F.l1_loss(random_density, sparsity).item(),
        "mean_sparsity_learned": learned_density.mean().item(),
        "mean_sparsity_target": sparsity.mean().item(),
    }


@torch.no_grad()
def evaluate_loader(
    d3pm,
    model,
    cfg,
    dataloader,
    survival_table: torch.Tensor,
    max_batches: int = 10,
    fixed_sparsity: float | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n = 0

    for batch_idx, (x, cond) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        if fixed_sparsity is not None:
            sparsity = torch.full((x.shape[0],), fixed_sparsity)
        else:
            sparsity = torch.full((x.shape[0],), 0.3)

        metrics = evaluate_batch(d3pm, model, cfg, x, cond, sparsity, survival_table)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_fastmri_loader(
    model,
    proj_head,
    cfg,
    dataloader,
    max_batches: int = 10,
    fixed_sparsity: float = 0.25,
) -> dict[str, float]:
    model.eval()
    proj_head.eval()
    totals: dict[str, float] = {}
    n = 0

    for batch_idx, (z, c, kspace) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        sparsity = torch.full((z.shape[0],), fixed_sparsity)
        metrics = evaluate_fastmri_batch(
            model, proj_head, cfg, z, c, kspace, sparsity
        )
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


def _nested_proxy_for_mask(
    d3pm_fine,
    d3pm_coarse,
    cfg,
    c: torch.Tensor,
    kspace: torch.Tensor,
    row_mask: torch.Tensor,
    sparsity: torch.Tensor,
    fine_survival: torch.Tensor,
    coarse_survival: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (h_coarse, h_fine, h_nested, t_fine, t_coarse) for a fixed row mask."""
    y_mag = apply_kspace_row_mask(kspace, row_mask)
    y_fine = magnitude_to_fine_disc(
        y_mag, size=cfg.fine_size, n_bins=cfg.num_classes
    )
    t_fine = sparsity_to_timestep(sparsity, fine_survival, cfg.n_t)

    c_row = magnitude_to_row_disc(c, n_bins=cfg.num_classes)
    y_row = apply_row_absorbing_observation(c_row, row_mask, cfg.num_classes)
    t_coarse = sparsity_to_timestep(sparsity, coarse_survival, cfg.n_t)

    cond0 = torch.zeros(c.shape[0], dtype=torch.long, device=cfg.device)
    h_fine = fine_cond_entropy_loss(d3pm_fine, y_fine, t_fine, cond0)
    h_coarse = coarse_cond_entropy_loss(
        d3pm_coarse, y_row, t_coarse, cond0, row_mask
    )
    h_nested = nested_cond_entropy_loss(
        h_coarse, h_fine, alpha=cfg.entropy_alpha, beta=cfg.entropy_beta
    )
    return h_coarse, h_fine, h_nested, t_fine, t_coarse


@torch.no_grad()
def evaluate_fastmri_nested_batch(
    model,
    d3pm_fine,
    d3pm_coarse,
    cfg,
    z: torch.Tensor,
    c: torch.Tensor,
    kspace: torch.Tensor,
    sparsity: torch.Tensor,
    fine_survival: torch.Tensor,
    coarse_survival: torch.Tensor,
    proj_head=None,
) -> dict[str, float]:
    """
    Compare learned vs random-row masks under nested D3PM entropy proxies.

    Optionally also reports InfoNCE if proj_head is provided.
    """
    z = z.to(cfg.device)
    c = c.to(cfg.device)
    kspace = kspace.to(cfg.device)
    sparsity = sparsity.to(cfg.device)
    fine_survival = fine_survival.to(cfg.device)
    coarse_survival = coarse_survival.to(cfg.device)

    row_logits = model(z, sparsity)
    learned_rows = gumbel_row_mask(row_logits, temperature=0.5, hard=True)
    random_rows = random_row_mask_batch(
        z.shape[0], cfg.image_size, sparsity, cfg.device
    )

    h_c_l, h_f_l, h_n_l, t_f_l, t_c_l = _nested_proxy_for_mask(
        d3pm_fine,
        d3pm_coarse,
        cfg,
        c,
        kspace,
        learned_rows,
        sparsity,
        fine_survival,
        coarse_survival,
    )
    h_c_r, h_f_r, h_n_r, t_f_r, t_c_r = _nested_proxy_for_mask(
        d3pm_fine,
        d3pm_coarse,
        cfg,
        c,
        kspace,
        random_rows,
        sparsity,
        fine_survival,
        coarse_survival,
    )

    learned_density = learned_rows.mean(dim=(1, 2, 3))
    random_density = random_rows.mean(dim=(1, 2, 3))

    metrics = {
        "h_coarse_learned": h_c_l.item(),
        "h_coarse_random": h_c_r.item(),
        "delta_h_coarse": h_c_r.item() - h_c_l.item(),
        "h_fine_learned": h_f_l.item(),
        "h_fine_random": h_f_r.item(),
        "delta_h_fine": h_f_r.item() - h_f_l.item(),
        "h_nested_learned": h_n_l.item(),
        "h_nested_random": h_n_r.item(),
        "delta_h_nested": h_n_r.item() - h_n_l.item(),
        "t_fine_mean": float(t_f_l.float().mean()),
        "t_coarse_mean": float(t_c_l.float().mean()),
        "sparsity_err_learned": F.l1_loss(learned_density, sparsity).item(),
        "sparsity_err_random": F.l1_loss(random_density, sparsity).item(),
        "mean_sparsity_learned": learned_density.mean().item(),
        "mean_sparsity_target": sparsity.mean().item(),
    }

    if proj_head is not None:
        y_learned = apply_kspace_row_mask(kspace, learned_rows)
        y_random = apply_kspace_row_mask(kspace, random_rows)
        info_nce = InfoNCELoss(temperature=cfg.infonce_temperature)
        metrics["mi_learned"] = info_nce(proj_head(y_learned), proj_head(c)).item()
        metrics["mi_random"] = info_nce(proj_head(y_random), proj_head(c)).item()
        metrics["delta_mi"] = metrics["mi_random"] - metrics["mi_learned"]

    return metrics


@torch.no_grad()
def evaluate_fastmri_nested_loader(
    model,
    d3pm_fine,
    d3pm_coarse,
    cfg,
    dataloader,
    fine_survival: torch.Tensor,
    coarse_survival: torch.Tensor,
    max_batches: int = 10,
    fixed_sparsity: float = 0.25,
    proj_head=None,
) -> dict[str, float]:
    model.eval()
    d3pm_fine.eval()
    d3pm_coarse.eval()
    if proj_head is not None:
        proj_head.eval()

    totals: dict[str, float] = {}
    n = 0
    for batch_idx, (z, c, kspace) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        sparsity = torch.full((z.shape[0],), fixed_sparsity)
        metrics = evaluate_fastmri_nested_batch(
            model,
            d3pm_fine,
            d3pm_coarse,
            cfg,
            z,
            c,
            kspace,
            sparsity,
            fine_survival,
            coarse_survival,
            proj_head=proj_head,
        )
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_fastmri_nested_ablation(
    model,
    d3pm_fine,
    d3pm_coarse,
    cfg,
    dataloader,
    fine_survival: torch.Tensor,
    coarse_survival: torch.Tensor,
    sparsities: tuple[float, ...] = (0.1, 0.25, 0.4),
    max_batches: int = 10,
    proj_head=None,
) -> dict[str, dict[str, float]]:
    """
    Run nested eval at several sparsities and report coarse-only / fine-only /
    nested deltas (via alpha/beta overrides on a shallow config copy).
    """
    from dataclasses import replace

    report: dict[str, dict[str, float]] = {}
    for s in sparsities:
        key = f"s={s:.2f}"
        report[key] = {}

        for name, alpha, beta in (
            ("nested", 1.0, 1.0),
            ("fine_only", 0.0, 1.0),
            ("coarse_only", 1.0, 0.0),
        ):
            cfg_ab = replace(cfg, entropy_alpha=alpha, entropy_beta=beta)
            metrics = evaluate_fastmri_nested_loader(
                model,
                d3pm_fine,
                d3pm_coarse,
                cfg_ab,
                dataloader,
                fine_survival,
                coarse_survival,
                max_batches=max_batches,
                fixed_sparsity=s,
                proj_head=proj_head if name == "nested" else None,
            )
            report[key][name] = metrics
    return report


def save_eval_report(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_mask_grid(
    x: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    title: str = "",
    save_path: str | Path | None = None,
) -> None:
    """Plot original, mask, and masked observation for up to 4 samples."""
    n_show = min(4, x.shape[0])
    scale = max(num_classes - 1, 1)

    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_show):
        if x.shape[1] == 1:
            axes[i, 0].imshow(x[i, 0].cpu().float() / scale, cmap="gray")
            axes[i, 1].imshow(mask[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(y[i, 0].cpu().float() / scale, cmap="gray")
        else:
            img = (x[i].permute(1, 2, 0).cpu().float() / scale).clamp(0, 1)
            obs = (y[i].permute(1, 2, 0).cpu().float() / scale).clamp(0, 1)
            axes[i, 0].imshow(img)
            axes[i, 1].imshow(mask[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(obs)
        axes[i, 0].set_ylabel(f"sample {i}")
        for j in range(3):
            axes[i, j].axis("off")

    cols = ["original", "mask", "observed"]
    for j, col in enumerate(cols):
        axes[0, j].set_title(col)
    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_fastmri_grid(
    z: torch.Tensor,
    c: torch.Tensor,
    row_mask: torch.Tensor,
    y: torch.Tensor,
    title: str = "",
    save_path: str | Path | None = None,
) -> None:
    """Plot scout Z, row mask, reconstruction Y, and full C for up to 4 samples."""
    n_show = min(4, z.shape[0])
    width = c.shape[-1]
    mask_2d = expand_row_mask(row_mask, width) if row_mask.shape[-1] == 1 else row_mask

    fig, axes = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_show):
        axes[i, 0].imshow(z[i, 0].cpu(), cmap="gray")
        axes[i, 1].imshow(mask_2d[i, 0].cpu(), cmap="gray", vmin=0, vmax=1, aspect="auto")
        axes[i, 2].imshow(y[i, 0].cpu(), cmap="gray")
        axes[i, 3].imshow(c[i, 0].cpu(), cmap="gray")
        axes[i, 0].set_ylabel(f"sample {i}")
        for j in range(4):
            axes[i, j].axis("off")

    for j, col in enumerate(["scout Z", "row mask X", "recon Y", "full C"]):
        axes[0, j].set_title(col)
    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
