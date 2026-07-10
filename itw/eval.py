"""Evaluation utilities for ITW mask generators."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .entropy import d3pm_cond_entropy_loss, masked_cond_entropy
from .masks import apply_masked_observation, gumbel_mask
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
    from .schedule import sparsity_to_timestep

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


def save_eval_report(metrics: dict[str, float], path: str | Path) -> None:
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
            axes[i, 0].imshow(
                x[i].permute(1, 2, 0).cpu().float() / scale, clip_range=(0, 1)
            )
            axes[i, 1].imshow(mask[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(
                y[i].permute(1, 2, 0).cpu().float() / scale, clip_range=(0, 1)
            )
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
