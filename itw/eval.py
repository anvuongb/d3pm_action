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
    nmse,
    psnr,
    ssim,
)
from .entropy import (
    coarse_cond_entropy_loss,
    d3pm_cond_entropy_loss,
    fine_cond_entropy_loss,
    nested_cond_entropy_loss,
)
from .infonce import InfoNCELoss
from .masks import (
    acs_bounds,
    apply_kspace_row_mask,
    apply_masked_observation,
    expand_row_mask,
    gumbel_mask,
    gumbel_row_mask,
    gumbel_row_mask_acs_locked,
    magnitude_from_kspace,
)
from .schedule import sparsity_to_timestep
from .train import coarse_profile, discretize, forward_mask_logits, mask_generator_cond


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
        k = _row_budget(sparsity[i], n_rows)
        m = torch.zeros(n_rows, device=device)
        if k > 0:
            idx = torch.randperm(n_rows, device=device)[:k]
            m[idx] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0).view(batch_size, 1, n_rows, 1)


def _row_budget(sparsity: torch.Tensor, n_rows: int) -> int:
    k = int(float(sparsity) * n_rows)
    return max(0, min(n_rows, k))


def equispaced_row_mask_batch(
    batch_size: int,
    n_rows: int,
    sparsity: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    """Equally spaced PE lines with a random integer offset per sample."""
    masks = []
    for i in range(batch_size):
        k = _row_budget(sparsity[i], n_rows)
        m = torch.zeros(n_rows, device=device)
        if k > 0:
            stride = n_rows / k
            offset = int(torch.randint(0, max(int(stride), 1), (1,), device=device))
            idx = (offset + (torch.arange(k, device=device).float() * stride).long()) % n_rows
            m[idx] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0).view(batch_size, 1, n_rows, 1)


def vd_gaussian_row_mask_batch(
    batch_size: int,
    n_rows: int,
    sparsity: torch.Tensor,
    device: str | torch.device,
    sigma: float = 0.3,
) -> torch.Tensor:
    """Sample rows without replacement with a center-heavy Gaussian density."""
    center = 0.5 * (n_rows - 1)
    scale = max(sigma * center, 1e-6)
    loc = torch.arange(n_rows, device=device, dtype=torch.float32)
    logits = -0.5 * ((loc - center) / scale) ** 2
    probs = torch.softmax(logits, dim=0)
    masks = []
    for i in range(batch_size):
        k = _row_budget(sparsity[i], n_rows)
        m = torch.zeros(n_rows, device=device)
        if k > 0:
            idx = torch.multinomial(probs, k, replacement=False)
            m[idx] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0).view(batch_size, 1, n_rows, 1)


def _outside_pe_indices(
    n_rows: int,
    acs_width: int,
    device: str | torch.device,
) -> torch.Tensor:
    lo, hi = acs_bounds(n_rows, acs_width)
    return torch.cat(
        (
            torch.arange(0, lo, device=device),
            torch.arange(hi, n_rows, device=device),
        )
    )


def _centered_k_block(n_rows: int, k: int, device: str | torch.device) -> torch.Tensor:
    m = torch.zeros(n_rows, device=device)
    if k > 0:
        inner_lo = (n_rows - k) // 2
        m[inner_lo : inner_lo + k] = 1.0
    return m


def acs_random_row_mask_batch(
    batch_size: int,
    n_rows: int,
    sparsity: torch.Tensor,
    device: str | torch.device,
    acs_width: int = 32,
) -> torch.Tensor:
    """Keep a centered ACS block, fill any remaining budget with random PE lines."""
    acs_width = min(max(int(acs_width), 0), n_rows)
    lo, hi = acs_bounds(n_rows, acs_width)
    outside = _outside_pe_indices(n_rows, acs_width, device)
    masks = []
    for i in range(batch_size):
        k = _row_budget(sparsity[i], n_rows)
        if k <= acs_width:
            masks.append(_centered_k_block(n_rows, k, device))
            continue
        m = torch.zeros(n_rows, device=device)
        m[lo:hi] = 1.0
        extra = k - acs_width
        if extra > 0:
            perm = outside[torch.randperm(outside.numel(), device=device)[:extra]]
            m[perm] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0).view(batch_size, 1, n_rows, 1)


def acs_vd_gaussian_row_mask_batch(
    batch_size: int,
    n_rows: int,
    sparsity: torch.Tensor,
    device: str | torch.device,
    acs_width: int = 32,
    sigma: float = 0.3,
) -> torch.Tensor:
    """ACS block plus VD-Gaussian remainder sampled only on non-ACS rows."""
    acs_width = min(max(int(acs_width), 0), n_rows)
    lo, hi = acs_bounds(n_rows, acs_width)
    center = 0.5 * (n_rows - 1)
    scale = max(sigma * center, 1e-6)
    loc = torch.arange(n_rows, device=device, dtype=torch.float32)
    logits = -0.5 * ((loc - center) / scale) ** 2
    probs = torch.softmax(logits, dim=0).clone()
    probs[lo:hi] = 0.0
    mass = float(probs.sum())
    if mass > 0:
        probs = probs / probs.sum()
    masks = []
    for i in range(batch_size):
        k = _row_budget(sparsity[i], n_rows)
        if k <= acs_width:
            masks.append(_centered_k_block(n_rows, k, device))
            continue
        m = torch.zeros(n_rows, device=device)
        m[lo:hi] = 1.0
        extra = k - acs_width
        n_out = n_rows - (hi - lo)
        if extra > 0 and n_out > 0 and mass > 0:
            take = min(extra, n_out)
            idx = torch.multinomial(probs, take, replacement=False)
            m[idx] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0).view(batch_size, 1, n_rows, 1)


def acs_equispaced_row_mask_batch(
    batch_size: int,
    n_rows: int,
    sparsity: torch.Tensor,
    device: str | torch.device,
    acs_width: int = 32,
) -> torch.Tensor:
    """ACS block plus equally spaced remainder on non-ACS rows only."""
    acs_width = min(max(int(acs_width), 0), n_rows)
    lo, hi = acs_bounds(n_rows, acs_width)
    outside = _outside_pe_indices(n_rows, acs_width, device)
    n_out = int(outside.numel())
    masks = []
    for i in range(batch_size):
        k = _row_budget(sparsity[i], n_rows)
        if k <= acs_width:
            masks.append(_centered_k_block(n_rows, k, device))
            continue
        m = torch.zeros(n_rows, device=device)
        m[lo:hi] = 1.0
        extra = k - acs_width
        if extra > 0 and n_out > 0:
            take = min(extra, n_out)
            stride = n_out / take
            offset = int(torch.randint(0, max(int(stride), 1), (1,), device=device))
            idx = (offset + (torch.arange(take, device=device).float() * stride).long()) % n_out
            m[outside[idx]] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0).view(batch_size, 1, n_rows, 1)


def acs_lock_topk_from_logits(
    logits: torch.Tensor,
    sparsity: torch.Tensor,
    acs_width: int,
) -> torch.Tensor:
    """Post-hoc ACS lock: hard ACS + top-k remainder by keep-logit.

    logits: [B, 1, H, 2] (index 1 = keep). Returns [B, 1, H, 1].
    When k <= acs_width, returns the same centered k-block as the ACS factories.
    """
    b, _, h, _ = logits.shape
    keep = logits[..., 1]
    lo, hi = acs_bounds(h, acs_width)
    masks = []
    for i in range(b):
        k = _row_budget(sparsity[i], h)
        if k <= acs_width:
            masks.append(_centered_k_block(h, k, logits.device))
            continue
        m = torch.zeros(h, device=logits.device, dtype=keep.dtype)
        m[lo:hi] = 1.0
        extra = k - acs_width
        scores = keep[i, 0].clone()
        scores[lo:hi] = float("-inf")
        _, idx = torch.topk(scores, extra)
        m[idx] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0).view(b, 1, h, 1)


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

    row_logits = model(mask_generator_cond(cfg, z, kspace), sparsity)
    learned_rows = _learned_gumbel_rows(cfg, row_logits, sparsity)
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Return (h_coarse, h_fine, h_nested, t_fine, t_coarse, recon_metrics).

    recon_metrics are NMSE / SSIM / PSNR on unnormalized magnitude:
    Y = apply_kspace_row_mask(K) vs magnitude_from_kspace(K).
    """
    y_mag = apply_kspace_row_mask(kspace, row_mask)
    c_mag = magnitude_from_kspace(kspace)
    recon = {
        "nmse": nmse(y_mag, c_mag),
        "ssim": ssim(y_mag, c_mag),
        "psnr": psnr(y_mag, c_mag),
    }

    cond0 = torch.zeros(c.shape[0], dtype=torch.long, device=cfg.device)
    t_fine = sparsity_to_timestep(sparsity, fine_survival, cfg.n_t)
    if d3pm_fine is not None and cfg.entropy_beta > 0:
        y_fine = magnitude_to_fine_disc(
            y_mag, size=cfg.fine_size, n_bins=cfg.num_classes
        )
        h_fine = fine_cond_entropy_loss(d3pm_fine, y_fine, t_fine, cond0)
    else:
        h_fine = torch.zeros((), device=cfg.device)

    c_row = coarse_profile(cfg, c, kspace)
    y_row = apply_row_absorbing_observation(c_row, row_mask, cfg.num_classes)
    t_coarse = sparsity_to_timestep(sparsity, coarse_survival, cfg.n_t)
    h_coarse = coarse_cond_entropy_loss(
        d3pm_coarse, y_row, t_coarse, cond0, row_mask
    )
    h_nested = nested_cond_entropy_loss(
        h_coarse, h_fine, alpha=cfg.entropy_alpha, beta=cfg.entropy_beta
    )
    return h_coarse, h_fine, h_nested, t_fine, t_coarse, recon


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

    row_logits = model(mask_generator_cond(cfg, z, kspace), sparsity)
    learned_rows = _learned_gumbel_rows(cfg, row_logits, sparsity)
    random_rows = random_row_mask_batch(
        z.shape[0], cfg.image_size, sparsity, cfg.device
    )

    h_c_l, h_f_l, h_n_l, t_f_l, t_c_l, recon_l = _nested_proxy_for_mask(
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
    h_c_r, h_f_r, h_n_r, t_f_r, t_c_r, recon_r = _nested_proxy_for_mask(
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
        "nmse_learned": recon_l["nmse"].item(),
        "nmse_random": recon_r["nmse"].item(),
        "delta_nmse": recon_r["nmse"].item() - recon_l["nmse"].item(),
        "ssim_learned": recon_l["ssim"].item(),
        "ssim_random": recon_r["ssim"].item(),
        "delta_ssim": recon_r["ssim"].item() - recon_l["ssim"].item(),
        "psnr_learned": recon_l["psnr"].item(),
        "psnr_random": recon_r["psnr"].item(),
        "delta_psnr": recon_r["psnr"].item() - recon_l["psnr"].item(),
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
    if d3pm_fine is not None:
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


def _mask_metrics(
    h_c,
    h_f,
    h_n,
    t_c,
    recon: dict[str, torch.Tensor],
    row_mask: torch.Tensor,
    sparsity: torch.Tensor,
) -> dict[str, float]:
    dens = row_mask.mean(dim=(1, 2, 3))
    return {
        "h_coarse": float(h_c),
        "h_fine": float(h_f),
        "h_nested": float(h_n),
        "t_coarse_mean": float(t_c.float().mean()),
        "nmse": float(recon["nmse"]),
        "ssim": float(recon["ssim"]),
        "psnr": float(recon["psnr"]),
        "mean_sparsity": float(dens.mean()),
        "sparsity_err": float(F.l1_loss(dens, sparsity)),
    }


def _learned_gumbel_rows(
    cfg,
    row_logits: torch.Tensor,
    sparsity: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Unconstrained Gumbel unless ``cfg.acs_lock`` (train-time ACS constraint)."""
    if getattr(cfg, "acs_lock", False):
        return gumbel_row_mask_acs_locked(
            row_logits,
            sparsity,
            acs_width=int(getattr(cfg, "scout_size", 32)),
            temperature=temperature,
            hard=True,
        )
    return gumbel_row_mask(row_logits, temperature=temperature, hard=True)


def baseline_row_mask_batch(
    name: str,
    batch_size: int,
    n_rows: int,
    sparsity: torch.Tensor,
    device: str | torch.device,
    acs_width: int = 32,
) -> torch.Tensor:
    if name == "random":
        return random_row_mask_batch(batch_size, n_rows, sparsity, device)
    if name == "equispaced":
        return equispaced_row_mask_batch(batch_size, n_rows, sparsity, device)
    if name == "vd_gaussian":
        return vd_gaussian_row_mask_batch(batch_size, n_rows, sparsity, device)
    if name == "acs_random":
        return acs_random_row_mask_batch(
            batch_size, n_rows, sparsity, device, acs_width=acs_width
        )
    if name == "acs_vd_gaussian":
        return acs_vd_gaussian_row_mask_batch(
            batch_size, n_rows, sparsity, device, acs_width=acs_width
        )
    if name == "acs_equispaced":
        return acs_equispaced_row_mask_batch(
            batch_size, n_rows, sparsity, device, acs_width=acs_width
        )
    raise ValueError(f"Unknown baseline mask {name}")


def sparsity_key(s: float) -> str:
    """JSON/report key for a sparsity. Two decimals when exact, else three (0.125)."""
    if abs(s * 100.0 - round(s * 100.0)) < 1e-9:
        return f"s={s:.2f}"
    return f"s={s:.3f}"


@torch.no_grad()
def evaluate_fastmri_baselines_loader(
    model,
    d3pm_fine,
    d3pm_coarse,
    cfg,
    dataloader,
    fine_survival: torch.Tensor,
    coarse_survival: torch.Tensor,
    sparsities: tuple[float, ...] = (0.1, 0.25, 0.4),
    max_batches: int = 10,
    baselines: tuple[str, ...] = ("random", "equispaced", "vd_gaussian", "acs_random"),
    include_learned_acs_lock: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """Learned policy vs MRI row-mask baselines at several sparsities.

    ``include_learned_acs_lock`` adds a post-hoc ACS + top-k remainder mask
    from the same logits (eval-only; independent of ``cfg.acs_lock``).
    """
    model.eval()
    if d3pm_fine is not None:
        d3pm_fine.eval()
    d3pm_coarse.eval()
    acs_width = int(getattr(cfg, "scout_size", 32))
    report: dict[str, dict[str, dict[str, float]]] = {}
    method_names = ("learned",) + baselines
    if include_learned_acs_lock:
        method_names = method_names + ("learned_acs_lock",)

    for s in sparsities:
        key = sparsity_key(s)
        buckets: dict[str, dict[str, float]] = {name: {} for name in method_names}
        n = 0
        for batch_idx, (z, c, kspace) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            z = z.to(cfg.device)
            c = c.to(cfg.device)
            kspace = kspace.to(cfg.device)
            sparsity = torch.full((z.shape[0],), s, device=cfg.device)
            fine_s = fine_survival.to(cfg.device)
            coarse_s = coarse_survival.to(cfg.device)

            row_logits = model(mask_generator_cond(cfg, z, kspace), sparsity)
            named_masks = {
                "learned": _learned_gumbel_rows(cfg, row_logits, sparsity)
            }
            if include_learned_acs_lock:
                named_masks["learned_acs_lock"] = acs_lock_topk_from_logits(
                    row_logits, sparsity, acs_width
                )
            for name in baselines:
                named_masks[name] = baseline_row_mask_batch(
                    name,
                    z.shape[0],
                    cfg.image_size,
                    sparsity,
                    cfg.device,
                    acs_width=acs_width,
                )

            for name, mask in named_masks.items():
                h_c, h_f, h_n, _t_f, t_c, recon = _nested_proxy_for_mask(
                    d3pm_fine,
                    d3pm_coarse,
                    cfg,
                    c,
                    kspace,
                    mask,
                    sparsity,
                    fine_s,
                    coarse_s,
                )
                mets = _mask_metrics(h_c, h_f, h_n, t_c, recon, mask, sparsity)
                acc = buckets[name]
                for mk, mv in mets.items():
                    acc[mk] = acc.get(mk, 0.0) + mv
            n += 1

        report[key] = {
            name: {mk: mv / max(n, 1) for mk, mv in mets.items()}
            for name, mets in buckets.items()
        }
        learned_nmse = report[key]["learned"]["nmse"]
        learned_hc = report[key]["learned"]["h_coarse"]
        learned_ssim = report[key]["learned"]["ssim"]
        learned_psnr = report[key]["learned"]["psnr"]
        for name in method_names:
            if name == "learned":
                continue
            report[key][name]["delta_nmse_vs_learned"] = (
                report[key][name]["nmse"] - learned_nmse
            )
            report[key][name]["delta_h_coarse_vs_learned"] = (
                report[key][name]["h_coarse"] - learned_hc
            )
            report[key][name]["delta_ssim_vs_learned"] = (
                report[key][name]["ssim"] - learned_ssim
            )
            report[key][name]["delta_psnr_vs_learned"] = (
                report[key][name]["psnr"] - learned_psnr
            )
    return report


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
        key = sparsity_key(s)
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
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _magnitude_clim(ref: torch.Tensor, vmax_percentile: float = 99.5) -> tuple[float, float]:
    """vmin=0, vmax=percentile of a single magnitude image (H, W)."""
    flat = ref.detach().float().reshape(-1)
    vmax = float(torch.quantile(flat, vmax_percentile / 100.0))
    vmin = 0.0
    if vmax <= vmin:
        vmax = max(float(flat.max()), 1e-6)
    return vmin, vmax


def plot_fastmri_grid(
    z: torch.Tensor,
    c: torch.Tensor,
    row_mask: torch.Tensor,
    y: torch.Tensor,
    title: str = "",
    save_path: str | Path | None = None,
    kspace: torch.Tensor | None = None,
    vmax_percentile: float = 99.5,
) -> None:
    """Plot scout Z, row mask, recon Y, and full magnitude for up to 4 samples.

    If ``kspace`` is given, Y and the full image are unnormalized magnitude
    (``apply_kspace_row_mask`` / ``magnitude_from_kspace``) with a **shared**
    per-sample clim (percentile clip of the full mag). Do not pass the
    dataloader's z-scored ``c`` as the visual GT next to raw Y.

    Scout Z stays as the policy input (typically z-scored). savefig only.
    """
    z = z.detach().cpu()
    c = c.detach().cpu()
    y = y.detach().cpu()
    row_mask = row_mask.detach().cpu()
    n_show = min(4, z.shape[0])
    width = c.shape[-1] if kspace is None else (
        kspace.shape[-1] if kspace.dim() >= 2 else c.shape[-1]
    )
    mask_2d = expand_row_mask(row_mask, width) if row_mask.shape[-1] == 1 else row_mask

    if kspace is not None:
        k_cpu = kspace.detach().cpu()
        if k_cpu.dim() == 3:
            k_cpu = k_cpu.unsqueeze(1)
        full_mag = magnitude_from_kspace(k_cpu)
        recon_mag = y.float()
        if recon_mag.dim() == 3:
            recon_mag = recon_mag.unsqueeze(1)
        full_label = "full mag"
    else:
        full_mag = c.float()
        recon_mag = y.float()
        if recon_mag.dim() == 3:
            recon_mag = recon_mag.unsqueeze(1)
        if full_mag.dim() == 3:
            full_mag = full_mag.unsqueeze(1)
        full_label = "full C"

    fig, axes = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_show):
        vmin, vmax = _magnitude_clim(full_mag[i, 0], vmax_percentile)
        axes[i, 0].imshow(z[i, 0].cpu(), cmap="gray")
        axes[i, 1].imshow(mask_2d[i, 0].cpu(), cmap="gray", vmin=0, vmax=1, aspect="auto")
        axes[i, 2].imshow(recon_mag[i, 0], cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 3].imshow(full_mag[i, 0], cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 0].set_ylabel(f"sample {i}")
        for j in range(4):
            axes[i, j].axis("off")

    for j, col in enumerate(["scout Z", "row mask X", "recon Y", full_label]):
        axes[0, j].set_title(col)
    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
