"""Discretization helpers for nested FastMRI D3PM priors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _per_sample_minmax(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Map each sample independently to [0, 1]. x: [B, ...]."""
    flat = x.reshape(x.shape[0], -1)
    lo = flat.min(dim=1).values.view(x.shape[0], *([1] * (x.dim() - 1)))
    hi = flat.max(dim=1).values.view(x.shape[0], *([1] * (x.dim() - 1)))
    return (x - lo) / (hi - lo + eps)


def quantize_unit(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Quantize values in [0, 1] to integer bins {0, ..., n_bins-1}."""
    return (x * (n_bins - 1)).round().long().clamp(0, n_bins - 1)


def quantize_unit_ste(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    Straight-through quantized bins in {0, ..., n_bins-1} as float.

    Forward: rounded integers. Backward: identity through the scaled input.
    """
    soft = x * (n_bins - 1)
    hard = soft.round().clamp(0, n_bins - 1)
    return (hard - soft).detach() + soft


def magnitude_to_fine_disc(
    c: torch.Tensor,
    size: int = 96,
    n_bins: int = 8,
    ste: bool = False,
) -> torch.Tensor:
    """
    Resize magnitude image and quantize for the fine D3PM.

    c: [B, 1, H, W] (or [B, H, W]) real magnitude
    returns: [B, 1, size, size] long (ste=False) or float STE bins (ste=True)
    """
    if c.dim() == 3:
        c = c.unsqueeze(1)
    unit = _per_sample_minmax(c.float())
    if unit.shape[-1] != size or unit.shape[-2] != size:
        unit = F.interpolate(unit, size=(size, size), mode="bilinear", align_corners=False)
    if ste:
        return quantize_unit_ste(unit, n_bins)
    return quantize_unit(unit, n_bins)


def magnitude_to_row_disc(
    c: torch.Tensor,
    n_bins: int = 8,
    ste: bool = False,
) -> torch.Tensor:
    """
    Per-image-row mean magnitude profile, quantized.

    c: [B, 1, H, W] (or [B, H, W])
    returns: [B, 1, H, 1] long (ste=False) or float STE bins (ste=True)
    """
    if c.dim() == 3:
        c = c.unsqueeze(1)
    row_mean = c.float().mean(dim=-1, keepdim=True)  # [B, 1, H, 1]
    unit = _per_sample_minmax(row_mean)
    if ste:
        return quantize_unit_ste(unit, n_bins)
    return quantize_unit(unit, n_bins)


def kspace_to_row_disc(
    kspace: torch.Tensor,
    n_bins: int = 8,
    ste: bool = False,
) -> torch.Tensor:
    """
    Per-PE-line k-space energy, log1p + min-max + quantize.

    kspace: [B, 1, H, W] (or [B, H, W]) complex
    returns: [B, 1, H, 1] long (ste=False) or float STE bins (ste=True)
    """
    if kspace.dim() == 3:
        kspace = kspace.unsqueeze(1)
    energy = torch.log1p(kspace.abs().mean(dim=-1, keepdim=True).float())
    unit = _per_sample_minmax(energy)
    if ste:
        return quantize_unit_ste(unit, n_bins)
    return quantize_unit(unit, n_bins)


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample ||pred-target||^2 / ||target||^2, then mean over batch."""
    pred = pred.float()
    target = target.float()
    num = (pred - target).pow(2).flatten(1).sum(dim=1)
    den = target.pow(2).flatten(1).sum(dim=1).clamp_min(eps)
    return (num / den).mean()


def apply_row_absorbing_observation(
    c_row: torch.Tensor,
    row_mask: torch.Tensor,
    n_bins: int,
) -> torch.Tensor:
    """
    Build coarse D3PM input: unselected rows -> absorbing state 0.

    Observed bins are shifted to {1, ..., n_bins-1} so 0 is mask-only
    (same convention as multichannel apply_masked_observation).
    Multiplicative gating preserves STE gradients through row_mask.

    c_row:    [B, 1, H, 1] long or float
    row_mask: [B, 1, H, 1] or broadcastable, in {0, 1}
    returns:  [B, 1, H, 1] float
    """
    if row_mask.shape[-1] != c_row.shape[-1]:
        row_mask = row_mask[..., :1]
    mask = row_mask.float()
    if n_bins > 2:
        observed = torch.clamp(c_row.float() + 1, 1, n_bins - 1)
        return observed * mask
    return c_row.float() * mask
