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


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Per-sample ||pred-target||^2 / ||target||^2, then mean over batch.

    ``eps`` is an absolute floor on the denominator and defaults to 0. It used
    to default to 1e-8, which is not a safe floor here: FastMRI magnitude is
    ~1e-6, so ||target||^2 is ~4e-9 for a median 300x300 slice and the floor
    replaced the denominator on ~2/3 of them. That turns NMSE into a
    constant-scaled MSE on exactly the low-energy slices, silently down-weights
    them, and reads ~4x lower than the true ratio. Only an all-zero target
    needs protection, so clamp at the dtype's tiny value instead.
    """
    pred = pred.float()
    target = target.float()
    num = (pred - target).pow(2).flatten(1).sum(dim=1)
    floor = max(eps, torch.finfo(target.dtype).tiny)
    den = target.pow(2).flatten(1).sum(dim=1).clamp_min(floor)
    return (num / den).mean()


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample PSNR (dB) from target peak, then mean over batch.

    Identity (zero MSE) is +inf. Peak is per-sample max(|target|).
    Does not clamp MSE to ``eps`` — FastMRI magnitude is ~1e-6, so MSE can
    be ~1e-15 and a 1e-8 floor would make every method look identical.
    """
    pred = pred.float()
    target = target.float()
    mse = (pred - target).pow(2).flatten(1).mean(dim=1)
    peak = target.abs().flatten(1).amax(dim=1).clamp_min(eps)
    finite = 10.0 * torch.log10(peak.pow(2) / mse.clamp_min(torch.finfo(mse.dtype).tiny))
    return torch.where(mse > 0, finite, torch.full_like(mse, float("inf"))).mean()


def _ssim_gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean SSIM over the batch (grayscale or 1-channel MRI magnitude).

    Uses an 11x11 Gaussian window. Each sample is divided by max(target)
    before the SSIM map (equivalent to data_range = peak, but stable when
    unnormalized FastMRI magnitude is ~1e-6). Identity scores ≈ 1.
    """
    pred = pred.float()
    target = target.float()
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    if pred.dim() != 4:
        raise ValueError(f"ssim expects [B,C,H,W] or [B,H,W], got {tuple(pred.shape)}")

    peak = target.flatten(1).amax(dim=1).clamp_min(eps).view(-1, 1, 1, 1)
    pred = pred / peak
    target = target / peak
    c1 = (k1**2)
    c2 = (k2**2)

    channels = pred.shape[1]
    window = _ssim_gaussian_window(
        window_size, sigma, channels, pred.device, pred.dtype
    )
    pad = window_size // 2
    mu_p = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_t = F.conv2d(target, window, padding=pad, groups=channels)
    mu_p2 = mu_p.pow(2)
    mu_t2 = mu_t.pow(2)
    mu_pt = mu_p * mu_t
    sigma_p2 = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_p2
    sigma_t2 = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_t2
    sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pt

    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2).clamp_min(eps)
    )
    return ssim_map.flatten(1).mean(dim=1).mean()


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
