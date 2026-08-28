"""Conditional entropy proxies via frozen D3PM."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pixel_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-pixel categorical entropy. logits: [..., num_classes]."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def masked_cond_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    H(X|Y) proxy on observed pixels only.

    logits: [B, C, H, W, num_classes]
    mask:   [B, 1, H, W] or [B, C, H, W] in {0, 1}

    Empty masks are undefined for masked-only entropy; they are assigned
    maximum categorical entropy log(N) so the objective cannot reward
    selecting zero sites (critical for nested FastMRI coarse proxy).
    """
    if mask.shape[1] == 1 and logits.shape[1] > 1:
        mask = mask.expand(-1, logits.shape[1], -1, -1)

    pixel_h = pixel_entropy(logits)
    masked_h = pixel_h * mask.float()
    denom_raw = mask.float().sum(dim=(1, 2, 3))
    denom = denom_raw.clamp_min(eps)
    per_sample = masked_h.sum(dim=(1, 2, 3))
    if normalize:
        per_sample = per_sample / denom

    n_classes = logits.shape[-1]
    max_h = float(torch.log(torch.tensor(float(n_classes), device=logits.device)))
    empty = denom_raw < eps
    if empty.any():
        per_sample = torch.where(
            empty, torch.full_like(per_sample, max_h), per_sample
        )

    return per_sample.mean()


def mean_cond_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean categorical entropy over all spatial sites."""
    return pixel_entropy(logits).mean()


def d3pm_cond_entropy_loss(
    d3pm,
    y: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    mask: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Conditional-entropy proxy. D3PM weights must be frozen; gradients flow
    through y (and thus the mask) via the forward pass.
    """
    logits = d3pm.model_predict(y, t, cond)
    return masked_cond_entropy(logits, mask, normalize=normalize)


def fine_cond_entropy_loss(
    d3pm_fine,
    y_fine: torch.Tensor,
    t_fine: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    """Full-image mean entropy for k-space-undersampled fine observations."""
    logits = d3pm_fine.model_predict(y_fine, t_fine, cond)
    return mean_cond_entropy(logits)


def coarse_cond_entropy_loss(
    d3pm_coarse,
    y_row: torch.Tensor,
    t_coarse: torch.Tensor,
    cond: torch.Tensor,
    row_mask: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Masked-only entropy on selected PE rows."""
    return d3pm_cond_entropy_loss(
        d3pm_coarse, y_row, t_coarse, cond, row_mask, normalize=normalize
    )


def nested_cond_entropy_loss(
    h_coarse: torch.Tensor,
    h_fine: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Weighted sum of coarse and fine conditional-entropy proxies."""
    return alpha * h_coarse + beta * h_fine
