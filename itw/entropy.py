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
    """
    if mask.shape[1] == 1 and logits.shape[1] > 1:
        mask = mask.expand(-1, logits.shape[1], -1, -1)

    pixel_h = pixel_entropy(logits)
    masked_h = pixel_h * mask.float()
    denom = mask.float().sum(dim=(1, 2, 3)).clamp_min(eps)
    per_sample = masked_h.sum(dim=(1, 2, 3))
    if normalize:
        per_sample = per_sample / denom
    return per_sample.mean()


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
