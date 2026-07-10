"""Mask generators and masking utilities."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsity_features(sparse: torch.Tensor, n_freq: int = 16) -> torch.Tensor:
    sparse = sparse.float().reshape(-1, 1)
    features = []
    for i in range(n_freq):
        angle = sparse * math.pi * (2.0**i)
        features.append(torch.sin(angle))
        features.append(torch.cos(angle))
    return torch.cat(features, dim=1)


def gumbel_mask(
    logits: torch.Tensor,
    temperature: float = 0.5,
    hard: bool = True,
) -> torch.Tensor:
    """logits: [B, 1, H, W, 2] -> mask [B, 1, H, W] in {0, 1}."""
    mask_2channel = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
    return mask_2channel[..., 1]


def mask_loss(
    output_mask: torch.Tensor,
    target_density: torch.Tensor,
    binarization_weight: float = 0.0,
) -> torch.Tensor:
    target_density = target_density.to(output_mask.device, dtype=output_mask.dtype)
    if target_density.dim() == 1:
        target_density = target_density.reshape(-1)

    current_density = output_mask.mean(dim=(1, 2, 3))
    sparsity_loss = F.mse_loss(current_density, target_density)

    if binarization_weight > 0:
        bin_loss = (output_mask * (1.0 - output_mask)).mean()
        return sparsity_loss + binarization_weight * bin_loss
    return sparsity_loss


def apply_masked_observation(
    x: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    multichannel: bool = False,
) -> torch.Tensor:
    """
    Build D3PM input y from clean x and binary mask.

    Masked pixels -> 0 (absorbing). For multichannel data (CIFAR), observed
    bins are shifted to 1..N-1 so bin 0 is never used for real observations.
    """
    if mask.shape[1] == 1 and x.shape[1] > 1:
        mask_expanded = mask.expand_as(x)
    else:
        mask_expanded = mask

    if multichannel and num_classes > 2:
        observed = torch.clamp(x + 1, 1, num_classes - 1)
        return torch.where(mask_expanded > 0, observed, torch.zeros_like(x))

    return x * mask_expanded


def conditioning_features(x: torch.Tensor, pool_size: int = 4) -> torch.Tensor:
    """Grayscale 8x8 summary for spatial mask generators."""
    if x.dtype == torch.long:
        denom = max(int(x.max().item()), 1)
        gray = x.float().mean(dim=1, keepdim=True) / denom
    else:
        gray = x.float().mean(dim=1, keepdim=True)
    return F.avg_pool2d(gray, pool_size, pool_size)


class MaskGeneratorMLP(nn.Module):
    """Class + sparsity conditioned mask (image-blind baseline)."""

    def __init__(self, num_classes: int = 10, hidden_dim: int = 128, size: int = 32):
        super().__init__()
        self.size = size
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        self.semb = nn.Linear(32, hidden_dim)
        out_dim = 2 * size * size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, labels: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        sx = sparsity_features(sparse).to(labels.device)
        embeds = self.embedding(labels) + self.semb(sx)
        logits = self.mlp(embeds)
        return logits.view(-1, 1, self.size, self.size, 2)


class SpatialMaskGenerator(nn.Module):
    """Image-conditioned mask generator (8x8 cond -> full-res logits)."""

    def __init__(self, in_channels: int = 1, scalar_dim: int = 16, size: int = 32):
        super().__init__()
        self.size = size
        self.scalar_mlp = nn.Sequential(
            nn.Linear(1, scalar_dim),
            nn.ReLU(),
            nn.Linear(scalar_dim, scalar_dim),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + scalar_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, x_cond: torch.Tensor, sparsity: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x_cond.shape
        s_feat = self.scalar_mlp(sparsity.reshape(b, 1))
        s_feat = s_feat.view(b, -1, 1, 1).expand(-1, -1, h, w)
        feat = self.encoder(torch.cat([x_cond, s_feat], dim=1))
        logits = self.decoder(feat)
        return logits.unsqueeze(1).permute(0, 1, 3, 4, 2)


class SpatialMaskGeneratorMNIST(SpatialMaskGenerator):
    """Spatial mask generator with optional class embedding for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        scalar_dim: int = 16,
        class_dim: int = 32,
        size: int = 32,
    ):
        super().__init__(in_channels=in_channels + class_dim, scalar_dim=scalar_dim, size=size)
        self.class_embed = nn.Embedding(num_classes, class_dim)

    def forward(
        self,
        x_cond: torch.Tensor,
        sparsity: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, h, w = x_cond.shape
        s_feat = self.scalar_mlp(sparsity.reshape(b, 1))
        s_feat = s_feat.view(b, -1, 1, 1).expand(-1, -1, h, w)

        if labels is not None:
            c_feat = self.class_embed(labels).view(b, -1, 1, 1).expand(-1, -1, h, w)
            x_cond = torch.cat([x_cond, c_feat], dim=1)

        feat = self.encoder(torch.cat([x_cond, s_feat], dim=1))
        logits = self.decoder(feat)
        return logits.unsqueeze(1).permute(0, 1, 3, 4, 2)
