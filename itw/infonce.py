"""InfoNCE mutual-information surrogate for continuous reconstructions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Conv encoder mapping [B,1,H,W] images to L2-normalized feature vectors."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat_y: torch.Tensor, feat_c: torch.Tensor) -> torch.Tensor:
        batch_size = feat_y.shape[0]
        feat_y = F.normalize(feat_y, dim=1)
        feat_c = F.normalize(feat_c, dim=1)
        logits = torch.matmul(feat_y, feat_c.T) / self.temperature
        labels = torch.arange(batch_size, device=feat_y.device)
        return F.cross_entropy(logits, labels)
