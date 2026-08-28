"""Backbones for FastMRI coarse (row) and fine D3PM priors."""

from __future__ import annotations

import torch
import torch.nn as nn

from d3pm_runner import DummyX0Model


def _fourier_time_features(t: torch.Tensor, n_freq: int = 16) -> torch.Tensor:
    t = t.float().reshape(-1, 1) / 1000.0
    feats = [torch.sin(t * 3.1415 * 2**i) for i in range(n_freq)] + [
        torch.cos(t * 3.1415 * 2**i) for i in range(n_freq)
    ]
    return torch.cat(feats, dim=1)


class RowX0Model(nn.Module):
    """
    1D absorbing-state x0 predictor over PE row profiles.

    Input x: [B, 1, H, 1] integer bins
    Output:  [B, 1, H, 1, N] logits
    """

    def __init__(self, n_bins: int = 8, hidden: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.N = n_bins
        self.input_proj = nn.Conv1d(1, hidden, kernel_size=5, padding=2)
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                    nn.GroupNorm(8, hidden),
                    nn.GELU(),
                ]
            )
        self.backbone = nn.Sequential(*layers)
        self.temb = nn.Linear(32, hidden)
        self.out = nn.Conv1d(hidden, n_bins, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        del cond  # unconditional
        if x.dim() != 4 or x.shape[1] != 1 or x.shape[-1] != 1:
            raise ValueError(f"Expected x shape [B,1,H,1], got {tuple(x.shape)}")

        x1 = (2.0 * x.float().squeeze(-1) / self.N) - 1.0  # [B, 1, H]
        h = self.input_proj(x1)
        t_emb = self.temb(_fourier_time_features(t).to(x.device)).unsqueeze(-1)
        h = self.backbone(h + t_emb)
        logits = self.out(h)  # [B, N, H]
        # [B, N, H] -> [B, 1, H, 1, N]
        return logits.permute(0, 2, 1).unsqueeze(1).unsqueeze(3).contiguous()


class FineX0Model(DummyX0Model):
    """Grayscale spatial x0 model for fine FastMRI D3PM (96x96, N bins)."""

    def __init__(self, n_bins: int = 8) -> None:
        super().__init__(n_channel=1, N=n_bins)


def build_fine_backbone(n_bins: int = 8) -> nn.Module:
    return FineX0Model(n_bins=n_bins)


def build_coarse_backbone(n_bins: int = 8, hidden: int = 64) -> nn.Module:
    return RowX0Model(n_bins=n_bins, hidden=hidden)
