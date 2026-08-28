"""Mask generators and masking utilities."""

from __future__ import annotations

import math

import torch
import torch.fft as fft
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


def acs_bounds(n_rows: int, acs_width: int) -> tuple[int, int]:
    """Centered ACS half-open slice ``[lo, hi)`` on ``n_rows`` PE lines."""
    acs_width = min(max(int(acs_width), 0), int(n_rows))
    lo = (n_rows - acs_width) // 2
    return lo, lo + acs_width


def _row_budget(sparsity: torch.Tensor | float, n_rows: int) -> int:
    k = int(float(sparsity) * n_rows)
    return max(0, min(n_rows, k))


def gumbel_row_mask(
    logits: torch.Tensor,
    temperature: float = 0.5,
    hard: bool = True,
) -> torch.Tensor:
    """logits: [B, 1, H, 2] -> row mask [B, 1, H, 1] in {0, 1}."""
    mask_2channel = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
    return mask_2channel[..., 1:].contiguous()


def gumbel_row_mask_ste(
    logits: torch.Tensor,
    temperature: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Soft keep-probs and STE hard mask from the same Gumbel sample.

    Returns (row_mask_soft, row_mask_hard), both [B, 1, H, 1].
    """
    soft = gumbel_row_mask(logits, temperature=temperature, hard=False)
    hard = ((soft > 0.5).to(soft.dtype) - soft).detach() + soft
    return soft, hard


def _acs_lock_drop_neg_inf(
    logits: torch.Tensor, acs_width: int
) -> torch.Tensor:
    """Set ACS drop-logits to -inf so keep-prob ≈ 1 before overwrite."""
    h = logits.shape[2]
    lo, hi = acs_bounds(h, acs_width)
    out = logits.clone()
    out[:, :, lo:hi, 0] = float("-inf")
    return out


def apply_acs_lock_overwrite(
    mask: torch.Tensor,
    sparsity: torch.Tensor,
    acs_width: int,
) -> torch.Tensor:
    """Hard-set ACS (or a centered k-block when k <= acs_width).

    ``mask``: [B, 1, H, 1]. Remainder rows are left unchanged when k > acs.
    """
    b, _, h, _ = mask.shape
    lo, hi = acs_bounds(h, acs_width)
    out = mask.clone()
    for i in range(b):
        k = _row_budget(sparsity[i], h)
        if k <= acs_width:
            out[i] = 0
            if k > 0:
                inner_lo = (h - k) // 2
                out[i, 0, inner_lo : inner_lo + k, 0] = 1.0
        else:
            out[i, 0, lo:hi, 0] = 1.0
    return out


def gumbel_row_mask_acs_locked(
    logits: torch.Tensor,
    sparsity: torch.Tensor,
    acs_width: int = 32,
    temperature: float = 0.5,
    hard: bool = True,
) -> torch.Tensor:
    """Gumbel row mask with ACS rows overwritten to 1. [B, 1, H, 2] -> [B, 1, H, 1]."""
    locked = _acs_lock_drop_neg_inf(logits, acs_width)
    mask = gumbel_row_mask(locked, temperature=temperature, hard=hard)
    return apply_acs_lock_overwrite(mask, sparsity, acs_width)


def gumbel_row_mask_ste_acs_locked(
    logits: torch.Tensor,
    sparsity: torch.Tensor,
    acs_width: int = 32,
    temperature: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """STE Gumbel with hard ACS overwrite on both soft and hard masks."""
    locked = _acs_lock_drop_neg_inf(logits, acs_width)
    soft, hard = gumbel_row_mask_ste(locked, temperature=temperature)
    soft = apply_acs_lock_overwrite(soft, sparsity, acs_width)
    hard = apply_acs_lock_overwrite(hard, sparsity, acs_width)
    return soft, hard


def expand_row_mask(row_mask: torch.Tensor, width: int) -> torch.Tensor:
    """Expand [B, 1, H, 1] row mask to [B, 1, H, W]."""
    return row_mask.expand(-1, -1, -1, width)


def row_mask_loss(
    row_mask: torch.Tensor,
    target_density: torch.Tensor,
) -> torch.Tensor:
    """MSE between mean selected-row fraction and target sparsity."""
    target_density = target_density.to(row_mask.device, dtype=row_mask.dtype)
    if target_density.dim() == 1:
        target_density = target_density.reshape(-1)
    current_density = row_mask.mean(dim=(1, 2, 3))
    return F.mse_loss(current_density, target_density)


def apply_kspace_row_mask(
    kspace: torch.Tensor,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Cartesian row mask in k-space and return magnitude reconstruction.

    kspace:   [B, 1, H, W] complex
    row_mask: [B, 1, H, 1] or [B, 1, H, W]
    returns:  [B, 1, H, W] real magnitude image
    """
    width = kspace.shape[-1]
    if row_mask.shape[-1] == 1:
        mask_2d = expand_row_mask(row_mask, width)
    else:
        mask_2d = row_mask

    masked = kspace * mask_2d.to(dtype=kspace.dtype)
    return torch.abs(fft.ifft2(fft.ifftshift(masked, dim=(-2, -1))))


def magnitude_from_kspace(kspace: torch.Tensor) -> torch.Tensor:
    """Full-sampled magnitude reconstruction. kspace: [B, 1, H, W] complex."""
    return torch.abs(fft.ifft2(fft.ifftshift(kspace, dim=(-2, -1))))


def extract_acs_kspace(
    kspace: torch.Tensor,
    scout_size: int = 32,
) -> torch.Tensor:
    """
    Center ACS PE strip as real+imag channels (phase preserved).

    kspace: [B, 1, H, W] or [B, H, W] complex
    returns: [B, 2, scout_size, W] float (channel 0 = real, 1 = imag)
    """
    if kspace.dim() == 3:
        kspace = kspace.unsqueeze(1)
    if kspace.dim() != 4 or kspace.shape[1] != 1:
        raise ValueError(f"expected kspace [B, 1, H, W], got {tuple(kspace.shape)}")
    if not torch.is_complex(kspace):
        raise ValueError("extract_acs_kspace expects complex k-space")
    _b, _c, h, _w = kspace.shape
    if scout_size > h:
        raise ValueError(f"scout_size={scout_size} exceeds PE height {h}")
    lo = h // 2 - scout_size // 2
    hi = lo + scout_size
    acs = kspace[:, 0, lo:hi, :]
    return torch.stack((acs.real, acs.imag), dim=1).contiguous()


def policy_condition(
    policy_input: str,
    z: torch.Tensor,
    kspace: torch.Tensor,
    scout_size: int = 32,
) -> torch.Tensor:
    """Conditioning tensor for CartesianRowMaskGenerator.forward."""
    if policy_input == "acs_kspace":
        return extract_acs_kspace(kspace, scout_size=scout_size)
    return z


class CartesianRowMaskGenerator(nn.Module):
    """
    Conditioned Cartesian row mask generator.

    ``in_kind="scout_image"`` (default): row-pool image-domain scout ``Z``
    and a 1D conv stack predicts keep/drop logits. Compatible with the
    nested parent checkpoint.

    ``in_kind="acs_kspace"``: 2D conv over the centered ACS k-space strip
    (real+imag), pool frequency, upsample to ``target_rows``, then the
    same 1D row-logit head. Outputs [B, 1, H, 2] over all PE lines.
    """

    def __init__(
        self,
        target_rows: int = 300,
        hidden: int = 64,
        in_channels: int = 1,
        in_kind: str = "scout_image",
        scout_size: int = 32,
    ):
        super().__init__()
        del in_channels
        if in_kind not in ("scout_image", "acs_kspace"):
            raise ValueError(f"unknown in_kind {in_kind}")
        self.target_rows = target_rows
        self.in_kind = in_kind
        self.scout_size = scout_size
        if in_kind == "acs_kspace":
            self.acs_enc = nn.Sequential(
                nn.Conv2d(2, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
            )
            row_in_ch = hidden + 1
        else:
            self.acs_enc = None
            row_in_ch = 2
        self.row_in = nn.Conv1d(row_in_ch, hidden, kernel_size=7, padding=3)
        self.backbone = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.out = nn.Conv1d(hidden, 2, kernel_size=1)

    def _row_logits(self, h1: torch.Tensor) -> torch.Tensor:
        logits = self.out(self.backbone(h1))  # [B, 2, H]
        return logits.permute(0, 2, 1).unsqueeze(1).contiguous()  # [B, 1, H, 2]

    def _prepare_acs(self, cond: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(cond):
            return extract_acs_kspace(cond, scout_size=self.scout_size)
        if cond.dim() != 4 or cond.shape[1] != 2:
            raise ValueError(
                "acs_kspace cond must be complex kspace [B,1,H,W] or "
                f"real+imag [B, 2, scout, W], got {tuple(cond.shape)} "
                f"dtype={cond.dtype}"
            )
        return cond.float()

    def forward(self, cond: torch.Tensor, sparsity: torch.Tensor) -> torch.Tensor:
        if self.in_kind == "acs_kspace":
            acs = self._prepare_acs(cond)
            b = acs.shape[0]
            feat = self.acs_enc(acs)  # [B, hidden, scout, W]
            row = feat.mean(dim=-1)  # [B, hidden, scout]
            row_h = F.interpolate(
                row,
                size=self.target_rows,
                mode="linear",
                align_corners=False,
            )
            s_ch = sparsity.reshape(b, 1, 1).float().expand(b, 1, self.target_rows)
            return self._row_logits(self.row_in(torch.cat([row_h, s_ch], dim=1)))

        z = cond.float()
        b, _, h, _w = z.shape
        row = z.mean(dim=-1)  # [B, 1, H]
        s_ch = sparsity.reshape(b, 1, 1).float().expand(b, 1, h)
        return self._row_logits(self.row_in(torch.cat([row, s_ch], dim=1)))

