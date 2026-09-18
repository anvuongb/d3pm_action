"""Small U-Net reconstructor for the B4 headroom test.

Zero-filled IFFT rewards raw k-space energy capture, which a per-row energy
average already solves (F7). Mask learning can only pay off if the
reconstructor carries a prior. This is the cheap stand-in for that prior: if
the mask ranking does not move under a trained reconstructor, the task has no
headroom for mask learning at all.

Trained under a mixture of *heuristic* masks only — never the learned masks —
so it cannot be biased toward the policy under test.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

TRAIN_MASK_FAMILIES = (
    "random",
    "equispaced",
    "vd_gaussian",
    "acs_random",
    "acs_vd_gaussian",
    "acs_equispaced",
)


def zero_filled_complex(kspace: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    """Masked k-space -> complex image [B, 1, H, W] (keeps phase)."""
    width = kspace.shape[-1]
    mask2d = row_mask.expand(-1, -1, -1, width) if row_mask.shape[-1] == 1 else row_mask
    return fft.ifft2(fft.ifftshift(kspace * mask2d.to(kspace.dtype), dim=(-2, -1)))


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


class ReconUNet(nn.Module):
    """[B,3,H,W] real+imag+mask zero-filled -> [B,1,H,W] magnitude residual.

    Predicts a correction to |zero-filled| rather than the image from scratch,
    which keeps the identity reachable and makes the model easy to train on
    ~1k slices. The third channel is the row mask: how much correction is
    appropriate depends entirely on which lines were actually acquired, and
    without it the net has to infer the sampling density from the artefacts.
    """

    def __init__(self, base: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1, self.enc2, self.enc3 = _block(3, c1), _block(c1, c2), _block(c2, c3)
        self.bottleneck = _block(c3, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _block(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _block(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _block(c1 * 2, c1)
        self.out = nn.Conv2d(c1, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        y = self.out(d1)
        if ph or pw:
            y = y[..., :h, :w]
        return y


def reconstruct(
    model: ReconUNet | None,
    kspace: torch.Tensor,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    """Magnitude reconstruction. ``model=None`` gives plain zero-filled."""
    zf = zero_filled_complex(kspace, row_mask)
    mag = zf.abs()
    if model is None:
        return mag
    scale = mag.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    width = kspace.shape[-1]
    mask2d = row_mask.expand(-1, -1, -1, width) if row_mask.shape[-1] == 1 else row_mask
    inp = torch.cat([zf.real / scale, zf.imag / scale, mask2d.to(mag.dtype)], dim=1)
    return (mag / scale + model(inp)).clamp_min(0.0) * scale


def sample_training_masks(
    batch_size: int,
    n_rows: int,
    device: str | torch.device,
    acs_width: int,
    sparsity_min: float,
    sparsity_max: float,
) -> torch.Tensor:
    """One random heuristic family + random sparsity per sample."""
    from .eval import baseline_row_mask_batch

    masks = []
    for _ in range(batch_size):
        fam = TRAIN_MASK_FAMILIES[int(torch.randint(len(TRAIN_MASK_FAMILIES), (1,)))]
        s = float(torch.rand(1)) * (sparsity_max - sparsity_min) + sparsity_min
        masks.append(
            baseline_row_mask_batch(
                fam, 1, n_rows, torch.full((1,), s, device=device), device,
                acs_width=acs_width,
            )
        )
    return torch.cat(masks, dim=0)


def train_recon_unet(
    dataloader,
    device: str,
    n_rows: int,
    acs_width: int = 32,
    n_epochs: int = 30,
    lr: float = 2e-4,
    base: int = 32,
    sparsity_min: float = 0.10,
    sparsity_max: float = 0.75,
    save_path: Path | None = None,
    log_path: Path | None = None,
) -> tuple[ReconUNet, list[dict]]:
    from .masks import magnitude_from_kspace

    model = ReconUNet(base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    history: list[dict] = []
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"recon unet params={n_params}\n", encoding="utf-8")

    for epoch in range(n_epochs):
        model.train()
        tot_loss = tot_zf = tot_un = 0.0
        n = 0
        pbar = tqdm(dataloader, desc=f"recon epoch {epoch}")
        for _z, _c, kspace in pbar:
            kspace = kspace.to(device)
            target = magnitude_from_kspace(kspace)
            mask = sample_training_masks(
                kspace.shape[0], n_rows, device, acs_width, sparsity_min, sparsity_max
            )
            pred = reconstruct(model, kspace, mask)
            denom = (target**2).sum(dim=(1, 2, 3)).clamp_min(1e-12)
            per_sample = ((pred - target) ** 2).sum(dim=(1, 2, 3)) / denom
            with torch.no_grad():
                zf = reconstruct(None, kspace, mask)
                zf_per_sample = ((zf - target) ** 2).sum(dim=(1, 2, 3)) / denom
                zf_nmse = zf_per_sample.mean()
            # Scale-free: NMSE relative to zero-filled, so a sample is worth the
            # same whether it is s=0.10 (NMSE ~0.3) or s=0.75 (NMSE ~0.002).
            # Absolute NMSE is ~100x larger at low s, so an unnormalised loss is
            # dominated by easy-to-improve samples and the net learns to damage
            # the high-s images it should leave alone.
            loss = (per_sample / zf_per_sample.clamp_min(1e-12)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss) * kspace.shape[0]
            tot_un += float(per_sample.mean()) * kspace.shape[0]
            tot_zf += float(zf_nmse) * kspace.shape[0]
            n += kspace.shape[0]
            pbar.set_postfix(rel=f"{tot_loss/n:.4f}", unet=f"{tot_un/n:.5f}",
                             zf=f"{tot_zf/n:.5f}")

        rec = {
            "epoch": epoch,
            "unet_nmse": tot_un / n,
            "zero_filled_nmse": tot_zf / n,
            "rel_to_zero_filled": tot_loss / n,
        }
        history.append(rec)
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"epoch: {epoch}, rel_to_zf: {rec['rel_to_zero_filled']:.4f}, "
                    f"unet_nmse: {rec['unet_nmse']:.6f}, "
                    f"zero_filled_nmse: {rec['zero_filled_nmse']:.6f}\n"
                )
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"recon_unet": model.state_dict(), "epoch": epoch,
                        "history": history, "base": base}, save_path)
    return model, history
