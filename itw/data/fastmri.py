"""fastMRI single-coil dataset with k-space scout side information."""

from __future__ import annotations

import os

import h5py
import torch
import torch.fft as fft
from torch.utils.data import Dataset


class FastMRIDataset(Dataset):
    """
    Returns (Z, C, kspace) each shaped [1, H, W]:

    - C: full magnitude reconstruction from cropped k-space
    - Z: magnitude reconstruction from a central scout k-space window
    - kspace: complex cropped k-space used for row-mask acquisition
    """

    def __init__(
        self,
        h5_root: str,
        scout_size: int = 32,
        target_dim: int = 300,
    ):
        self.files = sorted(
            os.path.join(h5_root, f)
            for f in os.listdir(h5_root)
            if f.endswith(".h5")
        )
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found under {h5_root}")
        self.scout_size = scout_size
        self.target_dim = target_dim

    def __len__(self) -> int:
        return len(self.files)

    def _center_crop(self, data: torch.Tensor) -> torch.Tensor:
        h, w = data.shape[-2:]
        start_h = (h - self.target_dim) // 2
        start_w = (w - self.target_dim) // 2
        return data[
            ...,
            start_h : start_h + self.target_dim,
            start_w : start_w + self.target_dim,
        ]

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean()) / (x.std() + 1e-7)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.files[idx], "r") as f:
            kspace_np = f["kspace"][f["kspace"].shape[0] // 2]
            kspace = torch.from_numpy(kspace_np).contiguous()

        kspace = self._center_crop(kspace)

        c = torch.abs(fft.ifft2(fft.ifftshift(kspace, dim=(-2, -1))))

        h, w = kspace.shape
        z_mask = torch.zeros_like(kspace)
        s = self.scout_size // 2
        z_mask[h // 2 - s : h // 2 + s, w // 2 - s : w // 2 + s] = 1
        z = torch.abs(fft.ifft2(fft.ifftshift(kspace * z_mask, dim=(-2, -1))))

        c = self._normalize(c)
        z = self._normalize(z)

        return z.unsqueeze(0), c.unsqueeze(0), kspace.unsqueeze(0)
