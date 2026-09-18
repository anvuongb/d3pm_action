"""PyTorch port of LOUPE (Bahadir et al. 2020), as a baseline for our masks.

Ported from https://github.com/cagladbahadir/LOUPE (Keras/TF1). The upstream
code cannot run here -- it imports ``keras.backend.tensorflow_backend`` and
``keras.layers.normalization``, both removed in TF2, and TF1 needs Python 3.7 --
so this reproduces its mask mechanism and U-Net in PyTorch.

Fidelity notes, since Keras defaults differ from PyTorch's and silently change
the model:

- ``LeakyReLU`` alpha is 0.3 in Keras, 0.01 in PyTorch.
- ``BatchNormalization`` is eps=1e-3, momentum=0.99 in Keras; PyTorch's momentum
  has the opposite sense, so 0.99 becomes 0.01.
- ``Conv2D`` initialises glorot_uniform with zero bias, not kaiming.
- Downsampling is *average* pooling and upsampling is nearest-neighbour.
- ``tf.fft2d``/``tf.ifft2d`` apply no fftshift, so DC sits at index 0 and the
  learned mask is in unshifted k-space. Our masks are fftshift-centred; see
  ``probmask_rows_to_centred``.

Verified against the upstream TF implementation on identical inputs; see
``tests/test_loupe_port.py``.
"""

from __future__ import annotations

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

PMASK_SLOPE = 5.0
SAMPLE_SLOPE = 12.0


def _conv(cin: int, cout: int, k: int = 3) -> nn.Conv2d:
    c = nn.Conv2d(cin, cout, k, padding=k // 2)
    nn.init.xavier_uniform_(c.weight)
    nn.init.zeros_(c.bias)
    return c


def _bn(c: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)


class _Block(nn.Module):
    """Conv-LeakyReLU-BN twice, as in LOUPE's hard-coded U-Net."""

    def __init__(self, cin: int, cout: int, kern: int = 3):
        super().__init__()
        self.c1, self.b1 = _conv(cin, cout, kern), _bn(cout)
        self.c2, self.b2 = _conv(cout, cout, kern), _bn(cout)
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.b1(self.act(self.c1(x)))
        return self.b2(self.act(self.c2(x)))


class LoupeUNet(nn.Module):
    """LOUPE's U-Net: 5 levels, average pooling, nearest upsampling, 1x1 head."""

    def __init__(self, cin: int = 2, filt: int = 64, kern: int = 3, cout: int = 1):
        super().__init__()
        f = filt
        self.e1 = _Block(cin, f, kern)
        self.e2 = _Block(f, f * 2, kern)
        self.e3 = _Block(f * 2, f * 4, kern)
        self.e4 = _Block(f * 4, f * 8, kern)
        self.e5 = _Block(f * 8, f * 16, kern)
        self.d4 = _Block(f * 16 + f * 8, f * 8, kern)
        self.d3 = _Block(f * 8 + f * 4, f * 4, kern)
        self.d2 = _Block(f * 4 + f * 2, f * 2, kern)
        self.d1 = _Block(f * 2 + f, f, kern)
        self.head = _conv(f, cout, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 16, (-w) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        c1 = self.e1(x)
        c2 = self.e2(F.avg_pool2d(c1, 2))
        c3 = self.e3(F.avg_pool2d(c2, 2))
        c4 = self.e4(F.avg_pool2d(c3, 2))
        c5 = self.e5(F.avg_pool2d(c4, 2))
        up = F.interpolate(c5, scale_factor=2, mode="nearest")
        d = self.d4(torch.cat([c4, up], 1))
        d = self.d3(torch.cat([c3, F.interpolate(d, scale_factor=2, mode="nearest")], 1))
        d = self.d2(torch.cat([c2, F.interpolate(d, scale_factor=2, mode="nearest")], 1))
        d = self.d1(torch.cat([c1, F.interpolate(d, scale_factor=2, mode="nearest")], 1))
        y = self.head(d)
        if ph or pw:
            y = y[..., :h, :w]
        return y


def rescale_prob_map(x: torch.Tensor, sparsity: float) -> torch.Tensor:
    """LOUPE's RescaleProbMap: force mean(x) == sparsity, staying in [0, 1]."""
    xbar = x.mean()
    r = sparsity / xbar
    beta = (1.0 - sparsity) / (1.0 - xbar)
    if r <= 1:
        return x * r
    return 1.0 - (1.0 - x) * beta


class ProbMask(nn.Module):
    """One logit per k-space location (or per PE row when ``rows``)."""

    def __init__(self, n_rows: int, n_cols: int, rows: bool = True,
                 slope: float = PMASK_SLOPE, eps: float = 0.01):
        super().__init__()
        shape = (1, 1, n_rows, 1) if rows else (1, 1, n_rows, n_cols)
        u = torch.empty(shape).uniform_(eps, 1.0 - eps)
        # LOUPE's initializer: logit(u) / slope, so sigmoid(slope*w) ~ U[eps,1-eps].
        self.logit = nn.Parameter(-torch.log(1.0 / u - 1.0) / slope)
        self.slope = slope

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.slope * self.logit)


class Loupe(nn.Module):
    """End-to-end LOUPE: learned probability mask + jointly trained U-Net.

    Input is a real magnitude image, as upstream: it is FFT'd with a zero
    imaginary channel, so k-space is Hermitian-symmetric. That is LOUPE's own
    forward model, not ours.
    """

    def __init__(self, n_rows: int, n_cols: int, sparsity: float,
                 rows: bool = True, filt: int = 64, kern: int = 3,
                 sample_slope: float = SAMPLE_SLOPE):
        super().__init__()
        self.prob = ProbMask(n_rows, n_cols, rows=rows)
        self.unet = LoupeUNet(2, filt, kern, 1)
        self.sparsity, self.sample_slope = sparsity, sample_slope

    def sampled_mask(self, batch: int, device, hard: bool = False) -> torch.Tensor:
        p = rescale_prob_map(self.prob(), self.sparsity).expand(batch, -1, -1, -1)
        thresh = torch.rand_like(p)
        if hard:
            return (p > thresh).float()
        return torch.sigmoid(self.sample_slope * (p - thresh))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        k = fft.fft2(torch.complex(img, torch.zeros_like(img)))   # unshifted, as tf.fft2d
        m = self.sampled_mask(img.shape[0], img.device)
        zf = fft.ifft2(k * m)
        two = torch.cat([zf.real, zf.imag], dim=1)
        return zf.abs() + self.unet(two)


def probmask_rows_to_centred(prob_rows: torch.Tensor) -> torch.Tensor:
    """Unshifted LOUPE row profile -> our fftshift-centred row indexing.

    LOUPE's FFT puts DC at row 0; our masks are applied to fftshift-centred
    k-space. Without this the learned low-frequency preference lands at the
    edges of our grid.
    """
    return torch.fft.fftshift(prob_rows.flatten())
