"""K-space PE-row discretization, NMSE, and 1D row-mask head."""

from __future__ import annotations

import torch

from itw.discrete import kspace_to_row_disc, nmse
from itw.masks import CartesianRowMaskGenerator, magnitude_from_kspace


def test_kspace_to_row_disc_shape_and_range() -> None:
    k = torch.randn(2, 1, 16, 8, dtype=torch.complex64)
    x = kspace_to_row_disc(k, n_bins=8)
    assert tuple(x.shape) == (2, 1, 16, 1)
    assert x.dtype == torch.long
    assert int(x.min()) >= 0
    assert int(x.max()) <= 7


def test_nmse_zero_on_identity() -> None:
    x = torch.randn(3, 1, 8, 8)
    assert float(nmse(x, x)) < 1e-6


def test_nmse_positive_when_different() -> None:
    x = torch.ones(2, 1, 4, 4)
    y = torch.zeros(2, 1, 4, 4)
    assert float(nmse(y, x)) == 1.0


def test_cartesian_row_head_1d_logits() -> None:
    model = CartesianRowMaskGenerator(target_rows=32, hidden=16)
    z = torch.randn(2, 1, 32, 32)
    s = torch.full((2,), 0.25)
    logits = model(z, s)
    assert tuple(logits.shape) == (2, 1, 32, 2)


def test_magnitude_from_kspace_matches_unmasked() -> None:
    k = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    mag = magnitude_from_kspace(k)
    assert mag.shape == k.shape
    assert not mag.is_complex()
