"""STE quantization keeps gradients through the observation path."""

from __future__ import annotations

import torch

from itw.discrete import apply_row_absorbing_observation, magnitude_to_fine_disc


def test_fine_disc_ste_requires_grad() -> None:
    row_mask = torch.rand(2, 1, 8, 1, requires_grad=True)
    kspace = torch.randn(2, 1, 8, 8)
    y_mag = (kspace * row_mask.expand(-1, -1, -1, 8)).abs()
    y_fine = magnitude_to_fine_disc(y_mag, size=8, n_bins=8, ste=True)
    assert y_fine.dtype.is_floating_point
    assert y_fine.requires_grad
    y_fine.sum().backward()
    assert row_mask.grad is not None
    assert float(row_mask.grad.abs().sum()) > 0.0


def test_row_absorb_multiplicative_grad() -> None:
    c_row = torch.randint(0, 8, (2, 1, 16, 1))
    row_mask = torch.rand(2, 1, 16, 1, requires_grad=True)
    y_row = apply_row_absorbing_observation(c_row, row_mask, n_bins=8)
    assert y_row.requires_grad
    y_row.sum().backward()
    assert row_mask.grad is not None
    assert float(row_mask.grad.abs().sum()) > 0.0
