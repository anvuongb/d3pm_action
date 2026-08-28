"""K-space PE-row discretization, NMSE, and 1D row-mask head."""

from __future__ import annotations

import torch

from itw.discrete import kspace_to_row_disc, nmse, psnr, ssim
from itw.masks import CartesianRowMaskGenerator, extract_acs_kspace, magnitude_from_kspace


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
    assert model.in_kind == "scout_image"


def test_extract_acs_kspace_shape_and_center_rows() -> None:
    h, w, scout = 16, 8, 4
    k = torch.zeros(2, 1, h, w, dtype=torch.complex64)
    lo = h // 2 - scout // 2
    hi = lo + scout
    k[:, :, lo:hi, :] = 1.0 + 2.0j
    k[:, :, 0, :] = 9.0 + 9.0j
    acs = extract_acs_kspace(k, scout_size=scout)
    assert tuple(acs.shape) == (2, 2, scout, w)
    assert not torch.is_complex(acs)
    assert torch.allclose(acs[:, 0], torch.ones(2, scout, w))
    assert torch.allclose(acs[:, 1], 2.0 * torch.ones(2, scout, w))


def test_acs_kspace_head_logits_from_complex_kspace() -> None:
    h, w, scout = 32, 24, 8
    model = CartesianRowMaskGenerator(
        target_rows=h, hidden=16, in_kind="acs_kspace", scout_size=scout
    )
    k = torch.randn(2, 1, h, w, dtype=torch.complex64)
    s = torch.full((2,), 0.25)
    logits = model(k, s)
    assert tuple(logits.shape) == (2, 1, h, 2)
    acs = extract_acs_kspace(k, scout_size=scout)
    logits_acs = model(acs, s)
    assert tuple(logits_acs.shape) == (2, 1, h, 2)
    assert torch.allclose(logits, logits_acs, atol=1e-5)


def test_scout_image_state_dict_has_no_acs_enc() -> None:
    model = CartesianRowMaskGenerator(target_rows=32, hidden=16)
    keys = model.state_dict().keys()
    assert any(k.startswith("row_in") for k in keys)
    assert not any("acs_enc" in k for k in keys)


def test_magnitude_from_kspace_matches_unmasked() -> None:
    k = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    mag = magnitude_from_kspace(k)
    assert mag.shape == k.shape
    assert not mag.is_complex()


def test_ssim_identity_near_one() -> None:
    x = torch.rand(2, 1, 32, 32)
    assert float(ssim(x, x)) > 0.99


def test_psnr_identity_is_inf() -> None:
    x = torch.rand(2, 1, 32, 32)
    val = float(psnr(x, x))
    assert val == float("inf")


def test_ssim_psnr_lower_when_different() -> None:
    x = torch.ones(2, 1, 32, 32)
    y = torch.zeros(2, 1, 32, 32)
    assert float(ssim(x, x)) > 0.99
    assert float(ssim(y, x)) < float(ssim(x, x))
    assert float(ssim(y, x)) < 0.1
    assert float(psnr(x, x)) == float("inf")
    assert float(psnr(y, x)) < 80.0
    noisy = x + 0.25 * torch.randn_like(x)
    assert float(ssim(noisy, x)) < float(ssim(x, x))
    assert float(psnr(noisy, x)) < float("inf")


def test_ssim_psnr_resolves_tiny_mri_magnitude() -> None:
    """Unnormalized FastMRI mag is ~1e-6; MSE must not be floored at 1e-8."""
    mag = torch.rand(2, 1, 32, 32) * 1e-6
    close = mag + 1e-9 * torch.randn_like(mag)
    far = mag * 0.2
    assert float(ssim(mag, mag)) > 0.99
    assert float(psnr(mag, mag)) == float("inf")
    assert float(psnr(close, mag)) > float(psnr(far, mag))
    assert float(ssim(close, mag)) > float(ssim(far, mag))
    assert float(psnr(close, mag)) > 0.0
