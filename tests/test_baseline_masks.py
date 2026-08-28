"""Exact-budget MRI row-mask baselines (equispaced, VD-Gaussian, ACS+random)."""

from __future__ import annotations

import torch

from itw.eval import (
    acs_equispaced_row_mask_batch,
    acs_random_row_mask_batch,
    acs_vd_gaussian_row_mask_batch,
    equispaced_row_mask_batch,
    random_row_mask_batch,
    sparsity_key,
    vd_gaussian_row_mask_batch,
)


def _budget(s: float, n_rows: int) -> int:
    return max(0, min(n_rows, int(s * n_rows)))


def _assert_shape_and_budget(
    mask: torch.Tensor, batch_size: int, n_rows: int, sparsity: torch.Tensor
) -> None:
    assert tuple(mask.shape) == (batch_size, 1, n_rows, 1)
    ones = mask.reshape(batch_size, -1).sum(dim=1)
    expected = torch.tensor(
        [_budget(float(sparsity[i]), n_rows) for i in range(batch_size)],
        dtype=ones.dtype,
        device=ones.device,
    )
    assert torch.equal(ones, expected)


def test_factories_shape_and_exact_floor_budget() -> None:
    torch.manual_seed(0)
    batch_size, n_rows = 4, 32
    factories = (
        random_row_mask_batch,
        equispaced_row_mask_batch,
        vd_gaussian_row_mask_batch,
        acs_random_row_mask_batch,
        acs_vd_gaussian_row_mask_batch,
        acs_equispaced_row_mask_batch,
    )
    for s in (0.1, 0.25, 0.4):
        sparsity = torch.full((batch_size,), s)
        k = _budget(s, n_rows)
        assert k == int(s * n_rows)
        for factory in factories:
            mask = factory(batch_size, n_rows, sparsity, "cpu")
            _assert_shape_and_budget(mask, batch_size, n_rows, sparsity)


def test_factories_exact_budget_fastmri_height() -> None:
    torch.manual_seed(1)
    batch_size, n_rows = 3, 300
    sparsity = torch.tensor([0.1, 0.25, 0.4])
    for factory in (
        random_row_mask_batch,
        equispaced_row_mask_batch,
        vd_gaussian_row_mask_batch,
        acs_random_row_mask_batch,
        acs_vd_gaussian_row_mask_batch,
        acs_equispaced_row_mask_batch,
    ):
        mask = factory(batch_size, n_rows, sparsity, "cpu")
        _assert_shape_and_budget(mask, batch_size, n_rows, sparsity)


def test_equispaced_unique_and_nearly_equal_spacing() -> None:
    torch.manual_seed(2)
    batch_size, n_rows = 8, 64
    for s in (0.1, 0.25, 0.4):
        sparsity = torch.full((batch_size,), s)
        k = _budget(s, n_rows)
        mask = equispaced_row_mask_batch(batch_size, n_rows, sparsity, "cpu")
        stride = n_rows / k
        for i in range(batch_size):
            idx = torch.where(mask[i, 0, :, 0] > 0.5)[0]
            assert int(idx.numel()) == k
            assert int(torch.unique(idx).numel()) == k
            ordered = idx.sort().values
            gaps = ordered[1:] - ordered[:-1]
            wrap = ordered[0] + n_rows - ordered[-1]
            all_gaps = torch.cat([gaps, wrap.view(1)])
            assert int(all_gaps.min()) >= max(int(stride) - 1, 1)
            assert int(all_gaps.max()) <= int(stride) + 2
            assert float(all_gaps.float().std()) < stride


def test_vd_gaussian_unique_and_center_heavy() -> None:
    torch.manual_seed(3)
    batch_size, n_rows, s = 256, 64, 0.25
    sparsity = torch.full((batch_size,), s)
    k = _budget(s, n_rows)
    mask = vd_gaussian_row_mask_batch(batch_size, n_rows, sparsity, "cpu")
    for i in range(min(16, batch_size)):
        idx = torch.where(mask[i, 0, :, 0] > 0.5)[0]
        assert int(idx.numel()) == k
        assert int(torch.unique(idx).numel()) == k
    freq = mask.squeeze(1).squeeze(-1).mean(dim=0)
    mid = n_rows // 2
    center = float(freq[mid - 8 : mid + 8].mean())
    edge = float(torch.cat([freq[:8], freq[-8:]]).mean())
    assert center > edge


def test_acs_random_keeps_center_block_when_k_ge_acs() -> None:
    torch.manual_seed(4)
    batch_size, n_rows, acs_width, s = 6, 64, 16, 0.4
    sparsity = torch.full((batch_size,), s)
    k = _budget(s, n_rows)
    assert k >= acs_width
    mask = acs_random_row_mask_batch(
        batch_size, n_rows, sparsity, "cpu", acs_width=acs_width
    )
    lo = (n_rows - acs_width) // 2
    hi = lo + acs_width
    assert torch.all(mask[:, 0, lo:hi, 0] == 1.0)
    _assert_shape_and_budget(mask, batch_size, n_rows, sparsity)


def test_acs_random_centered_block_when_k_lt_acs() -> None:
    torch.manual_seed(5)
    batch_size, n_rows, acs_width, s = 6, 64, 16, 0.1
    sparsity = torch.full((batch_size,), s)
    k = _budget(s, n_rows)
    assert 0 < k < acs_width
    mask = acs_random_row_mask_batch(
        batch_size, n_rows, sparsity, "cpu", acs_width=acs_width
    )
    inner_lo = (n_rows - k) // 2
    inner_hi = inner_lo + k
    assert torch.all(mask[:, 0, inner_lo:inner_hi, 0] == 1.0)
    assert torch.all(mask[:, 0, :inner_lo, 0] == 0.0)
    assert torch.all(mask[:, 0, inner_hi:, 0] == 0.0)
    _assert_shape_and_budget(mask, batch_size, n_rows, sparsity)


def test_sparsity_key_keeps_two_decimals_and_s0125() -> None:
    assert sparsity_key(0.1) == "s=0.10"
    assert sparsity_key(0.25) == "s=0.25"
    assert sparsity_key(0.4) == "s=0.40"
    assert sparsity_key(0.125) == "s=0.125"
