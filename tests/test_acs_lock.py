"""ACS-lock sampling constraint: factories, top-k post-hoc, STE Gumbel."""

from __future__ import annotations

import inspect

import torch

from itw.configs import FastMRIConfig
from itw.eval import (
    acs_equispaced_row_mask_batch,
    acs_lock_topk_from_logits,
    acs_random_row_mask_batch,
    acs_vd_gaussian_row_mask_batch,
    evaluate_fastmri_baselines_loader,
)
from itw.masks import (
    acs_bounds,
    gumbel_row_mask_acs_locked,
    gumbel_row_mask_ste_acs_locked,
)


def _budget(s: float, n_rows: int) -> int:
    return max(0, min(n_rows, int(s * n_rows)))


ACS_FACTORIES = (
    acs_random_row_mask_batch,
    acs_vd_gaussian_row_mask_batch,
    acs_equispaced_row_mask_batch,
)


def test_acs_bounds_matches_centered_block() -> None:
    n_rows, acs_width = 300, 32
    lo, hi = acs_bounds(n_rows, acs_width)
    assert hi - lo == acs_width
    assert lo == (n_rows - acs_width) // 2
    assert hi == lo + acs_width


def test_k_lt_acs_centered_block_factories_and_lock_agree() -> None:
    torch.manual_seed(0)
    batch_size, n_rows, acs_width, s = 6, 64, 16, 0.1
    sparsity = torch.full((batch_size,), s)
    k = _budget(s, n_rows)
    assert 0 < k < acs_width
    inner_lo = (n_rows - k) // 2
    inner_hi = inner_lo + k

    factory_masks = [
        f(batch_size, n_rows, sparsity, "cpu", acs_width=acs_width)
        for f in ACS_FACTORIES
    ]
    logits = torch.randn(batch_size, 1, n_rows, 2)
    topk = acs_lock_topk_from_logits(logits, sparsity, acs_width)
    gumbel = gumbel_row_mask_acs_locked(logits, sparsity, acs_width=acs_width)
    soft, hard = gumbel_row_mask_ste_acs_locked(
        logits, sparsity, acs_width=acs_width
    )

    for mask in factory_masks + [topk, gumbel, hard]:
        assert tuple(mask.shape) == (batch_size, 1, n_rows, 1)
        assert torch.all(mask[:, 0, inner_lo:inner_hi, 0] == 1.0)
        assert torch.all(mask[:, 0, :inner_lo, 0] == 0.0)
        assert torch.all(mask[:, 0, inner_hi:, 0] == 0.0)
        ones = mask.reshape(batch_size, -1).sum(dim=1)
        assert torch.equal(ones, torch.full((batch_size,), k, dtype=ones.dtype))

    assert torch.equal(factory_masks[0], factory_masks[1])
    assert torch.equal(factory_masks[0], factory_masks[2])
    assert torch.equal(factory_masks[0], topk)
    assert torch.equal(factory_masks[0], gumbel)
    assert torch.equal(soft, hard)


def test_k_gt_acs_block_ones_and_exact_remainder_cardinality() -> None:
    torch.manual_seed(1)
    batch_size, n_rows, acs_width, s = 6, 64, 16, 0.4
    sparsity = torch.full((batch_size,), s)
    k = _budget(s, n_rows)
    extra = k - acs_width
    assert extra > 0
    lo, hi = acs_bounds(n_rows, acs_width)

    for factory in ACS_FACTORIES:
        mask = factory(batch_size, n_rows, sparsity, "cpu", acs_width=acs_width)
        assert torch.all(mask[:, 0, lo:hi, 0] == 1.0)
        ones = mask.reshape(batch_size, -1).sum(dim=1)
        assert torch.equal(ones, torch.full((batch_size,), k, dtype=ones.dtype))
        remainder = mask.clone()
        remainder[:, 0, lo:hi, 0] = 0.0
        assert torch.equal(
            remainder.reshape(batch_size, -1).sum(dim=1),
            torch.full((batch_size,), extra, dtype=ones.dtype),
        )

    logits = torch.randn(batch_size, 1, n_rows, 2)
    topk = acs_lock_topk_from_logits(logits, sparsity, acs_width)
    assert torch.all(topk[:, 0, lo:hi, 0] == 1.0)
    ones = topk.reshape(batch_size, -1).sum(dim=1)
    assert torch.equal(ones, torch.full((batch_size,), k, dtype=ones.dtype))


def test_topk_remainder_are_highest_keep_logits_outside_acs() -> None:
    n_rows, acs_width, s = 16, 4, 0.5
    k = _budget(s, n_rows)
    extra = k - acs_width
    assert extra > 0
    lo, hi = acs_bounds(n_rows, acs_width)
    logits = torch.zeros(1, 1, n_rows, 2)
    keep = torch.arange(n_rows, dtype=torch.float32)
    logits[0, 0, :, 1] = keep
    logits[0, 0, :, 0] = 0.0
    mask = acs_lock_topk_from_logits(logits, torch.tensor([s]), acs_width)
    assert torch.all(mask[0, 0, lo:hi, 0] == 1.0)
    outside = torch.cat((torch.arange(0, lo), torch.arange(hi, n_rows)))
    rem = torch.where(mask[0, 0, :, 0] > 0.5)[0]
    rem = rem[(rem < lo) | (rem >= hi)]
    expected = outside[torch.topk(keep[outside], extra).indices].sort().values
    assert torch.equal(rem.sort().values, expected.sort().values)


def test_ste_acs_lock_acs_region_ones_and_shape() -> None:
    torch.manual_seed(2)
    batch_size, n_rows, acs_width = 4, 32, 8
    sparsity = torch.full((batch_size,), 0.5)
    assert _budget(0.5, n_rows) > acs_width
    logits = torch.randn(batch_size, 1, n_rows, 2)
    soft, hard = gumbel_row_mask_ste_acs_locked(
        logits, sparsity, acs_width=acs_width
    )
    lo, hi = acs_bounds(n_rows, acs_width)
    assert tuple(hard.shape) == (batch_size, 1, n_rows, 1)
    assert tuple(soft.shape) == (batch_size, 1, n_rows, 1)
    assert torch.all(hard[:, 0, lo:hi, 0] == 1.0)
    assert torch.all(soft[:, 0, lo:hi, 0] == 1.0)


def test_acs_lock_config_default_false() -> None:
    cfg = FastMRIConfig()
    assert cfg.acs_lock is False


def test_baselines_loader_default_omits_learned_acs_lock() -> None:
    sig = inspect.signature(evaluate_fastmri_baselines_loader)
    assert sig.parameters["include_learned_acs_lock"].default is False
    assert sig.parameters["baselines"].default == (
        "random",
        "equispaced",
        "vd_gaussian",
        "acs_random",
    )
