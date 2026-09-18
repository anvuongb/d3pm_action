"""B1 eval-protocol guardrails: split isolation, seeding, budget-exact masks."""

from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from itw.configs import FastMRIConfig, seed_everything
from itw.eval import learned_rows_topk, topk_row_mask_from_logits
from itw.train import build_dataloader, fastmri_split_root

N_ROWS = 300


def _budget(s: float, n_rows: int = N_ROWS) -> int:
    return max(0, min(n_rows, int(s * n_rows)))


def _write_fake_split(root, n_files: int, dim: int = 320) -> None:
    """Minimal fastMRI-shaped .h5 files: kspace [slices, H, W] complex."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n_files):
        k = (rng.normal(size=(3, dim, dim)) + 1j * rng.normal(size=(3, dim, dim)))
        with h5py.File(root / f"file{i:03d}.h5", "w") as f:
            f.create_dataset("kspace", data=k.astype(np.complex64))


@pytest.fixture(scope="module")
def split_cfg(tmp_path_factory):
    base = tmp_path_factory.mktemp("fastmri")
    _write_fake_split(base / "train", 12)
    _write_fake_split(base / "val", 6)
    return FastMRIConfig(
        device="cpu",
        data_root=str(base / "train"),
        val_root=str(base / "val"),
        batch_size=2,
        num_workers=0,
        image_size=64,
        scout_size=8,
    )


# --- budget-exact masks ------------------------------------------------------


@pytest.mark.parametrize("s", [0.1, 0.125, 0.25, 0.4, 0.5, 0.75])
def test_topk_density_is_exactly_the_budget(s):
    logits = torch.randn(4, 1, N_ROWS, 2)
    sparsity = torch.full((4,), s)
    mask = topk_row_mask_from_logits(logits, sparsity)
    k = _budget(s)
    assert mask.shape == (4, 1, N_ROWS, 1)
    assert torch.all(mask.sum(dim=(1, 2, 3)) == k)
    # exactness lives in the integer row count; mean is float32
    assert float(mask.mean()) == pytest.approx(k / N_ROWS, abs=1e-6)


@pytest.mark.parametrize("s", [0.1, 0.25, 0.5, 0.75])
def test_learned_rows_topk_exact_under_acs_lock(s):
    logits = torch.randn(4, 1, N_ROWS, 2)
    sparsity = torch.full((4,), s)
    cfg = FastMRIConfig(device="cpu", acs_lock=True, scout_size=32)
    mask = learned_rows_topk(cfg, logits, sparsity)
    assert torch.all(mask.sum(dim=(1, 2, 3)) == _budget(s))


def test_topk_is_deterministic_across_calls():
    logits = torch.randn(3, 1, N_ROWS, 2)
    sparsity = torch.full((3,), 0.25)
    a = topk_row_mask_from_logits(logits, sparsity)
    b = topk_row_mask_from_logits(logits, sparsity)
    assert torch.equal(a, b)


def test_topk_selects_the_highest_keep_logits():
    logits = torch.zeros(1, 1, N_ROWS, 2)
    logits[0, 0, :, 1] = torch.arange(N_ROWS, dtype=torch.float32)
    mask = topk_row_mask_from_logits(logits, torch.tensor([0.1]))
    chosen = mask[0, 0, :, 0].nonzero().flatten()
    assert torch.equal(chosen, torch.arange(N_ROWS - _budget(0.1), N_ROWS))


# --- split isolation ---------------------------------------------------------


def test_split_root_selects_the_configured_split(split_cfg):
    assert fastmri_split_root(split_cfg) == split_cfg.data_root
    assert fastmri_split_root(replace(split_cfg, data_split="val")) == split_cfg.val_root


def test_train_and_val_file_sets_are_disjoint(split_cfg):
    train = build_dataloader(split_cfg).dataset.files
    val = build_dataloader(replace(split_cfg, data_split="val")).dataset.files
    assert set(train).isdisjoint(set(val))
    assert len(train) == 12 and len(val) == 6


def test_val_loader_is_deterministic_and_keeps_every_slice(split_cfg):
    cfg = replace(split_cfg, data_split="val")
    dl = build_dataloader(cfg)
    assert dl.drop_last is False
    first = torch.cat([z for z, _, _ in dl])
    second = torch.cat([z for z, _, _ in build_dataloader(cfg)])
    assert torch.equal(first, second)
    assert first.shape[0] == 6


def test_train_loader_shuffles_and_drops_last(split_cfg):
    dl = build_dataloader(split_cfg)
    assert dl.drop_last is True
    assert isinstance(dl.sampler, torch.utils.data.RandomSampler)


# --- seeding -----------------------------------------------------------------


def test_same_seed_gives_identical_train_order(split_cfg):
    def order(seed):
        return torch.cat([z for z, _, _ in build_dataloader(replace(split_cfg, seed=seed))])

    assert torch.equal(order(0), order(0))


def test_different_seed_gives_different_train_order(split_cfg):
    def order(seed):
        return torch.cat([z for z, _, _ in build_dataloader(replace(split_cfg, seed=seed))])

    assert not torch.equal(order(0), order(1))


def test_unseeded_config_keeps_pre_b1_behaviour(split_cfg):
    assert split_cfg.seed is None
    assert build_dataloader(split_cfg).generator is None


def test_seed_everything_makes_mask_sampling_reproducible():
    def draw():
        seed_everything(7)
        return torch.rand(8)

    assert torch.equal(draw(), draw())


def test_nmse_denominator_not_clamped_at_fastmri_scale() -> None:
    """NMSE must stay a ratio at FastMRI's magnitude scale.

    ||target||^2 is ~4e-9 for a median 300x300 FastMRI slice. A denominator
    floor anywhere near that (the old default was 1e-8) replaces the
    denominator on most slices, turning NMSE into a constant-scaled MSE that
    reads ~4x low and silently down-weights low-energy images.
    """
    from itw.discrete import nmse

    torch.manual_seed(0)
    target = torch.rand(4, 1, 300, 300) * 3.6e-7
    den = target.pow(2).flatten(1).sum(1)
    # Median real FastMRI slice sits at ~4e-9, i.e. *below* the old 1e-8 floor.
    assert float(den.max()) < 1e-8

    # Scaling both arguments must not change a ratio.
    pred = target * 0.5
    ref = float(nmse(pred, target))
    for scale in (1e-3, 1e3, 1e6):
        assert float(nmse(pred * scale, target * scale)) == pytest.approx(ref, rel=1e-4)
    assert ref == pytest.approx(0.25, rel=1e-4)


def test_nmse_zero_target_does_not_explode() -> None:
    """The floor still has to protect the one case it exists for."""
    from itw.discrete import nmse

    z = torch.zeros(2, 1, 8, 8)
    assert torch.isfinite(nmse(z, z))
