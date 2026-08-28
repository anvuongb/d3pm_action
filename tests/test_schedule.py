"""Survival calibration and sparsity→timestep mapping."""

from __future__ import annotations

import torch

from itw.schedule import _build_survival_from_batches, sparsity_to_timestep


class _IdentityD3PM:
    n_T = 20
    num_classses = 2
    training = False

    def eval(self) -> None:
        return None

    def train(self) -> None:
        return None

    def q_sample(self, x, t, noise):
        del t, noise
        return x


def test_survival_fraction_independent_of_batch_size() -> None:
    d3pm = _IdentityD3PM()

    def batches_of(n: int):
        yield torch.zeros(n, 1, 4, 4, dtype=torch.long)

    t_small = _build_survival_from_batches(
        d3pm, batches_of(2), device="cpu", max_batches=1, timestep_stride=5, desc="s"
    )
    t_large = _build_survival_from_batches(
        d3pm, batches_of(8), device="cpu", max_batches=1, timestep_stride=5, desc="l"
    )
    assert float(t_small.min()) > 0.0
    assert float(t_small.max()) <= 1.0 + 1e-5
    assert torch.allclose(t_small, t_large, atol=1e-5)
    assert torch.allclose(t_small, torch.ones_like(t_small), atol=1e-5)


def test_sparsity_to_timestep_decreases_with_density() -> None:
    n_t = 1000
    table = torch.linspace(1.0, 0.01, n_t)
    s = torch.tensor([0.2, 0.5, 0.8])
    t = sparsity_to_timestep(s, table, n_t)
    assert t[0] > t[1] >= t[2] or t[0] > t[2]
    assert not all(int(v) == n_t for v in t.tolist())
    assert int(t.min()) >= 1
    assert int(t.max()) <= n_t


class _AbsorbNonzeroD3PM(_IdentityD3PM):
    def q_sample(self, x, t, noise):
        del t, noise
        return torch.zeros_like(x)


def test_survival_ignores_absorb_class_zero() -> None:
    """xt==x on bin-0 (absorb / dark) must not inflate survival."""
    d3pm = _AbsorbNonzeroD3PM()

    def batches():
        x = torch.zeros(2, 1, 4, 4, dtype=torch.long)
        x[..., :2, :] = 1
        yield x

    table = _build_survival_from_batches(
        d3pm, batches(), device="cpu", max_batches=1, timestep_stride=5, desc="z"
    )
    assert float(table.max()) <= 0.05
