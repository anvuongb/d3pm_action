"""Map target mask sparsity to D3PM timesteps."""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import torch
from tqdm import tqdm

from .discrete import magnitude_to_fine_disc, kspace_to_row_disc, magnitude_to_row_disc


def cumulative_survival(beta_t: torch.Tensor) -> torch.Tensor:
    """Per-timestep P(pixel not yet absorbed), shape [n_T]."""
    one_minus_beta = 1.0 - beta_t.double()
    cum = one_minus_beta.cumprod(dim=0)
    return cum.flip(0)


def _interpolate_survival_table(
    timesteps: list[int],
    sampled: dict[int, float],
    sample_counts: dict[int, int],
    n_t: int,
) -> torch.Tensor:
    ts = torch.tensor(timesteps, dtype=torch.float32)
    vs = torch.tensor(
        [sampled[t] / max(sample_counts[t], 1) for t in timesteps],
        dtype=torch.float32,
    )
    all_t = torch.arange(1, n_t + 1, dtype=torch.float32)
    table = torch.zeros(n_t)
    for i, t in enumerate(all_t):
        if t <= ts[0]:
            table[i] = vs[0]
        elif t >= ts[-1]:
            table[i] = vs[-1]
        else:
            j = torch.searchsorted(ts, t) - 1
            t0, t1 = ts[j], ts[j + 1]
            v0, v1 = vs[j], vs[j + 1]
            w = (t - t0) / (t1 - t0)
            table[i] = v0 * (1 - w) + v1 * w
    return table


def _validate_survival_table(table: torch.Tensor) -> None:
    lo = float(table.min())
    hi = float(table.max())
    if lo < 0.0 or hi > 1.0 + 1e-5:
        warnings.warn(
            f"Survival table expected in (0, 1], got min={lo:.4f} max={hi:.4f}",
            stacklevel=3,
        )
    diffs = table[1:] - table[:-1]
    if bool((diffs > 1e-4).any()):
        warnings.warn(
            "Survival table is not monotone non-increasing in t; "
            "sparsity_to_timestep assumes decreasing survival.",
            stacklevel=3,
        )


def _build_survival_from_batches(
    d3pm,
    batch_iter: Iterator[torch.Tensor],
    device: str,
    max_batches: int,
    timestep_stride: int,
    desc: str,
) -> torch.Tensor:
    n_t = d3pm.n_T
    num_classes = d3pm.num_classses
    timesteps = list(range(1, n_t + 1, timestep_stride))
    if timesteps[-1] != n_t:
        timesteps.append(n_t)

    sampled = {t: 0.0 for t in timesteps}
    sample_counts = {t: 0 for t in timesteps}

    was_training = d3pm.training
    d3pm.eval()

    with torch.no_grad():
        for batch_idx, x in enumerate(tqdm(batch_iter, desc=desc)):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            if x.dtype != torch.long:
                x = (x * (num_classes - 1)).round().long().clamp(0, num_classes - 1)

            batch_size = x.shape[0]

            for t in timesteps:
                t_vec = torch.full((batch_size,), t, device=device, dtype=torch.long)
                noise = torch.rand((*x.shape, num_classes), device=device)
                xt = d3pm.q_sample(x, t_vec, noise)
                # Class 0 is the absorbing state. Counting xt==x on those
                # sites treats "already dark" as survival, so FastMRI fine
                # tables floor near the bin-0 mass (~0.5) and t(s) is always T.
                valid = x != 0
                n_valid = valid.float().sum()
                if float(n_valid) < 1.0:
                    unchanged = 1.0
                else:
                    unchanged = ((xt == x) & valid).float().sum().item() / float(
                        n_valid
                    )
                sampled[t] += unchanged
                sample_counts[t] += 1

    if was_training:
        d3pm.train()

    table = _interpolate_survival_table(timesteps, sampled, sample_counts, n_t)
    _validate_survival_table(table)
    return table


def build_pixel_survival_table(
    d3pm,
    dataloader,
    device: str = "cuda",
    max_batches: int = 20,
    timestep_stride: int = 10,
) -> torch.Tensor:
    """
    Empirical E[fraction of pixels unchanged after q_sample(t)] per timestep.

    Returns tensor of shape [n_T] with survival fractions in (0, 1].
    Expects dataloader batches of (x, ...) where x is image-like.
    """

    def _iter() -> Iterator[torch.Tensor]:
        for batch in dataloader:
            yield batch[0]

    return _build_survival_from_batches(
        d3pm,
        _iter(),
        device=device,
        max_batches=max_batches,
        timestep_stride=timestep_stride,
        desc="calibrating schedule",
    )


def build_fine_survival_table(
    d3pm,
    dataloader,
    device: str = "cuda",
    fine_size: int = 96,
    n_bins: int = 8,
    max_batches: int = 20,
    timestep_stride: int = 10,
) -> torch.Tensor:
    """Survival table for fine FastMRI D3PM (quantized resized magnitude)."""

    def _iter() -> Iterator[torch.Tensor]:
        for batch in dataloader:
            # FastMRI: (z, c, kspace)
            c = batch[1]
            yield magnitude_to_fine_disc(c, size=fine_size, n_bins=n_bins)

    return _build_survival_from_batches(
        d3pm,
        _iter(),
        device=device,
        max_batches=max_batches,
        timestep_stride=timestep_stride,
        desc="calibrating fine schedule",
    )


def build_row_survival_table(
    d3pm,
    dataloader,
    device: str = "cuda",
    n_bins: int = 8,
    max_batches: int = 20,
    timestep_stride: int = 10,
) -> torch.Tensor:
    """Survival table for coarse FastMRI D3PM (PE-line k-space energy by default)."""

    def _iter() -> Iterator[torch.Tensor]:
        for batch in dataloader:
            kspace = batch[2]
            c = batch[1]
            if kspace.is_complex():
                yield kspace_to_row_disc(kspace, n_bins=n_bins)
            else:
                yield magnitude_to_row_disc(c, n_bins=n_bins)

    return _build_survival_from_batches(
        d3pm,
        _iter(),
        device=device,
        max_batches=max_batches,
        timestep_stride=timestep_stride,
        desc="calibrating coarse schedule",
    )


def sparsity_to_timestep(
    sparsity: torch.Tensor,
    survival_table: torch.Tensor,
    n_t: int,
) -> torch.Tensor:
    """
    Map target mask density (fraction observed) to diffusion timestep.

    Survival table is indexed by t=1..n_T and is decreasing in survival.
    searchsorted requires an ascending sequence, so the table is flipped
    before lookup (paper §4.3).     Higher sparsity (more kept) -> smaller t.
    """
    sparsity = sparsity.to(survival_table.device).clamp(0.0, 1.0)
    lo = float(survival_table.min())
    if lo > 0.05:
        warnings.warn(
            f"Survival table min={lo:.3f}; densities below that all map to t={n_t}.",
            stacklevel=2,
        )
    ascending = survival_table.flip(0)
    idx = torch.searchsorted(ascending, sparsity, right=False)
    idx = idx.clamp(0, n_t - 1)
    t = (n_t - idx).clamp(1, n_t)
    return t.long()
