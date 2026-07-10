"""Map target mask sparsity to D3PM timesteps."""

from __future__ import annotations

import torch
from tqdm import tqdm


def cumulative_survival(beta_t: torch.Tensor) -> torch.Tensor:
    """Per-timestep P(pixel not yet absorbed), shape [n_T]."""
    one_minus_beta = 1.0 - beta_t.double()
    cum = one_minus_beta.cumprod(dim=0)
    return cum.flip(0)


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
    """
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
        for batch_idx, (x, _) in enumerate(tqdm(dataloader, desc="calibrating schedule")):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            if x.dtype != torch.long:
                x = (x * (num_classes - 1)).round().long().clamp(0, num_classes - 1)

            batch_size = x.shape[0]
            flat_pixels = x.numel() // batch_size

            for t in timesteps:
                t_vec = torch.full((batch_size,), t, device=device, dtype=torch.long)
                noise = torch.rand((*x.shape, num_classes), device=device)
                xt = d3pm.q_sample(x, t_vec, noise)
                unchanged = (xt == x).float().sum().item() / flat_pixels
                sampled[t] += unchanged
                sample_counts[t] += 1

    if was_training:
        d3pm.train()

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


def sparsity_to_timestep(
    sparsity: torch.Tensor,
    survival_table: torch.Tensor,
    n_t: int,
) -> torch.Tensor:
    """
    Map target mask density (fraction observed) to diffusion timestep.

    Higher sparsity (more pixels kept) -> lower noise -> smaller t.
    """
    sparsity = sparsity.to(survival_table.device).clamp(0.0, 1.0)
    idx = torch.searchsorted(survival_table, sparsity, right=False)
    idx = idx.clamp(0, n_t - 1)
    return (n_t - idx).long()
