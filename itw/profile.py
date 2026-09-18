"""Static row-profile optimisation (stage B5).

B3 showed the conditional policy does not use its conditioning input: feeding a
slice another slice's scout *improves* NMSE, and collapsing the policy to its
own consensus mask beats every heuristic at s >= 0.40. So the object worth
learning is not a policy but a **static 300-number score vector**, exactly like
the F7 energy oracle -- only optimised against a reconstructor rather than
against raw k-space energy.

B4 showed why that matters: under a reconstructor that carries a prior, the
optimum moves away from the energy profile, because centre rows become
predictable and budget spent on them is redundant.

A profile is one parameter per phase-encode row. The mask at sparsity ``s`` is
the budget-exact top-k of the profile with the ACS block forced on, made
differentiable by a straight-through estimator.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .masks import acs_bounds


def _budget_threshold(scores: torch.Tensor, k: int, temperature: float,
                      iters: int = 60) -> torch.Tensor:
    """tau such that sum(sigmoid((scores - tau)/T)) == k, by bisection.

    Picking tau this way keeps the *soft* mask on-budget, so every row sits on
    the informative part of the sigmoid and receives gradient. A plain k-th
    largest threshold leaves rows deep inside or outside the selection with a
    vanishing derivative and the profile stops moving.
    """
    lo = scores.min() - 10.0 * temperature
    hi = scores.max() + 10.0 * temperature
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        over = torch.sigmoid((scores - mid) / temperature).sum() > k
        lo = torch.where(over, mid, lo)
        hi = torch.where(over, hi, mid)
    return 0.5 * (lo + hi)


class StaticProfile(nn.Module):
    """One learnable score per PE row; masks are budget-exact top-k of it."""

    def __init__(self, n_rows: int, acs_width: int, init: torch.Tensor | None = None):
        super().__init__()
        self.n_rows, self.acs_width = n_rows, acs_width
        if init is None:
            init = 0.01 * torch.randn(n_rows)
        self.theta = nn.Parameter(init.clone().float())
        lo, hi = acs_bounds(n_rows, acs_width)
        acs = torch.zeros(n_rows, dtype=torch.bool)
        acs[lo:hi] = True
        self.register_buffer("acs", acs)

    def scores(self) -> torch.Tensor:
        """Profile with the ACS block pinned above everything else.

        The pin is a constant, so no gradient reaches ACS rows -- correct, since
        they are kept at every sparsity regardless of what the profile says.
        """
        big = self.theta.detach().max() + 1e3
        return torch.where(self.acs, big.expand_as(self.theta), self.theta)

    def mask(self, sparsity: float, temperature: float = 0.1,
             hard: bool = True) -> torch.Tensor:
        """Row mask [1, 1, n_rows, 1] at ``sparsity``; STE-differentiable."""
        k = int(sparsity * self.n_rows)
        sc = self.scores()
        tau = _budget_threshold(sc.detach(), k, temperature)
        soft = torch.sigmoid((sc - tau) / temperature)
        if not hard:
            return soft.view(1, 1, -1, 1)
        idx = torch.topk(sc, k).indices
        hard_m = torch.zeros_like(sc).scatter(0, idx, 1.0)
        return (hard_m + soft - soft.detach()).view(1, 1, -1, 1)

    @torch.no_grad()
    def hard_mask(self, sparsity: float) -> torch.Tensor:
        k = int(sparsity * self.n_rows)
        idx = torch.topk(self.scores(), k).indices
        m = torch.zeros(self.n_rows, device=self.theta.device).scatter(0, idx, 1.0)
        return m.view(1, 1, -1, 1)


def energy_init(kspace_batches, n_rows: int, device) -> torch.Tensor:
    """Standardised log mean PE-row energy -- the F7 oracle, as a warm start."""
    energy = torch.zeros(n_rows, device=device)
    n = 0
    with torch.no_grad():
        for k in kspace_batches:
            k = k.to(device)
            energy += (k.abs() ** 2).sum(dim=(1, 3)).sum(0)
            n += k.shape[0]
    log_e = (energy / n).clamp_min(1e-20).log()
    return (log_e - log_e.mean()) / log_e.std().clamp_min(1e-12)


@torch.no_grad()
def evaluate_profile(profile, batches, recon, sparsity: float, device) -> float:
    """Mean NMSE of the *hard* mask over ``batches``.

    The optimiser's running loss mixes the profile values it held at different
    points in the epoch, so it cannot be used to pick the best epoch. This
    scores one fixed profile with the discrete mask that would actually be used.
    """
    from .discrete import nmse
    from .recon import reconstruct

    mask = profile.hard_mask(sparsity)
    tot, n = 0.0, 0
    for kspace, target in batches:
        kspace, target = kspace.to(device), target.to(device)
        b = kspace.shape[0]
        pred = reconstruct(recon, kspace, mask.expand(b, -1, -1, -1))
        # Use the shared metric so the objective and the reported number are the
        # same quantity -- they were not while nmse carried a 1e-8 denominator
        # floor that was active on ~2/3 of FastMRI slices.
        tot += float(nmse(pred, target)) * b
        n += b
    return tot / n


def train_static_profile(
    batches,
    recon,
    sparsity: float,
    device,
    n_rows: int = 300,
    acs_width: int = 32,
    n_epochs: int = 30,
    lr: float = 0.005,
    temperature: float = 0.25,
    init: torch.Tensor | None = None,
    log_path: Path | None = None,
    desc: str = "",
) -> tuple[StaticProfile, list[dict]]:
    """Minimise reconstruction NMSE at one sparsity over the row profile.

    ``recon`` is frozen; only the ``n_rows`` profile parameters move.
    ``batches`` is a list of (kspace, target) pairs.
    """
    from .discrete import nmse
    from .recon import reconstruct

    profile = StaticProfile(n_rows, acs_width, init=init).to(device)
    opt = torch.optim.Adam(profile.parameters(), lr=lr)
    # The mask is discrete, so the objective is a step function of the profile
    # and a fixed lr large enough to make progress keeps stepping over minima.
    # Decay lets it explore row swaps early and settle on one late.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1))
    for p in recon.parameters():
        p.requires_grad_(False)
    recon.eval()

    start = evaluate_profile(profile, batches, recon, sparsity, device)
    history = [{"epoch": -1, "train_nmse": start}]
    best = (start, profile.theta.detach().clone())
    pbar = tqdm(range(n_epochs), desc=desc or f"profile s={sparsity:.2f}")
    for epoch in pbar:
        for kspace, target in batches:
            kspace, target = kspace.to(device), target.to(device)
            b = kspace.shape[0]
            mask = profile.mask(sparsity, temperature).expand(b, -1, -1, -1)
            pred = reconstruct(recon, kspace, mask)
            loss = nmse(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                # Top-k is invariant to shift and positive scale, so the profile
                # is free to drift; renormalising keeps ``temperature`` a fixed
                # fraction of the spread instead of a slowly meaningless one.
                t = profile.theta
                t.sub_(t.mean()).div_(t.std().clamp_min(1e-6))
        sched.step()
        train_nmse = evaluate_profile(profile, batches, recon, sparsity, device)
        history.append({"epoch": epoch, "train_nmse": train_nmse})
        if train_nmse < best[0]:
            best = (train_nmse, profile.theta.detach().clone())
        pbar.set_postfix(nmse=f"{train_nmse:.6f}", best=f"{best[0]:.6f}")
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"s={sparsity:.2f} epoch={epoch} train_nmse={train_nmse:.6f}\n")

    # Keep the best epoch, not the last: the mask is discrete, so the objective
    # is a step function of the profile and the final step can be a worse one.
    with torch.no_grad():
        profile.theta.copy_(best[1])
    return profile, history
