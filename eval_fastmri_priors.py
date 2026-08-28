"""Eval + visualize FastMRI D3PM priors (fine 96x96 and coarse row profile)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from d3pm_runner_fastmri import FastMRIDiscDataset
from itw.configs import FastMRIConfig, default_device
from itw.data.fastmri import FastMRIDataset
from itw.discrete import magnitude_to_fine_disc, magnitude_to_row_disc
from itw.entropy import fine_cond_entropy_loss
from itw.eval import (
    acs_random_row_mask_batch,
    plot_fastmri_grid,
    random_row_mask_batch,
    save_eval_report,
    vd_gaussian_row_mask_batch,
)
from itw.masks import apply_kspace_row_mask, gumbel_row_mask
from itw.schedule import sparsity_to_timestep
from itw.train import (
    build_mask_model,
    load_d3pm_coarse,
    load_d3pm_fine,
    mask_generator_cond,
)

# D3PM q_sample indexes q_mats[t-1]; t=0 is invalid (wraps to t=n_T).
# 757 is dummy-linspace t at s=0.25 (nested operating point).
STAGE5_CE_TIMESTEPS = (1, 100, 500, 757, 999)
STAGE5_SPARSITY = 0.25
IMAGE_Z_POLICY = Path("models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth")
STAGE5_JSON_NESTED = Path(
    "models_mask_gen_fastmri_nested/eval/eval_fine_prior_stage5.json"
)
STAGE5_JSON_FINE = Path("models_d3pm_fastmri_fine/eval/eval_fine_prior_stage5.json")


def _bins_to_unit(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    return x.float() / max(n_bins - 1, 1)


@torch.no_grad()
def ce_at_t(d3pm, x: torch.Tensor, t_val: int, cond: torch.Tensor) -> float:
    t = torch.full((x.shape[0],), t_val, device=x.device, dtype=torch.long)
    noise = torch.rand((*x.shape, d3pm.num_classses), device=x.device)
    xt = d3pm.q_sample(x, t, noise)
    logits = d3pm.model_predict(xt, t, cond)
    return float(
        F.cross_entropy(
            logits.flatten(start_dim=0, end_dim=-2),
            x.flatten(start_dim=0, end_dim=-1),
        )
    )


@torch.no_grad()
def predict_x0(d3pm, x: torch.Tensor, t_val: int, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.full((x.shape[0],), t_val, device=x.device, dtype=torch.long)
    noise = torch.rand((*x.shape, d3pm.num_classses), device=x.device)
    xt = d3pm.q_sample(x, t, noise)
    logits = d3pm.model_predict(xt, t, cond)
    return xt, logits.argmax(dim=-1)


def plot_fine_denoise(
    c300: torch.Tensor,
    x0: torch.Tensor,
    xts: dict[int, torch.Tensor],
    preds: dict[int, torch.Tensor],
    samples: torch.Tensor | None,
    n_bins: int,
    save_path: Path,
) -> None:
    n = min(4, x0.shape[0])
    extra = 1 if samples is not None else 0
    cols = 3 + 2 * len(xts) + extra
    fig, axes = plt.subplots(n, cols, figsize=(3 * cols, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    titles = ["C 300", "C_fine 96"]
    for t in xts:
        titles += [f"x_t t={t}", f"x0-hat t={t}"]
    if samples is not None:
        titles.append("uncond sample")
    for i in range(n):
        axes[i, 0].imshow(c300[i, 0].cpu(), cmap="gray")
        axes[i, 1].imshow(_bins_to_unit(x0[i, 0], n_bins).cpu(), cmap="gray", vmin=0, vmax=1)
        col = 2
        for t in xts:
            axes[i, col].imshow(_bins_to_unit(xts[t][i, 0], n_bins).cpu(), cmap="gray", vmin=0, vmax=1)
            axes[i, col + 1].imshow(
                _bins_to_unit(preds[t][i, 0], n_bins).cpu(), cmap="gray", vmin=0, vmax=1
            )
            col += 2
        if samples is not None:
            axes[i, col].imshow(
                _bins_to_unit(samples[i, 0], n_bins).cpu(), cmap="gray", vmin=0, vmax=1
            )
        for j in range(cols):
            axes[i, j].axis("off")
    for j, title in enumerate(titles):
        axes[0, j].set_title(title)
    fig.suptitle("Fine D3PM: real magnitude, noised input, x0 prediction")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_coarse_profiles(
    true_row: torch.Tensor,
    xt_row: torch.Tensor,
    pred_row: torch.Tensor,
    samples: torch.Tensor | None,
    n_bins: int,
    t_val: int,
    save_path: Path,
) -> None:
    n = min(4, true_row.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    h = true_row.shape[-2]
    xs = range(h)
    for i, ax in enumerate(axes):
        ax.plot(xs, _bins_to_unit(true_row[i, 0, :, 0], n_bins).cpu(), label="true C_row", lw=1.5)
        ax.plot(xs, _bins_to_unit(xt_row[i, 0, :, 0], n_bins).cpu(), label=f"x_t t={t_val}", alpha=0.6)
        ax.plot(xs, _bins_to_unit(pred_row[i, 0, :, 0], n_bins).cpu(), label="x0-hat", lw=1.2)
        if samples is not None:
            ax.plot(
                xs,
                _bins_to_unit(samples[i, 0, :, 0], n_bins).cpu(),
                label="uncond sample",
                alpha=0.7,
                ls="--",
            )
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"s{i}")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("PE row")
    fig.suptitle("Coarse D3PM row-profile prior")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_ce_curves(fine_ce: dict[int, float], coarse_ce: dict[int, float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if fine_ce:
        ax.plot(list(fine_ce), list(fine_ce.values()), marker="o", label="fine 96x96")
    if coarse_ce:
        ax.plot(list(coarse_ce), list(coarse_ce.values()), marker="s", label="coarse row")
    ax.axhline(float(torch.log(torch.tensor(8.0))), color="gray", ls=":", label="chance log(8)")
    ax.set_xlabel("t")
    ax.set_ylabel("x0 CE (nats)")
    ax.set_title("D3PM prior CE vs diffusion timestep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _clamp_d3pm_t(t_val: int, n_t: int) -> int:
    """q_sample uses q_mats[t-1]; valid t is 1..n_t."""
    return max(1, min(int(t_val), n_t))


@torch.no_grad()
def h_fine_for_rows(
    d3pm_fine,
    kspace: torch.Tensor,
    row_mask: torch.Tensor,
    t_fine: torch.Tensor,
    cfg: FastMRIConfig,
) -> tuple[float, float]:
    """Mean cond entropy + realized row density. Always computed (even if β=0)."""
    y_mag = apply_kspace_row_mask(kspace, row_mask)
    y_fine = magnitude_to_fine_disc(y_mag, size=cfg.fine_size, n_bins=cfg.num_classes)
    cond0 = torch.zeros(y_fine.shape[0], dtype=torch.long, device=y_fine.device)
    h = float(fine_cond_entropy_loss(d3pm_fine, y_fine, t_fine, cond0))
    density = float(row_mask.float().mean())
    return h, density


def _try_load_image_z_policy(cfg: FastMRIConfig):
    if not IMAGE_Z_POLICY.is_file():
        print(f"no image-Z policy at {IMAGE_Z_POLICY}; skip learned H_fine")
        return None
    model = build_mask_model(cfg)
    ckpt = torch.load(IMAGE_Z_POLICY, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["mask_generator"])
    model.eval()
    print(f"loaded image-Z policy {IMAGE_Z_POLICY}")
    return model


def _named_row_masks(
    z: torch.Tensor,
    kspace: torch.Tensor,
    sparsity: torch.Tensor,
    cfg: FastMRIConfig,
    policy,
) -> dict[str, torch.Tensor]:
    b, _, h, _ = kspace.shape
    device = kspace.device
    masks: dict[str, torch.Tensor] = {
        "empty": torch.zeros(b, 1, h, 1, device=device),
        "full_ones": torch.ones(b, 1, h, 1, device=device),
        "random": random_row_mask_batch(b, h, sparsity, device),
        "vd_gaussian": vd_gaussian_row_mask_batch(b, h, sparsity, device),
        "acs_random": acs_random_row_mask_batch(
            b, h, sparsity, device, acs_width=int(cfg.scout_size)
        ),
    }
    if policy is not None:
        logits = policy(mask_generator_cond(cfg, z, kspace), sparsity)
        masks["learned_image_z"] = gumbel_row_mask(logits, temperature=0.5, hard=True)
    return masks


def plot_stage5_h_fine(h_by_mask: dict[str, dict], save_path: Path) -> None:
    names = list(h_by_mask)
    vals = [h_by_mask[n]["h_fine"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals, color="steelblue")
    ax.axhline(float(torch.log(torch.tensor(8.0))), color="gray", ls=":", label="chance log(8)")
    ax.set_ylabel("H_fine (nats)")
    ax.set_title(f"Fine cond entropy vs row mask at s={STAGE5_SPARSITY}")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _stage5_decision(
    ce_fine: dict[int, float],
    h_by_mask: dict[str, dict],
    h_density_matched_t: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Keep β=0 unless CE(t) and H_fine(mask) both move in the useful direction.

    Interior t (1..500) must move; a jump only at t≈n_T is absorb-end, not a
    usable CE(t) for mask densities 0.1–0.4. H_fine must move across real
    s=0.25 patterns or with density-matched empty vs full — not just empty
    vs anything at a shared dummy t.
    """
    interior = {t: v for t, v in ce_fine.items() if 1 <= t <= 500}
    interior_span = max(interior.values()) - min(interior.values()) if interior else 0.0
    t_lo = min(interior) if interior else min(ce_fine)
    t_hi = max(interior) if interior else max(ce_fine)
    ce_moves = interior_span > 0.05 and ce_fine[t_lo] < ce_fine[t_hi] - 0.03

    pattern_names = [n for n in h_by_mask if n not in {"empty", "full_ones"}]
    pattern_vals = [h_by_mask[n]["h_fine"] for n in pattern_names] or [0.0]
    pattern_span = max(pattern_vals) - min(pattern_vals)
    dens_span = 0.0
    if h_density_matched_t:
        dens_span = abs(
            h_density_matched_t["empty_h_fine"] - h_density_matched_t["full_h_fine"]
        )
    h_empty = h_by_mask.get("empty", {}).get("h_fine")
    h_full = h_by_mask.get("full_ones", {}).get("h_fine")
    same_t_empty_full = (
        abs(h_empty - h_full) if h_empty is not None and h_full is not None else 0.0
    )
    h_moves = pattern_span > 0.05 or dens_span > 0.05
    if ce_moves and h_moves:
        return (
            "consider_beta_gt_0",
            "Fine CE drops with t and H_fine tracks mask density/pattern.",
        )
    reasons = []
    if not ce_moves:
        reasons.append(
            f"fine CE is flat in t on 1..500 (span={interior_span:.4f} nats, "
            f"t={t_lo}:{ce_fine[t_lo]:.4f} vs t={t_hi}:{ce_fine[t_hi]:.4f})"
        )
    if not h_moves:
        reasons.append(
            f"H_fine does not track mask density/pattern "
            f"(s=0.25 pattern span={pattern_span:.4f}, "
            f"density-matched empty vs full span={dens_span:.4f}, "
            f"same-t empty vs full={same_t_empty_full:.4f})"
        )
    return "keep_beta_0", "; ".join(reasons) + "."


@torch.no_grad()
def run_stage5(cfg: FastMRIConfig, max_batches: int = 4) -> dict:
    """Diagnose frozen fine D3PM: CE(t), H_fine(mask) at s=0.25, bin-0 mass."""
    nested_out = STAGE5_JSON_NESTED.parent
    fine_out = STAGE5_JSON_FINE.parent
    nested_out.mkdir(parents=True, exist_ok=True)
    fine_out.mkdir(parents=True, exist_ok=True)
    print(f"device={cfg.device} stage5 max_batches={max_batches}")

    d3pm_fine = load_d3pm_fine(cfg)
    d3pm_fine.eval()
    n_t = int(cfg.n_t)
    chance = float(torch.log(torch.tensor(float(cfg.num_classes))))

    disc_dl = DataLoader(
        FastMRIDiscDataset(
            cfg.data_root, mode="fine", fine_size=cfg.fine_size, n_bins=cfg.num_classes
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    fine_batches = [batch for i, batch in enumerate(disc_dl) if i < max_batches]
    if not fine_batches:
        raise RuntimeError("no FastMRI disc batches; check data_root")

    ce_fine: dict[int, float] = {}
    for raw_t in (0,) + STAGE5_CE_TIMESTEPS:
        t = _clamp_d3pm_t(raw_t, n_t)
        vals = [ce_at_t(d3pm_fine, x.to(cfg.device), t, cd.to(cfg.device)) for x, cd in fine_batches]
        ce_fine[raw_t] = sum(vals) / len(vals)
        note = f" (q_sample t={t})" if raw_t != t else ""
        print(f"t={raw_t:4d}{note}  CE_fine={ce_fine[raw_t]:.4f}  chance={chance:.4f}")

    hist = torch.zeros(cfg.num_classes)
    n_pix = 0
    for x, _cd in fine_batches:
        hist += torch.bincount(x.flatten().cpu(), minlength=cfg.num_classes).float()
        n_pix += int(x.numel())
    hist = hist / hist.sum().clamp_min(1.0)
    bin0 = float(hist[0])
    print(f"bin0_fraction={bin0:.4f} n_pix={n_pix} hist={hist.tolist()}")

    # Nested protocol uses dummy linspace while entropy_beta=0.
    fine_survival = torch.linspace(1.0, 0.01, n_t, device=cfg.device)
    ds = FastMRIDataset(cfg.data_root)
    mag_dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    policy = _try_load_image_z_policy(cfg)

    acc: dict[str, dict[str, float]] = {}
    t_s025_vals: list[float] = []
    n_mag = 0
    for batch_idx, (z, _c, kspace) in enumerate(mag_dl):
        if batch_idx >= max_batches:
            break
        z = z.to(cfg.device)
        kspace = kspace.to(cfg.device)
        sparsity = torch.full((z.shape[0],), STAGE5_SPARSITY, device=cfg.device)
        t_s025 = sparsity_to_timestep(sparsity, fine_survival, n_t)
        t_s025_vals.append(float(t_s025.float().mean()))
        named = _named_row_masks(z, kspace, sparsity, cfg, policy)
        # Same t for every mask at the operating sparsity (pattern / observation test).
        t_empty = sparsity_to_timestep(torch.zeros_like(sparsity), fine_survival, n_t)
        t_full = sparsity_to_timestep(torch.ones_like(sparsity), fine_survival, n_t)
        for name, rows in named.items():
            h, dens = h_fine_for_rows(d3pm_fine, kspace, rows, t_s025, cfg)
            bucket = acc.setdefault(name, {"h_fine": 0.0, "density": 0.0, "t_fine": 0.0})
            bucket["h_fine"] += h
            bucket["density"] += dens
            bucket["t_fine"] += float(t_s025.float().mean())
        h_e, d_e = h_fine_for_rows(d3pm_fine, kspace, named["empty"], t_empty, cfg)
        h_f, d_f = h_fine_for_rows(d3pm_fine, kspace, named["full_ones"], t_full, cfg)
        dens_t = acc.setdefault(
            "_density_matched_t",
            {
                "empty_h_fine": 0.0,
                "empty_density": 0.0,
                "empty_t": 0.0,
                "full_h_fine": 0.0,
                "full_density": 0.0,
                "full_t": 0.0,
            },
        )
        dens_t["empty_h_fine"] += h_e
        dens_t["empty_density"] += d_e
        dens_t["empty_t"] += float(t_empty.float().mean())
        dens_t["full_h_fine"] += h_f
        dens_t["full_density"] += d_f
        dens_t["full_t"] += float(t_full.float().mean())
        n_mag += 1

    dens_matched_raw = acc.pop("_density_matched_t")
    h_by_mask = {
        name: {k: v / max(n_mag, 1) for k, v in stats.items()}
        for name, stats in acc.items()
    }
    h_density_matched_t = {
        k: v / max(n_mag, 1) for k, v in dens_matched_raw.items()
    }
    t_fine_mean = sum(t_s025_vals) / max(len(t_s025_vals), 1)
    for name, stats in h_by_mask.items():
        print(
            f"H_fine {name:16s}  {stats['h_fine']:.4f}  "
            f"density={stats['density']:.3f}  t={stats['t_fine']:.1f}"
        )
    print(
        "H_fine density-matched t: "
        f"empty={h_density_matched_t['empty_h_fine']:.4f} "
        f"(t={h_density_matched_t['empty_t']:.1f})  "
        f"full={h_density_matched_t['full_h_fine']:.4f} "
        f"(t={h_density_matched_t['full_t']:.1f})"
    )

    ce_for_rule = {t: ce_fine[t] for t in STAGE5_CE_TIMESTEPS}
    decision, reason = _stage5_decision(ce_for_rule, h_by_mask, h_density_matched_t)
    print(f"decision={decision}  {reason}")

    ce_plot = {t: ce_fine[t] for t in STAGE5_CE_TIMESTEPS}
    plot_ce_curves(ce_plot, {}, nested_out / "stage5_ce_vs_t.png")
    plot_stage5_h_fine(h_by_mask, nested_out / "stage5_h_fine_vs_mask.png")

    report = {
        "stage": 5,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "device": str(cfg.device),
        "fine_checkpoint": cfg.d3pm_fine_checkpoint,
        "image_z_policy": str(IMAGE_Z_POLICY) if policy is not None else None,
        "n_batches": max_batches,
        "batch_size": cfg.batch_size,
        "n_pix_bin_hist": n_pix,
        "sparsity": STAGE5_SPARSITY,
        "fine_survival": "dummy linspace(1, 0.01, n_t) (entropy_beta=0 protocol)",
        "t_fine_at_s025": t_fine_mean,
        "t0_note": (
            "D3PM q_sample indexes q_mats[t-1]; t=0 is invalid and is clamped to t=1."
        ),
        "ce_fine": {str(k): v for k, v in ce_fine.items()},
        "ce_chance_logN": chance,
        "fine_bin_fraction": hist.tolist(),
        "bin0_fraction": bin0,
        "h_fine_s025": h_by_mask,
        "h_fine_density_matched_t": h_density_matched_t,
        "decision": decision,
        "decision_reason": reason,
        "winning_recipe_stage6": {
            "policy_ckpt": str(IMAGE_Z_POLICY),
            "policy_input": "scout_image",
            "entropy_alpha": 1.0,
            "entropy_beta": 0.0,
            "recon_loss_weight": 1.0,
            "note": (
                "Stage 2/3 image-Z both. Stage 4 ACS-kspace is a failed "
                "architecture (PE upsample 32→300), not the protocol default."
            ),
            "pe_aligned_acs_head": (
                "Optional later: a PE-aligned ACS head (no 32→300 linear "
                "upsample) might address the ACS+random recon gap. Not Stage 5."
            ),
        },
    }
    save_eval_report(report, STAGE5_JSON_NESTED)
    save_eval_report(report, STAGE5_JSON_FINE)
    print("wrote", STAGE5_JSON_NESTED)
    print("wrote", STAGE5_JSON_FINE)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage5",
        action="store_true",
        help="Fine-prior diagnostic only (CE vs t, H_fine vs mask; no reverse sampling)",
    )
    parser.add_argument("--max-batches", type=int, default=4)
    args = parser.parse_args()
    if args.stage5:
        cfg = FastMRIConfig(device=default_device(), batch_size=8, num_workers=0)
        report = run_stage5(cfg, max_batches=args.max_batches)
        print(json.dumps(report, indent=2))
        return

    cfg = FastMRIConfig(device=default_device(), batch_size=4, num_workers=0)
    out = Path("models_d3pm_fastmri_fine/eval")
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={cfg.device} out={out}")

    d3pm_fine = load_d3pm_fine(cfg)
    d3pm_coarse = load_d3pm_coarse(cfg)
    d3pm_fine.eval()
    d3pm_coarse.eval()

    ds = FastMRIDataset(cfg.data_root)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    z, c, kspace = next(iter(dl))
    z, c, kspace = z.to(cfg.device), c.to(cfg.device), kspace.to(cfg.device)
    x_fine = magnitude_to_fine_disc(c, size=cfg.fine_size, n_bins=cfg.num_classes)
    x_row = magnitude_to_row_disc(c, n_bins=cfg.num_classes)
    cond = torch.zeros(c.shape[0], dtype=torch.long, device=cfg.device)

    timesteps = (50, 200, 400, 700, 900)
    fine_ce: dict[int, float] = {}
    coarse_ce: dict[int, float] = {}
    # CE on a few sequential batches so it is not one-batch noise
    disc_dl = DataLoader(
        FastMRIDiscDataset(cfg.data_root, mode="fine", fine_size=cfg.fine_size, n_bins=cfg.num_classes),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    coarse_dl = DataLoader(
        FastMRIDiscDataset(cfg.data_root, mode="coarse", n_bins=cfg.num_classes),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    fine_batches = [batch for i, batch in enumerate(disc_dl) if i < 4]
    coarse_batches = [batch for i, batch in enumerate(coarse_dl) if i < 4]
    for t in timesteps:
        fvals = [ce_at_t(d3pm_fine, x.to(cfg.device), t, cd.to(cfg.device)) for x, cd in fine_batches]
        cvals = [ce_at_t(d3pm_coarse, x.to(cfg.device), t, cd.to(cfg.device)) for x, cd in coarse_batches]
        fine_ce[t] = sum(fvals) / len(fvals)
        coarse_ce[t] = sum(cvals) / len(cvals)
        print(f"t={t:4d}  CE_fine={fine_ce[t]:.4f}  CE_coarse={coarse_ce[t]:.4f}")

    denoise_ts = (200, 500)
    xts, preds = {}, {}
    for t in denoise_ts:
        xts[t], preds[t] = predict_x0(d3pm_fine, x_fine, t, cond)
    xt_row, pred_row = predict_x0(d3pm_coarse, x_row, 400, cond)

    print("sampling fine prior (4 images, 999 reverse steps)...")
    x_init = torch.zeros(c.shape[0], 1, cfg.fine_size, cfg.fine_size, dtype=torch.long, device=cfg.device)
    samples_fine = d3pm_fine.sample(x_init, cond)

    print("sampling coarse prior...")
    row_init = torch.zeros(c.shape[0], 1, cfg.image_size, 1, dtype=torch.long, device=cfg.device)
    samples_row = d3pm_coarse.sample(row_init, cond)

    plot_fine_denoise(
        c.cpu(), x_fine.cpu(), {t: v.cpu() for t, v in xts.items()},
        {t: v.cpu() for t, v in preds.items()}, samples_fine.cpu(),
        cfg.num_classes, out / "fine_denoise_samples.png",
    )
    plot_coarse_profiles(
        x_row.cpu(), xt_row.cpu(), pred_row.cpu(), samples_row.cpu(),
        cfg.num_classes, 400, out / "coarse_profiles.png",
    )
    plot_ce_curves(fine_ce, coarse_ce, out / "ce_vs_t.png")

    sparsity = torch.full((z.shape[0],), 0.25, device=cfg.device)
    random_rows = random_row_mask_batch(z.shape[0], cfg.image_size, sparsity, cfg.device)
    y_random = apply_kspace_row_mask(kspace, random_rows)
    plot_fastmri_grid(
        z.cpu(), c.cpu(), random_rows.cpu(), y_random.cpu(),
        title="random row mask s=0.25 (no learned policy yet)",
        save_path=out / "acquisition_random_s025.png",
        kspace=kspace.cpu(),
    )
    plt.close("all")

    hist = torch.bincount(x_fine.flatten().cpu(), minlength=cfg.num_classes).float()
    hist = hist / hist.sum()
    report = {
        "device": cfg.device,
        "fine_checkpoint": cfg.d3pm_fine_checkpoint,
        "coarse_checkpoint": cfg.d3pm_coarse_checkpoint,
        "ce_fine": {str(k): v for k, v in fine_ce.items()},
        "ce_coarse": {str(k): v for k, v in coarse_ce.items()},
        "fine_bin_fraction": hist.tolist(),
        "note": (
            "Learned-vs-random nested mask eval needs train_fastmri first. "
            "models_mask_gen_fastmri/ is pre-fix and invalid."
        ),
    }
    save_eval_report(report, out / "prior_eval.json")
    print(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
