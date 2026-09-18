"""Compare every static row mask on held-out val: NMSE, PSNR and SSIM.

Two passes, so plots can be redrawn without a GPU:

  --score   re-scores every mask on the 199 val slices under zf, U-Net A and
            the held-out judge U-Net B, and caches per-method metrics plus a
            few example reconstructions to OUT_DIR.
  (default) draws the figures from that cache.

Methods: our hill-climbed mask (B5b), our STE profile (B5), LOUPE's row mask
(filt 64 if trained, else the filt 32 capacity control), the energy oracle
(F7), and the three ACS heuristics (5 seeds each, since they draw per slice).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from itw.discrete import nmse, psnr, ssim
from itw.eval import baseline_row_mask_batch
from itw.profile import StaticProfile, energy_init
from itw.recon import reconstruct
from train_fastmri_static_profile import (
    CachedBatches, _load_recon, _oracle_mask, RECON_A, RECON_B, H, ACS)

OUT_DIR = Path("models_mask_gen_fastmri_nested/eval_val/mask_comparison")
CACHE = OUT_DIR / "metrics.json"
EXAMPLES = OUT_DIR / "examples_s25.npz"
S_GRID = (0.25, 0.40, 0.50, 0.75)
STOCHASTIC = ("acs_vd_gaussian", "acs_equispaced", "acs_random")
METRICS = {"nmse": nmse, "psnr": psnr, "ssim": ssim}

# Fixed order = fixed colour (dataviz reference palette, validated: all checks
# pass, contrast WARN mitigated by per-method markers + direct labels).
STYLE = {
    "ours_hillclimb":  ("hill climb (B5b, ours)", "#2a78d6", "o"),
    "loupe_rows":      ("LOUPE rows",             "#eb6834", "s"),
    "b5_profile":      ("STE profile (B5, ours)", "#1baf7a", "^"),
    "acs_vd_gaussian": ("ACS + VD Gaussian",      "#eda100", "D"),
    "energy_oracle":   ("energy oracle (F7)",     "#e87ba4", "v"),
    "acs_equispaced":  ("ACS + equispaced",       "#008300", "P"),
    "acs_random":      ("ACS + random",           "#4a3aa7", "X"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def _loupe_mask(s: float, dev):
    from train_loupe_baseline import budget_exact_rows
    for tag in ("", "_filt32"):
        f = Path(f"models_loupe/loupe_rows_s{int(s*100):02d}{tag}.pth")
        if f.is_file():
            p = torch.load(f, map_location="cpu", weights_only=False)["prob"]
            return budget_exact_rows(p, s, dev), f"filt32" if tag else "filt64"
    return None, None


def _deterministic_masks(s: float, log_e, dev) -> tuple[dict, dict]:
    masks, notes = {}, {}
    hc = Path(f"models_hill_climb/hillclimb_s{int(s*100):02d}.pth")
    masks["ours_hillclimb"] = torch.load(hc, map_location=dev, weights_only=False)["mask"].to(dev)
    st = torch.load(f"models_static_profile/profile_s{int(s*100):02d}.pth",
                    map_location=dev, weights_only=False)
    prof = StaticProfile(H, ACS).to(dev)
    prof.theta.data.copy_(st["theta"].to(dev))
    masks["b5_profile"] = prof.hard_mask(s).view(1, 1, H, 1)
    masks["energy_oracle"] = _oracle_mask(log_e, s)
    m, note = _loupe_mask(s, dev)
    if m is not None:
        masks["loupe_rows"], notes["loupe_rows"] = m, note
    return masks, notes


@torch.no_grad()
def _score(mask_fn, val, recons, dev) -> dict:
    acc = {f"{r}_{q}": 0.0 for r in recons for q in METRICS}
    for _z, k, t in val.items:
        k, t = k.to(dev), t.to(dev)
        b = k.shape[0]
        mk = mask_fn(b)
        for r, rm in recons.items():
            p = reconstruct(rm, k, mk)
            for q, fn in METRICS.items():
                acc[f"{r}_{q}"] += float(fn(p, t)) * b
    return {kk: v / val.n for kk, v in acc.items()}


def score(dev) -> None:
    from itw.configs import FastMRIConfig, seed_everything
    cfg = FastMRIConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading train (for the energy oracle) and val...")
    train = CachedBatches(cfg.data_root, 8)
    val = CachedBatches(cfg.val_root, 8)
    log_e = energy_init((k for k, _t in train.pairs()), H, dev)
    recons = {"zf": None, "unetA": _load_recon(RECON_A, dev), "unetB": _load_recon(RECON_B, dev)}

    out = {}
    for s in S_GRID:
        masks, notes = _deterministic_masks(s, log_e, dev)
        entry = {}
        for name, m in masks.items():
            entry[name] = _score(lambda b, m=m: m.expand(b, -1, -1, -1), val, recons, dev)
            if name in notes:
                entry[name]["variant"] = notes[name]
        for name in STOCHASTIC:
            per_seed = []
            for seed in range(5):
                seed_everything(seed)
                per_seed.append(_score(
                    lambda b: baseline_row_mask_batch(
                        name, b, H, torch.full((b,), s, device=dev), dev, acs_width=ACS),
                    val, recons, dev))
            entry[name] = {k: float(np.mean([d[k] for d in per_seed])) for k in per_seed[0]}
            entry[name].update({f"{k}_std": float(np.std([d[k] for d in per_seed], ddof=1))
                                for k in per_seed[0]})
        seed_everything(0)
        entry["_rows"] = {n: m.flatten().nonzero().flatten().tolist() for n, m in masks.items()}
        entry["_rows"]["acs_vd_gaussian"] = baseline_row_mask_batch(
            "acs_vd_gaussian", 1, H, torch.full((1,), s, device=dev), dev,
            acs_width=ACS).flatten().nonzero().flatten().tolist()
        out[f"s={s:.2f}"] = entry
        print(f"s={s:.2f}: " + "  ".join(
            f"{n} {entry[n]['unetB_nmse']:.5f}" for n in STYLE if n in entry))
    CACHE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {CACHE}")
    _examples(val, log_e, recons["unetB"], out["s=0.25"], dev)


@torch.no_grad()
def _examples(val, log_e, judge, entry, dev) -> None:
    """Two val slices, chosen by the heuristic's own difficulty (median and
    75th-percentile acs_vd_gaussian NMSE) so the choice cannot favour a method."""
    from itw.configs import seed_everything
    s = 0.25
    ks = torch.cat([k for _z, k, _t in val.items])
    ts = torch.cat([t for _z, _k, t in val.items])
    seed_everything(0)
    ref = baseline_row_mask_batch("acs_vd_gaussian", len(ks), H,
                                  torch.full((len(ks),), s, device=dev), dev, acs_width=ACS)
    per = torch.stack([nmse(reconstruct(judge, ks[i:i+1].to(dev), ref[i:i+1]), ts[i:i+1].to(dev))
                       for i in range(len(ks))]).cpu()
    order = torch.argsort(per)
    idx = [int(order[len(order) // 2]), int(order[3 * len(order) // 4])]

    masks, _ = _deterministic_masks(s, log_e, dev)
    masks["acs_vd_gaussian"] = ref[0:1]
    save = {"idx": np.array(idx), "target": ts[idx].numpy()}
    for name, m in masks.items():
        k = ks[idx].to(dev)
        save[name] = reconstruct(judge, k, m.expand(len(idx), -1, -1, -1)).cpu().numpy()
    np.savez_compressed(EXAMPLES, **save)
    print(f"wrote {EXAMPLES} (val slices {idx})")


# ----------------------------------------------------------------------------
# plotting

def _style_ax(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)


def plot_metrics(data: dict, recon: str, dest: Path) -> None:
    import matplotlib.pyplot as plt
    s_keys = [k for k in data if k.startswith("s=")]
    xs = [float(k[2:]) for k in s_keys]
    names = [n for n in STYLE if any(n in data[k] for k in s_keys)]
    val = lambda k, n, key: data[k][n][key] if n in data[k] else np.nan
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharex=True)
    spec = [("nmse", "NMSE", True), ("psnr", "PSNR (dB)", False), ("ssim", "SSIM", False)]
    for col, (q, label, log) in enumerate(spec):
        top, bot = axes[0, col], axes[1, col]
        ref = np.array([data[k]["acs_vd_gaussian"][f"{recon}_{q}"] for k in s_keys])
        ref_sd = np.array([data[k]["acs_vd_gaussian"][f"{recon}_{q}_std"] for k in s_keys])
        for n in names:
            lab, c, mk = STYLE[n]
            y = np.array([val(k, n, f"{recon}_{q}") for k in s_keys])
            lw, z = (2.4, 5) if n in ("ours_hillclimb", "loupe_rows") else (1.6, 3)
            top.plot(xs, y, color=c, marker=mk, ms=7, lw=lw, label=lab, zorder=z,
                     mec="white", mew=1.0)
            # "better than ACS+VD" is always up: NMSE as % reduction, others as a difference.
            gain = (ref - y) / ref * 100 if q == "nmse" else (y - ref) * (100 if q == "ssim" else 1)
            bot.plot(xs, gain, color=c, marker=mk, ms=7, lw=lw, zorder=z, mec="white", mew=1.0)
        band = 2 * ref_sd / ref * 100 if q == "nmse" else 2 * ref_sd * (100 if q == "ssim" else 1)
        bot.fill_between(xs, -band, band, color=GRID, alpha=0.9, lw=0, zorder=1,
                         label="ACS+VD ±2 sd (seeds)")
        bot.axhline(0, color=MUTED, lw=1, zorder=2)
        if log:
            top.set_yscale("log")
        top.set_ylabel(label, color=INK)
        bot.set_ylabel({"nmse": "NMSE reduction vs ACS+VD (%)",
                        "psnr": "PSNR gain vs ACS+VD (dB)",
                        "ssim": "SSIM gain vs ACS+VD (×100)"}[q], color=INK)
        bot.set_xlabel("sampling fraction s", color=INK)
        bot.set_xticks(xs)
        for ax in (top, bot):
            _style_ax(ax)
        top.set_title(f"{label}  ({'lower' if q == 'nmse' else 'higher'} is better)",
                      color=INK, fontsize=11, loc="left")
    # Direct labels on the two methods the comparison is about.
    ax = axes[1, 0]
    for n in ("ours_hillclimb", "loupe_rows"):
        if n in names:
            y = [(data[k]["acs_vd_gaussian"][f"{recon}_nmse"] - val(k, n, f"{recon}_nmse"))
                 / data[k]["acs_vd_gaussian"][f"{recon}_nmse"] * 100 for k in s_keys]
            ax.annotate(STYLE[n][0].split(" (")[0], (xs[0], y[0]), xytext=(6, 6 if y[0] > 0 else -12),
                        textcoords="offset points", color=INK, fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    hb, lb = axes[1, 0].get_legend_handles_labels()
    fig.legend(h + hb, l + lb, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    rname = {"unetB": "held-out judge U-Net B", "unetA": "U-Net A",
             "zf": "zero-filled (no network)"}[recon]
    variant = {data[k].get("loupe_rows", {}).get("variant") for k in s_keys} - {None}
    fig.suptitle(f"Static row masks on 199 held-out val slices, reconstructed by {rname}"
                 + (f"   [LOUPE: {', '.join(sorted(variant))}]" if variant else ""),
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(dest, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {dest}")


def plot_masks(data: dict, dest: Path) -> None:
    """Which PE rows each method samples, centred at DC, with the ±f mirror count."""
    import matplotlib.pyplot as plt
    s_keys = [k for k in data if k.startswith("s=")]
    names = [n for n in STYLE if any(n in data[k]["_rows"] for k in s_keys)]
    fig, axes = plt.subplots(len(names), len(s_keys), figsize=(16, 0.62 * len(names) + 1.6),
                             sharex=True, squeeze=False)
    f = np.arange(H) - H // 2
    for j, k in enumerate(s_keys):
        for i, n in enumerate(names):
            ax = axes[i, j]
            rows = data[k]["_rows"].get(n)
            if rows is None:
                ax.text(0.5, 0.5, "not trained yet", transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color=MUTED)
                ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
                continue
            ax.bar(f, np.ones(H), width=1.0, color="#efeee8", lw=0)
            ax.bar(f[rows], np.ones(len(rows)), width=1.0, color=STYLE[n][1], lw=0)
            fr = {r - H // 2 for r in rows}
            pairs = sum(1 for x in fr if x > 0 and -x in fr)
            ax.text(1.0, 0.5, f" {pairs} ±f pairs", transform=ax.transAxes, va="center",
                    ha="left", fontsize=8, color=MUTED)
            ax.set_yticks([])
            ax.set_xlim(-H // 2, H // 2)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
            if j == 0:
                ax.set_ylabel(STYLE[n][0], rotation=0, ha="right", va="center", fontsize=9,
                              color=INK)
            if i == 0:
                ax.set_title(k.replace("s=", "s = "), color=INK, fontsize=11)
        axes[-1, j].set_xlabel("phase-encode frequency f (0 = DC)", color=INK, fontsize=9)
    fig.suptitle("Sampled k-space rows per method (ACS+VD: one seed-0 draw; it redraws per slice)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.96, 0.95))
    fig.savefig(dest, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {dest}")


def plot_examples(dest: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ex = np.load(EXAMPLES)
    names = [n for n in STYLE if n in ex.files]
    # magnitude_from_kspace (itw/masks.py) leaves the image origin at the corner,
    # i.e. circularly shifted by H/2 vertically. Every metric is shift-invariant
    # (or, for SSIM, nearly), so only the display is re-centred here.
    centre = lambda a: np.roll(a, a.shape[-2] // 2, axis=-2)
    err_cmap = LinearSegmentedColormap.from_list("err", ["#ffffff", "#86b6ef", "#1c5cab", "#0d366b"])
    n_img = len(ex["idx"])
    fig, axes = plt.subplots(2 * n_img, len(names) + 1,
                             figsize=(2.3 * (len(names) + 1), 2.5 * 2 * n_img), squeeze=False)
    for r in range(n_img):
        t = centre(ex["target"][r, 0])
        vmax = np.percentile(t, 99.5)
        emax = 0.25 * vmax
        a_img, a_err = axes[2 * r], axes[2 * r + 1]
        a_img[0].imshow(t, cmap="gray", vmin=0, vmax=vmax)
        a_img[0].set_title(f"fully sampled\nval slice {int(ex['idx'][r])}", fontsize=9, color=INK)
        a_err[0].text(0.5, 0.5, "|error|\n(0 to 25% of peak)", ha="center", va="center",
                      fontsize=9, color=MUTED, transform=a_err[0].transAxes)
        for c, n in enumerate(names, start=1):
            p = centre(ex[n][r, 0])
            tt, pp = torch.from_numpy(t)[None, None], torch.from_numpy(p)[None, None]
            a_img[c].imshow(p, cmap="gray", vmin=0, vmax=vmax)
            a_img[c].set_title(f"{STYLE[n][0]}\nPSNR {float(psnr(pp, tt)):.2f} dB"
                               f"  NMSE {float(nmse(pp, tt)):.4f}", fontsize=8, color=INK)
            a_err[c].imshow(np.abs(p - t), cmap=err_cmap, vmin=0, vmax=emax)
            for sp in a_img[c].spines.values():
                sp.set_color(STYLE[n][1]); sp.set_linewidth(3)
        for ax in list(a_img) + list(a_err):
            ax.set_xticks([]); ax.set_yticks([])
        for sp in a_err[0].spines.values():
            sp.set_visible(False)
    fig.suptitle("s = 0.25, reconstructed by the held-out judge U-Net B "
                 "(slices = median and 75th-percentile difficulty for ACS+VD)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(dest, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--score", action="store_true", help="re-score masks on GPU first")
    args = ap.parse_args()
    if args.score:
        from itw.configs import default_device
        score(default_device())
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    for recon in ("unetB", "zf", "unetA"):
        plot_metrics(data, recon, OUT_DIR / f"metrics_vs_s_{recon}.png")
    plot_masks(data, OUT_DIR / "sampled_rows.png")
    if EXAMPLES.is_file():
        plot_examples(OUT_DIR / "examples_s25.png")


if __name__ == "__main__":
    main()
