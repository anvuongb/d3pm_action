"""Draw the MNIST/CIFAR-10 figures from the cached evaluation JSONs (no GPU).

    python plot_image_metrics.py

Reads ``eval_image_masks/{mnist,cifar10}_allH_*.json`` and writes vector
figures into ``docs/latex/figures/``:

  metrics_vs_budget.pdf   learned vs random against the budget s, on the
                          criteria the mask never optimized.
  training_divergence.pdf the surrogate and the reconstruction criterion over
                          training, in separate panels (never a shared axis):
                          the surrogate keeps improving while reconstruction
                          does not.

Colors, marks and labels follow the dataviz reference palette already used by
plot_mask_comparison.py, so every figure in the paper reads as one system:
slot 1 (blue) is always the learned mask, slot 2 (orange) always random.
Validated with the skill's validator (all six checks pass on this pair).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL = Path("eval_image_masks")
DEST = Path("docs/latex/figures")
LEARNED, RANDOM = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3dc", "#fcfcfb"
S_GRID = (0.1, 0.3, 0.5, 0.7)
# Checkpoint -> epochs trained. save_every=10 writes 9e after 10 epochs.
MNIST_CKPTS = {"9e": 10, "29e": 30, "49e": 50, "99e": 100, "149e": 150, "final": 200}
CIFAR_CKPTS = {"39e": 40, "79e": 80, "99e": 100, "final": 200}

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
})


def load(dataset: str, tag: str) -> dict:
    return json.loads((EVAL / f"{dataset}_allH_{tag}.json").read_text())


def series(res: dict, metric: str, arm: str) -> list[float]:
    return [res[f"s={s:.2f}"][f"{metric}_{arm}"] for s in S_GRID]


def dress(ax, xlabel: str, ylabel: str, title: str | None = None) -> None:
    """Recessive grid and axes; the data stays the most salient thing."""
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=3)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, color=INK, loc="left", pad=4)


def line(ax, xs, ys, color, label, marker) -> None:
    ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=5.5, mec="white", mew=0.9,
            ls="-" if color == LEARNED else (0, (4, 2)),
            label=label, zorder=3, clip_on=False)


def metrics_vs_budget() -> None:
    """Three criteria the masks never optimized, against the budget."""
    mnist, cifar = load("mnist", "final"), load("cifar10", "final")
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95))
    panels = [
        (axes[0], mnist, "recall_fg", "digit recovery", "MNIST"),
        (axes[1], mnist, "acc_unobserved_fg", "hidden digit-pixel acc.", "MNIST"),
        (axes[2], cifar, "psnr", "PSNR (dB)", "CIFAR-10"),
    ]
    for ax, res, metric, ylabel, title in panels:
        line(ax, S_GRID, series(res, metric, "learned"), LEARNED, "learned", "o")
        line(ax, S_GRID, series(res, metric, "random"), RANDOM, "random", "s")
        dress(ax, "budget $s$", ylabel, title)
        ax.set_xticks(S_GRID)
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.6,
                   labelcolor=INK, borderpad=0.2)
    fig.tight_layout(pad=0.5)
    fig.savefig(DEST / "metrics_vs_budget.pdf")
    plt.close(fig)


def training_divergence() -> None:
    """Surrogate and reconstruction over training, one panel each.

    Separate panels rather than two y-scales on one: the point is the shape of
    each curve over training, and a twin axis would let the crossing point be
    set by the scaling rather than by the data.
    """
    cols = [
        ("MNIST, $s = 0.7$", "mnist", MNIST_CKPTS, "recall_fg", "digit recovery"),
        ("CIFAR-10, $s = 0.1$", "cifar10", CIFAR_CKPTS, "psnr", "PSNR (dB)"),
    ]
    key = {"mnist": "s=0.70", "cifar10": "s=0.10"}
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 2.45), sharex="col")
    for j, (title, dataset, ckpts, metric, ylabel) in enumerate(cols):
        res = {e: load(dataset, t) for t, e in ckpts.items()}
        epochs = sorted(res)
        k = key[dataset]
        top, bot = axes[0][j], axes[1][j]

        line(top, epochs, [res[e][k]["h_learned"] for e in epochs], LEARNED,
             "learned", "o")
        rnd_h = res[epochs[-1]][k]["h_random"]
        top.axhline(rnd_h, color=RANDOM, lw=1.6, ls=(0, (4, 2)), zorder=2,
                    label="random")
        dress(top, "", r"$\mathcal{H}_{\mathrm{all}}$", title)

        line(bot, epochs, [res[e][k][f"{metric}_learned"] for e in epochs], LEARNED,
             "learned", "o")
        rnd_m = res[epochs[-1]][k][f"{metric}_random"]
        bot.axhline(rnd_m, color=RANDOM, lw=1.6, ls=(0, (4, 2)), zorder=2)
        dress(bot, "epochs trained", ylabel)
    # One legend, in the empty lower-left of the first panel: direct labels
    # collide with the curves here because the two series cross.
    axes[0][0].legend(frameon=False, loc="lower left", handlelength=2.2,
                      labelcolor=INK, borderpad=0.2)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=1.6)
    fig.savefig(DEST / "training_divergence.pdf")
    plt.close(fig)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    metrics_vs_budget()
    training_divergence()
    print(f"wrote {DEST}/metrics_vs_budget.pdf and {DEST}/training_divergence.pdf")
