"""Mask and reconstruction galleries for MNIST and CIFAR-10.

Two passes, so the figure can be redrawn without a GPU:

  --sample   runs the mask generator and the frozen D3PM on held-out test
             images and caches masks, observations and reconstructions to
             eval_image_masks/examples_{dataset}.npz
  (default)  draws docs/latex/figures/examples_{dataset}.pdf from that cache

The sampling path is the evaluation path of itw.eval.evaluate_batch, so what
the figure shows is what the tables score: the same Gumbel mask at temperature
0.5, the same budget-exact random baseline, the same absorbing observation, and
the same reconstruction (observed pixels at their measured value, hidden pixels
at the D3PM's posterior mean).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from itw import (CIFAR10Config, MNISTConfig, build_dataloader, build_mask_model,
                 load_d3pm)
from itw.configs import seed_everything
from itw.eval import random_mask_batch
from itw.masks import apply_masked_observation, gumbel_mask
from itw.train import discretize, forward_mask_logits
from itw.schedule import sparsity_to_timestep

CKPT = {"mnist": "models_mask_gen_mnist_allH/mask_gen_mnist_final.pth",
        "cifar10": "models_mask_gen_cifar10_allH/mask_gen_cifar10_final.pth"}
D3PM = {"mnist": "models/mnist/model_absorb_cosine_399.pth",
        "cifar10": "models/cifar10/model_absorb_cosine_499.pth"}
SURVIVAL = {d: Path(CKPT[d]).parent / "survival_table.pt" for d in CKPT}
CACHE = Path("eval_image_masks")
DEST = Path("docs/latex/figures")
S_SHOWN = (0.1, 0.3, 0.7)
N_IMAGES = 4

LEARNED, RANDOM = "#2a78d6", "#eb6834"
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a64", "#fcfcfb"


def _cfg(dataset: str):
    base = MNISTConfig if dataset == "mnist" else CIFAR10Config
    return base(device="cuda", mask_arch="spatial", entropy_region="all",
                sparsity_loss_weight=10.0, d3pm_checkpoint=D3PM[dataset],
                data_split="val", batch_size=N_IMAGES, seed=0)


@torch.no_grad()
def sample(dataset: str) -> None:
    cfg = _cfg(dataset)
    d3pm = load_d3pm(cfg)
    model = build_mask_model(cfg)
    model.load_state_dict(torch.load(CKPT[dataset], map_location=cfg.device,
                                     weights_only=False))
    model.to(cfg.device).eval()
    survival = torch.load(SURVIVAL[dataset], map_location=cfg.device,
                          weights_only=False).to(cfg.device)

    seed_everything(0)
    x, cond = next(iter(build_dataloader(cfg)))
    x, cond = x[:N_IMAGES].to(cfg.device), cond[:N_IMAGES].to(cfg.device)
    x_disc = discretize(x, cfg.num_classes)
    levels = torch.linspace(0, 1, cfg.num_classes, device=cfg.device)

    out = {"target": levels[x_disc].cpu().numpy()}
    for s in S_SHOWN:
        sparsity = torch.full((x.shape[0],), s, device=cfg.device)
        t = sparsity_to_timestep(sparsity, survival, cfg.n_t)
        logits = forward_mask_logits(model, cfg, x_disc, cond, sparsity)
        masks = {"learned": gumbel_mask(logits, temperature=0.5, hard=True),
                 "random": random_mask_batch(
                     x.shape[0], (cfg.image_channels, cfg.image_size, cfg.image_size),
                     sparsity, cfg.device)}
        for arm, mask in masks.items():
            y = apply_masked_observation(x_disc, mask, cfg.num_classes,
                                         multichannel=cfg.multichannel)
            probs = d3pm.model_predict(y, t, cond).float().softmax(dim=-1)
            m = mask.expand_as(x_disc) if mask.shape[1] != x_disc.shape[1] else mask
            recon = torch.where(m > 0.5, levels[x_disc], probs @ levels)
            tag = f"{arm}_s{int(s * 100):02d}"
            out[f"mask_{tag}"] = mask.cpu().numpy()
            out[f"recon_{tag}"] = recon.cpu().numpy()
            out[f"density_{tag}"] = mask.mean(dim=(1, 2, 3)).cpu().numpy()
    dest = CACHE / f"examples_{dataset}.npz"
    np.savez_compressed(dest, **out)
    print(f"wrote {dest}")


def _show(ax, img: np.ndarray, tint: str | None = None) -> None:
    """One panel. A mask is drawn in its arm's color so identity is not carried
    by position alone; images are drawn as they are."""
    import matplotlib.colors as mcolors
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(MUTED)
        sp.set_linewidth(0.4)
    if tint is not None:
        cmap = mcolors.LinearSegmentedColormap.from_list("m", ["#14140f", tint])
        ax.imshow(img[0], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    elif img.shape[0] == 1:
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        ax.imshow(np.transpose(img, (1, 2, 0)).clip(0, 1), interpolation="nearest")


def draw(dataset: str) -> None:
    """Rows are what is shown, columns are the budget, blocks are examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "pdf.fonttype": 42})

    d = np.load(CACHE / f"examples_{dataset}.npz")
    rows = [("mask", "learned"), ("recon", "learned"), ("mask", "random"),
            ("recon", "random")]
    labels = ["learned\nmask", "learned\nreconstruction", "random\nmask",
              "random\nreconstruction"]
    n_blocks, ncol = 2, 1 + len(S_SHOWN)
    fig, axes = plt.subplots(len(rows), n_blocks * ncol,
                             figsize=(5.5, 0.66 * len(rows) + 0.5), squeeze=False)
    for b in range(n_blocks):
        for r, (kind, arm) in enumerate(rows):
            color = LEARNED if arm == "learned" else RANDOM
            ax0 = axes[r][b * ncol]
            if kind == "recon":
                _show(ax0, d["target"][b])
                if r == 1:
                    ax0.set_title("target", color=INK, fontsize=6.5, pad=3)
            else:
                ax0.axis("off")
            for c, s_ in enumerate(S_SHOWN, start=1):
                ax = axes[r][b * ncol + c]
                img = d[f"{kind}_{arm}_s{int(s_ * 100):02d}"][b]
                _show(ax, img, tint=color if kind == "mask" else None)
                if r == 0:
                    ax.set_title(f"$s = {s_}$", color=INK, fontsize=6.5, pad=3)
            if b == 0:
                # Row label hangs off the left edge; for the mask rows the
                # first cell stays blank, so the label is drawn into it.
                axes[r][0].text(-0.09, 0.5, labels[r], transform=axes[r][0].transAxes,
                                ha="right", va="center", fontsize=6, color=color)
    fig.tight_layout(pad=0.2, w_pad=0.12, h_pad=0.12, rect=(0.135, 0, 1, 1))
    dest = DEST / f"examples_{dataset}.pdf"
    fig.savefig(dest)
    plt.close(fig)
    print(f"wrote {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", action="store_true", help="re-run the GPU pass")
    ap.add_argument("--datasets", nargs="*", default=["mnist", "cifar10"])
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        if args.sample:
            sample(ds)
        draw(ds)
