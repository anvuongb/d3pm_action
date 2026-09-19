"""Train and evaluate MNIST / CIFAR-10 mask generators, headless.

The notebooks (20260415_itw_mnist.ipynb, 20260416_itw_cifar10.ipynb) trained
with the original ``entropy_region="observed"`` objective, which is degenerate:
it scores entropy only on observed pixels, so the MNIST mask learned to observe
background only (0% of the digit at s <= 0.5) and the CIFAR mask shrank to
empty. This script trains with the chosen region and evaluates on the held-out
*test* split, reporting the proxy alongside criteria the mask never saw:
the D3PM's accuracy on the hidden pixels, and the share of observed pixels
that carry information (value != absorbing token).

    python train_image_masks.py --dataset mnist --entropy-region all
    python train_image_masks.py --dataset cifar10 --entropy-region all --sparsity-weight 10
    python train_image_masks.py --dataset mnist --eval-only \\
        --ckpt models_mask_gen_mnist/mask_gen_mnist_final.pth --entropy-region observed
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from itw import (CIFAR10Config, MNISTConfig, build_dataloader, build_mask_model,
                 build_pixel_survival_table,
                 evaluate_loader, load_d3pm, train_mask_generator)
from itw.configs import seed_everything

D3PM = {"mnist": "models/mnist/model_absorb_cosine_399.pth",
        "cifar10": "models/cifar10/model_absorb_cosine_499.pth"}


def _cfg(args) -> MNISTConfig | CIFAR10Config:
    base = MNISTConfig if args.dataset == "mnist" else CIFAR10Config
    kw = dict(device="cuda", mask_arch="spatial", n_epochs=args.epochs, save_every=10,
              d3pm_checkpoint=D3PM[args.dataset], entropy_region=args.entropy_region,
              save_dir=args.save_dir)
    if args.sparsity_weight is not None:
        kw["sparsity_loss_weight"] = args.sparsity_weight
    if args.batch is not None:
        kw["batch_size"] = args.batch
    return base(**kw)


def evaluate(cfg, model, d3pm, survival, sparsities, max_batches, eval_batch) -> dict:
    """Held-out test split; proxy scored in ``cfg.entropy_region``."""
    vcfg = replace(cfg, data_split="val", batch_size=eval_batch, seed=0)
    loader = build_dataloader(vcfg)
    out = {}
    for s in sparsities:
        seed_everything(0)
        out[f"s={s:.2f}"] = evaluate_loader(d3pm, model, vcfg, loader, survival,
                                            max_batches=max_batches, fixed_sparsity=s)
        r = out[f"s={s:.2f}"]
        print(f"  s={s:.2f}: H learned {r['h_learned']:.5f} random {r['h_random']:.5f} | "
              f"acc_unobs learned {r['acc_unobserved_learned']:.4f} "
              f"random {r['acc_unobserved_random']:.4f} | acc_fg learned "
              f"{r['acc_unobserved_fg_learned']:.4f} random {r['acc_unobserved_fg_random']:.4f} | "
              f"full acc learned {r['acc_full_learned']:.4f} random {r['acc_full_random']:.4f} | "
              f"fg recall learned {r['recall_fg_learned']:.4f} random {r['recall_fg_random']:.4f} | "
              f"psnr learned {r['psnr_learned']:.2f} random {r['psnr_random']:.2f} | "
              f"informative learned "
              f"{r['informative_frac_learned']:.3f} random {r['informative_frac_random']:.3f} | "
              f"density {r['mean_sparsity_learned']:.3f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    ap.add_argument("--entropy-region", choices=["observed", "all"], default="all")
    ap.add_argument("--sparsity-weight", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None, help="mask generator to evaluate")
    ap.add_argument("--survival", default=None, help="survival table (default: next to ckpt)")
    ap.add_argument("--eval-sparsities", type=float, nargs="*", default=[0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--out", default=None, help="eval JSON path")
    args = ap.parse_args()
    if args.save_dir is None:
        args.save_dir = f"models_mask_gen_{args.dataset}_{args.entropy_region}H"

    cfg = _cfg(args)
    d3pm = load_d3pm(cfg)
    for p in d3pm.parameters():
        p.requires_grad_(False)

    if args.eval_only:
        model = build_mask_model(cfg)
        model.load_state_dict(torch.load(args.ckpt, map_location=cfg.device, weights_only=False))
        model.to(cfg.device)
        surv = Path(args.survival or Path(args.ckpt).parent / "survival_table.pt")
        if surv.is_file():
            survival = torch.load(surv, map_location=cfg.device, weights_only=False)
        else:                      # e.g. a run that stopped before saving it
            print(f"no {surv}; calibrating a survival table on the train split")
            survival = build_pixel_survival_table(
                d3pm, build_dataloader(cfg), device=cfg.device,
                max_batches=cfg.schedule_calibration_batches)
        source = args.ckpt
    else:
        seed_everything(0)
        print(f"training {args.dataset}: region={cfg.entropy_region} "
              f"sparsity_weight={cfg.sparsity_loss_weight} batch={cfg.batch_size} "
              f"epochs={cfg.n_epochs} -> {cfg.save_dir}", flush=True)
        model, survival = train_mask_generator(cfg, d3pm=d3pm)
        source = str(Path(cfg.save_dir) / f"mask_gen_{cfg.dataset}_final.pth")
        torch.cuda.empty_cache()

    print(f"evaluating {source} on the held-out test split "
          f"(proxy region: {cfg.entropy_region})", flush=True)
    res = evaluate(cfg, model, d3pm, survival.to(cfg.device), args.eval_sparsities,
                   args.eval_batches, args.eval_batch_size)
    out = Path(args.out or Path(args.save_dir) / "eval_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "_meta": {
        "dataset": args.dataset, "checkpoint": source, "entropy_region": cfg.entropy_region,
        "sparsity_loss_weight": cfg.sparsity_loss_weight, "batch_size": cfg.batch_size,
        "epochs": cfg.n_epochs, "split": "test", "eval_batches": args.eval_batches,
        "eval_batch_size": args.eval_batch_size}}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
