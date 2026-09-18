"""LOUPE baseline (Bahadir et al. 2020) under our evaluation protocol.

Trains the ported LOUPE (itw/loupe.py: learned probability mask + jointly
trained U-Net, MAE loss on magnitude) on our training split, then scores the
*mask* it learned under our protocol: budget-exact, on held-out val, against
the held-out judge reconstructor.

Comparability caveats, all forced by LOUPE's own design:

- LOUPE's forward model FFTs a *real* magnitude image, so its k-space is
  Hermitian-symmetric -- an easier problem than our true complex k-space. We
  train it under its own assumption and transfer the mask.
- LOUPE trains its reconstructor jointly; ours is frozen. We therefore compare
  only the masks, each scored by the same judge.
- LOUPE's budget holds in expectation (RescaleProbMap sets mean(p)=s). We take a
  budget-exact top-k of its learned probability map, since F4 showed density
  mismatch roughly doubles apparent margins.
- LOUPE has no ACS constraint; our masks lock a 32-row ACS block. Its mask is
  used exactly as learned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from itw.configs import FastMRIConfig, default_device, seed_everything
from itw.discrete import nmse, ssim
from itw.eval import baseline_row_mask_batch
from itw.loupe import Loupe, probmask_rows_to_centred, rescale_prob_map
from itw.recon import reconstruct
from train_fastmri_static_profile import (
    CachedBatches, _load_recon, RECON_A, RECON_B, H, ACS)

OUT_DIR = Path("models_loupe")
OUT_JSON = Path("models_mask_gen_fastmri_nested/eval_val/eval_loupe.json")


def _topk_set(prob, sparsity):
    k = int(sparsity * prob.numel())
    return set(torch.topk(prob.detach().flatten(), k).indices.tolist())


def train_loupe(images, sparsity, rows, dev, epochs, filt, batch, lr=1e-3, log=print):
    """Train to *mask* convergence, not a fixed epoch count.

    Upstream trains 60 epochs over 60k images (~112k steps). Our split has 973
    images, so a fixed 60 epochs would be ~500x less optimisation and would
    understate LOUPE. We therefore track how much the selected row set still
    moves per epoch and report it, so convergence is evidence rather than an
    assumption.
    """
    model = Loupe(H, H, sparsity, rows=rows, filt=filt).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(images)
    prev = _topk_set(rescale_prob_map(model.prob(), sparsity), sparsity)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            x = images[perm[i:i + batch]].to(dev)
            loss = (model(x) - x).abs().mean()        # keras loss='mae'
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * x.shape[0]
        cur = _topk_set(rescale_prob_map(model.prob(), sparsity), sparsity)
        churn = len(cur - prev)
        prev = cur
        history.append({"epoch": ep, "mae": tot / n, "mask_churn": churn})
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  epoch {ep}: mae {tot / n:.6f}  mask churn {churn}/{len(cur)}")
    return model, history


def budget_exact_rows(prob_rows: torch.Tensor, sparsity: float, dev) -> torch.Tensor:
    """LOUPE row probabilities -> our centred, budget-exact row mask."""
    centred = probmask_rows_to_centred(prob_rows.detach().cpu()).to(dev)
    m = torch.zeros(H, device=dev)
    m[torch.topk(centred, int(sparsity * H)).indices] = 1.0
    return m.view(1, 1, H, 1)


def budget_exact_2d(prob2d: torch.Tensor, sparsity: float, dev) -> torch.Tensor:
    """Native LOUPE 2D point mask -> centred, budget-exact. Not Cartesian."""
    c = torch.fft.fftshift(prob2d.detach().cpu()[0, 0], dim=(0, 1)).to(dev)
    k = int(sparsity * H * H)
    flat = torch.zeros(H * H, device=dev)
    flat[torch.topk(c.flatten(), k).indices] = 1.0
    return flat.view(1, 1, H, H)


def main() -> None:
    ap = argparse.ArgumentParser(description="LOUPE baseline under our protocol")
    ap.add_argument("--sparsities", type=float, nargs="*", default=[0.25, 0.40, 0.50, 0.75])
    ap.add_argument("--mask", choices=["rows", "native", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--filt", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev = default_device()
    cfg = FastMRIConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = CachedBatches(cfg.data_root, 8)
    val = CachedBatches(cfg.val_root, 8)

    # LOUPE consumes real magnitude images in [0,1], per-slice normalised.
    imgs = torch.cat([t for _z, _k, t in train.items])
    imgs = imgs / imgs.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    if args.limit:
        imgs = imgs[:args.limit]
    print(f"LOUPE baseline: {len(imgs)} train images, val {val.n}, device {dev}")

    recon_a = _load_recon(RECON_A, dev)
    recon_b = _load_recon(RECON_B, dev) if RECON_B.is_file() else None
    recons = {"zf": None, "unetA": recon_a, **({"unetB": recon_b} if recon_b else {})}
    hc = {s: torch.load(f"models_hill_climb/hillclimb_s{int(s*100):02d}.pth",
                        map_location=dev, weights_only=False)["mask"].to(dev)
          for s in args.sparsities
          if Path(f"models_hill_climb/hillclimb_s{int(s*100):02d}.pth").is_file()}

    modes = ["rows", "native"] if args.mask == "both" else [args.mask]
    report = {}
    for s in args.sparsities:
        entry = {}
        for mode in modes:
            print(f"\n=== LOUPE {mode} s={s:.2f} ===")
            seed_everything(0)
            model, hist = train_loupe(imgs, s, mode == "rows", dev, args.epochs,
                                      args.filt, args.batch)
            tail = [h["mask_churn"] for h in hist[-10:]]
            print(f"  mask churn over last 10 epochs: {tail}")
            p = rescale_prob_map(model.prob(), s)
            torch.save({"prob": p.detach().cpu(), "sparsity": s, "mode": mode},
                       OUT_DIR / f"loupe_{mode}_s{int(s*100):02d}{args.tag}.pth")
            mask = (budget_exact_rows(p, s, dev) if mode == "rows"
                    else budget_exact_2d(p, s, dev))
            entry[f"loupe_{mode}"] = _score(mask, val, recons, dev)
            entry[f"loupe_{mode}"]["final_mae"] = hist[-1]["mae"]
            entry[f"loupe_{mode}"]["mask_churn_last10"] = tail
            print("  scored: " + "  ".join(
                f"{k} {v:.5f}" for k, v in entry[f"loupe_{mode}"].items()
                if k.endswith("nmse") and isinstance(v, float)))

        if s in hc:
            entry["ours_hillclimb"] = _score(hc[s], val, recons, dev)
        entry["acs_vd_gaussian"] = _score_stochastic(
            "acs_vd_gaussian", s, val, recons, dev, args.seeds)
        report[f"s={s:.2f}"] = entry

        judge = "unetB" if recon_b else "unetA"
        print(f"\n--- s={s:.2f} ranked by {judge} ---")
        for name in sorted(entry, key=lambda n: entry[n][f"{judge}_nmse"]):
            sd = entry[name].get(f"{judge}_nmse_std")
            print(f"  {name:>18} {entry[name][f'{judge}_nmse']:.5f}"
                  + (f" +-{sd:.5f}" if sd else ""))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({**report, "_meta": {
        "stage": "LOUPE baseline", "epochs": args.epochs, "filt": args.filt,
        "batch": args.batch, "n_train": len(imgs), "n_val": val.n,
        "port": "itw/loupe.py, validated against upstream TF in tests/test_loupe_port.py",
        "caveats": "LOUPE trained on real magnitude images (Hermitian k-space) with a "
                   "jointly trained U-Net; masks made budget-exact by top-k; no ACS lock.",
    }}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


@torch.no_grad()
def _score(mask, val, recons, dev) -> dict:
    out = {f"{r}_{q}": 0.0 for r in recons for q in ("nmse", "ssim")}
    for _z, k, t in val.items:
        k, t = k.to(dev), t.to(dev)
        b = k.shape[0]
        mk = mask.expand(b, -1, -1, -1) if mask.shape[-1] == 1 else mask.expand(b, -1, -1, -1)
        for rname, rmodel in recons.items():
            p = reconstruct(rmodel, k, mk)
            out[f"{rname}_nmse"] += float(nmse(p, t)) * b
            out[f"{rname}_ssim"] += float(ssim(p, t)) * b
    return {kk: v / val.n for kk, v in out.items()}


@torch.no_grad()
def _score_stochastic(name, s, val, recons, dev, seeds) -> dict:
    per = {r: [] for r in recons}
    for sd in seeds:
        seed_everything(sd)
        acc = {r: 0.0 for r in recons}
        for _z, k, t in val.items:
            k, t = k.to(dev), t.to(dev)
            b = k.shape[0]
            mk = baseline_row_mask_batch(name, b, H, torch.full((b,), s, device=dev),
                                         dev, acs_width=ACS)
            for r, rm in recons.items():
                acc[r] += float(nmse(reconstruct(rm, k, mk), t)) * b
        for r in recons:
            per[r].append(acc[r] / val.n)
    out = {}
    for r, vals in per.items():
        m = sum(vals) / len(vals)
        out[f"{r}_nmse"] = m
        out[f"{r}_nmse_std"] = (sum((x - m) ** 2 for x in vals) / max(len(vals) - 1, 1)) ** 0.5
    return out


if __name__ == "__main__":
    main()
