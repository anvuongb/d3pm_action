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
RUN_DIR = OUT_JSON.parent / "loupe"


def _topk_set(prob, sparsity):
    k = int(sparsity * prob.numel())
    return set(torch.topk(prob.detach().flatten(), k).indices.tolist())


def train_loupe(images, sparsity, rows, dev, epochs, filt, batch, lr=1e-3, log=print,
                patience=0, min_epochs=0, monitor=None, eval_every=10,
                ckpt=None, ckpt_every=10, churn_tol=0.0):
    """Train to *mask* convergence, not a fixed epoch count.

    Upstream trains 60 epochs over 60k images (~112k steps). Our split has 973
    images, so a fixed 60 epochs would be ~500x less optimisation and would
    understate LOUPE. We therefore track how much the selected row set still
    moves per epoch and stop once it has stayed at or below ``churn_tol`` (a
    fraction of the row budget) for ``patience`` epochs, so convergence is
    evidence rather than an assumption. ``epochs`` is a hard cap.

    A tolerance is needed at s >= 0.50: there LOUPE leaves ~100 rows with
    near-tied probabilities (gaps ~1e-3) around the top-k cut, so 1-3 of them
    trade places every epoch indefinitely while val NMSE is flat.

    ``monitor(model) -> dict`` is logged every ``eval_every`` epochs; it must not
    use the held-out judge. ``ckpt`` is saved every ``ckpt_every`` epochs and
    resumed from if present.
    """
    model = Loupe(H, H, sparsity, rows=rows, filt=filt).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(images)
    prev = _topk_set(rescale_prob_map(model.prob(), sparsity), sparsity)
    tol = int(churn_tol * len(prev))
    history, start, still = [], 0, 0
    if ckpt is not None and Path(ckpt).is_file():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        history, start, prev = st["history"], st["epoch"] + 1, st["prev"]
        still = 0                       # recount under the current tolerance
        for h in reversed(history):
            if h["mask_churn"] > tol:
                break
            still += 1
        torch.set_rng_state(st["rng"].cpu())   # map_location moves it to the GPU
        log(f"  resumed from {ckpt} at epoch {start}")
        if patience and still >= patience and start >= min_epochs:
            log("  checkpoint already converged; not training further")
            return model, history
    for ep in range(start, epochs):
        model.train()
        perm = torch.randperm(n)
        tot, gsum, gcount = 0.0, 0.0, 0
        for i in range(0, n, batch):
            x = images[perm[i:i + batch]].to(dev)
            loss = (model(x) - x).abs().mean()        # keras loss='mae'
            opt.zero_grad(); loss.backward()
            gsum += float(model.prob.logit.grad.abs().mean()); gcount += 1
            opt.step()
            tot += loss.item() * x.shape[0]
        p = rescale_prob_map(model.prob(), sparsity).detach()
        cur = _topk_set(p, sparsity)
        churn = len(cur - prev)
        prev = cur
        still = still + 1 if churn <= tol else 0
        rec = {"epoch": ep, "mae": tot / n, "mask_churn": churn,
               "logit_grad": gsum / max(gcount, 1),
               "p_saturated": float(((p < 1e-3) | (p > 1 - 1e-3)).float().mean())}
        line = (f"  epoch {ep}: mae {rec['mae']:.6f}  churn {churn}/{len(cur)}"
                f"  |dL/dlogit| {rec['logit_grad']:.2e}  p_sat {rec['p_saturated']:.2f}")
        if monitor is not None and (ep % eval_every == 0 or ep == epochs - 1):
            rec.update(monitor(model))
            line += "  " + "  ".join(f"{k} {v:.5f}" for k, v in rec.items()
                                     if k.endswith("nmse"))
        history.append(rec)
        log(line, flush=True)
        done = patience and still >= patience and ep + 1 >= min_epochs
        if ckpt is not None and (ep % ckpt_every == 0 or done or ep == epochs - 1):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "history": history, "epoch": ep, "still": still, "prev": prev,
                        "rng": torch.get_rng_state()}, ckpt)
        if done:
            log(f"  converged: churn <= {tol} for {still} epochs, stopping at epoch {ep}")
            break
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


def _run_json(mode: str, s: float, tag: str) -> Path:
    return RUN_DIR / f"eval_loupe_{mode}_s{int(s*100):02d}{tag}.json"


def merge_runs() -> dict:
    """Fold every per-run JSON into OUT_JSON, keyed by sparsity.

    Each run writes its own file so that separate invocations (rows vs native,
    one sparsity per GPU) cannot overwrite each other's results.
    """
    report = {}
    for f in sorted(RUN_DIR.glob("eval_loupe_*.json")):
        run = json.loads(f.read_text(encoding="utf-8"))
        entry = report.setdefault(run["sparsity_key"], {})
        entry.update(run["entries"])
        entry.setdefault("_runs", {})[run["name"]] = run["meta"]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({**report, "_meta": {
        "stage": "LOUPE baseline", "runs_dir": str(RUN_DIR),
        "port": "itw/loupe.py, validated against upstream TF in tests/test_loupe_port.py",
        "caveats": "LOUPE trained on real magnitude images (Hermitian k-space) with a "
                   "jointly trained U-Net; masks made budget-exact by top-k; no ACS lock.",
    }}, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="LOUPE baseline under our protocol")
    ap.add_argument("--sparsities", type=float, nargs="*", default=[0.25, 0.40, 0.50, 0.75])
    ap.add_argument("--mask", choices=["rows", "native", "both"], default="rows")
    ap.add_argument("--epochs", type=int, default=1000, help="hard cap")
    ap.add_argument("--patience", type=int, default=20,
                    help="stop after this many consecutive low-churn epochs (0 = off)")
    ap.add_argument("--churn-tol", type=float, default=0.02,
                    help="churn counted as settled if <= this fraction of the row budget")
    ap.add_argument("--min-epochs", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--filt", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="")
    ap.add_argument("--merge-only", action="store_true",
                    help="just rebuild OUT_JSON from the per-run files")
    args = ap.parse_args()

    if args.merge_only:
        merge_runs()
        print(f"wrote {OUT_JSON}")
        return

    dev = default_device()
    cfg = FastMRIConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
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
    judge = "unetB" if recon_b else "unetA"
    hc = {s: torch.load(f"models_hill_climb/hillclimb_s{int(s*100):02d}.pth",
                        map_location=dev, weights_only=False)["mask"].to(dev)
          for s in args.sparsities
          if Path(f"models_hill_climb/hillclimb_s{int(s*100):02d}.pth").is_file()}

    modes = ["rows", "native"] if args.mask == "both" else [args.mask]
    for s in args.sparsities:
        # Baselines are deterministic given seeds; score them once per sparsity.
        baselines = {"acs_vd_gaussian": _score_stochastic(
            "acs_vd_gaussian", s, val, recons, dev, args.seeds)}
        if s in hc:
            baselines["ours_hillclimb"] = _score(hc[s], val, recons, dev)
        for mode in modes:
            name = f"loupe_{mode}{args.tag}"
            stem = f"loupe_{mode}_s{int(s*100):02d}{args.tag}"
            print(f"\n=== {name} s={s:.2f} ===")
            to_mask = budget_exact_rows if mode == "rows" else budget_exact_2d
            # Monitoring uses zf and unetA only: the judge stays held out.
            monitor = lambda m, s=s, f=to_mask: {
                f"val_{k}": v for k, v in _score(
                    f(rescale_prob_map(m.prob(), s), s, dev), val,
                    {"zf": None, "unetA": recon_a}, dev).items() if k.endswith("nmse")}
            seed_everything(0)
            model, hist = train_loupe(
                imgs, s, mode == "rows", dev, args.epochs, args.filt, args.batch,
                patience=args.patience, min_epochs=args.min_epochs, monitor=monitor,
                eval_every=args.eval_every, ckpt=OUT_DIR / f"ckpt_{stem}.pt",
                churn_tol=args.churn_tol)
            tail = [h["mask_churn"] for h in hist[-20:]]
            print(f"  mask churn over last 20 epochs: {tail}")
            p = rescale_prob_map(model.prob(), s)
            torch.save({"prob": p.detach().cpu(), "sparsity": s, "mode": mode,
                        "history": hist}, OUT_DIR / f"{stem}.pth")
            res = _score(to_mask(p, s, dev), val, recons, dev)
            res.update(final_mae=hist[-1]["mae"], mask_churn_last20=tail)
            steps = len(hist) * -(-len(imgs) // args.batch)
            meta = {"epochs_run": len(hist), "epoch_cap": args.epochs,
                    "patience": args.patience, "churn_tol": args.churn_tol,
                    "converged": len(tail) >= 20
                                 and max(tail) <= int(args.churn_tol * int(s * H)),
                    "optimizer_steps": steps, "filt": args.filt, "batch": args.batch,
                    "n_train": len(imgs), "n_val": val.n, "judge": judge, "history": hist}
            _run_json(mode, s, args.tag).write_text(json.dumps({
                "name": name, "sparsity_key": f"s={s:.2f}",
                "entries": {name: res, **baselines}, "meta": meta}, indent=2),
                encoding="utf-8")

            entry = {name: res, **baselines}
            print(f"\n--- s={s:.2f} ranked by {judge} ---")
            for k in sorted(entry, key=lambda n: entry[n][f"{judge}_nmse"]):
                sd = entry[k].get(f"{judge}_nmse_std")
                print(f"  {k:>22} {entry[k][f'{judge}_nmse']:.5f}"
                      + (f" +-{sd:.5f}" if sd else ""))

    merge_runs()
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
