"""Stage B5b: is the gap to variable density real, or this optimiser's ceiling?

B5 optimised a 300-parameter profile by STE gradient and landed 1.5-2.0% short
of ``acs_vd_gaussian``, but that optimiser oscillated 15% between epochs and
reverted to its initialisation at half the sparsity grid -- too weak to
conclude the headroom is absent.

This attacks the discrete object directly: greedy row-swap hill climbing on the
hard mask, gradient only to *propose* swaps, exact evaluation to accept them.
No STE, no relaxation, no budget confound.

It also runs the control B5 was missing. ``acs_vd_gaussian`` draws a fresh mask
per slice, so a fixed learned mask must beat an average over random draws. If a
single fixed VD draw does as well as the per-slice random version, the target is
a fixed mask and the comparison is fair; if per-slice randomisation is what
carries it, no fixed profile can match it and B5's negative means something
entirely different.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from itw.configs import FastMRIConfig, default_device, seed_everything
from itw.discrete import nmse
from itw.eval import baseline_row_mask_batch, save_eval_report
from itw.masks import acs_bounds
from itw.profile import StaticProfile
from itw.recon import reconstruct
from train_fastmri_static_profile import (
    CachedBatches, _load_recon, _oracle_mask, RECON_A, RECON_B, H, ACS)
from itw.profile import energy_init

PARENT = Path("models_mask_gen_fastmri_nested")
OUT_DIR = Path("models_hill_climb")
OUT_JSON = PARENT / "eval_val" / "eval_b5b_hillclimb.json"
PROTECTED = (RECON_A, RECON_B, PARENT / "eval_val" / "eval_b5_profile.json")


def _fp(p: Path):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    st = p.stat()
    return h.hexdigest(), st.st_mtime, st.st_size


@torch.no_grad()
def score(mask: torch.Tensor, batches, recon, dev) -> float:
    """Mean NMSE of one fixed mask over ``batches``."""
    tot, n = 0.0, 0
    for k, t in batches:
        k, t = k.to(dev), t.to(dev)
        b = k.shape[0]
        tot += float(nmse(reconstruct(recon, k, mask.expand(b, -1, -1, -1)), t)) * b
        n += b
    return tot / n


def mask_grad(mask: torch.Tensor, batches, recon, dev) -> torch.Tensor:
    """dL/dmask at the current hard mask, summed over ``batches``.

    Only used to rank which swaps are worth evaluating exactly -- a first-order
    estimate is fine for proposals and wrong often enough that accepting on it
    directly would be unsound.
    """
    m = mask.clone().detach().requires_grad_(True)
    for k, t in batches:
        k, t = k.to(dev), t.to(dev)
        b = k.shape[0]
        nmse(reconstruct(recon, k, m.expand(b, -1, -1, -1)), t).backward()
    return m.grad[0, 0, :, 0].detach()


def hill_climb(mask, prop_batches, eval_batches, recon, dev, acs, rounds, n_add,
               n_drop, log=print):
    """Greedy swap: propose by gradient, accept only a verified improvement."""
    cur = mask.clone()
    best = score(cur, eval_batches, recon, dev)
    log(f"    start {best:.6f}")
    history = [{"round": -1, "nmse": best, "swap": None}]
    for r in range(rounds):
        g = mask_grad(cur, prop_batches, recon, dev)
        rows = cur[0, 0, :, 0]
        inside = (rows > 0.5) & ~acs          # ACS rows are never swappable
        outside = rows < 0.5
        # Dropping row i changes L by about -g_i, so drop the largest g_i.
        # Adding row j changes L by about +g_j, so add the smallest g_j.
        drop = torch.topk(torch.where(inside, g, torch.full_like(g, -1e30)),
                          min(n_drop, int(inside.sum()))).indices
        add = torch.topk(torch.where(outside, -g, torch.full_like(g, -1e30)),
                         min(n_add, int(outside.sum()))).indices
        cand, best_c = None, best
        for i in drop:
            for j in add:
                trial = cur.clone()
                trial[0, 0, i, 0] = 0.0
                trial[0, 0, j, 0] = 1.0
                v = score(trial, eval_batches, recon, dev)
                if v < best_c:
                    best_c, cand = v, (int(i), int(j), trial)
        if cand is None:
            log(f"    round {r}: no improving swap among {len(drop)}x{len(add)} "
                f"-> local optimum at {best:.6f}")
            break
        i, j, cur = cand
        gain = 100 * (best - best_c) / best
        best = best_c
        history.append({"round": r, "nmse": best, "swap": [i, j]})
        log(f"    round {r}: drop {i:3d} add {j:3d} -> {best:.6f} ({gain:+.3f}%)")
    return cur, best, history


def main() -> None:
    ap = argparse.ArgumentParser(description="B5b discrete row-swap hill climb")
    ap.add_argument("--sparsities", type=float, nargs="*", default=[0.25, 0.40, 0.50, 0.75])
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--n-add", type=int, default=4)
    ap.add_argument("--n-drop", type=int, default=3)
    ap.add_argument("--prop-batches", type=int, default=40,
                    help="batches used for the gradient proposal only")
    ap.add_argument("--eval-batches", type=int, default=0,
                    help="batches used to accept a swap (0 = full train split)")
    ap.add_argument("--vd-draws", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    before = {str(p): _fp(p) for p in PROTECTED if p.is_file()}
    cfg = FastMRIConfig()
    dev = default_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = CachedBatches(cfg.data_root, 8)
    val = CachedBatches(cfg.val_root, 8)
    pairs = train.pairs()
    eval_b = pairs if args.eval_batches == 0 else pairs[:args.eval_batches]
    prop_b = pairs[:args.prop_batches]
    recon_a = _load_recon(RECON_A, dev)
    recon_b = _load_recon(RECON_B, dev) if RECON_B.is_file() else None
    log_e = energy_init((k for k, _t in pairs), H, dev)
    lo, hi = acs_bounds(H, ACS)
    acs = torch.zeros(H, dtype=torch.bool, device=dev)
    acs[lo:hi] = True
    print(f"B5b hill climb (device={dev}); accept on {len(eval_b)} batches, "
          f"propose on {len(prop_b)}")

    report, masks_out = {}, {}
    for s in args.sparsities:
        print(f"\n=== s={s:.2f} ===")
        # --- control: is VD's strength its density, or the per-slice draw? ---
        fixed_vd = []
        for d in range(args.vd_draws):
            seed_everything(1000 + d)
            m = baseline_row_mask_batch("acs_vd_gaussian", 1, H,
                                        torch.full((1,), s, device=dev), dev,
                                        acs_width=ACS)
            fixed_vd.append((score(m, eval_b, recon_a, dev), m))
        fixed_vd.sort(key=lambda x: x[0])
        vd_best, vd_worst = fixed_vd[0][0], fixed_vd[-1][0]
        vd_mean = sum(v for v, _ in fixed_vd) / len(fixed_vd)
        per_slice = 0.0
        for sd in args.seeds:
            seed_everything(sd)
            tot, n = 0.0, 0
            with torch.no_grad():
                for k, t in eval_b:
                    k, t = k.to(dev), t.to(dev)
                    b = k.shape[0]
                    mk = baseline_row_mask_batch("acs_vd_gaussian", b, H,
                                                 torch.full((b,), s, device=dev),
                                                 dev, acs_width=ACS)
                    tot += float(nmse(reconstruct(recon_a, k, mk), t)) * b
                    n += b
            per_slice += tot / n
        per_slice /= len(args.seeds)
        print(f"  VD control (train): per-slice-random {per_slice:.6f} | "
              f"fixed draw best {vd_best:.6f} mean {vd_mean:.6f} worst {vd_worst:.6f}")
        print(f"    -> a FIXED draw beats per-slice randomisation by "
              f"{100*(per_slice-vd_best)/per_slice:+.2f}% (best) / "
              f"{100*(per_slice-vd_mean)/per_slice:+.2f}% (mean draw); "
              f"positive means fixing the mask helps")

        seeds = {"fixed_vd_best": fixed_vd[0][1], "energy_oracle": _oracle_mask(log_e, s)}
        pf = Path(f"models_static_profile/profile_s{int(s*100):02d}.pth")
        if pf.is_file():
            st = torch.load(pf, map_location=dev, weights_only=False)
            seeds["b5_profile"] = StaticProfile(H, ACS, init=st["theta"]).to(dev).hard_mask(s)
        start = {k: score(m, eval_b, recon_a, dev) for k, m in seeds.items()}
        pick = min(start, key=lambda k: start[k])
        print(f"  seeds: " + ", ".join(f"{k} {v:.6f}" for k, v in start.items())
              + f"  -> climbing from {pick}")

        final, best, hist = hill_climb(seeds[pick], prop_b, eval_b, recon_a, dev,
                                       acs, args.rounds, args.n_add, args.n_drop)
        moved = int((final[0, 0, :, 0] != seeds[pick][0, 0, :, 0]).sum()) // 2
        print(f"  climbed {start[pick]:.6f} -> {best:.6f} "
              f"({100*(start[pick]-best)/start[pick]:+.2f}%, {moved} swaps)")
        masks_out[s] = final
        torch.save({"mask": final.cpu(), "sparsity": s, "seed_from": pick,
                    "train_nmse": best, "history": hist},
                   OUT_DIR / f"hillclimb_s{int(s*100):02d}.pth")
        report[f"s={s:.2f}"] = {
            "vd_control_train": {"per_slice_random": per_slice, "fixed_best": vd_best,
                                 "fixed_mean": vd_mean, "fixed_worst": vd_worst},
            "seed_train_nmse": start, "seed_from": pick,
            "climb_train_nmse": best, "n_swaps": moved, "history": hist,
        }

    # ---- val, under both reconstructors, against the seeded baselines ----
    print("\n" + "=" * 74)
    for s in args.sparsities:
        recons = {"zf": None, "unetA": recon_a, **({"unetB": recon_b} if recon_b else {})}
        vpairs = val.pairs()
        row = {}
        for rname, rmodel in recons.items():
            row[f"hillclimb_{rname}"] = score(masks_out[s], vpairs, rmodel, dev)
            vals = []
            for sd in args.seeds:
                seed_everything(sd)
                tot, n = 0.0, 0
                with torch.no_grad():
                    for k, t in vpairs:
                        k, t = k.to(dev), t.to(dev)
                        b = k.shape[0]
                        mk = baseline_row_mask_batch("acs_vd_gaussian", b, H,
                                                     torch.full((b,), s, device=dev),
                                                     dev, acs_width=ACS)
                        tot += float(nmse(reconstruct(rmodel, k, mk), t)) * b
                        n += b
                vals.append(tot / n)
            mean = sum(vals) / len(vals)
            sd_ = (sum((x - mean) ** 2 for x in vals) / max(len(vals) - 1, 1)) ** 0.5
            row[f"acs_vd_{rname}"] = mean
            row[f"acs_vd_{rname}_std"] = sd_
        report[f"s={s:.2f}"]["val"] = row
        judge = "unetB" if recon_b else "unetA"
        hc, vd, sd_ = row[f"hillclimb_{judge}"], row[f"acs_vd_{judge}"], row[f"acs_vd_{judge}_std"]
        verdict = ("BEATS acs_vd" if hc < vd - 2 * sd_ else
                   "ties acs_vd" if hc < vd + 2 * sd_ else "loses to acs_vd")
        print(f"s={s:.2f} [{judge}] hillclimb {hc:.5f} vs acs_vd {vd:.5f}+-{sd_:.5f} "
              f"({100*(vd-hc)/vd:+.2f}%)  {verdict}")
        report[f"s={s:.2f}"]["verdict"] = verdict

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    save_eval_report({**report, "_meta": {
        "stage": "B5b", "split_search": "train", "split_eval": "val",
        "rounds": args.rounds, "n_add": args.n_add, "n_drop": args.n_drop,
        "recon_in_loop": str(RECON_A), "recon_judge": str(RECON_B),
        "note": "greedy row swap on the hard mask; gradient proposes, exact eval accepts",
    }}, OUT_JSON)
    print(f"\nwrote {OUT_JSON}")
    for k, v in before.items():
        if _fp(Path(k)) != v:
            raise SystemExit(f"protected artifact changed: {k}")
    print("STAGE B5b DONE (protected artifacts unchanged)")


if __name__ == "__main__":
    main()
