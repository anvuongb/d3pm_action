"""Stage B5: optimise a static row profile against a reconstructor.

B3 killed the conditional path (the policy ignores its conditioning input and
its own consensus mask beats it). B4 showed the ranking of masks is
reconstructor-dependent, so there is something real to learn -- but it must be
learned against a reconstructor, not against zero-filled NMSE.

So: 300 parameters, trained on TRAIN against the frozen B4 U-Net, evaluated on
VAL against

  zero-filled        -- does it survive without any prior at all?
  U-Net A            -- the net it was optimised against (in-loop, optimistic)
  U-Net B            -- an independently trained judge (different seed, width,
                        and mask draws) that never took part in the
                        optimisation. This is the honest number: a profile that
                        wins on A but not on B exploited A's quirks.

Baselines: energy oracle (F7), acs_vd_gaussian (the B4 winner at s=0.25), and
the B3 static/adaptive policy masks.

Writes only models_static_profile/ and eval_val/eval_b5_profile.json.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from itw.configs import FastMRIConfig, default_device, seed_everything
from itw.data.fastmri import FastMRIDataset
from itw.discrete import nmse, ssim
from itw.eval import acs_lock_topk_from_logits, baseline_row_mask_batch, save_eval_report
from itw.masks import acs_bounds, magnitude_from_kspace
from itw.profile import StaticProfile, energy_init, train_static_profile
from itw.recon import ReconUNet, reconstruct, train_recon_unet
from itw.train import build_mask_model, mask_generator_cond

PARENT = Path("models_mask_gen_fastmri_nested")
OUT_DIR = Path("models_static_profile")
OUT_JSON = PARENT / "eval_val" / "eval_b5_profile.json"
RECON_A = Path("models_recon_unet/recon_unet.pth")
RECON_B = Path("models_recon_unet/recon_unet_judge.pth")
H, ACS = 300, 32
S_GRID = (0.25, 0.40, 0.50, 0.75)  # s=0.10 is the forced centred block
POLICIES = {
    "P3": PARENT / "protocol_50ep_acs_lock" / "mask_gen_fastmri_final.pth",
    "A2": PARENT / "acs_lock_10ep" / "mask_gen_fastmri_final.pth",
}
PROTECTED = (
    PARENT / "mask_gen_fastmri_final.pth", *POLICIES.values(), RECON_A,
    PARENT / "eval_val" / "eval_b2.json",
    PARENT / "eval_val" / "eval_b3_adaptivity.json",
    PARENT / "eval_val" / "eval_b4_recon.json",
)


def _std(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def _fp(p: Path):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    st = p.stat()
    return h.hexdigest(), st.st_mtime, st.st_size


class CachedBatches:
    """Decoded k-space held in host RAM.

    The profile optimiser makes many passes over the same 973 slices; re-reading
    and re-cropping the h5 files every epoch would dominate the runtime.
    """

    def __init__(self, root: str, batch_size: int = 8, shuffle: bool = False):
        loader = DataLoader(FastMRIDataset(root, scout_size=ACS, target_dim=H),
                            batch_size=batch_size, shuffle=False, num_workers=0)
        self.items = []
        for z, _c, k in loader:
            self.items.append((z, k, magnitude_from_kspace(k)))
        self.shuffle = shuffle
        self.n = sum(z.shape[0] for z, _, _ in self.items)

    def pairs(self):
        return [(k, t) for _z, k, t in self.items]

    def __iter__(self):
        """(z, target, kspace) triples, in train_recon_unet's argument order."""
        order = torch.randperm(len(self.items)).tolist() if self.shuffle \
            else range(len(self.items))
        for i in order:
            z, k, t = self.items[i]
            yield z, t, k


def _cfg() -> FastMRIConfig:
    return FastMRIConfig(
        device=default_device(), batch_size=8, num_workers=0,
        policy_input="scout_image", acs_lock=True, scout_size=ACS,
        image_size=H, entropy_beta=0.0, data_split="val",
    )


def _load_recon(path: Path, dev):
    st = torch.load(path, map_location=dev, weights_only=False)
    m = ReconUNet(base=st.get("base", 32)).to(dev)
    m.load_state_dict(st["recon_unet"])
    m.eval()
    return m


def _oracle_mask(log_energy: torch.Tensor, s: float) -> torch.Tensor:
    lo, hi = acs_bounds(H, ACS)
    sc = log_energy.clone()
    sc[lo:hi] = float("inf")
    m = torch.zeros_like(log_energy)
    m[torch.topk(sc, int(s * H)).indices] = 1.0
    return m.view(1, 1, H, 1)


def _policy_masks(net, cfg, z, kspace, s):
    sp = torch.full((z.shape[0],), s, device=cfg.device)
    return acs_lock_topk_from_logits(net(mask_generator_cond(cfg, z, kspace), sp), sp, ACS)


def _policy_static(net, cfg, train_cache, s) -> torch.Tensor:
    """B3's static_train mask: the policy's own top-k selection frequency."""
    freq = torch.zeros(H, device=cfg.device)
    n = 0
    with torch.no_grad():
        for z, _t, k in train_cache:
            z, k = z.to(cfg.device), k.to(cfg.device)
            freq += _policy_masks(net, cfg, z, k, s)[:, 0, :, 0].sum(0)
            n += z.shape[0]
    lo, hi = acs_bounds(H, ACS)
    sc = freq / n
    sc[lo:hi] = float("inf")
    m = torch.zeros(H, device=cfg.device)
    m[torch.topk(sc, int(s * H)).indices] = 1.0
    return m.view(1, 1, H, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B5 static profile optimisation")
    ap.add_argument("--epochs", type=int, default=25, help="profile optimisation epochs")
    ap.add_argument("--judge-epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4],
                    help="seeds for the stochastic baselines")
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--sparsities", type=float, nargs="*", default=list(S_GRID))
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()

    before = {str(p): _fp(p) for p in PROTECTED if p.is_file()}
    cfg = _cfg()
    dev = cfg.device
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"B5: static profile optimisation (device={dev})")

    train = CachedBatches(cfg.data_root, cfg.batch_size, shuffle=True)
    val = CachedBatches(cfg.val_root, cfg.batch_size)
    print(f"    cached {train.n} train / {val.n} val slices")

    recon_a = _load_recon(RECON_A, dev)
    if not args.skip_judge and not RECON_B.is_file():
        print(f"\ntraining held-out judge reconstructor -> {RECON_B}")
        seed_everything(1)          # different seed, width and mask draws than A
        train_recon_unet(train, device=dev, n_rows=H, acs_width=ACS,
                         n_epochs=args.judge_epochs, base=48,
                         save_path=RECON_B, log_path=OUT_DIR / "judge_train.log")
    recon_b = _load_recon(RECON_B, dev) if RECON_B.is_file() else None
    recons = {"zf": None, "unetA": recon_a, **({"unetB": recon_b} if recon_b else {})}
    print(f"    reconstructors: {list(recons)}")

    log_e = energy_init(  # standardised, so it doubles as a warm start
        (k for k, _t in train.pairs()), H, dev)

    nets = {}
    for name, path in POLICIES.items():
        m = build_mask_model(cfg)
        m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["mask_generator"])
        m.eval()
        nets[name] = m

    pairs = train.pairs()
    profiles, opt_hist = {}, {}
    for s in args.sparsities:
        cands = {}
        for init_name, init in (("energy", log_e.cpu()), ("random", None)):
            seed_everything(0)
            prof, hist = train_static_profile(
                pairs, recon_a, s, dev, n_rows=H, acs_width=ACS,
                n_epochs=args.epochs, lr=args.lr, temperature=args.temperature,
                init=init, log_path=OUT_DIR / f"profile{args.tag}.log",
                desc=f"s={s:.2f} init={init_name}")
            cands[init_name] = (min(h["train_nmse"] for h in hist), prof, hist)
        # Selection is on TRAIN; every number reported below is on val.
        pick = min(cands, key=lambda k: cands[k][0])
        best_nmse, prof, hist = cands[pick]
        print(f"  s={s:.2f}: init={pick} wins (train NMSE {best_nmse:.6f} vs "
              f"{cands['random' if pick == 'energy' else 'energy'][0]:.6f})")
        profiles[s] = prof
        opt_hist[f"s={s:.2f}"] = {
            "init_selected": pick,
            "train_nmse": {k: v[0] for k, v in cands.items()},
            "history": {k: v[2] for k, v in cands.items()},
        }
        torch.save({"theta": prof.theta.detach().cpu(), "sparsity": s,
                    "init": pick, "train_nmse": best_nmse},
                   OUT_DIR / f"profile_s{int(s*100):02d}{args.tag}.pth")

    print("\nbuilding baseline masks...")
    static_pol = {n: {s: _policy_static(nets[n], cfg, train, s) for s in args.sparsities}
                  for n in nets}

    # acs_vd_gaussian and friends draw a fresh mask per sample, so a single
    # draw is not a measurement -- the s=0.50 gap between the best heuristic and
    # the best learned mask is ~0.1%, far inside one draw's spread.
    STOCHASTIC = ("acs_vd_gaussian", "acs_random", "acs_equispaced", "vd_gaussian")
    report = {}
    for s in args.sparsities:
        det = ["b5_profile", "energy_oracle"] + \
              [f"{n}_{v}" for n in nets for v in ("adaptive", "static_train")]
        per_seed = {m: {r: [] for r in recons} for m in STOCHASTIC}
        det_acc = {m: dict.fromkeys([f"{r}_{q}" for r in recons for q in ("nmse", "ssim")]
                                    + ["density"], 0.0) for m in det}
        for si, seed in enumerate(args.seeds):
            seed_everything(seed)
            acc = {m: dict.fromkeys([f"{r}_nmse" for r in recons], 0.0) for m in STOCHASTIC}
            with torch.no_grad():
                for z, target, k in val:
                    z, k, target = z.to(dev), k.to(dev), target.to(dev)
                    b = z.shape[0]
                    sp = torch.full((b,), s, device=dev)
                    masks = {h: baseline_row_mask_batch(h, b, H, sp, dev, acs_width=ACS)
                             for h in STOCHASTIC}
                    if si == 0:
                        masks["b5_profile"] = profiles[s].hard_mask(s).expand(b, -1, -1, -1)
                        masks["energy_oracle"] = _oracle_mask(log_e, s).expand(b, -1, -1, -1)
                        for n, net in nets.items():
                            masks[f"{n}_adaptive"] = _policy_masks(net, cfg, z, k, s)
                            masks[f"{n}_static_train"] = static_pol[n][s].expand(b, -1, -1, -1)
                    for m, mk in masks.items():
                        for rname, rmodel in recons.items():
                            pred = reconstruct(rmodel, k, mk)
                            v = float(nmse(pred, target)) * b
                            if m in STOCHASTIC:
                                acc[m][f"{rname}_nmse"] += v
                            else:
                                det_acc[m][f"{rname}_nmse"] += v
                                det_acc[m][f"{rname}_ssim"] += float(ssim(pred, target)) * b
                        if m in det:
                            det_acc[m]["density"] += float(mk.mean()) * b
            for m in STOCHASTIC:
                for r in recons:
                    per_seed[m][r].append(acc[m][f"{r}_nmse"] / val.n)

        res = {m: {kk: v / val.n for kk, v in d.items()} for m, d in det_acc.items()}
        for m in STOCHASTIC:
            res[m] = {"density": s}
            for r in recons:
                vals = per_seed[m][r]
                res[m][f"{r}_nmse"] = sum(vals) / len(vals)
                res[m][f"{r}_nmse_std"] = _std(vals)
                res[m][f"{r}_nmse_seeds"] = vals
        report[f"s={s:.2f}"] = res

        judge = "unetB_nmse" if recon_b else "unetA_nmse"
        print(f"\n--- s={s:.2f} --- (ranked by held-out judge {judge}; "
              f"+/- is over {len(args.seeds)} seeds, stochastic baselines only)")
        print(f"{'method':>18}" + "".join(f"{r + ' NMSE':>22}" for r in recons))
        for m in sorted(res, key=lambda m: res[m][judge]):
            cells = ""
            for r in recons:
                sd = res[m].get(f"{r}_nmse_std")
                cells += (f"{res[m][f'{r}_nmse']:>15.5f}"
                          + (f" +-{sd:.5f}" if sd is not None else " " * 8))
            print(f"{m:>18}{cells}" + ("  <<<" if m == "b5_profile" else ""))

    print("\n" + "=" * 74)
    verdicts = {}
    for s in args.sparsities:
        res = report[f"s={s:.2f}"]
        row = {}
        for rname in recons:
            key = f"{rname}_nmse"
            rank = sorted(res, key=lambda m: res[m][key])
            others = [m for m in rank if m != "b5_profile"]
            best_other = others[0]
            gain = 100 * (res[best_other][key] - res["b5_profile"][key]) / res[best_other][key]
            sd = res[best_other].get(f"{key}_std")
            row[rname] = {
                "rank": rank.index("b5_profile") + 1, "of": len(rank),
                "best_other": best_other,
                "gain_vs_best_other": gain,
                # A win inside the rival's own seed spread is not a win.
                "beyond_seed_noise": None if sd is None else
                abs(res[best_other][key] - res["b5_profile"][key]) > 2 * sd,
            }
        verdicts[f"s={s:.2f}"] = row
        for rname, d in row.items():
            sig = d["beyond_seed_noise"]
            tag = "" if sig is None else ("  (beyond 2 seed sd)" if sig
                                          else "  (INSIDE seed noise)")
            print(f"s={s:.2f} [{rname:>5}] b5_profile rank {d['rank']}/{d['of']}  "
                  f"vs best other ({d['best_other']}) {d['gain_vs_best_other']:+.1f}%{tag}")

    judge = "unetB" if recon_b else "unetA"
    wins = [s for s in args.sparsities if verdicts[f"s={s:.2f}"][judge]["rank"] == 1]
    survives_zf = [s for s in args.sparsities if verdicts[f"s={s:.2f}"]["zf"]["rank"] == 1]
    if len(wins) == len(args.sparsities):
        verdict = (f"static profile is best under the held-out judge at every s "
                   f"-> B5 succeeds; headline claim is a learned static profile")
    elif wins:
        verdict = (f"static profile wins under the held-out judge at {wins} only "
                   f"(loses at {[s for s in args.sparsities if s not in wins]})")
    else:
        verdict = ("static profile never wins under the held-out judge -> the "
                   "gains against U-Net A were in-loop overfitting")
    print(f"\nzero-filled wins at: {survives_zf}")
    print(f"VERDICT: {verdict}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    save_eval_report({**report, "_verdicts": verdicts, "_verdict": verdict,
                      "_optimisation": opt_hist,
                      "_meta": {"stage": "B5", "split": "val", "n_val_slices": val.n,
                                "sparsities": list(args.sparsities),
                                "profile_params": H, "epochs": args.epochs,
                                "lr": args.lr, "temperature": args.temperature,
                                "baseline_seeds": list(args.seeds),
                                "recon_in_loop": str(RECON_A),
                                "recon_judge": str(RECON_B) if recon_b else None,
                                "note": "profile trained on train vs U-Net A; "
                                        "judge U-Net B never used in optimisation"}},
                     Path(str(OUT_JSON).replace(".json", f"{args.tag}.json")))
    print(f"\nwrote {OUT_JSON}")
    for k, v in before.items():
        if _fp(Path(k)) != v:
            raise SystemExit(f"protected artifact changed: {k}")
    print("STAGE B5 DONE (protected artifacts unchanged)")


if __name__ == "__main__":
    main()
