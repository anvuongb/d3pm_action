"""Stage B4: does a real reconstructor create headroom for mask learning?

Trains a small U-Net reconstructor on the TRAIN split under heuristic masks
only (never the learned masks), then re-ranks every mask method on VAL under
both zero-filled and U-Net reconstruction.

Reads the policy checkpoints; writes only models_recon_unet/ and
models_mask_gen_fastmri_nested/eval_val/eval_b4_recon.json.

Decision: if the ranking moves (learned or any non-energy mask overtakes the
energy oracle) there is headroom and the fine-D3PM-as-reconstructor pivot is
justified. If the energy oracle still wins, this task has no headroom for mask
learning and the policy should not be tuned further.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from itw.configs import FastMRIConfig, default_device, seed_everything
from itw.data.fastmri import FastMRIDataset
from itw.discrete import nmse, ssim
from itw.eval import acs_lock_topk_from_logits, baseline_row_mask_batch, save_eval_report
from itw.masks import acs_bounds, magnitude_from_kspace
from itw.recon import ReconUNet, reconstruct, train_recon_unet
from itw.train import build_mask_model, mask_generator_cond

PARENT = Path("models_mask_gen_fastmri_nested")
RECON_DIR = Path("models_recon_unet")
OUT_JSON = PARENT / "eval_val" / "eval_b4_recon.json"
H, ACS = 300, 32
S_GRID = (0.10, 0.25, 0.40, 0.50, 0.75)
HEURISTICS = ("acs_vd_gaussian", "acs_random", "acs_equispaced", "vd_gaussian")
POLICIES = {
    "learned_P3": PARENT / "protocol_50ep_acs_lock" / "mask_gen_fastmri_final.pth",
    "learned_A2": PARENT / "acs_lock_10ep" / "mask_gen_fastmri_final.pth",
}
PROTECTED = (
    PARENT / "mask_gen_fastmri_final.pth",
    *POLICIES.values(),
    PARENT / "eval_val" / "eval_b2.json",
    PARENT / "eval_val" / "eval_b2_train.json",
)


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _fp(p: Path):
    st = p.stat()
    return _md5(p), st.st_mtime, st.st_size


def _snapshot():
    return {str(p): _fp(p) for p in PROTECTED if p.is_file()}


def _assert_unchanged(before):
    for k, v in before.items():
        if _fp(Path(k)) != v:
            raise SystemExit(f"protected artifact changed: {k}")


def _cfg(split: str) -> FastMRIConfig:
    return FastMRIConfig(
        device=default_device(), batch_size=8, num_workers=0,
        policy_input="scout_image", acs_lock=True, scout_size=ACS,
        image_size=H, entropy_beta=0.0, data_split=split,
    )


def _loader(cfg: FastMRIConfig, split: str) -> DataLoader:
    root = cfg.val_root if split == "val" else cfg.data_root
    return DataLoader(
        FastMRIDataset(root, scout_size=ACS, target_dim=H),
        batch_size=cfg.batch_size, shuffle=(split == "train"), num_workers=0,
    )


def _energy_profile(cfg: FastMRIConfig) -> torch.Tensor:
    """Mean PE-row energy over TRAIN (the F7 oracle), fit once."""
    dev = cfg.device
    energy = torch.zeros(H, device=dev)
    n = 0
    with torch.no_grad():
        for _z, _c, k in _loader(cfg, "train"):
            k = k.to(dev)
            energy += (k.abs() ** 2).sum(dim=(1, 3)).sum(0)
            n += k.shape[0]
    return (energy / n).clamp_min(1e-20).log()


def _oracle_mask(log_energy: torch.Tensor, s: float, dev) -> torch.Tensor:
    lo, hi = acs_bounds(H, ACS)
    scores = log_energy.clone()
    scores[lo:hi] = float("inf")
    m = torch.zeros(H, device=dev)
    m[torch.topk(scores, int(s * H)).indices] = 1.0
    return m.view(1, 1, H, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B4 reconstructor headroom test")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse models_recon_unet/recon_unet.pth")
    args = ap.parse_args()

    before = _snapshot()
    cfg = _cfg("val")
    dev = cfg.device
    ckpt_path = RECON_DIR / "recon_unet.pth"

    print(f"B4: reconstructor headroom test (device={dev})")
    print(f"    train masks = heuristics only {list(__import__('itw.recon', fromlist=['x']).TRAIN_MASK_FAMILIES)}")

    if args.skip_train and ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model = ReconUNet(base=state.get("base", args.base)).to(dev)
        model.load_state_dict(state["recon_unet"])
        history = state.get("history", [])
        print(f"    loaded {ckpt_path} (epoch {state.get('epoch')})")
    else:
        seed_everything(0)
        model, history = train_recon_unet(
            _loader(cfg, "train"), device=dev, n_rows=H, acs_width=ACS,
            n_epochs=args.epochs, base=args.base,
            save_path=ckpt_path, log_path=RECON_DIR / "train.log",
        )
    model.eval()

    print("\nfitting energy oracle on train...")
    log_energy = _energy_profile(cfg)

    nets = {}
    for name, path in POLICIES.items():
        m = build_mask_model(cfg)
        m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["mask_generator"])
        m.eval()
        nets[name] = m

    val_batches = []
    with torch.no_grad():
        for z, _c, k in _loader(cfg, "val"):
            z, k = z.to(dev), k.to(dev)
            val_batches.append((z, k, magnitude_from_kspace(k)))
    n_val = sum(z.shape[0] for z, _, _ in val_batches)
    print(f"evaluating on {n_val} val slices\n")

    methods = list(POLICIES) + ["energy_oracle"] + list(HEURISTICS)
    report: dict = {}
    for s in S_GRID:
        oracle = _oracle_mask(log_energy, s, dev)
        acc = {m: {"zf_nmse": 0.0, "unet_nmse": 0.0, "zf_ssim": 0.0,
                   "unet_ssim": 0.0, "density": 0.0} for m in methods}
        seed_everything(0)
        with torch.no_grad():
            for z, k, target in val_batches:
                b = z.shape[0]
                sp = torch.full((b,), s, device=dev)
                masks = {}
                for name, net in nets.items():
                    masks[name] = acs_lock_topk_from_logits(
                        net(mask_generator_cond(cfg, z, k), sp), sp, ACS)
                masks["energy_oracle"] = oracle.expand(b, -1, -1, -1)
                for hname in HEURISTICS:
                    masks[hname] = baseline_row_mask_batch(
                        hname, b, H, sp, dev, acs_width=ACS)
                for name, mk in masks.items():
                    zf = reconstruct(None, k, mk)
                    un = reconstruct(model, k, mk)
                    a = acc[name]
                    a["zf_nmse"] += float(nmse(zf, target)) * b
                    a["unet_nmse"] += float(nmse(un, target)) * b
                    a["zf_ssim"] += float(ssim(zf, target)) * b
                    a["unet_ssim"] += float(ssim(un, target)) * b
                    a["density"] += float(mk.mean()) * b
        res = {m: {kk: v / n_val for kk, v in d.items()} for m, d in acc.items()}
        report[f"s={s:.2f}"] = res

        best_zf = min(res, key=lambda m: res[m]["zf_nmse"])
        best_un = min(res, key=lambda m: res[m]["unet_nmse"])
        print(f"--- s={s:.2f} ---   best zero-filled: {best_zf}   best U-Net: {best_un}"
              + ("   <<< RANKING MOVED" if best_zf != best_un else ""))
        print(f"{'method':>18} {'zf NMSE':>9} {'unet NMSE':>10} {'gain':>7} "
              f"{'zf SSIM':>8} {'unet SSIM':>10} {'dens':>7}")
        for m in sorted(res, key=lambda m: res[m]["unet_nmse"]):
            d = res[m]
            gain = 100 * (d["zf_nmse"] - d["unet_nmse"]) / max(d["zf_nmse"], 1e-12)
            print(f"{m:>18} {d['zf_nmse']:>9.5f} {d['unet_nmse']:>10.5f} {gain:>6.1f}% "
                  f"{d['zf_ssim']:>8.4f} {d['unet_ssim']:>10.4f} {d['density']:>7.4f}")
        print()

    moved = {
        k: {
            "best_zero_filled": min(v, key=lambda m: v[m]["zf_nmse"]),
            "best_unet": min(v, key=lambda m: v[m]["unet_nmse"]),
        }
        for k, v in report.items()
    }
    any_moved = any(d["best_zero_filled"] != d["best_unet"] for d in moved.values())

    # Gate: a reconstructor that loses to zero-filled reorders masks by being
    # bad, not by carrying a prior. Without this the "ranking moved" readout is
    # meaningless.
    valid = {}
    for k, v in report.items():
        best = min(v, key=lambda m: v[m]["zf_nmse"])
        valid[k] = v[best]["unet_nmse"] < v[best]["zf_nmse"]
    n_valid = sum(valid.values())

    print("=" * 70)
    print("Reconstructor sanity (U-Net must beat zero-filled on the best mask):")
    for k, ok in valid.items():
        b = min(report[k], key=lambda m: report[k][m]["zf_nmse"])
        d = report[k][b]
        print(f"  {k}: {'PASS' if ok else 'FAIL'}  best-mask zf {d['zf_nmse']:.5f} "
              f"-> unet {d['unet_nmse']:.5f}")
    print("\nRanking, and whether it counts (movement at an INVALID sparsity is "
          "the reconstructor being bad, not a prior):")
    for k, d in moved.items():
        mv = d["best_zero_filled"] != d["best_unet"]
        counts = "counts" if (mv and valid[k]) else ("DISCARD (invalid)" if mv else "-")
        print(f"  {k}: zf -> {d['best_zero_filled']:<16} unet -> {d['best_unet']:<16}"
              f" moved={mv} valid={valid[k]}  {counts}")

    moved_and_valid = [
        k for k, d in moved.items()
        if valid[k] and d["best_zero_filled"] != d["best_unet"]
    ]
    if n_valid == 0:
        verdict = ("INVALID — reconstructor never beat zero-filled. Train it "
                   "longer/differently before reading anything into the ranking.")
    elif moved_and_valid:
        verdict = (f"headroom exists — ranking moved at {moved_and_valid} where the "
                   "reconstructor is valid -> fine-D3PM-as-reconstructor justified")
    elif n_valid < len(valid):
        verdict = (f"INCONCLUSIVE, leaning negative — at the {n_valid}/{len(valid)} "
                   "valid sparsities the ranking did NOT move (energy oracle still "
                   "wins); the rest are invalid. Fix the reconstructor at high s.")
    else:
        verdict = ("no headroom — reconstructor valid everywhere and the ranking "
                   "never moved -> do not tune the policy; change the problem")
    print(f"\nVERDICT: {verdict}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    save_eval_report(
        {**report, "_best": moved,
         "_ranking_moved": any_moved,
         "_recon_beats_zero_filled": valid,
         "_moved_and_valid": moved_and_valid,
         "_verdict": verdict,
         "_recon_history": history,
         "_meta": {"stage": "B4", "split": "val", "n_val_slices": n_val,
                   "sparsities": list(S_GRID), "recon_ckpt": str(ckpt_path),
                   "recon_train_masks": "heuristics only (no learned masks)",
                   "note": "learned masks budget-exact top-k; oracle fit on train."}},
        OUT_JSON)
    print(f"\nwrote {OUT_JSON}")
    _assert_unchanged(before)
    print("STAGE B4 DONE (protected artifacts unchanged)")


if __name__ == "__main__":
    main()
