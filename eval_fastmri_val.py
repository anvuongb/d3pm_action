"""Stage B2: held-out re-decide of A2 vs P3 on singlecoil_val. Never trains.

Load-only. Uses the B1 protocol: val split (deterministic, all 199 slices),
budget-exact top-k as the primary learned mask, multi-seed paired reporting.
Writes only models_mask_gen_fastmri_nested/eval_val/.

NMSE / SSIM / PSNR are cross-comparable between the two policies (pure recon).
H_coarse is NOT: each policy is scored under the coarse prior it was trained
against (A2 = 50ep k-space coarse, P3 = 100ep), so cross-policy H_c differences
mix policy quality with prior identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from itw.configs import FastMRIConfig, default_device
from itw.eval import (
    evaluate_fastmri_baselines_seeded,
    fastmri_loader_factory,
    save_eval_report,
    sparsity_key,
)
from itw.train import build_dataloader, build_mask_model, load_d3pm_coarse

PARENT = Path("models_mask_gen_fastmri_nested")
A2_DIR = PARENT / "acs_lock_10ep"
P3_DIR = PARENT / "protocol_50ep_acs_lock"
OLD_COARSE = Path("models_d3pm_fastmri_coarse_kspace/model_absorb_cosine_final.pth")
NEW_COARSE = Path("models_d3pm_fastmri_coarse_kspace_100ep/model_absorb_cosine_final.pth")
OUT_DIR = PARENT / "eval_val"

B2_S = (0.10, 0.25, 0.40, 0.50, 0.75)
B2_SEEDS = (0, 1, 2, 3, 4)
B2_BASELINES = (
    "random",
    "equispaced",
    "vd_gaussian",
    "acs_random",
    "acs_vd_gaussian",
    "acs_equispaced",
)

POLICIES = {
    "A2_10ep": {"dir": A2_DIR, "coarse": OLD_COARSE, "init": "warm_start_parent, 10ep, s in [0.10,0.40]"},
    "P3_50ep": {"dir": P3_DIR, "coarse": NEW_COARSE, "init": "from_scratch, 50ep, s in [0.10,0.75]"},
}

PROTECTED = (
    PARENT / "mask_gen_fastmri_final.pth",
    PARENT / "coarse_survival_table.pt",
    A2_DIR / "mask_gen_fastmri_final.pth",
    A2_DIR / "coarse_survival_table.pt",
    A2_DIR / "eval" / "eval_a2.json",
    P3_DIR / "mask_gen_fastmri_final.pth",
    P3_DIR / "coarse_survival_table.pt",
    P3_DIR / "fine_survival_table.pt",
    P3_DIR / "train_status.json",
    P3_DIR / "eval" / "eval_p4.json",
    OLD_COARSE,
    NEW_COARSE,
)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(path: Path) -> tuple[str, float, int]:
    st = path.stat()
    return _md5(path), st.st_mtime, st.st_size


def _snapshot() -> dict[str, tuple[str, float, int]]:
    snap = {}
    for p in PROTECTED:
        if not p.is_file():
            raise SystemExit(f"missing protected artifact {p}")
        snap[str(p)] = _fingerprint(p)
    return snap


def _assert_unchanged(before: dict) -> None:
    for key, prev in before.items():
        if _fingerprint(Path(key)) != prev:
            raise SystemExit(f"protected artifact changed: {key}")


def _make_cfg(save_dir: Path, coarse: Path, split: str = "val") -> FastMRIConfig:
    # The train-split control must use the *same* sampler semantics as val
    # (sequential, drop_last=False, full coverage) or the two are not
    # comparable. data_split="val" selects those semantics; pointing val_root
    # at the train dir is what makes it walk the train files.
    val_root = FastMRIConfig().data_root if split == "train" else None
    return FastMRIConfig(
        device=default_device(),
        mask_objective="nested_d3pm",
        batch_size=8,
        # forkserver flakes when loaders are rebuilt per seed (py3.14); val is
        # small enough that in-process loading is not the bottleneck.
        num_workers=0,
        policy_input="scout_image",
        acs_lock=True,
        scout_size=32,
        image_size=300,
        entropy_alpha=1.0,
        entropy_beta=0.0,
        recon_loss_weight=1.0,
        d3pm_coarse_checkpoint=str(coarse),
        data_split="val",
        **({"val_root": val_root} if val_root else {}),
        save_dir=str(save_dir),
    )


def _load_policy(cfg: FastMRIConfig, ckpt: Path):
    model = build_mask_model(cfg)
    state = torch.load(ckpt, map_location=cfg.device, weights_only=False)
    model.load_state_dict(state["mask_generator"])
    model.eval()
    return model


def _run_policy(name: str, spec: dict, split: str = "val") -> dict:
    save_dir = spec["dir"]
    cfg = _make_cfg(save_dir, spec["coarse"], split=split)
    ckpt = save_dir / "mask_gen_fastmri_final.pth"
    model = _load_policy(cfg, ckpt)
    d3pm_coarse = load_d3pm_coarse(cfg)
    coarse_surv = torch.load(
        save_dir / "coarse_survival_table.pt", map_location=cfg.device, weights_only=False
    )
    fine_surv_path = save_dir / "fine_survival_table.pt"
    if not fine_surv_path.is_file():
        fine_surv_path = PARENT / "fine_survival_table.pt"
    fine_surv = torch.load(fine_surv_path, map_location=cfg.device, weights_only=False)

    n_slices = len(build_dataloader(cfg).dataset)
    print(f"\n=== {name} === ckpt {ckpt} (md5 {_md5(ckpt)[:8]}...)")
    print(f"    coarse {spec['coarse']}  survival {save_dir/'coarse_survival_table.pt'}")
    print(f"    {len(B2_SEEDS)} seeds x {split}({n_slices} slices) at s={B2_S}")

    report = evaluate_fastmri_baselines_seeded(
        model,
        None,
        d3pm_coarse,
        cfg,
        fastmri_loader_factory(cfg),
        fine_surv,
        coarse_surv,
        sparsities=B2_S,
        baselines=B2_BASELINES,
        seeds=B2_SEEDS,
        learned_mask="topk",
        include_learned_gumbel=True,
        reference="learned",
    )
    report["_policy"] = {
        "ckpt": str(ckpt),
        "ckpt_md5": _md5(ckpt),
        "coarse_ckpt": str(spec["coarse"]),
        "init": spec["init"],
    }
    return report


def _print_tables(reports: dict) -> None:
    for name, rep in reports.items():
        print(f"\n### {name} — val, budget-exact top-k, {len(B2_SEEDS)} seeds")
        print(f"{'s':>5} {'method':>18} {'NMSE':>9} {'std':>8} {'dens':>7} "
              f"{'SSIM':>7} {'H_c':>7}  paired dNMSE vs learned")
        for s in B2_S:
            k = sparsity_key(s)
            for m, d in rep[k].items():
                p = d.get("paired_vs_learned", {}).get("nmse")
                ptxt = (
                    f"{p['delta_mean']:+.5f} ({p['n_seeds_method_lower']}/{p['n_seeds']} lower)"
                    if p else "—"
                )
                print(f"{s:>5.2f} {m:>18} {d['nmse']['mean']:>9.5f} {d['nmse']['std']:>8.5f} "
                      f"{d['mean_sparsity']['mean']:>7.4f} {d['ssim']['mean']:>7.4f} "
                      f"{d['h_coarse']['mean']:>7.4f}  {ptxt}")
            print()


def _head_to_head(reports: dict) -> dict:
    """A2 vs P3 on recon only. top-k on a deterministic val pass => exact."""
    a2, p3 = reports["A2_10ep"], reports["P3_50ep"]
    out = {}
    print("\n### Head-to-head (learned, budget-exact; val pass is deterministic)")
    print(f"{'s':>5} {'A2 NMSE':>10} {'P3 NMSE':>10} {'P3-A2':>10} {'A2 SSIM':>9} {'P3 SSIM':>9}  winner")
    for s in B2_S:
        k = sparsity_key(s)
        na, np_ = a2[k]["learned"]["nmse"]["mean"], p3[k]["learned"]["nmse"]["mean"]
        sa, sp = a2[k]["learned"]["ssim"]["mean"], p3[k]["learned"]["ssim"]["mean"]
        d = np_ - na
        win = "tie" if abs(d) < 1e-6 else ("P3" if d < 0 else "A2")
        out[k] = {"a2_nmse": na, "p3_nmse": np_, "delta_p3_minus_a2": d,
                  "a2_ssim": sa, "p3_ssim": sp, "winner": win}
        print(f"{s:>5.2f} {na:>10.5f} {np_:>10.5f} {d:>+10.5f} {sa:>9.4f} {sp:>9.4f}  {win}")
    return out


def _best_baseline(reports: dict) -> dict:
    """Does the learned policy still beat the best ACS heuristic on val?"""
    out = {}
    print("\n### Learned vs best ACS heuristic (budget-matched)")
    print(f"{'policy':>9} {'s':>5} {'learned':>9} {'best heuristic':>22} {'NMSE':>9} {'gap':>9}  verdict")
    acs = ("acs_random", "acs_vd_gaussian", "acs_equispaced")
    for name, rep in reports.items():
        out[name] = {}
        for s in B2_S:
            k = sparsity_key(s)
            ln = rep[k]["learned"]["nmse"]["mean"]
            best = min(acs, key=lambda m: rep[k][m]["nmse"]["mean"])
            bn = rep[k][best]["nmse"]["mean"]
            gap = ln - bn
            verdict = "learned wins" if gap < 0 else "heuristic wins"
            if abs(gap) < 1e-6:
                verdict = "tie"
            out[name][k] = {"learned": ln, "best_heuristic": best,
                            "heuristic_nmse": bn, "gap": gap, "verdict": verdict}
            print(f"{name:>9} {s:>5.2f} {ln:>9.5f} {best:>22} {bn:>9.5f} {gap:>+9.5f}  {verdict}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B2 held-out eval")
    parser.add_argument(
        "--split",
        choices=("val", "train"),
        default="val",
        help="'train' is the B2 control: identical protocol over the train "
             "files, to separate overfitting from a density artifact.",
    )
    args = parser.parse_args()
    split = args.split

    before = _snapshot()
    print(f"B2: eval on {split}, seeds={B2_SEEDS}, s={B2_S}")
    print(f"mode=eval-only  protected={len(PROTECTED)} artifacts snapshotted")

    reports = {
        name: _run_policy(name, spec, split=split) for name, spec in POLICIES.items()
    }
    _print_tables(reports)
    h2h = _head_to_head(reports)
    vs_base = _best_baseline(reports)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / ("eval_b2.json" if split == "val" else "eval_b2_train.json")
    save_eval_report(
        {
            **reports,
            "_head_to_head": h2h,
            "_learned_vs_best_heuristic": vs_base,
            "_meta": {
                "stage": "B2" if split == "val" else "B2-control",
                "split": split,
                "seeds": list(B2_SEEDS),
                "sparsities": list(B2_S),
                "baselines": list(B2_BASELINES),
                "learned_mask": "topk (budget-exact, deterministic)",
                "note": (
                    "NMSE/SSIM/PSNR cross-comparable between policies; H_coarse "
                    "is not (each policy scored under its own coarse prior)."
                ),
            },
        },
        json_path,
    )
    print(f"\nwrote {json_path}")
    _assert_unchanged(before)
    print("STAGE B2 DONE (protected artifacts unchanged)")


if __name__ == "__main__":
    main()
