"""Stage B3: is the row policy instance-adaptive, or one static profile?

Three load-only tests on val, each scored under BOTH zero-filled and the B4
U-Net (adaptivity may only pay off once the reconstructor carries a prior):

  adaptive       per-slice mask from the correct scout Z
  shuffled_cond  per-slice mask from ANOTHER slice's scout Z, applied to the
                 original k-space. No degradation => the conditioning input is
                 not being used.
  static_train   one mask for every slice: the policy's own top-k most
                 frequently selected rows, measured on the TRAIN split. Fixes
                 the in-sample caveat on F6.

Reference: energy_oracle (F7) and acs_vd_gaussian (the B4 winner).

Decides the form of the B5 pivot: if conditioning is unused, optimise a
300-parameter static profile against the reconstructor instead of retraining a
conditional policy that does not condition.
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
from itw.recon import ReconUNet, reconstruct
from itw.train import build_mask_model, mask_generator_cond

PARENT = Path("models_mask_gen_fastmri_nested")
RECON_CKPT = Path("models_recon_unet/recon_unet.pth")
OUT_JSON = PARENT / "eval_val" / "eval_b3_adaptivity.json"
H, ACS = 300, 32
S_GRID = (0.25, 0.40, 0.50, 0.75)  # s=0.10 is the forced centred block
POLICIES = {
    "P3": PARENT / "protocol_50ep_acs_lock" / "mask_gen_fastmri_final.pth",
    "A2": PARENT / "acs_lock_10ep" / "mask_gen_fastmri_final.pth",
}
PROTECTED = (PARENT / "mask_gen_fastmri_final.pth", *POLICIES.values())


def _fp(p: Path):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    st = p.stat()
    return h.hexdigest(), st.st_mtime, st.st_size


def _cfg() -> FastMRIConfig:
    return FastMRIConfig(
        device=default_device(), batch_size=8, num_workers=0,
        policy_input="scout_image", acs_lock=True, scout_size=ACS,
        image_size=H, entropy_beta=0.0, data_split="val",
    )


def _loader(cfg, split):
    root = cfg.val_root if split == "val" else cfg.data_root
    return DataLoader(FastMRIDataset(root, scout_size=ACS, target_dim=H),
                      batch_size=cfg.batch_size, shuffle=False, num_workers=0)


def _policy_masks(net, cfg, z, kspace, s):
    sp = torch.full((z.shape[0],), s, device=cfg.device)
    return acs_lock_topk_from_logits(net(mask_generator_cond(cfg, z, kspace), sp), sp, ACS)


def _static_profile_from_train(net, cfg, s) -> torch.Tensor:
    """Policy's own consensus mask, measured on TRAIN, applied to val."""
    freq = torch.zeros(H, device=cfg.device)
    n = 0
    with torch.no_grad():
        for z, _c, k in _loader(cfg, "train"):
            z, k = z.to(cfg.device), k.to(cfg.device)
            m = _policy_masks(net, cfg, z, k, s)
            freq += m[:, 0, :, 0].sum(0)
            n += z.shape[0]
    freq /= n
    lo, hi = acs_bounds(H, ACS)
    scores = freq.clone()
    scores[lo:hi] = float("inf")            # ACS always kept, as in the masks
    out = torch.zeros(H, device=cfg.device)
    out[torch.topk(scores, int(s * H)).indices] = 1.0
    return out.view(1, 1, H, 1)


def _energy_oracle(cfg, s) -> torch.Tensor:
    energy = torch.zeros(H, device=cfg.device)
    n = 0
    with torch.no_grad():
        for _z, _c, k in _loader(cfg, "train"):
            k = k.to(cfg.device)
            energy += (k.abs() ** 2).sum(dim=(1, 3)).sum(0)
            n += k.shape[0]
    lo, hi = acs_bounds(H, ACS)
    scores = (energy / n).clamp_min(1e-20).log()
    scores[lo:hi] = float("inf")
    m = torch.zeros(H, device=cfg.device)
    m[torch.topk(scores, int(s * H)).indices] = 1.0
    return m.view(1, 1, H, 1)


def main() -> None:
    argparse.ArgumentParser(description="Stage B3 adaptivity diagnostic").parse_args()
    before = {str(p): _fp(p) for p in PROTECTED}
    cfg = _cfg()
    dev = cfg.device

    recon = None
    if RECON_CKPT.is_file():
        st = torch.load(RECON_CKPT, map_location=dev, weights_only=False)
        recon = ReconUNet(base=st.get("base", 32)).to(dev)
        recon.load_state_dict(st["recon_unet"])
        recon.eval()
        print(f"loaded B4 reconstructor {RECON_CKPT}")

    nets = {}
    for name, path in POLICIES.items():
        m = build_mask_model(cfg)
        m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["mask_generator"])
        m.eval()
        nets[name] = m

    val = []
    with torch.no_grad():
        for z, _c, k in _loader(cfg, "val"):
            z, k = z.to(dev), k.to(dev)
            val.append((z, k, magnitude_from_kspace(k)))
    n_val = sum(z.shape[0] for z, _, _ in val)

    # Global derangement of scout Z across the whole val set.
    all_z = torch.cat([z for z, _, _ in val])
    g = torch.Generator().manual_seed(0)
    perm = (torch.randperm(n_val, generator=g) + 1) % n_val   # no fixed points
    shuffled_z = all_z[perm]
    print(f"val {n_val} slices; scout Z deranged (0 fixed points)\n")

    print("fitting energy oracle + static profiles on train...")
    oracle = {s: _energy_oracle(cfg, s) for s in S_GRID}
    static = {name: {s: _static_profile_from_train(nets[name], cfg, s) for s in S_GRID}
              for name in nets}

    report = {}
    for s in S_GRID:
        variants = ["adaptive", "shuffled_cond", "static_train"]
        methods = [f"{p}_{v}" for p in nets for v in variants] + \
                  ["energy_oracle", "acs_vd_gaussian"]
        acc = {m: {"zf_nmse": 0.0, "unet_nmse": 0.0, "zf_ssim": 0.0,
                   "unet_ssim": 0.0, "density": 0.0} for m in methods}
        seed_everything(0)
        off = 0
        with torch.no_grad():
            for z, k, target in val:
                b = z.shape[0]
                z_sh = shuffled_z[off:off + b]
                off += b
                sp = torch.full((b,), s, device=dev)
                masks = {}
                for name, net in nets.items():
                    masks[f"{name}_adaptive"] = _policy_masks(net, cfg, z, k, s)
                    masks[f"{name}_shuffled_cond"] = _policy_masks(net, cfg, z_sh, k, s)
                    masks[f"{name}_static_train"] = static[name][s].expand(b, -1, -1, -1)
                masks["energy_oracle"] = oracle[s].expand(b, -1, -1, -1)
                masks["acs_vd_gaussian"] = baseline_row_mask_batch(
                    "acs_vd_gaussian", b, H, sp, dev, acs_width=ACS)
                for m, mk in masks.items():
                    zf = reconstruct(None, k, mk)
                    a = acc[m]
                    a["zf_nmse"] += float(nmse(zf, target)) * b
                    a["zf_ssim"] += float(ssim(zf, target)) * b
                    a["density"] += float(mk.mean()) * b
                    if recon is not None:
                        un = reconstruct(recon, k, mk)
                        a["unet_nmse"] += float(nmse(un, target)) * b
                        a["unet_ssim"] += float(ssim(un, target)) * b
        res = {m: {kk: v / n_val for kk, v in d.items()} for m, d in acc.items()}
        report[f"s={s:.2f}"] = res

        print(f"\n--- s={s:.2f} ---")
        print(f"{'method':>20} {'zf NMSE':>9} {'unet NMSE':>10} {'dens':>7}")
        for m in sorted(res, key=lambda m: res[m]["unet_nmse"] or res[m]["zf_nmse"]):
            d = res[m]
            print(f"{m:>20} {d['zf_nmse']:>9.5f} {d['unet_nmse']:>10.5f} {d['density']:>7.4f}")
        for name in nets:
            a, sh, st_ = (res[f"{name}_{v}"] for v in
                          ("adaptive", "shuffled_cond", "static_train"))
            for tag, key in (("zero-filled", "zf_nmse"), ("U-Net", "unet_nmse")):
                if recon is None and tag == "U-Net":
                    continue
                d_sh = 100 * (sh[key] - a[key]) / a[key]
                d_st = 100 * (st_[key] - a[key]) / a[key]
                print(f"  [{name} {tag}] shuffled cond {d_sh:+.1f}% vs adaptive | "
                      f"static-train {d_st:+.1f}% vs adaptive")

    print("\n" + "=" * 70)
    for name in nets:
        for tag, key in (("zero-filled", "zf_nmse"), ("U-Net", "unet_nmse")):
            if recon is None and tag == "U-Net":
                continue
            sh = [100 * (report[f"s={s:.2f}"][f"{name}_shuffled_cond"][key]
                         - report[f"s={s:.2f}"][f"{name}_adaptive"][key])
                  / report[f"s={s:.2f}"][f"{name}_adaptive"][key] for s in S_GRID]
            st_ = [100 * (report[f"s={s:.2f}"][f"{name}_static_train"][key]
                          - report[f"s={s:.2f}"][f"{name}_adaptive"][key])
                   / report[f"s={s:.2f}"][f"{name}_adaptive"][key] for s in S_GRID]
            print(f"{name} {tag:>11}: mean shuffle penalty {sum(sh)/len(sh):+.2f}%  "
                  f"mean static penalty {sum(st_)/len(st_):+.2f}%")
    print("\nshuffle penalty ~0%  => conditioning input unused (policy is static)")
    print("static penalty  <=0% => a single fixed profile is as good or better")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    save_eval_report({**report, "_meta": {
        "stage": "B3", "split": "val", "n_val_slices": n_val,
        "sparsities": list(S_GRID), "recon_ckpt": str(RECON_CKPT),
        "static_profile": "policy top-k selection frequency measured on TRAIN",
        "shuffled_cond": "scout Z deranged across val; k-space unchanged",
    }}, OUT_JSON)
    print(f"\nwrote {OUT_JSON}")
    for p in PROTECTED:
        if _fp(p) != before[str(p)]:
            raise SystemExit(f"protected artifact changed: {p}")
    print("STAGE B3 DONE (protected artifacts unchanged)")


if __name__ == "__main__":
    main()
