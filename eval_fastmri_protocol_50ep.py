"""Stage P4: detailed eval of the 50-epoch ACS-lock policy. Never trains.

Loads protocol_50ep_acs_lock/ (P3) + 100ep coarse. Does not overwrite P3
ckpts, survival tables, train_status, or older nested / D3PM artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from itw.configs import FastMRIConfig, default_device
from itw.data.fastmri import FastMRIDataset
from itw.eval import (
    acs_lock_topk_from_logits,
    baseline_row_mask_batch,
    evaluate_fastmri_baselines_loader,
    plot_fastmri_grid,
    save_eval_report,
    sparsity_key,
)
from itw.masks import apply_kspace_row_mask, gumbel_row_mask_acs_locked
from itw.train import (
    build_dataloader,
    build_mask_model,
    load_d3pm_coarse,
    mask_generator_cond,
)

PARENT = Path("models_mask_gen_fastmri_nested")
PARENT_FINAL = PARENT / "mask_gen_fastmri_final.pth"
OLD_COARSE_SURVIVAL = PARENT / "coarse_survival_table.pt"
ACS_LOCK_10EP = PARENT / "acs_lock_10ep"
OLD_COARSE_DIR = Path("models_d3pm_fastmri_coarse_kspace")
OLD_FINE_DIR = Path("models_d3pm_fastmri_fine")
NEW_COARSE_CKPT = Path(
    "models_d3pm_fastmri_coarse_kspace_100ep/model_absorb_cosine_final.pth"
)
SAVE_DIR = PARENT / "protocol_50ep_acs_lock"
P4_S = (0.10, 0.25, 0.50, 0.75)
P4_BASELINES = (
    "random",
    "equispaced",
    "vd_gaussian",
    "acs_random",
    "acs_vd_gaussian",
    "acs_equispaced",
)
PLOT_METHODS = ("learned", "acs_random", "acs_vd_gaussian", "random")
CURVE_METHODS = ("learned", "acs_random", "acs_vd_gaussian", "random")
A2_S025 = {
    "density": 0.258,
    "h_coarse": 0.080,
    "nmse": 0.0071,
    "ssim": 0.764,
    "psnr": 32.16,
    "max_batches": 10,
    "source": "acs_lock_10ep/eval/eval_a2.json",
}
P3_SANITY_S025 = {
    "density": 0.255,
    "h_coarse": 0.084,
    "nmse": 0.0079,
    "ssim": 0.758,
    "psnr": 31.75,
    "max_batches": 5,
    "source": "protocol_50ep_acs_lock/eval/sanity_s025.json",
}
PROTECTED = (
    PARENT_FINAL,
    ACS_LOCK_10EP / "mask_gen_fastmri_final.pth",
    OLD_COARSE_DIR / "model_absorb_cosine_final.pth",
    OLD_FINE_DIR / "model_absorb_cosine_final.pth",
    OLD_COARSE_SURVIVAL,
    ACS_LOCK_10EP / "coarse_survival_table.pt",
    ACS_LOCK_10EP / "train_status.json",
    ACS_LOCK_10EP / "eval" / "eval_a2.json",
    SAVE_DIR / "mask_gen_fastmri_final.pth",
    SAVE_DIR / "coarse_survival_table.pt",
    SAVE_DIR / "fine_survival_table.pt",
    SAVE_DIR / "train_status.json",
    SAVE_DIR / "train.log",
    SAVE_DIR / "eval" / "sanity_s025.json",
    NEW_COARSE_CKPT,
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


def _p3_epoch_ckpts() -> tuple[Path, ...]:
    return tuple(sorted(SAVE_DIR.glob("mask_gen_fastmri_*e.pth")))


def _snapshot_protected() -> dict[str, tuple[str, float, int]]:
    snap: dict[str, tuple[str, float, int]] = {}
    for path in (*PROTECTED, *_p3_epoch_ckpts()):
        if not path.is_file():
            raise SystemExit(f"missing protected artifact {path}")
        snap[str(path)] = _fingerprint(path)
    return snap


def _assert_protected_unchanged(before: dict[str, tuple[str, float, int]]) -> None:
    after = _snapshot_protected()
    if set(after) != set(before):
        raise SystemExit(
            f"protected path set changed: extra={set(after) - set(before)} "
            f"missing={set(before) - set(after)}"
        )
    for key, prev in before.items():
        if after[key] != prev:
            raise SystemExit(f"protected artifact changed: {key}")


def _make_cfg() -> FastMRIConfig:
    return FastMRIConfig(
        device=default_device(),
        mask_objective="nested_d3pm",
        n_epochs=50,
        batch_size=8,
        num_workers=4,
        save_every=5,
        lr=1e-4,
        sparsity_min=0.1,
        sparsity_max=0.75,
        sparsity_loss_weight=50.0,
        entropy_alpha=1.0,
        entropy_beta=0.0,
        recon_loss_weight=1.0,
        policy_input="scout_image",
        acs_lock=True,
        scout_size=32,
        image_size=300,
        d3pm_coarse_checkpoint=str(NEW_COARSE_CKPT),
        save_dir=str(SAVE_DIR),
    )


def _load_mask_net(cfg: FastMRIConfig) -> torch.nn.Module:
    ckpt_path = Path(cfg.save_dir) / "mask_gen_fastmri_final.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    if ckpt_path.resolve() == PARENT_FINAL.resolve():
        raise SystemExit("refusing to load parent nested checkpoint as P3 ckpt")
    if ckpt_path.resolve() == (ACS_LOCK_10EP / "mask_gen_fastmri_final.pth").resolve():
        raise SystemExit("refusing to load acs_lock_10ep as P3 ckpt")
    model = build_mask_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["mask_generator"])
    model.eval()
    return model


def _s_tag(s: float) -> str:
    return f"s{int(round(s * 100)):03d}"


def _plot_grids(cfg: FastMRIConfig, model, out: Path) -> list[str]:
    viz_ds = FastMRIDataset(cfg.data_root)
    viz_dl = DataLoader(viz_ds, batch_size=4, shuffle=False, num_workers=0)
    z, c, kspace = next(iter(viz_dl))
    acs_width = int(cfg.scout_size)
    written: list[str] = []
    model.eval()
    with torch.no_grad():
        z_d = z.to(cfg.device)
        k_d = kspace.to(cfg.device)
        cond = mask_generator_cond(cfg, z_d, k_d)
        for s in P4_S:
            sparsity = torch.full((z.shape[0],), s, device=cfg.device)
            row_logits = model(cond, sparsity)
            named_masks = {
                "learned": gumbel_row_mask_acs_locked(
                    row_logits,
                    sparsity,
                    acs_width=acs_width,
                    temperature=0.5,
                    hard=True,
                ),
                "learned_acs_lock": acs_lock_topk_from_logits(
                    row_logits, sparsity, acs_width
                ),
            }
            for name in ("acs_random", "acs_vd_gaussian", "random"):
                named_masks[name] = baseline_row_mask_batch(
                    name,
                    z.shape[0],
                    cfg.image_size,
                    sparsity,
                    cfg.device,
                    acs_width=acs_width,
                )
            titles = {
                "learned": f"learned Gumbel ACS-lock s={s:.2f} (P4)",
                "acs_random": f"acs_random s={s:.2f} (P4)",
                "acs_vd_gaussian": f"acs_vd_gaussian s={s:.2f} (P4)",
                "random": f"random s={s:.2f} (P4)",
            }
            tag = _s_tag(s)
            for name in PLOT_METHODS:
                rows = named_masks[name]
                y = apply_kspace_row_mask(k_d, rows)
                dest = out / f"{name}_{tag}.png"
                plot_fastmri_grid(
                    z,
                    c,
                    rows.cpu(),
                    y.cpu(),
                    title=titles[name],
                    save_path=dest,
                    kspace=kspace,
                )
                written.append(str(dest))
    plt.close("all")
    print("wrote plots", written)
    return written


def _s_from_key(key: str) -> float:
    return float(key.split("=", 1)[1])


def _plot_curves(report: dict, out: Path) -> Path:
    s_vals = [_s_from_key(k) for k in report if k.startswith("s=")]
    s_vals.sort()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    style = {
        "learned": {"label": "learned Gumbel ACS-lock", "marker": "o", "color": "C0"},
        "acs_random": {"label": "ACS+random", "marker": "s", "color": "C1"},
        "acs_vd_gaussian": {"label": "ACS+VD", "marker": "^", "color": "C2"},
        "random": {"label": "random", "marker": "x", "color": "C3"},
    }
    panels = (
        ("nmse", "NMSE", True),
        ("ssim", "SSIM", False),
        ("h_coarse", r"$H_c$", False),
    )
    for ax, (metric, ylabel, logy) in zip(axes, panels):
        for name in CURVE_METHODS:
            ys = [report[sparsity_key(s)][name][metric] for s in s_vals]
            ax.plot(s_vals, ys, **style[name])
        ax.set_xlabel("sparsity s")
        ax.set_ylabel(ylabel)
        ax.set_xticks(s_vals)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("P4: 50-epoch ACS-lock vs baselines")
    fig.tight_layout()
    dest = out / "nmse_ssim_hc_vs_s.png"
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print("wrote", dest)
    return dest


def _print_table(report: dict) -> None:
    methods = ("learned", "learned_acs_lock") + P4_BASELINES
    print("P4 table (20 batches)")
    print(
        f"{'s':>6} {'method':<18} {'dens':>7} {'H_c':>8} "
        f"{'NMSE':>9} {'SSIM':>7} {'PSNR':>7}"
    )
    for s in P4_S:
        block = report[sparsity_key(s)]
        for name in methods:
            m = block[name]
            print(
                f"{s:6.2f} {name:<18} {m['mean_sparsity']:7.3f} "
                f"{m['h_coarse']:8.4f} {m['nmse']:9.4f} "
                f"{m['ssim']:7.3f} {m['psnr']:7.2f}"
            )


def main() -> None:
    if not NEW_COARSE_CKPT.is_file():
        raise SystemExit(f"missing 100ep coarse ckpt {NEW_COARSE_CKPT}")
    final = SAVE_DIR / "mask_gen_fastmri_final.pth"
    coarse_surv_path = SAVE_DIR / "coarse_survival_table.pt"
    fine_surv_path = SAVE_DIR / "fine_survival_table.pt"
    if not final.is_file():
        raise SystemExit(f"missing P3 ckpt {final}; P4 does not train")
    if not coarse_surv_path.is_file():
        raise SystemExit(f"missing P3 coarse survival {coarse_surv_path}")
    if not fine_surv_path.is_file():
        raise SystemExit(f"missing P3 fine survival {fine_surv_path}")
    if PARENT_FINAL.resolve() == final.resolve():
        raise SystemExit("parent checkpoint path collided with P3 save_dir")

    protected_before = _snapshot_protected()
    print("protected snapshot")
    for path, (md5, mtime, size) in protected_before.items():
        print(f"  {path} md5={md5} mtime={mtime} size={size}")

    cfg = _make_cfg()
    if not bool(cfg.acs_lock):
        raise SystemExit("P4 requires cfg.acs_lock=True for train-consistent Gumbel")
    if abs(float(cfg.entropy_beta) - 0.0) > 1e-12:
        raise SystemExit(f"P4 requires entropy_beta=0, got {cfg.entropy_beta}")
    if Path(cfg.d3pm_coarse_checkpoint).resolve() != NEW_COARSE_CKPT.resolve():
        raise SystemExit("P4 must use 100ep coarse ckpt, not the old 50ep coarse")
    if int(cfg.scout_size) != 32 or int(cfg.image_size) != 300:
        raise SystemExit("P4 requires scout_size=32, image_size=300")
    print(
        f"device={cfg.device} policy_input={cfg.policy_input} "
        f"acs_lock={cfg.acs_lock} alpha={cfg.entropy_alpha} "
        f"beta={cfg.entropy_beta} recon={cfg.recon_loss_weight} "
        f"scout_size={cfg.scout_size} image_size={cfg.image_size} "
        f"coarse={cfg.d3pm_coarse_checkpoint} save_dir={cfg.save_dir} "
        "mode=eval-only"
    )

    dataloader = build_dataloader(cfg)
    d3pm_coarse = load_d3pm_coarse(cfg)
    coarse_survival = torch.load(
        coarse_surv_path, map_location=cfg.device, weights_only=False
    )
    fine_survival = torch.load(
        fine_surv_path, map_location=cfg.device, weights_only=False
    )

    model = _load_mask_net(cfg)
    out = SAVE_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "eval_p4.json"
    print(
        "P4 eval: 20 batches at",
        P4_S,
        "baselines",
        P4_BASELINES,
        "include_learned_acs_lock=True",
    )
    report = evaluate_fastmri_baselines_loader(
        model,
        None,
        d3pm_coarse,
        cfg,
        dataloader,
        fine_survival,
        coarse_survival,
        sparsities=P4_S,
        max_batches=20,
        baselines=P4_BASELINES,
        include_learned_acs_lock=True,
    )
    combined = {
        **report,
        "acs_lock": True,
        "entropy_alpha": cfg.entropy_alpha,
        "entropy_beta": cfg.entropy_beta,
        "recon_loss_weight": cfg.recon_loss_weight,
        "policy_input": cfg.policy_input,
        "scout_size": cfg.scout_size,
        "image_size": cfg.image_size,
        "max_batches": 20,
        "d3pm_coarse_checkpoint": cfg.d3pm_coarse_checkpoint,
        "policy_ckpt": str(final),
        "coarse_survival": str(coarse_surv_path),
        "note": (
            "P4 20-batch grid. learned = ACS-locked Gumbel (train-consistent); "
            "learned_acs_lock = ACS + top-k remainder from the same logits. "
            "s=0.10 k=30<32 is centered ACS block."
        ),
        "a2_s025": A2_S025,
        "p3_sanity_s025": P3_SANITY_S025,
    }
    save_eval_report(combined, json_path)
    print("wrote", json_path)
    _print_table(report)
    plot_paths = _plot_grids(cfg, model, out)
    curve_path = _plot_curves(report, out)
    print("plots", plot_paths)
    print("curve", curve_path)
    _assert_protected_unchanged(protected_before)
    print("STAGE P4 DONE wrote", json_path)


if __name__ == "__main__":
    main()
