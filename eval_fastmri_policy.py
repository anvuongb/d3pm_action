"""Train nested row-mask policy (if needed), then eval + plot learned vs random."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from itw.configs import FastMRIConfig, default_device
from itw.data.fastmri import FastMRIDataset
from itw.eval import (
    evaluate_fastmri_nested_loader,
    plot_fastmri_grid,
    random_row_mask_batch,
    save_eval_report,
)
from itw.masks import apply_kspace_row_mask, gumbel_row_mask
from itw.train import (
    build_dataloader,
    build_mask_model,
    load_d3pm_coarse,
    load_d3pm_fine,
    train_fastmri,
)


def _load_or_train(cfg: FastMRIConfig):
    save = Path(cfg.save_dir)
    final = save / "mask_gen_fastmri_final.pth"
    fine_pt = save / "fine_survival_table.pt"
    coarse_pt = save / "coarse_survival_table.pt"
    print("no valid nested checkpoint; training mask policy")
    return train_fastmri(cfg)


def main() -> None:
    cfg = FastMRIConfig(
        device=default_device(),
        mask_objective="nested_d3pm",
        n_epochs=10,
        batch_size=8,
        num_workers=4,
        save_every=1,
    )
    out = Path(cfg.save_dir) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={cfg.device} save_dir={cfg.save_dir}")

    model, tables = _load_or_train(cfg)
    fine_s = tables["fine_survival"]
    coarse_s = tables["coarse_survival"]
    print(
        "survival fine",
        float(fine_s.min()),
        float(fine_s.max()),
        "coarse",
        float(coarse_s.min()),
        float(coarse_s.max()),
    )

    d3pm_fine = load_d3pm_fine(cfg) if cfg.entropy_beta > 0 else None
    d3pm_coarse = load_d3pm_coarse(cfg)
    dataloader = build_dataloader(cfg)

    metrics = evaluate_fastmri_nested_loader(
        model,
        d3pm_fine,
        d3pm_coarse,
        cfg,
        dataloader,
        fine_s,
        coarse_s,
        max_batches=10,
        fixed_sparsity=0.25,
    )
    print("eval s=0.25", metrics)
    save_eval_report(metrics, out / "eval_nested.json")

    from dataclasses import replace

    ablation = {}
    for s in (0.1, 0.25, 0.4):
        ablation[f"s={s:.2f}"] = evaluate_fastmri_nested_loader(
            model,
            d3pm_fine,
            d3pm_coarse,
            replace(cfg, entropy_alpha=1.0, entropy_beta=cfg.entropy_beta),
            dataloader,
            fine_s,
            coarse_s,
            max_batches=5,
            fixed_sparsity=s,
        )
    save_eval_report(ablation, out / "eval_ablation.json")
    print("ablation", ablation)

    viz_ds = FastMRIDataset(cfg.data_root)
    viz_dl = DataLoader(viz_ds, batch_size=4, shuffle=False, num_workers=0)
    z, c, kspace = next(iter(viz_dl))
    sparsity = torch.full((z.shape[0],), 0.25, device=cfg.device)
    model.eval()
    with torch.no_grad():
        z_d = z.to(cfg.device)
        k_d = kspace.to(cfg.device)
        row_logits = model(z_d, sparsity)
        learned_rows = gumbel_row_mask(row_logits, temperature=0.5, hard=True)
        random_rows = random_row_mask_batch(
            z.shape[0], cfg.image_size, sparsity, cfg.device
        )
        y_learned = apply_kspace_row_mask(k_d, learned_rows)
        y_random = apply_kspace_row_mask(k_d, random_rows)

    print(
        "learned density",
        float(learned_rows.mean()),
        "nmse learned vs random will be in eval json",
        "y mean",
        float(y_learned.mean()),
    )
    plot_fastmri_grid(
        z,
        c,
        learned_rows.cpu(),
        y_learned.cpu(),
        title="learned row mask s=0.25",
        save_path=out / "learned_s025.png",
    )
    plot_fastmri_grid(
        z,
        c,
        random_rows.cpu(),
        y_random.cpu(),
        title="random row mask s=0.25",
        save_path=out / "random_s025.png",
    )
    plt.close("all")
    print("wrote", out)


if __name__ == "__main__":
    main()
