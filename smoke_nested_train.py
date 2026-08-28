"""2-epoch nested FastMRI smoke after collapse fixes."""

from __future__ import annotations

import torch

from itw.configs import FastMRIConfig, default_device
from itw.schedule import sparsity_to_timestep
from itw.train import train_fastmri


def main() -> None:
    cfg = FastMRIConfig(
        device=default_device(),
        mask_objective="nested_d3pm",
        n_epochs=2,
        batch_size=8,
        num_workers=4,
        save_every=1,
        save_dir="models_mask_gen_fastmri_nested",
    )
    print("device", cfg.device)
    _model, tables = train_fastmri(cfg)
    fs, cs = tables["fine_survival"], tables["coarse_survival"]
    s = torch.tensor([0.1, 0.25, 0.4])
    print(
        "fine survival",
        float(fs.min()),
        float(fs.max()),
        "t",
        sparsity_to_timestep(s, fs.cpu(), cfg.n_t).tolist(),
    )
    print(
        "coarse survival",
        float(cs.min()),
        float(cs.max()),
        "t",
        sparsity_to_timestep(s, cs.cpu(), cfg.n_t).tolist(),
    )
    print("SMOKE DONE")


if __name__ == "__main__":
    main()
