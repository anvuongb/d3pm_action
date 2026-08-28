"""Training loop for ITW mask generators."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from tqdm import tqdm

from d3pm_runner import D3PM, DummyX0Model
from dit import DiT_Llama

from .configs import CIFAR10Config, FastMRIConfig, ITWConfig, MNISTConfig, config_to_dict
from .data.fastmri import FastMRIDataset
from .discrete import (
    apply_row_absorbing_observation,
    kspace_to_row_disc,
    magnitude_to_fine_disc,
    magnitude_to_row_disc,
    nmse,
)
from .entropy import (
    coarse_cond_entropy_loss,
    d3pm_cond_entropy_loss,
    fine_cond_entropy_loss,
    nested_cond_entropy_loss,
)
from .infonce import InfoNCELoss, ProjectionHead
from .masks import (
    CartesianRowMaskGenerator,
    MaskGeneratorMLP,
    SpatialMaskGenerator,
    SpatialMaskGeneratorMNIST,
    apply_kspace_row_mask,
    apply_masked_observation,
    conditioning_features,
    gumbel_mask,
    gumbel_row_mask_ste,
    magnitude_from_kspace,
    mask_loss,
    row_mask_loss,
)
from .row_d3pm import build_coarse_backbone, build_fine_backbone
from .schedule import (
    build_fine_survival_table,
    build_pixel_survival_table,
    build_row_survival_table,
    sparsity_to_timestep,
)


def _reset_train_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")


def _append_train_log(log_path: Path, line: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def _gumbel_temperature(cfg: ITWConfig, epoch: int) -> float:
    if cfg.n_epochs <= 1:
        return cfg.gumbel_temperature_end
    frac = epoch / (cfg.n_epochs - 1)
    return cfg.gumbel_temperature_start + frac * (
        cfg.gumbel_temperature_end - cfg.gumbel_temperature_start
    )


def _freeze_d3pm(d3pm: D3PM) -> D3PM:
    d3pm.eval()
    for p in d3pm.parameters():
        p.requires_grad_(False)
    return d3pm


def load_d3pm(cfg: ITWConfig) -> D3PM:
    if cfg.dataset == "mnist":
        backbone = DummyX0Model(cfg.image_channels, cfg.num_classes)
    elif cfg.dataset == "cifar10":
        backbone = DiT_Llama(cfg.image_channels, cfg.num_classes, dim=1024)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    d3pm = D3PM(
        backbone,
        cfg.n_t,
        num_classes=cfg.num_classes,
        hybrid_loss_coeff=0.0,
        forward_type="absorb",
        schedule="cosine",
    ).to(cfg.device)
    d3pm.load_state_dict(torch.load(cfg.d3pm_checkpoint, map_location=cfg.device))
    return _freeze_d3pm(d3pm)


def load_d3pm_fine(cfg: FastMRIConfig) -> D3PM:
    """Load frozen fine (96x96) FastMRI D3PM prior."""
    d3pm = D3PM(
        build_fine_backbone(n_bins=cfg.num_classes),
        cfg.n_t,
        num_classes=cfg.num_classes,
        hybrid_loss_coeff=0.0,
        forward_type="absorb",
        schedule="cosine",
    ).to(cfg.device)
    d3pm.load_state_dict(
        torch.load(cfg.d3pm_fine_checkpoint, map_location=cfg.device)
    )
    return _freeze_d3pm(d3pm)


def load_d3pm_coarse(cfg: FastMRIConfig) -> D3PM:
    """Load frozen coarse (row-profile) FastMRI D3PM prior."""
    d3pm = D3PM(
        build_coarse_backbone(n_bins=cfg.num_classes, hidden=cfg.coarse_hidden),
        cfg.n_t,
        num_classes=cfg.num_classes,
        hybrid_loss_coeff=0.0,
        forward_type="absorb",
        schedule="cosine",
    ).to(cfg.device)
    d3pm.load_state_dict(
        torch.load(cfg.d3pm_coarse_checkpoint, map_location=cfg.device)
    )
    return _freeze_d3pm(d3pm)


def coarse_profile(
    cfg: FastMRIConfig,
    c: torch.Tensor,
    kspace: torch.Tensor,
    ste: bool = False,
) -> torch.Tensor:
    """Discrete PE-row profile for the coarse prior (k-space energy by default)."""
    if cfg.coarse_from_kspace:
        return kspace_to_row_disc(kspace, n_bins=cfg.num_classes, ste=ste)
    return magnitude_to_row_disc(c, n_bins=cfg.num_classes, ste=ste)


def build_mask_model(cfg: ITWConfig) -> torch.nn.Module:
    if cfg.dataset == "fastmri" or cfg.mask_arch == "cartesian_row":
        return CartesianRowMaskGenerator(target_rows=cfg.image_size).to(cfg.device)

    if cfg.mask_arch == "mlp":
        return MaskGeneratorMLP(size=cfg.image_size).to(cfg.device)

    if cfg.dataset == "mnist":
        return SpatialMaskGeneratorMNIST(size=cfg.image_size).to(cfg.device)
    return SpatialMaskGenerator(in_channels=1, size=cfg.image_size).to(cfg.device)


def build_dataloader(cfg: ITWConfig) -> DataLoader:
    if cfg.dataset == "mnist":
        dataset = MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Pad(2)]
            ),
        )
    elif cfg.dataset == "cifar10":
        dataset = CIFAR10(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            ),
        )
    elif cfg.dataset == "fastmri":
        assert isinstance(cfg, FastMRIConfig)
        dataset = FastMRIDataset(
            cfg.data_root,
            scout_size=cfg.scout_size,
            target_dim=cfg.image_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=cfg.dataset == "fastmri",
    )


def discretize(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    return (x * (num_classes - 1)).round().long().clamp(0, num_classes - 1)


def forward_mask_logits(model, cfg: ITWConfig, x_disc, cond, sparsity):
    x_cond = conditioning_features(x_disc)
    sparsity_col = sparsity.reshape(-1, 1).to(cfg.device)

    if isinstance(model, SpatialMaskGeneratorMNIST):
        return model(x_cond, sparsity_col, labels=cond)
    if isinstance(model, SpatialMaskGenerator):
        return model(x_cond, sparsity_col)
    return model(cond, sparsity)


def train_mask_generator(
    cfg: ITWConfig,
    model: torch.nn.Module | None = None,
    d3pm: D3PM | None = None,
    dataloader: DataLoader | None = None,
    survival_table: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, torch.Tensor]:
    os.makedirs(cfg.save_dir, exist_ok=True)
    log_path = Path(cfg.save_dir) / cfg.log_file

    d3pm = d3pm or load_d3pm(cfg)
    model = model or build_mask_model(cfg)
    dataloader = dataloader or build_dataloader(cfg)

    if survival_table is None:
        survival_table = build_pixel_survival_table(
            d3pm,
            dataloader,
            device=cfg.device,
            max_batches=cfg.schedule_calibration_batches,
        )
    survival_table = survival_table.to(cfg.device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.n_epochs):
        model.train()
        temperature = _gumbel_temperature(cfg, epoch)
        loss_ema = None
        pbar = tqdm(dataloader, desc=f"epoch {epoch}")

        for x, cond in pbar:
            d3pm.eval()
            sparsity = (
                torch.rand(x.shape[0]) * (cfg.sparsity_max - cfg.sparsity_min)
                + cfg.sparsity_min
            )
            t = sparsity_to_timestep(sparsity, survival_table, cfg.n_t)

            optim.zero_grad()
            x = x.to(cfg.device)
            cond = cond.to(cfg.device)
            x_disc = discretize(x, cfg.num_classes)

            mask_logits = forward_mask_logits(model, cfg, x_disc, cond, sparsity)
            mask = gumbel_mask(mask_logits, temperature=temperature, hard=True)
            y = apply_masked_observation(
                x_disc,
                mask,
                cfg.num_classes,
                multichannel=cfg.multichannel,
            )

            hxy_loss = d3pm_cond_entropy_loss(d3pm, y, t.to(cfg.device), cond, mask)
            sparsity_loss = mask_loss(
                mask,
                sparsity,
                binarization_weight=cfg.binarization_weight,
            )
            loss = hxy_loss + cfg.sparsity_loss_weight * sparsity_loss
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()

            pbar.set_description(
                f"epoch {epoch} loss {loss_ema:.4f} H {hxy_loss.item():.4f} "
                f"sp {sparsity_loss.item():.4f} tau {temperature:.2f}"
            )
            optim.step()

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"epoch: {epoch}, loss: {loss_ema:.6f}, H(X|Y): {hxy_loss.item():.6f}, "
                f"sparsity_loss: {sparsity_loss.item():.6f}, tau: {temperature:.3f}\n"
            )

        if (epoch + 1) % cfg.save_every == 0:
            ckpt = Path(cfg.save_dir) / f"mask_gen_{cfg.dataset}_{epoch}e.pth"
            torch.save(model.state_dict(), ckpt)

    final_ckpt = Path(cfg.save_dir) / f"mask_gen_{cfg.dataset}_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    torch.save(survival_table.cpu(), Path(cfg.save_dir) / "survival_table.pt")
    return model, survival_table


def train_mnist(**kwargs):
    cfg = MNISTConfig(**kwargs)
    return train_mask_generator(cfg)


def train_cifar10(**kwargs):
    cfg = CIFAR10Config(**kwargs)
    return train_mask_generator(cfg)


def train_fastmri_infonce(
    cfg: FastMRIConfig,
    model: torch.nn.Module | None = None,
    proj_head: torch.nn.Module | None = None,
    dataloader: DataLoader | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Train Cartesian row-mask generator with InfoNCE + sparsity."""
    os.makedirs(cfg.save_dir, exist_ok=True)
    log_path = Path(cfg.save_dir) / cfg.log_file
    _reset_train_log(log_path)

    model = model or build_mask_model(cfg)
    proj_head = proj_head or ProjectionHead(feature_dim=cfg.feature_dim).to(cfg.device)
    dataloader = dataloader or build_dataloader(cfg)
    info_nce = InfoNCELoss(temperature=cfg.infonce_temperature).to(cfg.device)

    optim = torch.optim.Adam(
        list(model.parameters()) + list(proj_head.parameters()),
        lr=cfg.lr,
    )

    for epoch in range(cfg.n_epochs):
        model.train()
        proj_head.train()
        temperature = _gumbel_temperature(cfg, epoch)
        loss_ema = None
        density_ema = None
        pbar = tqdm(dataloader, desc=f"epoch {epoch}")

        for z, c, kspace in pbar:
            sparsity = (
                torch.rand(z.shape[0]) * (cfg.sparsity_max - cfg.sparsity_min)
                + cfg.sparsity_min
            ).to(cfg.device)

            optim.zero_grad()
            z = z.to(cfg.device)
            c = c.to(cfg.device)
            kspace = kspace.to(cfg.device)

            row_logits = model(z, sparsity)
            row_mask_soft, row_mask = gumbel_row_mask_ste(
                row_logits, temperature=temperature
            )
            y = apply_kspace_row_mask(kspace, row_mask)

            mi_loss = info_nce(proj_head(y), proj_head(c))
            sparsity_loss = row_mask_loss(row_mask_soft, sparsity)
            loss = mi_loss + cfg.sparsity_loss_weight * sparsity_loss
            loss.backward()
            optim.step()

            density = float(row_mask.detach().mean())
            if loss_ema is None:
                loss_ema = loss.item()
                density_ema = density
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()
                density_ema = 0.99 * density_ema + 0.01 * density

            pbar.set_description(
                f"epoch {epoch} loss {loss_ema:.4f} MI {mi_loss.item():.4f} "
                f"sp {sparsity_loss.item():.4f} dens {density_ema:.3f} tau {temperature:.2f}"
            )

        _append_train_log(
            log_path,
            f"epoch: {epoch}, loss: {loss_ema:.6f}, MI: {mi_loss.item():.6f}, "
            f"sparsity_loss: {sparsity_loss.item():.6f}, density: {density_ema:.6f}, "
            f"tau: {temperature:.3f}\n",
        )

        if (epoch + 1) % cfg.save_every == 0:
            ckpt = Path(cfg.save_dir) / f"mask_gen_fastmri_{epoch}e.pth"
            torch.save(
                {
                    "mask_generator": model.state_dict(),
                    "proj_head": proj_head.state_dict(),
                    "epoch": epoch,
                    "mask_objective": "infonce",
                    "cfg": config_to_dict(cfg),
                },
                ckpt,
            )

    final_ckpt = Path(cfg.save_dir) / "mask_gen_fastmri_final.pth"
    torch.save(
        {
            "mask_generator": model.state_dict(),
            "proj_head": proj_head.state_dict(),
            "mask_objective": "infonce",
            "cfg": config_to_dict(cfg),
        },
        final_ckpt,
    )
    return model, proj_head


def train_fastmri_nested(
    cfg: FastMRIConfig,
    model: torch.nn.Module | None = None,
    dataloader: DataLoader | None = None,
    d3pm_fine: D3PM | None = None,
    d3pm_coarse: D3PM | None = None,
    fine_survival: torch.Tensor | None = None,
    coarse_survival: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """
    Train Cartesian row-mask generator with nested frozen D3PM entropy proxies.

    Returns (mask_generator, {"fine_survival": ..., "coarse_survival": ...}).
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    log_path = Path(cfg.save_dir) / cfg.log_file
    _reset_train_log(log_path)

    model = model or build_mask_model(cfg)
    dataloader = dataloader or build_dataloader(cfg)
    d3pm_coarse = d3pm_coarse or load_d3pm_coarse(cfg)
    use_fine = cfg.entropy_beta > 0
    if use_fine:
        d3pm_fine = d3pm_fine or load_d3pm_fine(cfg)
    else:
        d3pm_fine = None

    if use_fine and fine_survival is None:
        fine_survival = build_fine_survival_table(
            d3pm_fine,
            dataloader,
            device=cfg.device,
            fine_size=cfg.fine_size,
            n_bins=cfg.num_classes,
            max_batches=cfg.schedule_calibration_batches,
        )
    if not use_fine:
        fine_survival = torch.linspace(1.0, 0.01, cfg.n_t)
    if coarse_survival is None:
        coarse_survival = build_row_survival_table(
            d3pm_coarse,
            dataloader,
            device=cfg.device,
            n_bins=cfg.num_classes,
            max_batches=cfg.schedule_calibration_batches,
        )
    fine_survival = fine_survival.to(cfg.device)
    coarse_survival = coarse_survival.to(cfg.device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    cond0_cache: dict[int, torch.Tensor] = {}

    for epoch in range(cfg.n_epochs):
        model.train()
        temperature = _gumbel_temperature(cfg, epoch)
        loss_ema = None
        density_ema = None
        pbar = tqdm(dataloader, desc=f"epoch {epoch}")

        for z, c, kspace in pbar:
            sparsity = (
                torch.rand(z.shape[0]) * (cfg.sparsity_max - cfg.sparsity_min)
                + cfg.sparsity_min
            ).to(cfg.device)

            optim.zero_grad()
            z = z.to(cfg.device)
            c = c.to(cfg.device)
            kspace = kspace.to(cfg.device)

            row_logits = model(z, sparsity)
            row_mask_soft, row_mask = gumbel_row_mask_ste(
                row_logits, temperature=temperature
            )

            y_mag = apply_kspace_row_mask(kspace, row_mask)
            c_mag = magnitude_from_kspace(kspace)
            recon_loss = nmse(y_mag, c_mag)

            b = z.shape[0]
            if b not in cond0_cache:
                cond0_cache[b] = torch.zeros(b, dtype=torch.long, device=cfg.device)
            cond0 = cond0_cache[b]

            h_fine = torch.zeros((), device=cfg.device)
            if use_fine:
                y_fine = magnitude_to_fine_disc(
                    y_mag, size=cfg.fine_size, n_bins=cfg.num_classes, ste=True
                )
                t_fine = sparsity_to_timestep(sparsity, fine_survival, cfg.n_t)
                h_fine = fine_cond_entropy_loss(d3pm_fine, y_fine, t_fine, cond0)

            c_row = coarse_profile(cfg, c, kspace, ste=True)
            y_row = apply_row_absorbing_observation(
                c_row, row_mask, cfg.num_classes
            )
            t_coarse = sparsity_to_timestep(sparsity, coarse_survival, cfg.n_t)

            h_coarse = coarse_cond_entropy_loss(
                d3pm_coarse, y_row, t_coarse, cond0, row_mask
            )
            h_loss = nested_cond_entropy_loss(
                h_coarse, h_fine, alpha=cfg.entropy_alpha, beta=cfg.entropy_beta
            )
            sparsity_loss = row_mask_loss(row_mask_soft, sparsity)
            loss = (
                h_loss
                + cfg.sparsity_loss_weight * sparsity_loss
                + cfg.recon_loss_weight * recon_loss
            )
            loss.backward()
            optim.step()

            density = float(row_mask.detach().mean())
            if loss_ema is None:
                loss_ema = loss.item()
                density_ema = density
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()
                density_ema = 0.99 * density_ema + 0.01 * density

            pbar.set_description(
                f"epoch {epoch} loss {loss_ema:.4f} "
                f"Hc {h_coarse.item():.4f} Hf {h_fine.item():.4f} "
                f"nmse {recon_loss.item():.4f} sp {sparsity_loss.item():.4f} "
                f"dens {density_ema:.3f} tau {temperature:.2f}"
            )

        _append_train_log(
            log_path,
            f"epoch: {epoch}, loss: {loss_ema:.6f}, "
            f"H_coarse: {h_coarse.item():.6f}, H_fine: {h_fine.item():.6f}, "
            f"nmse: {recon_loss.item():.6f}, "
            f"sparsity_loss: {sparsity_loss.item():.6f}, density: {density_ema:.6f}, "
            f"tau: {temperature:.3f}\n",
        )

        if (epoch + 1) % cfg.save_every == 0:
            ckpt = Path(cfg.save_dir) / f"mask_gen_fastmri_{epoch}e.pth"
            torch.save(
                {
                    "mask_generator": model.state_dict(),
                    "epoch": epoch,
                    "mask_objective": "nested_d3pm",
                    "entropy_alpha": cfg.entropy_alpha,
                    "entropy_beta": cfg.entropy_beta,
                    "cfg": config_to_dict(cfg),
                },
                ckpt,
            )

    final_ckpt = Path(cfg.save_dir) / "mask_gen_fastmri_final.pth"
    torch.save(
        {
            "mask_generator": model.state_dict(),
            "mask_objective": "nested_d3pm",
            "entropy_alpha": cfg.entropy_alpha,
            "entropy_beta": cfg.entropy_beta,
            "cfg": config_to_dict(cfg),
        },
        final_ckpt,
    )
    torch.save(fine_survival.cpu(), Path(cfg.save_dir) / "fine_survival_table.pt")
    torch.save(coarse_survival.cpu(), Path(cfg.save_dir) / "coarse_survival_table.pt")
    return model, {"fine_survival": fine_survival, "coarse_survival": coarse_survival}


def train_fastmri(
    cfg: FastMRIConfig | None = None,
    model: torch.nn.Module | None = None,
    proj_head: torch.nn.Module | None = None,
    dataloader: DataLoader | None = None,
    **kwargs,
):
    """
    Train Cartesian row-mask generator on fastMRI.

    Dispatches on cfg.mask_objective:
      - "infonce": InfoNCE + sparsity (returns mask_generator, proj_head)
      - "nested_d3pm": nested frozen D3PM entropy proxies (returns mask_generator, tables)
    """
    cfg = cfg or FastMRIConfig(**kwargs)
    if cfg.mask_objective == "infonce":
        return train_fastmri_infonce(
            cfg, model=model, proj_head=proj_head, dataloader=dataloader
        )
    if cfg.mask_objective == "nested_d3pm":
        return train_fastmri_nested(cfg, model=model, dataloader=dataloader)
    raise ValueError(f"Unknown mask_objective: {cfg.mask_objective}")

