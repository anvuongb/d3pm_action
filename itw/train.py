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

from .configs import CIFAR10Config, ITWConfig, MNISTConfig
from .entropy import d3pm_cond_entropy_loss
from .masks import (
    MaskGeneratorMLP,
    SpatialMaskGenerator,
    SpatialMaskGeneratorMNIST,
    apply_masked_observation,
    conditioning_features,
    gumbel_mask,
    mask_loss,
)
from .schedule import build_pixel_survival_table, sparsity_to_timestep


def _gumbel_temperature(cfg: ITWConfig, epoch: int) -> float:
    if cfg.n_epochs <= 1:
        return cfg.gumbel_temperature_end
    frac = epoch / (cfg.n_epochs - 1)
    return cfg.gumbel_temperature_start + frac * (
        cfg.gumbel_temperature_end - cfg.gumbel_temperature_start
    )


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
    d3pm.eval()
    for p in d3pm.parameters():
        p.requires_grad_(False)
    return d3pm


def build_mask_model(cfg: ITWConfig) -> torch.nn.Module:
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
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
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
