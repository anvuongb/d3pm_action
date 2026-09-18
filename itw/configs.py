"""ITW mask-generator training configuration."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_fastmri_root() -> str:
    return os.environ.get(
        "FASTMRI_ROOT", "/home/anvuong/data/fast_mri/singlecoil_train"
    )


def default_fastmri_val_root() -> str:
    return os.environ.get(
        "FASTMRI_VAL_ROOT", "/home/anvuong/data/fast_mri/singlecoil_val"
    )


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNGs. Call before building a seeded dataloader."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ITWConfig:
    dataset: Literal["mnist", "cifar10", "fastmri"]
    num_classes: int
    n_t: int = 1000
    d3pm_checkpoint: str = ""
    image_channels: int = 1
    image_size: int = 32
    multichannel: bool = False

    # mask model
    mask_arch: Literal["spatial", "mlp", "cartesian_row"] = "spatial"
    hidden_dim: int = 128

    # training
    batch_size: int = 256
    num_workers: int = 8
    lr: float = 2e-5
    n_epochs: int = 200
    sparsity_min: float = 0.1
    sparsity_max: float = 0.9
    sparsity_loss_weight: float = 1.0
    gumbel_temperature_start: float = 1.0
    gumbel_temperature_end: float = 0.5
    binarization_weight: float = 0.0

    # io
    save_dir: str = "models_mask_gen"
    log_file: str = "train.log"
    save_every: int = 10
    schedule_calibration_batches: int = 20

    # ``None`` keeps the pre-B1 nondeterministic behaviour; eval paths set it.
    seed: int | None = None

    device: str = field(default_factory=default_device)


@dataclass
class MNISTConfig(ITWConfig):
    dataset: Literal["mnist"] = "mnist"
    num_classes: int = 2
    image_channels: int = 1
    multichannel: bool = False
    mask_arch: Literal["spatial", "mlp", "cartesian_row"] = "spatial"
    batch_size: int = 512
    num_workers: int = 8
    lr: float = 3e-4
    sparsity_min: float = 0.05
    sparsity_max: float = 0.95
    sparsity_loss_weight: float = 10.0
    d3pm_checkpoint: str = "models/model_absorb_cosine_399.pth"
    save_dir: str = "models_mask_gen_mnist"


@dataclass
class CIFAR10Config(ITWConfig):
    dataset: Literal["cifar10"] = "cifar10"
    num_classes: int = 8
    image_channels: int = 3
    multichannel: bool = True
    mask_arch: Literal["spatial", "mlp", "cartesian_row"] = "spatial"
    batch_size: int = 256
    lr: float = 2e-5
    sparsity_min: float = 0.1
    sparsity_max: float = 0.9
    sparsity_loss_weight: float = 1.0
    d3pm_checkpoint: str = "models/cifar10/model_absorb_cosine_499.pth"
    save_dir: str = "models_mask_gen_cifar10"


@dataclass
class FastMRIConfig(ITWConfig):
    dataset: Literal["fastmri"] = "fastmri"
    num_classes: int = 8
    image_channels: int = 1
    image_size: int = 300
    multichannel: bool = False
    mask_arch: Literal["spatial", "mlp", "cartesian_row"] = "cartesian_row"
    data_root: str = field(default_factory=default_fastmri_root)
    val_root: str = field(default_factory=default_fastmri_val_root)
    # "train" shuffles and drops the last partial batch; "val" is deterministic
    # order over the full held-out split (B1).
    data_split: Literal["train", "val"] = "train"
    scout_size: int = 32
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-4
    n_epochs: int = 10
    sparsity_min: float = 0.1
    sparsity_max: float = 0.4
    sparsity_loss_weight: float = 50.0
    save_every: int = 1
    save_dir: str = ""
    d3pm_checkpoint: str = ""
    feature_dim: int = 128
    infonce_temperature: float = 0.1

    # Nested D3PM entropy-proxy path (Option 2)
    mask_objective: Literal["infonce", "nested_d3pm"] = "nested_d3pm"
    fine_size: int = 96
    d3pm_fine_checkpoint: str = "models_d3pm_fastmri_fine/model_absorb_cosine_final.pth"
    d3pm_coarse_checkpoint: str = (
        "models_d3pm_fastmri_coarse_kspace/model_absorb_cosine_final.pth"
    )
    entropy_alpha: float = 1.0
    entropy_beta: float = 0.0
    recon_loss_weight: float = 1.0
    coarse_from_kspace: bool = True
    coarse_hidden: int = 64
    policy_input: Literal["scout_image", "acs_kspace"] = "scout_image"
    acs_lock: bool = False

    def __post_init__(self) -> None:
        if not self.save_dir or self.save_dir == "models_mask_gen_fastmri":
            self.save_dir = (
                "models_mask_gen_fastmri_nested"
                if self.mask_objective == "nested_d3pm"
                else "models_mask_gen_fastmri_infonce"
            )


def config_to_dict(cfg: ITWConfig) -> dict:
    return asdict(cfg)
