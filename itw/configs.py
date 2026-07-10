"""ITW mask-generator training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ITWConfig:
    dataset: Literal["mnist", "cifar10"]
    num_classes: int
    n_t: int = 1000
    d3pm_checkpoint: str = ""
    image_channels: int = 1
    image_size: int = 32
    multichannel: bool = False

    # mask model
    mask_arch: Literal["spatial", "mlp"] = "spatial"
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

    device: str = "cuda"


@dataclass
class MNISTConfig(ITWConfig):
    dataset: Literal["mnist"] = "mnist"
    num_classes: int = 2
    image_channels: int = 1
    multichannel: bool = False
    mask_arch: Literal["spatial", "mlp"] = "spatial"
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
    mask_arch: Literal["spatial", "mlp"] = "spatial"
    batch_size: int = 256
    lr: float = 2e-5
    sparsity_min: float = 0.1
    sparsity_max: float = 0.9
    sparsity_loss_weight: float = 1.0
    d3pm_checkpoint: str = "models/cifar10/model_absorb_cosine_499.pth"
    save_dir: str = "models_mask_gen_cifar10"
