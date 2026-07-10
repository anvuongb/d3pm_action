"""ITW: information-theoretic mask generation for active sensing."""

from .configs import CIFAR10Config, ITWConfig, MNISTConfig
from .entropy import d3pm_cond_entropy_loss, masked_cond_entropy, pixel_entropy
from .eval import evaluate_batch, evaluate_loader, plot_mask_grid, save_eval_report
from .masks import (
    MaskGeneratorMLP,
    SpatialMaskGenerator,
    SpatialMaskGeneratorMNIST,
    apply_masked_observation,
    conditioning_features,
    gumbel_mask,
    mask_loss,
)
from .schedule import build_pixel_survival_table, sparsity_to_timestep, cumulative_survival
from .train import build_dataloader, build_mask_model, load_d3pm, train_cifar10, train_mask_generator, train_mnist

__all__ = [
    "ITWConfig",
    "MNISTConfig",
    "CIFAR10Config",
    "pixel_entropy",
    "masked_cond_entropy",
    "d3pm_cond_entropy_loss",
    "MaskGeneratorMLP",
    "SpatialMaskGenerator",
    "SpatialMaskGeneratorMNIST",
    "gumbel_mask",
    "mask_loss",
    "apply_masked_observation",
    "conditioning_features",
    "build_pixel_survival_table",
    "sparsity_to_timestep",
    "cumulative_survival",
    "load_d3pm",
    "build_mask_model",
    "build_dataloader",
    "train_mask_generator",
    "train_mnist",
    "train_cifar10",
    "evaluate_batch",
    "evaluate_loader",
    "plot_mask_grid",
    "save_eval_report",
]
