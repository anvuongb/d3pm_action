# ITW + D3PM

Information-theoretic active sensing with frozen absorbing-state D3PM priors
([Austin et al., 2021](https://arxiv.org/abs/2107.03006)). MNIST/CIFAR-10 train
spatial pixel masks; FastMRI trains Cartesian **row** masks under a PE-line budget.

Upstream D3PM runners are based on [cloneofsimo/d3pm](https://github.com/cloneofsimo/d3pm).

## Setup

```bash
uv sync
```

CUDA is used when available; otherwise CPU (`itw.configs.default_device()`).

FastMRI data path (single-coil train `.h5` files):

```bash
export FASTMRI_ROOT=/path/to/singlecoil_train
```

Default if unset: `/home/anvuong/data/fast_mri/singlecoil_train`.

## Pretrain D3PM priors

```bash
# MNIST absorbing D3PM
uv run python d3pm_runner.py

# CIFAR-10
uv run python d3pm_runner_cifar10.py

# FastMRI fine (96x96) + coarse (row-profile) priors
uv run python d3pm_runner_fastmri.py --mode both --n-epochs 50
```

Checkpoints:

- MNIST: `models/model_absorb_cosine_*.pth`
- CIFAR-10: `models/cifar10/model_absorb_cosine_*.pth`
- FastMRI: `models_d3pm_fastmri_fine/` and `models_d3pm_fastmri_coarse/`

## Train ITW mask generators

```python
from itw import train_mnist, train_cifar10, train_fastmri, FastMRIConfig

train_mnist()
train_cifar10()

# Nested D3PM entropy proxies (primary FastMRI path)
train_fastmri(FastMRIConfig(mask_objective="nested_d3pm", n_epochs=30))

# InfoNCE baseline (separate save dir)
train_fastmri(FastMRIConfig(mask_objective="infonce", n_epochs=30))
```

Notebooks:

- `20260415_itw_mnist.ipynb`
- `20260416_itw_cifar10.ipynb`
- `20260710_itw_fastmri.ipynb`

Save dirs:

| Objective | Directory |
|-----------|-----------|
| Nested D3PM | `models_mask_gen_fastmri_nested/` |
| InfoNCE | `models_mask_gen_fastmri_infonce/` |

**Do not reuse** checkpoints or eval JSON under `models_mask_gen_fastmri/` from
before the survival-table / empty-mask / STE fixes. Retrain nested from scratch
so survival tables are rebuilt.

## Retrain nested FastMRI (after code fixes)

Success criteria:

| Metric | Pass |
|--------|------|
| Survival table range | subset of (0, 1] |
| `t(0.1) > t(0.4)` | true |
| `mean_sparsity_learned` at target 0.25 | in [0.20, 0.30] |
| `H_coarse` | > 0 when mask nonempty; empty ≈ log(8) |
| `H_fine` | varies with sparsity / epoch (not stuck) |
| Plot | visible PE lines; recon Y not all black |

Smoke (1–2 epochs): `dens` in the progress bar should track the sampled budget
(around 0.1–0.4), and `H_fine` should change if you evaluate at two sparsities.

```bash
uv run pytest tests/ -q
```

## Package layout

- `itw/` — mask generators, entropy proxies, FastMRI data, train/eval
- `d3pm_runner.py` — `D3PM` class + MNIST pretrain
- `d3pm_runner_fastmri.py` — FastMRI prior pretrain
- `docs/paper.md` — MNIST/CIFAR ITW write-up
