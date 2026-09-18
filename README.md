# ITW + D3PM

Information-theoretic active sensing with frozen absorbing-state D3PM priors
([Austin et al., 2021](https://arxiv.org/abs/2107.03006)). MNIST/CIFAR-10 train
spatial pixel masks; FastMRI trains Cartesian **row** masks under a PE-line budget.

**FastMRI results: see [`docs/fastmri_results.md`](docs/fastmri_results.md).** A
learned *static* row mask beats variable-density sampling by 1.5-3.8% NMSE on
held-out data under a reconstructor it was never optimised against; learned
*instance-adaptive* acquisition does not pay, and the D3PM entropy proxy does not
contribute to that result. That document supersedes the three staged FastMRI
docs, which are kept as development records behind supersession banners.

Upstream D3PM runners are based on [cloneofsimo/d3pm](https://github.com/cloneofsimo/d3pm).

## Setup

```bash
uv sync
```

CUDA is used when available; otherwise CPU (`itw.configs.default_device()`).

FastMRI data paths (single-coil `.h5` files). **Both splits are required** —
every reported number is measured on the held-out validation split:

```bash
export FASTMRI_ROOT=/path/to/singlecoil_train
export FASTMRI_VAL_ROOT=/path/to/singlecoil_val
```

Defaults if unset: `/home/anvuong/data/fast_mri/singlecoil_{train,val}`.

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

## FastMRI evaluation and mask search

Every script below is load-only with respect to the policy checkpoints: each
snapshots the protected artifacts by md5 and fails if any changed.

```bash
uv run python eval_fastmri_val.py            # held-out re-decide, budget-exact
uv run python eval_fastmri_adaptivity.py     # scout-derangement / static / adaptive
uv run python train_fastmri_recon_rerank.py  # train a U-Net, re-rank every mask
uv run python train_fastmri_static_profile.py  # 300-param profile + held-out judge U-Net
uv run python hill_climb_mask.py             # discrete row-swap search (best masks)
```

Evaluation rules these enforce, and which earlier results did not:

- masks are **budget-exact** top-k, so densities are comparable across methods;
- results are on **`singlecoil_val`**, with `seed_everything` before each pass;
- stochastic baselines are averaged over 5 seeds and a win inside 2 sd is not a
  win;
- masks optimised against one reconstructor are scored by an independently
  trained **judge** that took no part in the optimisation.

## Package layout

- `itw/` — mask generators, entropy proxies, FastMRI data, train/eval
- `itw/recon.py` — U-Net reconstructor (zero-filled residual, mask-conditioned)
- `itw/profile.py` — static row-profile parameterisation and optimiser
- `d3pm_runner.py` — `D3PM` class + MNIST pretrain
- `d3pm_runner_fastmri.py` — FastMRI prior pretrain

## Docs

| Document | Contents |
|----------|----------|
| [`docs/fastmri_results.md`](docs/fastmri_results.md) | **Corrected FastMRI results.** Start here. |
| [`docs/fastmri_eval_integrity.md`](docs/fastmri_eval_integrity.md) | Full audit trail: findings F1-F9, stages B0-B6, decision rules |
| `docs/fastmri_scratch_protocol.md` | Superseded — from-scratch prior/policy protocol (P1-P4) |
| `docs/fastmri_policy_improvements.md` | Superseded — six-stage baseline/ablation run |
| `docs/fastmri_acs_remainder.md` | Superseded — ACS-locked remainder policy (A1-A4) |
| `docs/paper.md` | MNIST/CIFAR ITW write-up (predates the FastMRI arm) |

Note on metrics: `itw.discrete.nmse` previously carried a `1e-8` absolute floor
on the denominator, which was active on ~2/3 of FastMRI slices and read ~4x low.
It is fixed and guarded by tests; any NMSE quoted in the superseded docs above
is on the old scale.
