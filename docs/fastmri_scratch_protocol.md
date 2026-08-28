# FastMRI from-scratch protocol (A4 winning recipe)

From-scratch priors, then a 50-epoch ACS-lock policy. Do **not** overwrite:

- `models_d3pm_fastmri_coarse_kspace/`
- `models_d3pm_fastmri_fine/`
- `models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth`
- `models_mask_gen_fastmri_nested/acs_lock_10ep/`

```
P1 coarse D3PM 100ep → P2 fine D3PM 100ep → P3 policy 50ep ACS-lock → P4 analysis at s=0.10,0.25,0.50,0.75
```

Stop after each stage: eval, report numbers, then continue.

## Winning recipe (policy = P3)

A4 winner, trained from scratch on new 100ep priors:

- `policy_input="scout_image"` (image-\(Z\) cond; not ACS-kspace upsample)
- `acs_lock=True` (Cartesian row Gumbel + ACS hard-1 / centered-block remainder)
- \(\alpha=1\), **\(\beta=0\)**, recon \(=1\) (nested \(H_c\) + NMSE)
- Sparsity train range **0.10–0.75** so eval at 10/25/50/75% is in-distribution
- Policy from scratch after new priors (no warm-start from `acs_lock_10ep/`)

## Discretizations

- **Coarse** disc = k-space PE-row energy (`kspace_to_row_disc`), 8 bins (current working coarse).
- **Fine** = \(96\times 96\) 8-bin magnitude. Trained 100ep as asked; **policy keeps \(\beta=0\)**. P2 CE vs \(t\) did not overturn A5 (flat \(\approx 0.45\) on \(t=1\)–\(757\); only \(t=999\) rises).

## Policy (P3)

Cartesian row Gumbel + ACS lock, image-\(Z\) cond, nested \(H_c\) + NMSE.

At \(s=0.10\), \(k=30<32\) ACS lock = centered 30-line block (learned \(\approx\) ACS factories). Still eval it.

## New save dirs

| Stage | Path |
|-------|------|
| P1 coarse | `models_d3pm_fastmri_coarse_kspace_100ep/` |
| P2 fine | `models_d3pm_fastmri_fine_100ep/` |
| P3 policy | `models_mask_gen_fastmri_nested/protocol_50ep_acs_lock/` |

## Status

- [x] P1: coarse D3PM 100ep (`kspace_to_row_disc`, 8 bins) → brief CE eval
- [x] P2: fine D3PM 100ep (\(96\times 96\) mag) → brief CE eval; **keep \(\beta=0\)** (A5 not overturned)
- [x] P3: policy 50ep ACS-lock, image-\(Z\), \(\alpha=1\), \(\beta=0\), recon=1, \(s\in[0.10,0.75]\)
- [x] P4: analysis at \(s=0.10,0.25,0.50,0.75\) vs ACS+random / ACS+VD / ACS+equispaced

---

## P1 — Coarse D3PM 100ep

From scratch (do **not** load the 50ep ckpt). CLI: `--coarse-save-dir`.

```
uv run python d3pm_runner_fastmri.py --mode coarse --n-epochs 100 --save-every 10 --coarse-save-dir models_d3pm_fastmri_coarse_kspace_100ep --batch-size 8 --num-workers 4
```

Data: `FASTMRI_ROOT` or `/home/anvuong/data/fast_mri/singlecoil_train/`.

Protect: old `models_d3pm_fastmri_coarse_kspace/model_absorb_cosine_final.pth` must keep mtime/md5.

### P1 results

**Train:** 100 epochs, from scratch, cuda. `mode=coarse` params=85704, 121 batches/epoch, batch size 8. Wall time **153 s** (~2.6 min). Script: `d3pm_runner_fastmri.py`. Disc: `kspace_to_row_disc`, 8 bins.

**New ckpt:** `models_d3pm_fastmri_coarse_kspace_100ep/model_absorb_cosine_final.pth` (md5 `c9caf510…`, mtime 01:45). Intermediate saves every 10 epochs (`model_absorb_cosine_{9,19,…,99}.pth`). Log: `models_d3pm_fastmri_coarse_kspace_100ep/train.log`.

**Old 50ep intact:** `models_d3pm_fastmri_coarse_kspace/model_absorb_cosine_final.pth` md5 `8e8db4df35a6cae2afe7094dda00b387`, mtime 2026-08-28 00:07:58 (unchanged). Fine / nested parent / `acs_lock_10ep/` also untouched.

Last-batch CE is noisy (one batch of 8). Loss EMA is the stable signal. Extra 50 epochs did **not** clearly beat the 50ep plateau (~0.046 EMA).

| run | epoch | loss EMA | CE (last batch) |
|-----|-------|----------|-----------------|
| old 50ep | 0 | 0.903 | 0.140 |
| old 50ep | 49 | **0.046** | 0.075 |
| new 100ep | 0 | 0.825 | 0.229 |
| new 100ep | 49 | 0.046 | 0.127 |
| new 100ep | 99 | 0.051 | **0.043** |

Cheap CE vs \(t\) on 2 held-order batches (`eval/p1_ce_vs_t.json`). Healthy: near-zero at small \(t\), rises toward absorb.

| \(t\) | CE |
|-------|-----|
| 1 | 0.0005 |
| 100 | 0.0027 |
| 500 | 0.0265 |
| 999 | 0.5507 |

**Do not start P2 until this report is accepted.**

---

## P2 — Fine D3PM 100ep

From scratch (do **not** load the 50ep ckpt). CLI: `--fine-save-dir`.

```
uv run python d3pm_runner_fastmri.py --mode fine --n-epochs 100 --save-every 10 --fine-save-dir models_d3pm_fastmri_fine_100ep --batch-size 8 --num-workers 4
```

Protect: old `models_d3pm_fastmri_fine/model_absorb_cosine_final.pth` must keep mtime/md5.

### P2 results

**Train:** 100 epochs, from scratch, cuda. `mode=fine` params=63278176, 121 batches/epoch, batch size 8. Wall time **899 s** (~15.0 min). Script: `d3pm_runner_fastmri.py`. Disc: `magnitude_to_fine_disc`, \(96\times 96\), 8 bins.

**New ckpt:** `models_d3pm_fastmri_fine_100ep/model_absorb_cosine_final.pth` (md5 `97e1d753…`, mtime 02:02). Intermediate saves every 10 epochs (`model_absorb_cosine_{9,19,…,99}.pth`). Log: `models_d3pm_fastmri_fine_100ep/train.log`.

**Old 50ep intact:** `models_d3pm_fastmri_fine/model_absorb_cosine_final.pth` md5 `cb096461dddbd33239459b9ff5311c0b`, mtime 2026-08-27 23:36:26 (unchanged). Coarse 50ep / nested parent / `acs_lock_10ep/` also untouched.

Last-batch CE is noisy (one batch of 8). Loss EMA is the stable signal. Extra 50 epochs slowly lowered EMA (unlike coarse, which plateaued). Old 50ep log has two concatenated 50ep runs; the second ends ~0.60 (the “flat ~0.60” prior).

| run | epoch | loss EMA | CE (last batch) |
|-----|-------|----------|-----------------|
| old 50ep (2nd run) | 0 | 1.361 | 1.030 |
| old 50ep (2nd run) | 49 | **0.601** | 0.586 |
| new 100ep | 0 | 1.551 | 1.084 |
| new 100ep | 49 | 0.596 | 0.547 |
| new 100ep | 99 | **0.478** | 0.464 |

Cheap CE vs \(t\) on 4 held-order batches (`eval/p2_ce_vs_t.json`). **Not healthy as a denoiser:** CE is flat \(\approx 0.45\) from \(t=1\) through the nested operating point \(t=757\) (\(s=0.25\)); only \(t=999\) (absorb) rises. Old 50ep Stage-5 CE was the same shape at \(\approx 0.61\) then \(0.97\) at \(t=999\). Extra epochs lowered the floor but did **not** create \(t\)-dependence on the operating range.

| \(t\) | CE (new 100ep) | CE (old 50ep Stage 5) |
|-------|----------------|------------------------|
| 1 | 0.455 | 0.614 |
| 100 | 0.448 | 0.614 |
| 500 | 0.448 | 0.612 |
| 757 | 0.456 | 0.614 |
| 999 | 0.960 | 0.967 |

**P3 \(\beta\) go/no-go: keep \(\beta=0\).** A5 is not overturned. CE does not move with \(t\) on the operating range (only at absorb \(t=999\)). Frozen fine prior remains not mask-sensitive; do not turn on \(H_\text{fine}\).

**Do not start P3 until this report is accepted.**

---

## P3 — Policy 50ep ACS-lock

From scratch after new priors (do **not** load `acs_lock_10ep/` or parent `mask_gen_fastmri_final.pth`). Script: `train_fastmri_protocol_50ep.py`.

```
uv run python train_fastmri_protocol_50ep.py
```

Protect: parent nested final, `acs_lock_10ep/`, old `models_d3pm_fastmri_coarse_kspace/`, old `models_d3pm_fastmri_fine/`. Rebuild coarse survival with `build_row_survival_table` on the **new** 100ep coarse; do not copy `models_mask_gen_fastmri_nested/coarse_survival_table.pt`. Fine survival = dummy linspace (\(\beta=0\)).

### P3 results

**Train:** 50 epochs, from scratch, cuda. ACS-lock STE confirmed (`gumbel_row_mask_ste_acs_locked`, ACS `[134:166]` hard-1). Recipe: image-\(Z\) `scout_image`, `acs_lock=True`, \(\alpha=1\), \(\beta=0\), recon \(=1\), \(s\in[0.10,0.75]\), `sparsity_loss_weight=50`. Coarse prior: `models_d3pm_fastmri_coarse_kspace_100ep/model_absorb_cosine_final.pth`. Wall time **82 s**. **No density collapse** (`collapsed: false`; density stayed \(\approx 0.37\)–\(0.47\)).

**New ckpt:** `models_mask_gen_fastmri_nested/protocol_50ep_acs_lock/mask_gen_fastmri_final.pth` (md5 `453fb236…`, mtime 02:07). Intermediate saves every 5 epochs (`mask_gen_fastmri_{4,9,…,49}e.pth`). Log: `protocol_50ep_acs_lock/train.log`. Status: 50/50 epochs, `entropy_beta=0.0`.

**Survival:** coarse rebuilt with `build_row_survival_table` on the 100ep coarse (md5 `61a3f11d…`, **not** equal to old 50ep table `880a7c2b…`; max abs diff \(0.027\)). Fine = dummy `linspace(1, 0.01, 1000)`. Calibration warned that the 100ep coarse table is not strictly monotone decreasing in \(t\).

**Old artifacts intact:**

| path | md5 | mtime |
|------|-----|-------|
| parent nested `mask_gen_fastmri_final.pth` | `508742a8934d6795ffbdaa759a8362e1` | 2026-08-28 00:10:20 |
| `acs_lock_10ep/mask_gen_fastmri_final.pth` | `ee56a0b65fb9a524bb1e0f2ed58e46d4` | 2026-08-28 01:34:12 |
| old 50ep coarse final | `8e8db4df35a6cae2afe7094dda00b387` | 2026-08-28 00:07:58 |
| old 50ep fine final | `cb096461dddbd33239459b9ff5311c0b` | 2026-08-27 23:36:26 |

Train density / \(H_c\) / NMSE (mean over the epoch; density is EMA). Train NMSE is over random \(s\in[0.10,0.75]\), not a fixed eval \(s\).

| epoch | density | \(H_c\) | NMSE | loss EMA |
|-------|---------|---------|------|----------|
| 0 | 0.466 | 0.255 | 0.0063 | 1.671 |
| 9 | 0.372 | 0.177 | 0.0060 | 0.209 |
| 24 | 0.383 | 0.144 | 0.0055 | 0.169 |
| 49 | **0.417** | **0.124** | **0.0054** | **0.160** |

Optional 5-batch sanity at \(s=0.25\) only (`eval/sanity_s025.json`; **not** the P4 10/25/50/75 grid):

| method | density | \(H_c\) | NMSE | SSIM | PSNR |
|--------|---------|---------|------|------|------|
| learned (ACS-lock Gumbel) | 0.255 | **0.084** | **0.0079** | **0.758** | **31.75** |
| acs_random | 0.250 | 0.239 | 0.0098 | 0.745 | 30.84 |

Learned beats ACS+random on this 5-batch \(s=0.25\) slice (NMSE \(0.0079\) vs \(0.0098\); \(H_c\) \(0.084\) vs \(0.239\)). Numbers are a sanity check, not the P4 table.

**Do not start P4 until this report is accepted.**

---

## P4 — Analysis

Eval at \(s=0.10,0.25,0.50,0.75\). At \(s=0.10\) learned \(\approx\) ACS factories (centered 30-line block); still report it.

**Load only (no train).** Script: `eval_fastmri_protocol_50ep.py`. Policy `protocol_50ep_acs_lock/mask_gen_fastmri_final.pth`, 100ep coarse, P3-rebuilt `coarse_survival_table.pt`, dummy fine linspace (\(\beta=0\)). Config: `acs_lock=True`, `policy_input="scout_image"`, \(\alpha=1\), \(\beta=0\), recon \(=1\), `scout_size=32`, `image_size=300`. `learned` = ACS-locked Gumbel (train-consistent); `learned_acs_lock` = ACS + top-\(k\) from the same logits.

JSON: `models_mask_gen_fastmri_nested/protocol_50ep_acs_lock/eval/eval_p4.json` (20 batches). Curve: `eval/nmse_ssim_hc_vs_s.png`. Honest `plot_fastmri_grid` (kspace clim) at each \(s\): `learned_s0{10,25,50,75}.png`, `acs_random_s0*.png`, `acs_vd_gaussian_s0*.png`, `random_s0*.png`.

**Protected artifacts unchanged** (md5/mtime): P3 final `453fb236…`, P3 coarse survival `61a3f11d…`, P3 `sanity_s025.json`, parent nested `508742a8…`, `acs_lock_10ep/` `ee56a0b6…`, old 50ep coarse `8e8db4df…`, old 50ep fine `cb096461…`, 100ep coarse `c9caf510…`, 100ep fine `97e1d753…`. Intermediate P3 epoch ckpts and `train_status.json` / `train.log` also untouched.

**\(s=0.10\) sanity (\(k=30<32\)):** learned Gumbel, top-\(k\), `acs_random`, `acs_vd_gaussian`, and `acs_equispaced` are **identical** (centered 30-line block). NMSE \(0.0132\), \(H_c\) \(0.052\), SSIM \(0.668\), PSNR \(29.84\). Pass. Unconstrained random/equispaced/VD remain \(\sim 0.33\)–\(0.42\) NMSE.

**\(s=0.25\) vs P3 5-batch sanity and A2 10ep:** P4 20-batch learned NMSE **0.0081** / SSIM 0.745 / \(H_c\) 0.079 vs P3 sanity **0.0079** / 0.758 / 0.084 (5 batches; 20-batch numbers differ slightly) vs A2 10ep **0.0071** / 0.764 / 0.080 (10 batches). **50ep + 100ep priors do not beat A2 10ep ACS-lock** at the overlapping \(s\). Learned still beats ACS+random (\(0.0101\)) and ACS+VD (\(0.0085\)) on this grid.

Primary \(s\) (density / \(H_c\) / NMSE / SSIM / PSNR), 20 batches:

| \(s\) | method | density | \(H_c\) | NMSE | SSIM | PSNR |
|------|--------|---------|---------|------|------|------|
| 0.10 | learned | 0.100 | 0.052 | **0.0132** | **0.668** | **29.84** |
| 0.10 | learned_acs_lock | 0.100 | 0.052 | 0.0132 | 0.668 | 29.84 |
| 0.10 | acs_random | 0.100 | 0.052 | 0.0132 | 0.668 | 29.84 |
| 0.10 | acs_vd_gaussian | 0.100 | 0.052 | 0.0132 | 0.668 | 29.84 |
| 0.10 | acs_equispaced | 0.100 | 0.052 | 0.0132 | 0.668 | 29.84 |
| 0.10 | random | 0.100 | 0.351 | 0.4149 | 0.380 | 16.58 |
| 0.10 | equispaced | 0.100 | 0.234 | 0.4169 | 0.370 | 16.40 |
| 0.10 | vd_gaussian | 0.100 | 0.253 | 0.3315 | 0.401 | 17.62 |
| 0.25 | learned | 0.257 | 0.079 | **0.0081** | **0.745** | **31.38** |
| 0.25 | learned_acs_lock | 0.250 | **0.070** | 0.0091 | 0.740 | 31.02 |
| 0.25 | acs_random | 0.250 | 0.235 | 0.0101 | 0.732 | 30.48 |
| 0.25 | acs_vd_gaussian | 0.250 | 0.177 | 0.0085 | 0.738 | 31.18 |
| 0.25 | acs_equispaced | 0.250 | 0.116 | 0.0101 | 0.730 | 30.47 |
| 0.25 | random | 0.250 | 0.366 | 0.2962 | 0.419 | 17.58 |
| 0.25 | equispaced | 0.250 | 0.289 | 0.2948 | 0.411 | 17.49 |
| 0.25 | vd_gaussian | 0.250 | 0.213 | 0.1484 | 0.520 | 21.40 |
| 0.50 | learned | 0.478 | 0.115 | 0.0036 | 0.856 | 35.13 |
| 0.50 | learned_acs_lock | 0.500 | 0.112 | **0.0034** | **0.866** | **35.37** |
| 0.50 | acs_random | 0.500 | 0.242 | 0.0055 | 0.829 | 33.35 |
| 0.50 | acs_vd_gaussian | 0.500 | 0.177 | 0.0035 | 0.857 | 35.25 |
| 0.50 | acs_equispaced | 0.500 | **0.073** | 0.0055 | 0.819 | 33.30 |
| 0.50 | random | 0.500 | 0.300 | 0.1867 | 0.555 | 20.70 |
| 0.50 | equispaced | 0.500 | 0.263 | 0.2161 | 0.562 | 19.24 |
| 0.50 | vd_gaussian | 0.500 | 0.180 | 0.0179 | 0.808 | 32.42 |
| 0.75 | learned | 0.732 | 0.185 | 0.0018 | 0.917 | 37.87 |
| 0.75 | learned_acs_lock | 0.750 | 0.387 | 0.0017 | 0.930 | 38.31 |
| 0.75 | acs_random | 0.750 | 0.213 | 0.0026 | 0.900 | 36.32 |
| 0.75 | acs_vd_gaussian | 0.750 | 0.341 | **0.0016** | **0.928** | **38.47** |
| 0.75 | acs_equispaced | 0.750 | **0.056** | 0.0025 | 0.895 | 36.43 |
| 0.75 | random | 0.750 | 0.240 | 0.0868 | 0.698 | 25.12 |
| 0.75 | equispaced | 0.750 | 0.061 | 0.0926 | 0.717 | 24.01 |
| 0.75 | vd_gaussian | 0.750 | 0.338 | 0.0016 | 0.928 | 38.46 |

**ACS lock at 50/75%?** Yes vs unconstrained random/equispaced: at \(s=0.50\) ACS cluster NMSE \(0.0034\)–\(0.0055\) vs random \(0.187\); at \(s=0.75\) ACS cluster \(0.0016\)–\(0.0026\) vs random \(0.087\). At 75% unconstrained VD joins the ACS cluster (\(0.0016\)) because the budget covers the center. Remainder quality still shows in \(H_c\) (learned \(0.185\) vs `acs_equispaced` \(0.056\) vs `acs_vd` \(0.341\)).

**Conclusion.** From-scratch 50ep ACS-lock on 100ep priors is a working ACS-locked policy: \(s=0.10\) matches ACS factories; learned Gumbel beats ACS+random at \(0.25/0.50/0.75\) and is tied with ACS+VD at \(0.50\). It does **not** beat A2 10ep warm-start ACS-lock at overlapping \(s=0.25\) (\(0.0081\) vs \(0.0071\)). ACS lock remains the recon constraint that matters; extra prior/policy epochs did not overturn that. Ship A2 `acs_lock_10ep` for the overlapping operating point; this P3/P4 run is the from-scratch confirmation at the wider \(s\) grid.
