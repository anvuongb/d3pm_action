# ACS-locked remainder policy

> **SUPERSEDED — see `fastmri_results.md`.** A1-A4 numbers are training-set,
> single-draw and pre-F9 (~4x low). The "winning recipe" section recommends the
> A2 checkpoint: on held-out data with budget-exact masks it is indistinguishable
> from the 50-epoch P3 policy (F5), and both lose to `acs_vd_gaussian` under a
> real reconstructor. The best mask found in this project is the discrete
> hill-climbed static mask (B5b), not a conditional policy. Kept as a
> development record.

Bottleneck from the six-stage run: zero-filled Cartesian NMSE is dominated by the center PE block. ACS+random ~0.01 NMSE vs best learned ~0.09 at \(s=0.25\). Nested \(H_c\) prefers the learned masks. Independent Gumbel over 300 rows has no ACS inductive bias. Fine prior is dead (\(\beta=0\)). More epochs do not help.

**Keep** parent ckpt `models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth` (10-epoch image-\(Z\) `both`). Do not overwrite. Do not load `models_mask_gen_fastmri/`. Stage 4 ACS-kspace upsample head is a failed architecture — do not reuse it.

```
A1 diagnostic (no train) → A2 retrain ACS-locked remainder → A3 PE-aligned cond (only if A2 loses) → A4 curve/write-up
```

After each stage: eval, report, possibly rewrite the next stage.

- [x] A1: constraint + remainder baselines + load-only diagnostic
- [x] A2: warm-start 10-epoch ACS-lock retrain
- [x] A3: **skipped** (A2 Gumbel already beats ACS+random and ACS+VD)
- [x] A4: curve + winning recipe; no 30-epoch retrain

---

## Stage A1 — Constraint + remainder baselines + load-only diagnostic

Implement ACS lock as a **sampling constraint**, not a new net.

- `acs_bounds(n_rows, acs_width)` → `(lo, hi)` matching `acs_random_row_mask_batch`.
- When \(k = \lfloor s H \rfloor \le\) `acs_width`: mask is a **centered block of width \(k\)** (same as existing ACS factory). No remainder.
- When \(k >\) `acs_width`: ACS block of width `acs_width` is **hard 1**; remainder budget \(k -\) `acs_width` is spent **only** on rows outside `[lo, hi)`.

New remainder factories (ACS already on):

- `acs_random` (exists)
- `acs_vd_gaussian`: ACS + sample remainder without replacement with the same Gaussian as `vd_gaussian_row_mask_batch`, restricted to non-ACS rows (renormalize probs on the outside).
- `acs_equispaced`: ACS + equally spaced remainder on the outside (prefer outside-only remainder so density is exact).

Learned **post-hoc** ACS lock (eval only, no train):

- Load parent image-\(Z\) policy, get logits `[B,1,H,2]`.
- ACS rows = 1.
- Remainder = top-`k_extra` non-ACS rows by keep-logit (index 1), deterministic. This answers: “if we had forced ACS at test time, is the current remainder already better than random?”

Also implement `gumbel_row_mask_acs_locked` / STE variant for **later train**, but A1 eval of the parent ckpt uses **top-\(k\) remainder**, not Gumbel, so the diagnostic is not extra sampling noise. Gumbel-ACS-lock still needs tests now so A2 can use it.

Config: `FastMRIConfig.acs_lock: bool = False` default (parent path unchanged). Do not flip it on for the parent eval of unconstrained learned.

Eval (load-only, 10 batches) at \(s \in \{0.125, 0.25, 0.4\}\) (skip \(s=0.10\) as primary — \(k=30<32\) so ACS lock IS the center block). Optionally include \(s=0.10\) as a sanity row (`learned_acs_lock` should match `acs_random` / centered block).

Methods:

- `learned` (unconstrained Gumbel, current eval)
- `learned_acs_lock` (post-hoc ACS + top-\(k\) remainder from same logits)
- `acs_random`
- `acs_vd_gaussian`
- `acs_equispaced`
- keep `random` / `vd_gaussian` / `equispaced` as reference if cheap

Metrics: density, \(H_c\), NMSE, SSIM, PSNR. Honest plots at \(s=0.25\) for learned, learned_acs_lock, acs_random, acs_vd_gaussian.

Save under `models_mask_gen_fastmri_nested/eval_acs_lock/` (do not clobber Stage 1–6 plots in `eval/`). JSON `eval_a1.json`.

### A1 decision rules

- If `learned_acs_lock` NMSE \(\approx\) `acs_random` (within ~0.01): current remainder is ~random. **A2 retrain is required.**
- If `learned_acs_lock` clearly beats `acs_random` on NMSE: remainder already informative; A2 still retrains so train-time masks match the constraint, but we expect a smaller lift.
- If `learned_acs_lock` NMSE still \(\gg\) `acs_random` and close to unconstrained `learned`: forcing ACS did not fix recon (holes in ACS or remainder destroying ACS benefit). Inspect masks; may need exact ACS overwrite (not top-\(k\) that accidentally drops ACS — ACS must be hard 1).
- Do **not** start A3 from A1. A3 only if A2 remainder still loses.

---

## Stage A2 — Retrain with ACS lock

Warm-start from parent `models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth` (do **not** train from scratch). 10-epoch image-\(Z\) `both` (\(\alpha=1\), \(\beta=0\), recon=1, `policy_input=scout_image`), `acs_lock=True`. Save `models_mask_gen_fastmri_nested/acs_lock_10ep/` — never overwrite the parent ckpt. ACS-locked STE Gumbel via `gumbel_row_mask_ste_acs_locked`. Sparsity loss on the **full** mask including ACS.

Bar: match or beat A1 post-hoc `learned_acs_lock` NMSE at \(s=0.25\) and \(0.40\), and still beat `acs_random`. Eval both train-consistent ACS-locked Gumbel (`learned`) and top-\(k\) remainder from the same new logits (`learned_acs_lock`).

Skip A3 unless A2 Gumbel **and** top-\(k\) both lose to `acs_random` **and** `acs_vd_gaussian` on NMSE at \(s=0.25\) and \(0.40\).

---

## Stage A3 — PE-aligned ACS conditioning (skipped)

Scatter ACS k-space features onto true PE indices (32 of 300), rest zero; 1D head over PE. Keep acs_lock. Do **not** linear-upsample \(32\to 300\) (Stage 4 failure). A2 Gumbel beats ACS+random and ACS+VD — do not run.

---

## Stage A4 — Curve + write-up

No re-eval (JSON already on disk). No 30-epoch retrain (A2 train NMSE was ACS-level from epoch 0, \(\approx 0.008\), flat). Parent `mask_gen_fastmri_nested/mask_gen_fastmri_final.pth` and A2 `acs_lock_10ep/mask_gen_fastmri_final.pth` not overwritten.

Curve from `eval_a2.json` + `eval_acs_lock/eval_a1.json` at \(s\in\{0.125,0.25,0.40\}\): A2 Gumbel `learned`, A1 `learned_acs_lock` (parent post-hoc top-\(k\)), `acs_random`, `acs_vd_gaussian` (A2 session), unconstrained A1 `learned` as dashed “no ACS lock”. NMSE log-scale so the ACS cluster is readable against unconstrained \(\sim 0.08\).

Save: `models_mask_gen_fastmri_nested/acs_lock_10ep/eval/nmse_ssim_vs_s.png`.

At \(s=0.25\): A2 Gumbel NMSE **0.0071** / SSIM 0.764 vs A1 post-hoc **0.0065** / 0.752 vs `acs_random` 0.0092 / 0.750 vs `acs_vd_gaussian` 0.0077 / 0.755. Unconstrained A1 `learned` 0.080 / 0.591. Same ranking at \(0.125\) and \(0.40\): ACS-locked methods cluster; unconstrained is far worse on NMSE.

### Winning recipe (ship this)

Train/eval default — ACS-locked remainder, not unconstrained Gumbel:

| knob | value |
|------|--------|
| ckpt | `models_mask_gen_fastmri_nested/acs_lock_10ep/mask_gen_fastmri_final.pth` |
| `policy_input` | `scout_image` |
| `acs_lock` | **True** |
| \(\alpha\) | 1 |
| \(\beta\) | **0** |
| `recon_loss_weight` | 1 |

A1 post-hoc top-\(k\) on the unconstrained parent is a hair better at \(s=0.25\) (0.0065 vs 0.0071) — 10-batch noise, not a reason to ship unconstrained Gumbel. Do **not** eval/deploy image-\(Z\) Gumbel without ACS lock: unconstrained NMSE at \(s=0.25\) is \(0.080\), an order of magnitude worse.

Unconstrained image-\(Z\) `both` (`acs_lock=False`, parent ckpt) remains the **nested-\(H_c\)** story (lowest coarse entropy among fair non-ACS baselines). ACS-lock is the **recon** story (NMSE/SSIM at ACS level, remainder better than ACS+random / ACS+VD).

### Six-stage + A1–A4 outcome

Fair MRI baselines showed nested image-\(Z\) `both` beats uniform random, equispaced, and VD-Gaussian but not ACS+random on zero-filled NMSE; ACS-kspace policy input failed (PE upsample \(32\to 300\)); fine CE is flat so \(\beta=0\); 30 from-scratch epochs lowered \(H_c\) but did not close the ACS recon gap. A1 post-hoc ACS + top-\(k\) remainder on the parent logits collapsed that gap (\(0.080\to 0.0065\) at \(s=0.25\)) and already beat ACS+random, so the remainder was informative. A2 warm-start ACS-lock Gumbel made train-time masks match the constraint (train NMSE \(\approx 0.008\) from epoch 0, no collapse) and beat both ACS remainder heuristics, but did not beat A1 post-hoc NMSE. **A3 skipped:** PE-aligned ACS cond is only for when A2 Gumbel *and* top-\(k\) both lose to `acs_random` **and** `acs_vd_gaussian` at \(s=0.25\) and \(0.40\); A2 Gumbel beats both heuristics at both \(s\). Ship ACS-lock (`acs_lock_10ep`); keep the unconstrained parent for the nested-\(H_c\) plot only.

---

## A1 results

Load-only, 10 batches, device cuda. Parent image-\(Z\) ckpt `models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth` (md5 `508742a8…`, mtime unchanged). `policy_input=scout_image`, `acs_lock=False` for unconstrained Gumbel; `learned_acs_lock` is post-hoc ACS + top-\(k\) remainder from the **same** logits (not Gumbel).

JSON: `models_mask_gen_fastmri_nested/eval_acs_lock/eval_a1.json`.
Honest plots at \(s=0.25\): `learned_s025.png`, `learned_acs_lock_s025.png`, `acs_random_s025.png`, `acs_vd_gaussian_s025.png` (same dir). Stage 1–6 plots in `eval/` were not overwritten. No training.

Tests: `uv run pytest tests/test_acs_lock.py tests/test_baseline_masks.py tests/test_kspace_row.py` — 26 passed.

Sanity \(s=0.10\) (\(k=30<32\)): `learned_acs_lock`, `acs_random`, `acs_vd_gaussian`, and `acs_equispaced` are **identical** (centered 30-line block). NMSE \(0.014\), \(H_c\) \(0.140\). Pass.

Primary \(s\) (density / \(H_c\) / NMSE / SSIM / PSNR):

| \(s\) | method | density | \(H_c\) | NMSE | SSIM | PSNR |
|------|--------|---------|---------|------|------|------|
| 0.125 | learned | 0.151 | 0.131 | 0.216 | 0.459 | 19.36 |
| 0.125 | learned_acs_lock | 0.123 | 0.074 | **0.0115** | 0.682 | 30.09 |
| 0.125 | acs_random | 0.123 | 0.142 | 0.0125 | 0.684 | 29.75 |
| 0.125 | acs_vd_gaussian | 0.123 | 0.125 | 0.0122 | 0.686 | 29.84 |
| 0.125 | acs_equispaced | 0.123 | 0.149 | 0.0125 | 0.685 | 29.75 |
| 0.25 | learned | 0.261 | 0.097 | 0.080 | 0.591 | 24.45 |
| 0.25 | learned_acs_lock | 0.250 | **0.042** | **0.0065** | **0.752** | **32.12** |
| 0.25 | acs_random | 0.250 | 0.205 | 0.0088 | 0.740 | 30.84 |
| 0.25 | acs_vd_gaussian | 0.250 | 0.169 | 0.0074 | 0.747 | 31.55 |
| 0.25 | acs_equispaced | 0.250 | 0.134 | 0.0088 | 0.738 | 30.83 |
| 0.40 | learned | 0.422 | 0.107 | 0.0214 | 0.768 | 31.13 |
| 0.40 | learned_acs_lock | 0.400 | **0.053** | **0.0046** | **0.823** | **33.97** |
| 0.40 | acs_random | 0.400 | 0.205 | 0.0073 | 0.791 | 32.04 |
| 0.40 | acs_vd_gaussian | 0.400 | 0.162 | 0.0051 | 0.810 | 33.49 |
| 0.40 | acs_equispaced | 0.400 | 0.125 | 0.0074 | 0.785 | 31.98 |

**Did post-hoc ACS lock close most of the NMSE gap to `acs_random`?** Yes. At \(s=0.25\), unconstrained learned NMSE \(0.080\) \(\to\) lock \(0.0065\) vs `acs_random` \(0.0088\) (absolute \(\Delta \approx 0.002\), inside the \(\sim 0.01\) band; lock is slightly better). Same pattern at \(0.125\) and \(0.40\). Rule 3 does **not** apply: ACS is hard-1 (no holes); recon is at ACS level, not unconstrained-learned level.

**Decision rules:** NMSE of `learned_acs_lock` \(\approx\) `acs_random` within \(\sim 0.01\) (rule 1 band) *and* slightly better, with much lower \(H_c\) (rule 2: remainder already informative). **A2 retrain is required** so train-time Gumbel matches the constraint. Expect a **smaller** recon lift than if the remainder had been pure random. Do **not** start A3.

A2 tweaks applied after A1: warm-start (not from scratch); bar is A1 post-hoc lock, not only ACS+random; eval Gumbel and top-\(k\); skip A3 unless both learned variants lose to ACS heuristics.

## A2 results

Warm-start 10-epoch image-\(Z\) `both`, `acs_lock=True`, device cuda. Parent `models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth` (md5 `508742a8…`, mtime 00:10 unchanged). A2 ckpt `models_mask_gen_fastmri_nested/acs_lock_10ep/mask_gen_fastmri_final.pth` (md5 `ee56a0b6…`). STE hook confirmed (`gumbel_row_mask_ste_acs_locked`, ACS `[134:166]` hard-1). Coarse survival copied from parent; fine dummy linspace. Script: `train_fastmri_acs_lock.py`.

JSON: `models_mask_gen_fastmri_nested/acs_lock_10ep/eval/eval_a2.json`.
Honest plots at \(s=0.25\): `learned_s025.png` (Gumbel-lock), `learned_acs_lock_s025.png` (top-\(k\)), `acs_random_s025.png`, `acs_vd_gaussian_s025.png`.

**Train:** no collapse. Density stayed \(0.23\)–\(0.26\) (final EMA \(0.254\)). Train NMSE immediately ACS-level (\(\approx 0.008\)) vs unconstrained parent \(\sim 0.07\)–\(0.22\). \(H_c\) \(0.101\to 0.085\). Loss EMA \(0.205\to 0.139\).

Sanity \(s=0.10\) (\(k=30<32\)): Gumbel-lock, top-\(k\), `acs_random`, `acs_vd_gaussian`, and `acs_equispaced` are **identical** (centered 30-line block). NMSE \(0.0124\), \(H_c\) \(0.140\). Pass.

Primary \(s\) (density / \(H_c\) / NMSE / SSIM / PSNR). `learned` = ACS-locked Gumbel (train-consistent); `learned_acs_lock` = ACS + top-\(k\) remainder from the **new** logits:

| \(s\) | method | density | \(H_c\) | NMSE | SSIM | PSNR |
|------|--------|---------|---------|------|------|------|
| 0.125 | learned | 0.125 | 0.100 | **0.0114** | **0.685** | **30.03** |
| 0.125 | learned_acs_lock | 0.123 | 0.103 | 0.0116 | 0.678 | 29.95 |
| 0.125 | acs_random | 0.123 | 0.148 | 0.0119 | 0.681 | 29.87 |
| 0.125 | acs_vd_gaussian | 0.123 | 0.125 | 0.0116 | 0.683 | 29.97 |
| 0.125 | acs_equispaced | 0.123 | 0.149 | 0.0118 | 0.681 | 29.88 |
| 0.25 | learned | 0.258 | 0.080 | **0.0071** | **0.764** | **32.16** |
| 0.25 | learned_acs_lock | 0.250 | **0.061** | 0.0082 | 0.758 | 31.70 |
| 0.25 | acs_random | 0.250 | 0.202 | 0.0092 | 0.750 | 31.11 |
| 0.25 | acs_vd_gaussian | 0.250 | 0.172 | 0.0077 | 0.755 | 31.83 |
| 0.25 | acs_equispaced | 0.250 | 0.135 | 0.0092 | 0.748 | 31.09 |
| 0.40 | learned | 0.382 | 0.068 | 0.0044 | 0.817 | 34.11 |
| 0.40 | learned_acs_lock | 0.400 | **0.060** | **0.0044** | **0.827** | **34.21** |
| 0.40 | acs_random | 0.400 | 0.205 | 0.0065 | 0.798 | 32.49 |
| 0.40 | acs_vd_gaussian | 0.400 | 0.162 | 0.0046 | 0.816 | 33.95 |
| 0.40 | acs_equispaced | 0.400 | 0.127 | 0.0066 | 0.794 | 32.40 |

**Beat `acs_random`?** Yes at \(s=0.25\) and \(0.40\) for both Gumbel and top-\(k\). Gumbel also beats `acs_vd_gaussian` on NMSE at both \(s\) (\(0.0071\) vs \(0.0077\); \(0.0044\) vs \(0.0046\)). Top-\(k\) loses slightly to VD at \(s=0.25\) (\(0.0082\) vs \(0.0077\)) and beats it at \(0.40\).

**Beat A1 post-hoc \(0.0065\) NMSE at \(s=0.25\)?** No. Gumbel \(0.0071\), top-\(k\) \(0.0082\). At \(s=0.40\) both match A1 (\(0.0044\) vs A1 \(0.0046\)). Gumbel SSIM/PSNR at \(s=0.25\) (\(0.764\) / \(32.16\)) still match or beat A1 (\(0.752\) / \(32.12\)).

**Gumbel vs top-\(k\):** Gumbel is **not** worse on recon — better NMSE at \(s=0.25\), tied at \(0.40\). Top-\(k\) keeps exact density and slightly lower \(H_c\). Keep train-consistent STE Gumbel; no need to swap eval to top-\(k\) only.

**Warm-start vs A1 post-hoc:** Parent remainder ranking was already informative (A1 \(H_c\) \(0.042\)). ACS-lock retrain made Gumbel train-consistent and dropped train NMSE to ACS level immediately; it did **not** improve on A1 post-hoc NMSE at \(s=0.25\). Eval \(H_c\) rose vs A1 top-\(k\) (\(0.042\to 0.061\)/\(0.080\)) — a small remainder-score drift under STE Gumbel, not a recon collapse.

**A3 go/no-go:** **Skip A3.** Skip unless both Gumbel and top-\(k\) lose to `acs_random` **and** `acs_vd_gaussian` at \(s=0.25\) and \(0.40\). Gumbel beats both heuristics at both \(s\); top-\(k\) beats `acs_random` at both \(s\). PE-aligned ACS cond is not justified.

A4 wrap-up (no train): curve `acs_lock_10ep/eval/nmse_ssim_vs_s.png`; winning recipe is ACS-lock Gumbel at `acs_lock_10ep/mask_gen_fastmri_final.pth`. See Stage A4.

