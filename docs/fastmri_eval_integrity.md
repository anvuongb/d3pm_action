# FastMRI eval integrity + next directions (B-stages)

Follow-on to `fastmri_scratch_protocol.md` (P1–P4). Measurements in "Findings"
below were taken 2026-08-28 with load-only scripts and **supersede the P4
conclusion**. Nothing in this doc has been run yet beyond the Findings section.

Do **not** overwrite:

- `models_mask_gen_fastmri_nested/mask_gen_fastmri_final.pth`
- `models_mask_gen_fastmri_nested/acs_lock_10ep/`
- `models_mask_gen_fastmri_nested/protocol_50ep_acs_lock/`
- `models_d3pm_fastmri_coarse_kspace{,_100ep}/`, `models_d3pm_fastmri_fine{,_100ep}/`

```
B0 record/retract → B1 eval protocol fix → B2 held-out re-decide
   → B3 adaptivity diagnostic → B4 bottleneck attack → B5 fine-prior decision → B6 write-up
```

Stop after each stage: eval, report numbers, then continue.

## Status

- [x] B0: record findings; retract the survival-table concern; mark P4 superseded
- [x] B1: held-out split + seeded, paired, budget-exact eval protocol
- [x] B2: re-decide A2 vs P3 vs baselines on `singlecoil_val`
- [x] B3: instance-adaptivity diagnostic — **headline retracted by F9**; corrected
      result is that conditioning is mildly *used* at s=0.25 and mildly harmful above it
- [x] B4: **does a real reconstructor create headroom?** — yes; policy does not exploit it
- [x] F9: **NMSE denominator floor bug** — invalidates the level of every
      pre-F9 number and two B3/B4 rankings
- [x] B5: static 300-parameter profile by STE gradient — negative, but the
      optimiser was the limit, not the idea (superseded by B5b)
- [x] B5b: **discrete row-swap hill climb — positive.** Beats `acs_vd_gaussian`
      at s=0.25/0.40/0.50 beyond 2 seed sd, and is the best mask tested at every s
- [x] B6: write-up with corrected claims -> `docs/fastmri_results.md`

---

## Findings that motivate this plan

All measured load-only; no checkpoint or repo artifact was modified.

### F1 — The survival-table concern is retracted

The P3 calibration warning ("100ep coarse table is not strictly monotone") is a
**false alarm**, not a defect:

| table | violations (>1e-4) | max increase | t range |
|-------|--------------------|--------------|---------|
| P3 100ep | 50 / 999 | 0.00032 | 71–650 |
| parent 50ep | 40 / 999 | 0.00049 | 251–780 |
| A2 10ep | 40 / 999 | 0.00049 | 251–780 |

The winning A2 table violates monotonicity by *more* than P3's. Magnitudes are
Monte-Carlo noise from 20-batch sampling in `_build_survival_from_batches`, an
order below the 1e-4 threshold in `_validate_survival_table` (`itw/schedule.py:56`).
Resulting `sparsity_to_timestep` maps differ by ≤4 steps out of 1000:

| s | t (P3) | t (A2) |
|---|--------|--------|
| 0.10 | 935 | 933 |
| 0.25 | 841 | 839 |
| 0.50 | 665 | 660 |
| 0.75 | 457 | 456 |

Optional cleanup only: raise the warning threshold to ~1e-3 or report violation
magnitude, so it stops firing on MC noise.

### F2 — Every number in every doc is a training-set number

`build_dataloader` (`itw/train.py:203`) builds one `FastMRIDataset` over
`singlecoil_train` (973 files, one mid-slice each) with `shuffle=True`, and
**both training and every eval script use it**. There is no train/val split
anywhere in the repo, and no `manual_seed` outside `lm_deepspeed.py`.

`/home/anvuong/data/fast_mri/singlecoil_val/` exists with **199 files** and has
never been used.

### F3 — Eval noise is the same size as the gaps being interpreted

8 seeds × 20 batches at s=0.25, learned Gumbel:

| method | mean NMSE | std | min | max |
|--------|-----------|-----|-----|-----|
| P3 | 0.00757 | 0.00035 | 0.00718 | 0.00806 |
| A2 | 0.00726 | 0.00030 | 0.00691 | 0.00765 |
| acs_random | 0.00939 | 0.00046 | 0.00889 | 0.00993 |

The doc's headline pair (P3 0.0081, A2 0.0071) sits near opposite extremes of
these distributions. The reported gap **+0.0010 overstates the paired effect
(+0.00031) by ~3×**. Single-draw, unpaired, unseeded eval is not adequate to
separate methods whose true differences are ≤0.001.

### F4 — Learned masks miss the budget, and that confounds every comparison

Gumbel masks do not hit exact density, baselines do. Paired, 6 seeds × 20 batches:

| s | P3 density | A2 density | P3 NMSE | A2 NMSE |
|---|-----------|-----------|---------|---------|
| 0.25 | 0.2549 | 0.2584 | 0.00750 | 0.00719 |
| 0.40 | 0.3943 | 0.3828 | 0.00458 | 0.00471 |
| 0.50 | 0.4794 | 0.4501 | 0.00370 | 0.00399 |
| 0.75 | 0.7328 | 0.6235 | 0.00175 | 0.00259 |

Whichever policy overshoots wins. A2 undershoots badly at s=0.75 (0.62 for a
0.75 budget) because it was trained only on s∈[0.10,0.40].

### F5 — Budget-matched, P3 and A2 are indistinguishable everywhere

Same protocol with budget-exact `acs_lock_topk_from_logits` (density exact at
every s):

| s | P3 | A2 | acs_random | paired P3−A2 |
|---|-----|-----|-----------|--------------|
| 0.10 | 0.01256 | 0.01256 | 0.01256 | 0.00000 |
| 0.25 | 0.00829 | 0.00827 | 0.00930 | +0.00001 |
| 0.40 | 0.00457 | 0.00464 | 0.00698 | −0.00008 |
| 0.50 | 0.00354 | 0.00354 | 0.00557 | −0.00001 |
| 0.75 | 0.00159 | 0.00161 | 0.00254 | −0.00002 |

**50 epochs from scratch on 100-epoch priors buys nothing over 10 warm-start
epochs.** All apparent P3/A2 differences in P4 were density artifacts. The
learned-vs-`acs_random` win survives, but the margin roughly halves once budget
is matched (s=0.25: 11% better, not the 20% the P4 table implies).

### F6 — The policy is not usefully instance-adaptive

s=0.25, 80 slices, budget-exact top-k. "Fixed" = one static mask built from the
top-k most frequently selected rows, applied to every slice:

| ckpt | mean pairwise Jaccard | NMSE adaptive | NMSE single fixed mask | acs_random |
|------|----------------------|---------------|------------------------|------------|
| P3 | 0.530 | 0.00820 | **0.00817** | 0.00917 |
| A2 | 0.525 | 0.00884 | **0.00795** | 0.00983 |

Masks do vary across slices (Jaccard 0.53, not 1.0), but **that variation buys
nothing** — a single static profile matches P3 and beats A2 by 10%. Of the
75-row budget, 32 are the hard-locked ACS block and 133 rows are never selected
at all.

So the demonstrated contribution is *"nested-D3PM-entropy + NMSE finds a good
**static** PE sampling profile,"* not instance-adaptive active sensing. This
also explains F5: there is no per-slice signal being learned, so extra epochs
cannot help.

Caveat to re-check in B3: the fixed mask here was derived from the same 80
slices it was tested on. B3 must derive it on train and test on val.

### F7 — A 300-number average beats the entire pipeline

> **Margins restated under F9.** The oracle still beats the learned policy at
> every sparsity under zero-filled, but by +5.1% / +0.5% / +2.9% / +3.6% at
> s=0.25/0.40/0.50/0.75 — not the 17% recorded below, which was the clamped
> metric.

"Energy oracle" = mean per-row k-space energy over the 973 **train** slices,
ACS-locked, top-k. No SGD, no diffusion prior, no entropy, no conditioning —
300 numbers. Evaluated on **val**, budget-exact:

| s | learned P3 | energy oracle | acs_vd | acs_random |
|---|-----------|---------------|--------|------------|
| 0.10 | 0.01340 | **0.01321** | 0.01340 | 0.01340 |
| 0.25 | 0.00895 | **0.00740** (−17%) | 0.00837 | 0.00991 |
| 0.40 | 0.00497 | **0.00496** | 0.00540 | 0.00746 |
| 0.50 | 0.00389 | **0.00383** | 0.00400 | 0.00597 |
| 0.75 | 0.00175 | 0.00174 | **0.00171** | 0.00274 |

The oracle matches or beats the learned policy at every sparsity and beats it
by 17% at s=0.25. Fit on train, tested on val — no leakage.

### F8 — The learned policy is approximating that profile, slightly worse

At s=0.25, learned selection frequency vs reference profiles (all 300 rows):

| comparison | Pearson r |
|------------|-----------|
| learned freq vs VD-Gaussian prior | **+0.93** |
| learned freq vs log row-energy | +0.83 |
| log row-energy vs VD prior | +0.77 |

The learned row set overlaps the energy oracle's by **77% (P3) / 81% (A2)** of
75 rows. Both checkpoints give nearly identical profiles.

Partial mechanism: zero-filled NMSE is largely, but not only, "how much
k-space energy did you miss." Over 5970 (mask, slice) pairs spanning 6 mask
families × 5 sparsities, r(missed-energy fraction, NMSE) = **0.695**,
R² = 0.48. The magnitude operation makes it nonlinear, so this is the dominant
factor rather than a closed form — enough to explain why an energy average is
hard to beat, not enough to call the problem solved analytically.

**Consequence.** Under zero-filled reconstruction the objective rewards one
thing — capture energy — and averaging finds that better than SGD does. This
explains F5 (extra epochs do nothing), F6 (adaptivity does nothing), and B2
(learned loses to VD at s=0.25). It is not a tuning problem: not adaptivity,
not training length, not overfitting, not capacity, not data. **The
reconstructor is the bottleneck**, and it is what B4 now tests.

---

### F9 — The NMSE denominator floor was active on two thirds of all slices

Found while building B5, when a profile pinned to the energy oracle scored
0.01756 under the B5 objective but 0.00431 under `itw.discrete.nmse` — the same
mask, data and reconstructor, differing 4x.

`nmse` (`itw/discrete.py:95`) took `eps: float = 1e-8` as an **absolute** floor
on the per-sample denominator `||target||^2`. On FastMRI that floor is not a
safety net, it is the denominator:

| split | median `\|\|target\|\|^2` | fraction below the 1e-8 floor | fraction below 1e-7 |
|-------|--------------------|------------------------------|---------------------|
| train (973) | 4.17e-9 | **64.3%** | 99.4% |
| val (199)   | 4.21e-9 | **67.3%** | 99.5% |

For those slices the metric is not NMSE at all but `||pred-target||^2 / 1e-8`, a
constant-scaled MSE. Because the clamp only ever *raises* the denominator, it
silently down-weights low-energy slices — an energy weighting nobody chose — and
reads 3.8–4.3x below the true ratio. The `psnr` docstring immediately below
already warned that "FastMRI magnitude is ~1e-6, so ... a 1e-8 floor would make
every method look identical"; `nmse` never got the same treatment.

**Blast radius**, checked call site by call site:

- Every NMSE reported in B2, B3, B4 and the P-series docs is the hybrid metric.
- `itw/train.py:564` uses `nmse` as the mask policy's reconstruction loss, so
  P3 and A2 were **trained** against the distorted objective, not merely
  evaluated with it. Their training signal down-weighted the low-energy two
  thirds of the data by up to 4x, which is a plausible contributor to their
  converging on a near-static, centre-heavy profile.
- **Not affected:** the B4 reconstructor. `itw/recon.py:170` computes its loss
  inline with a 1e-12 floor and never calls `nmse`, so U-Net A and the B5 judge
  are valid as trained, and only the numbers *reported about* them were wrong.
- Not affected: `psnr`/`ssim`, which have no such floor.

**Fix.** `eps` now defaults to 0 and the denominator is clamped at the dtype's
tiny value; only an all-zero target needs protection. Guarded by two tests in
`tests/test_eval_protocol.py`: NMSE must be invariant to rescaling both
arguments at FastMRI's magnitude scale (the old default returns 0.097 instead of
0.25 there), and an all-zero target must stay finite. Suite is 54 -> 56.

**What changes.** Rankings are not preserved, because the clamp is a per-slice
reweighting. Under the U-Net reconstructor on val:

| s | winner (old, clamped) | winner (corrected) |
|---|-----------------------|--------------------|
| 0.25 | acs_vd_gaussian | acs_vd_gaussian |
| 0.40 | P3_static_train | **acs_vd_gaussian** |
| 0.50 | A2_static_train | **acs_vd_gaussian** |
| 0.75 | A2_static_train | A2_static_train |

so **B3's "the policy's own static profile is the best mask in the study" is
retracted**: `acs_vd_gaussian`, a zero-parameter heuristic, is best at three of
four sparsities, and the learned static profile survives only at s=0.75 (0.00771
vs 0.00778 for plain `vd_gaussian`, 0.9%).

B4's core finding is *strengthened*: the energy oracle wins at s=0.25 under
zero-filled and wins nowhere under the U-Net, so mask ranking is strongly
reconstructor-dependent.

**Caveat on the flips.** `acs_vd_gaussian` draws a fresh mask per sample. Its
s=0.50 margin over `A2_static_train` is 0.00002 absolute against a seed sd of
~1e-5, i.e. ~2 sd — marginal. All stochastic baselines are now evaluated over 5
seeds with the spread reported (`--seeds`), and a B5 win inside 2 sd of the
rival's spread is not counted as a win.

## B1 — Eval protocol fix

Everything downstream is untrustworthy until this lands. Code change, then
re-eval existing checkpoints; **no training**.

1. **Held-out split.** Add `FastMRIConfig.data_split: Literal["train","val"] = "train"`
   and `val_root` defaulting to `singlecoil_val`. `build_dataloader` uses
   `shuffle=(split=="train")`. Eval scripts request `split="val"`.
2. **Seeding.** Add `FastMRIConfig.seed: int = 0` and a `seed_everything(seed)`
   helper; seed the DataLoader `generator` and torch RNG in every eval entry point.
3. **Budget-exact masks are the default eval mask.** Report top-k as the primary
   `learned` number; keep Gumbel as a clearly-labelled secondary. Every reported
   row must carry its realized density.
4. **Paired, multi-seed reporting.** `evaluate_fastmri_baselines_loader` gains
   `seeds: tuple[int,...] = (0,1,2,3,4)`; it reports mean, std, and paired deltas
   vs a named reference method rather than a single scalar.
5. **Guardrail test.** `tests/test_eval_protocol.py`: same seed ⇒ identical
   metrics; different seed ⇒ different; val loader never returns a train file;
   top-k density equals `floor(s*300)/300` exactly.

**Pass:** re-running B1 eval twice with the same seed is bit-identical; density
column is exact for every learned row; val and train file sets are disjoint.

### B1 results

All five items landed. **No pre-B1 default changed**, so every P1–P4 script
reproduces its old behaviour: `learned_mask="gumbel"`,
`include_learned_gumbel=False`, `data_split="train"`, `seed=None`.

| item | where |
|------|-------|
| split + `val_root` + `FASTMRI_VAL_ROOT` | `itw/configs.py`, `itw/train.py:fastmri_split_root` |
| `seed` + `seed_everything` | `itw/configs.py`, seeded `generator` in `build_dataloader` |
| budget-exact masks | `itw/eval.py:topk_row_mask_from_logits`, `learned_rows_topk` |
| paired multi-seed reporting | `itw/eval.py:evaluate_fastmri_baselines_seeded`, `fastmri_loader_factory` |
| guardrails | `tests/test_eval_protocol.py` (20 tests; suite 34 → 54, all pass) |

`build_dataloader` now branches on split: train keeps `shuffle=True` +
`drop_last=True`; **val is `SequentialSampler`, `drop_last=False`**, so a val
pass is deterministic and covers all 199 slices (25 batches at batch 8). Seeds
therefore vary *only* mask sampling, not data order — which is what makes the
paired deltas meaningful.

Verified on real data: train 973 / val 199 files, **disjoint**; two val passes
bit-identical; `learned` (top-k) std exactly 0.00000 across seeds; realized
density exactly 0.2500 for learned and every baseline, vs 0.2581 for
`learned_gumbel`.

**Preview (not B2):** the smoke ran s=0.25 on val, 3 seeds, budget-matched:

| method | NMSE | std | density | paired vs learned |
|--------|------|-----|---------|-------------------|
| learned (top-k) | 0.00895 | 0.00000 | 0.2500 | — |
| acs_vd_gaussian | **0.00835** | 0.00005 | 0.2500 | −0.00060 (3/3 lower) |
| acs_random | 0.00993 | 0.00002 | 0.2500 | +0.00098 (0/3 lower) |
| learned_gumbel | 0.00797 | 0.00002 | 0.2581 | not budget-matched |

On held-out data at s=0.25, **`acs_vd_gaussian` beats the learned policy** —
consistently (3/3 seeds, delta 12× the seed std). On train the learned policy
won this matchup. One sparsity, one checkpoint; B2 runs the full grid before
this is treated as a result. But it is the first direct sign that the
learned-vs-heuristic claim may not survive the held-out split at low s.

Operational note: repeatedly constructing loaders with `num_workers>0` hit
forkserver `ConnectionResetError` on this box (Python 3.14). The B2 script
should use `num_workers=0` for val, or build the loader once per seed.

## B2 — Re-decide A2 vs P3 on held-out data

Load-only, `singlecoil_val`, 5 seeds, budget-exact, all baselines, s ∈
{0.10, 0.25, 0.40, 0.50, 0.75}. Script: `eval_fastmri_val.py` (new), writing
`models_mask_gen_fastmri_nested/eval_val/eval_b2.json`.

**Decision rules:**

- If paired |P3−A2| < 2× seed std at every s: declare them equivalent and
  **ship A2** on cost (10 epochs, warm-start) — but state it as *equivalent*,
  not as *A2 wins*. This is what F5 predicts on train.
- If P3 wins only at s > 0.40: ship P3 for the wide-s operating range and note
  that A2's loss there is a training-range artifact, not a capability gap.
- If train and val rankings disagree: the policies are overfitting 973 slices;
  that becomes the top priority ahead of B4.

### B2 results

Load-only. Script `eval_fastmri_val.py`, JSON
`models_mask_gen_fastmri_nested/eval_val/eval_b2.json`. 199 val slices (25
batches), 5 seeds, s ∈ {0.10, 0.25, 0.40, 0.50, 0.75}, six baselines,
budget-exact top-k. All 12 protected artifacts verified unchanged.

Because the val pass is deterministic and top-k carries no sampling noise, the
`learned` rows have **std exactly 0** — the head-to-head is exact, not estimated.

**A2 vs P3 — equivalent, confirming F5 on held-out data:**

| s | A2 NMSE | P3 NMSE | P3−A2 | winner |
|---|---------|---------|-------|--------|
| 0.10 | 0.01340 | 0.01340 | +0.00000 | tie |
| 0.25 | 0.00893 | 0.00895 | +0.00002 | A2 |
| 0.40 | 0.00507 | 0.00497 | −0.00009 | P3 |
| 0.50 | 0.00389 | 0.00389 | +0.00000 | tie |
| 0.75 | 0.00177 | 0.00175 | −0.00002 | P3 |

Every difference is ≤1% relative, split 2–2 with two exact ties. **50 epochs
from scratch on 100-epoch priors is equivalent to 10 warm-start epochs on
held-out data.** Ship A2 on cost; state it as *equivalent*, not as A2 winning.

**Learned vs the best ACS heuristic — the headline correction:**

| s | learned (P3) | best heuristic | verdict |
|---|--------------|----------------|---------|
| 0.10 | 0.01340 | acs_random 0.01340 | tie (forced centered block) |
| 0.25 | 0.00895 | **acs_vd_gaussian 0.00837** | **heuristic wins** (5/5 seeds) |
| 0.40 | **0.00497** | acs_vd_gaussian 0.00538 | learned wins (−8%) |
| 0.50 | **0.00389** | acs_vd_gaussian 0.00399 | learned wins (−3%) |
| 0.75 | 0.00175 | **acs_vd_gaussian 0.00170** | heuristic wins |

Same pattern for A2. **On held-out data at matched budget, the learned policy
beats a tuned variable-density ACS heuristic only in the middle of the range
(s = 0.40–0.50), and loses at 0.25 and 0.75.** The docs' claim that learned
beats `acs_vd_gaussian` at s=0.25 (A2 train: 0.0071 vs 0.0077) **does not
survive the held-out split**. At s=0.75 even unconstrained `vd_gaussian` ties
the ACS cluster (0.00170), so the ACS constraint stops mattering once the
budget covers the center.

Unconstrained `random`/`equispaced` remain catastrophic everywhere
(0.076–0.383). The center-weighting is doing nearly all the work.

**The objective disagrees with the metric.** The learned policy has by far the
lowest coarse entropy at low s — at s=0.25, H_c 0.068 vs `acs_vd_gaussian`
0.179 — yet loses to it on NMSE. Minimizing nested H_c is not buying
reconstruction on held-out data. At s=0.75 the ordering even flips
(learned 0.389 vs `acs_equispaced` 0.055). This is now the central problem with
the objective, not a detail.

**Density confound quantified:** at s=0.25 `learned_gumbel` scores 0.00797 at
density 0.2587 versus 0.00895 for the same policy at exactly 0.2500. A 3.5%
budget overshoot buys an 11% NMSE improvement — which is larger than every
learned-vs-heuristic gap in the table. Every pre-B1 Gumbel number is inflated
by this.

### B2 control — train split, identical protocol

`uv run python eval_fastmri_val.py --split train` → `eval_val/eval_b2_train.json`.
`--split train` walks the 973 train files with the **same sequential,
`drop_last=False` sampler as val** (122 batches), so the split is the only
difference; a plain `data_split="train"` would have shuffled and dropped the
tail and would not be comparable.

**The rankings are identical on both splits. There is no overfitting.**

| s | train verdict | val verdict |
|---|---------------|-------------|
| 0.10 | tie | tie |
| 0.25 | heuristic wins +0.00050 | heuristic wins +0.00058 |
| 0.40 | learned wins −0.00041 | learned wins −0.00041 |
| 0.50 | learned wins −0.00010 | learned wins −0.00010 |
| 0.75 | heuristic wins +0.00006 | heuristic wins +0.00005 |

Gap magnitudes match to within a few 1e-5. The apparent train/val disagreement
was **not** a ranking flip — it was a comparison against the old,
density-unmatched numbers. Budget-matched, `acs_vd_gaussian` beat the learned
policy at s=0.25 on train too. The docs' A2 claim (learned 0.0071 vs acs_vd
0.0077) was the density confound, and nothing else.

Train→val degradation is a property of the **data**, not the policy: the
learned generator (~174K params) degrades by the same amount as a
zero-parameter heuristic.

| s | learned | acs_vd_gaussian | acs_random |
|---|---------|-----------------|------------|
| 0.25 | +6.2% | +5.6% | +5.0% |
| 0.40 | +7.3% | +6.8% | +5.0% |
| 0.50 | +8.1% | +7.7% | +5.3% |
| 0.75 | +8.3% | +9.2% | +5.8% |

A2 vs P3 head-to-head is again equivalent on train (largest gap 0.00008).

**Consequence for the plan:** the "overfitting becomes top priority" branch of
the B2 decision rule is **closed — it does not apply**. Proceed to B3 as
written. F6 gets stronger, not weaker: a policy that is not overfitting, and
not beating a static heuristic outside s=0.40–0.50, is most consistent with
having learned a static profile.

## B3 — Instance-adaptivity diagnostic

The scientific question F6 raises. Load-only, val split.

1. Build the static profile from **train** slices (per-row selection frequency,
   top-k), evaluate on **val**. Compare: conditioned per-slice policy vs static
   profile vs `acs_vd_gaussian` vs `acs_random`, budget-matched, 5 seeds.
2. Shuffle-control: feed each slice a *different* slice's scout `Z`. If NMSE
   does not degrade, the policy is provably ignoring its conditioning input.
3. Report per-row selection frequency and the learned static profile as a plot
   against a fitted variable-density curve — is the learned profile just VD?

**Decision rules:**

- Static ≥ conditioned on val (F6's expectation): stop claiming active sensing.
  Reframe the contribution as a learned static profile, and make the static
  profile the shipped artifact — it is simpler, faster, and reproducible.
  Then B4 is about *whether conditioning can be made to work at all*.
- Conditioned clearly beats static on val: F6 was an in-sample artifact;
  proceed to B4 to strengthen adaptivity.
- Shuffle-control shows no degradation: the conditioning path is dead code in
  practice; fixing it is B4's whole content.

### B3 results

> **Superseded in part by F9.** Every number in this section was measured with
> the clamped NMSE. The corrected table follows the original one below.

`eval_fastmri_adaptivity.py` → `eval_val/eval_b3_adaptivity.json`. Load-only,
val, budget-exact, scored under both zero-filled and the B4 U-Net. `static_train`
is the policy's own top-k selection frequency measured on **train** (fixes the
F6 in-sample caveat); `shuffled_cond` deranges scout `Z` across val (0 fixed
points) while leaving k-space untouched.

**The conditioning input is not used, and using it correctly is net-harmful.**

Mean over s ∈ {0.25, 0.40, 0.50, 0.75}, relative to the adaptive policy
(negative = better than adaptive):

| policy | recon | shuffle penalty | static penalty |
|--------|-------|-----------------|----------------|
| P3 | zero-filled | **−1.63%** | **−2.16%** |
| P3 | U-Net | −1.60% | −1.73% |
| A2 | zero-filled | −1.79% | −2.57% |
| A2 | U-Net | −1.75% | −2.15% |

Giving each slice **another slice's** scout image makes reconstruction *better*.
The effect is concentrated at s=0.25 (shuffle −6.4% P3 / −7.1% A2) and reverses
to a negligible +0.2…+0.9% at s=0.50/0.75. So at the operating point of
interest the conditioning is actively counterproductive, not merely ignored.

**The policy's own static profile is the best mask tested** — better than the
energy oracle and better than `acs_vd_gaussian` — at every s except 0.25:

| s | best method (U-Net) | NMSE | adaptive policy | oracle | acs_vd |
|---|---------------------|------|-----------------|--------|--------|
| 0.25 | acs_vd_gaussian | 0.00642 | 0.00771 | 0.00671 | 0.00642 |
| 0.40 | **P3_static_train** | **0.00453** | 0.00460 | 0.00460 | 0.00462 |
| 0.50 | **P3/A2_static_train** | **0.00355** | 0.00365 | 0.00362 | 0.00362 |
| 0.75 | **A2_static_train** | **0.00167** | 0.00172 | 0.00174 | 0.00171 |

The training found a good static profile and then **degraded it** with useless
per-slice variation. Collapsing the policy to its own consensus mask recovers
the loss and produces the strongest mask in the study.

**Consequence for B5 (as written pre-F9; conclusion stands, numbers do not).** The
conditional architecture is not merely unhelpful; it costs 1.6–2.6% and, at
s=0.25, up to 7%. Optimise a 300-parameter static profile directly against the
reconstructor: no Gumbel/STE, no density confound, no conditioning encoder,
trainable in seconds, and it already beats every heuristic at s ≥ 0.40 before
any such optimisation.

Adaptivity is not salvaged by the better reconstructor either — the shuffle and
static penalties are essentially identical under zero-filled and U-Net, so this
is not an artefact of the weak reconstructor.

### B3 results, corrected for F9

Re-scored on val with the fixed denominator. The old column reproduces the
published means exactly, which is what certifies the re-score.

Penalty relative to `adaptive` (negative = better than adaptive):

| s | recon | shuffle old | **shuffle corrected** | static old | **static corrected** |
|---|-------|-------------|----------------------|------------|---------------------|
| 0.25 | zf | −6.8 / −7.1% | **+2.3 / +2.4%** | −2.4 / −1.0% | **+4.3 / +5.5%** |
| 0.25 | U-Net | −6.4 / −6.6% | **+0.7 / +0.7%** | −0.2 / +1.2% | **+4.1 / +5.1%** |
| 0.40 | U-Net | −0.9 / −1.5% | −1.1 / −1.1% | −1.5 / −3.4% | −0.9 / −1.4% |
| 0.50 | U-Net | +0.2 / +0.2% | −1.7 / −1.7% | −2.6 / −2.6% | −3.6 / −3.6% |
| 0.75 | U-Net | +0.7 / +0.9% | −0.9 / −1.1% | −2.5 / −3.8% | −3.0 / −4.2% |

(pairs are P3 / A2)

Mean shuffle penalty over s falls from −1.60…−1.79% to **−0.35…−0.80%**.

**The headline claim is retracted.** B3 reported that deranging the scout
*improved* NMSE by 6.4–7.1% at s=0.25 — the single most striking number in the
study, and pure metric artifact. Corrected, the sign reverses to +0.7% (U-Net)
and +2.3–2.4% (zero-filled): at the tightest budget the correct scout is
genuinely better, so the conditioning input **is** mildly used, exactly where
budget pressure is highest.

What survives:

- Conditioning is mildly *harmful* at s ≥ 0.40 (−0.9% to −1.7%), and the whole
  effect is bounded by ~2.4% in either direction.
- Static beats adaptive at s ≥ 0.40 (−0.9% to −4.2%) but **loses at s=0.25**
  (+4.1 to +5.5%), where B3 had it winning. "Collapse the policy to its
  consensus mask" is sparsity-dependent, not universal.

**Consequence for B5 is unchanged in form but not in justification.** The
conditional path is still not worth rebuilding — it buys at most ~2% and only at
s=0.25 — so B5 remains a static-profile optimisation. But it no longer starts
from a winning mask: the target to beat is now `acs_vd_gaussian`, which no
learned mask beats below s=0.75.

## B4 — Does a real reconstructor create headroom?

**The decisive experiment.** F7/F8 say the zero-filled objective rewards energy
capture, which a 300-number average already solves. Mask learning can only
matter if the reconstructor has a prior: then centre rows become *predictable*,
spending budget on them is wasteful, and the optimum moves away from the energy
profile toward complementary information.

Train a small U-Net reconstructor on the **train** split under a mixture of
*heuristic* masks (deliberately excluding learned masks, so the reconstructor
is unbiased), then re-rank every mask method under it on **val**.

Methods: learned P3, learned A2, energy oracle, acs_vd_gaussian, acs_random,
acs_equispaced, vd_gaussian — each scored both zero-filled and U-Net.

**Decision rules:**

- **Ranking changes** (learned or a non-energy mask overtakes the energy
  oracle): headroom exists. The pivot is justified — promote the frozen fine
  D3PM from a dead entropy term to the reconstructor (posterior sampling /
  k-space inpainting) and retrain the policy against *that* objective. The
  information-theoretic framing becomes load-bearing: sample where the prior is
  uncertain.
- **Energy oracle still wins under the U-Net**: this dataset and task have no
  headroom for mask learning at all. Do not tune the policy further. Either
  change the problem (sequential acquisition; heterogeneous anatomy; a
  task-driven objective instead of global NMSE) or write up the negative
  result.

### B4 results

> **Levels superseded by F9** (clamped NMSE, ~4x low). The ranking conclusion —
> mask choice is reconstructor-dependent — survives and strengthens: corrected,
> the energy oracle wins nowhere under the U-Net.

`train_fastmri_recon_rerank.py` → `models_recon_unet/`,
`eval_val/eval_b4_recon.json`. 1.93M-param U-Net, 40 epochs on the train split,
trained under heuristic masks only. Val, budget-exact, protected artifacts
unchanged.

**Answer: headroom exists — and the current policy does not exploit it.**

Reconstructor sanity gate: PASS at s = 0.10 / 0.25 / 0.40 / 0.50. "FAIL" at
s=0.75 is a tie (0.00169 → 0.00170); at 75% sampling there is nothing to fix
and the net correctly learned to leave the image alone.

Ranking moved at **every sparsity where movement is possible** (s=0.10 cannot
move — all ACS masks are the same forced centred block):

| s | best zero-filled | best under U-Net |
|---|------------------|------------------|
| 0.25 | energy_oracle | **acs_vd_gaussian** |
| 0.40 | energy_oracle | **learned_P3** |
| 0.50 | energy_oracle | **acs_vd_gaussian** |

The energy oracle loses its crown once the reconstructor has a prior. This
confirms the F7 mechanism: the oracle's dominance was an artefact of
zero-filled reconstruction, not a property of the sampling problem.

**But the learned policy gains the least from the reconstructor.** At s=0.25 it
drops to *5th of 7*:

| method | zf NMSE | U-Net NMSE | gain |
|--------|---------|-----------|------|
| acs_vd_gaussian | 0.00843 | **0.00643** | 23.8% |
| energy_oracle | 0.00740 | 0.00671 | 9.2% |
| acs_equispaced | 0.00992 | 0.00710 | **28.4%** |
| acs_random | 0.00990 | 0.00726 | 26.7% |
| learned_A2 | 0.00894 | 0.00770 | 13.9% |
| learned_P3 | 0.00895 | 0.00771 | 13.9% |

The masks that *spread out* gain most (equispaced 28%, random 27%); the
centre-concentrated ones gain least (learned 14%, oracle 9%). Mechanism: a
reconstructor with a prior can already predict the low-frequency centre, so
budget spent there is redundant. The learned policy is maximally
centre-concentrated **because it was trained on zero-filled NMSE + coarse
entropy, both of which reward energy capture** — precisely the wrong signal
once the reconstructor improves. At s=0.40/0.50 learned, oracle and acs_vd are
a three-way tie (0.00460/0.00460/0.00461 and 0.00362/0.00362/0.00365).

**Verdict: the pivot is justified.** Mask choice is reconstructor-dependent, so
there is something real to learn — but it must be learned *against the
reconstructor*, not against zero-filled NMSE. Promote the frozen fine D3PM from
dead entropy term to reconstructor (B5) and retrain the policy in that loop.

Caveats: the U-Net saw only heuristic masks in training, so it is mildly
out-of-distribution on learned masks (the energy oracle was also unseen and
does fine, so the effect looks small); and a 1.9M-param U-Net on 973 slices
understates what a real diffusion posterior sampler would give.

**Methodological note (first B4 run, kept as `*_absloss.*`).** The initial run
used absolute NMSE as the reconstruction loss and produced a *false positive*:
it reported "ranking moved" while the U-Net was worse than zero-filled almost
everywhere (−166% at s=0.75). Training NMSE spans ~100× across s∈[0.10,0.75],
so an unnormalised loss is dominated by low-s samples and the net learns to
damage the high-s images it should leave alone. Two fixes: loss is now NMSE
**relative to zero-filled** (every sample ≈1.0 at the identity), and the mask is
a third input channel. The verdict logic was also wrong — it counted movement
at sparsities where the reconstructor had failed its gate. Movement now only
counts where the gate passes.

**Deprioritised by F7/F8** — these were the pre-F7 B4 candidates and are no
longer the bottleneck; none of them can beat an average under a zero-filled
objective. Revisit only if B4 finds headroom: Gumbel anneal (τ never went below
0.5, `itw/train.py:79`), PE-aligned conditioning, 2× capacity, and directly
optimising a 300-parameter static profile.

## B5 — Fine prior: repurpose, not retire

β=0 has survived three independent checks (Stage 5, P2, the 100-epoch retrain),
so the fine prior is dead **as an entropy term**. F7 reframes it: it is the
obvious candidate *reconstructor*, which is where a diffusion prior earns its
keep. Do not retrain it as an entropy term; do not delete it either.

Gate: only if B4 finds headroom. Caveat to check first — the flat CE(t) that
killed it as an entropy term is also evidence it may be too weak a model to
serve as a reconstructor; compare it against the B4 U-Net before committing.

### B5 results

300 parameters, optimised on train against the frozen B4 U-Net ("U-Net A"),
evaluated on val under three reconstructors. **U-Net B** is an independently
trained judge (seed 1, width 48, its own mask draws) that took no part in the
optimisation; stochastic baselines are averaged over 5 seeds with the spread
reported.

Ranked by the held-out judge:

| s | winner (judge) | b5_profile | rank | vs `acs_vd_gaussian` |
|---|----------------|-----------|------|----------------------|
| 0.25 | acs_vd_gaussian 0.02450 | 0.02487 | 2/10 | −1.50% |
| 0.40 | acs_vd_gaussian 0.01873 | 0.01910 | 2/10 | −1.97% |
| 0.50 | acs_vd_gaussian 0.01538 | 0.01573 | 4/10 | −2.25% |
| 0.75 | A2_static_train 0.00764 | 0.00782 | 5/10 | −2.34% |

All four margins exceed 2 seed sd of the rival. **A directly optimised static
profile does not beat a zero-parameter variable-density heuristic.**

**The judge earned its place.** Scoring only against the net it was optimised
against would have overstated the profile by roughly 1 pp:

| s | vs acs_vd, in-loop (U-Net A) | vs acs_vd, judge (U-Net B) | transfer loss |
|---|------------------------------|----------------------------|---------------|
| 0.25 | −1.78% | −1.50% | −0.28 pp |
| 0.40 | −0.49% | −1.97% | **+1.49 pp** |
| 0.50 | −1.34% | −2.25% | +0.91 pp |
| 0.75 | −1.51% | −2.34% | +0.83 pp |

**Major limitation — half the grid was never actually tested.** `best epoch`
includes the epoch −1 initialisation, and at s=0.50 and s=0.75 *no epoch beat
the energy warm start*, so the optimiser returned it untouched. The saved
profiles produce masks identical to the energy oracle there:

| s | init selected | kept rows differing from the energy-oracle mask |
|---|---------------|------------------------------------------------|
| 0.25 | energy | 64 of 75 |
| 0.40 | energy | 14 of 120 |
| 0.50 | energy | **0 of 150** |
| 0.75 | energy | **0 of 225** |

The identical val numbers for `b5_profile` and `energy_oracle` at s=0.50/0.75
are that, not a coincidence. So B5's negative verdict rests on s=0.25 and
s=0.40 only.

**What did work.** Where the optimiser moved, it produced the best learned mask
in the study by a wide margin. At s=0.25 under the judge:

| mask | judge NMSE | vs b5_profile |
|------|-----------|---------------|
| acs_vd_gaussian | 0.02450 | −1.5% |
| **b5_profile** | **0.02487** | — |
| energy_oracle | 0.02703 | +8.7% |
| A2/P3_adaptive | 0.02785 | +12.0% |
| P3_static_train | 0.02897 | +16.5% |

It lifts the best learned mask from rank 5–7 to **rank 2 of 10**, a 10.7%
improvement over the previous best learned mask.

**It also confirms the B4 mechanism constructively.** At s=0.25 the optimised
profile is *13% worse* than the energy oracle under zero-filled (0.03557 vs
0.03147) while being *8.0% better* under the judge (0.02487 vs 0.02703).
Optimising against a reconstructor that carries a prior moves the mask away
from the energy-optimal one, exactly as B4 predicted — the profile trades raw
energy capture for information the reconstructor cannot already supply.

> **B5b answers the open question: the headroom was real.** A discrete row-swap
> hill climb beats `acs_vd_gaussian` where this optimiser lost. The negative
> verdict above is an artefact of the STE-gradient search, not a property of the
> task. Read B5 as a lower bound on what a static profile can do.

**What is and is not established.** Established: this STE-gradient optimiser,
tuned, lands 1.5–2.0% short of `acs_vd_gaussian` at the two sparsities where it
moved. Not established: that no static profile can win. The optimiser is weak —
it oscillates 15% between epochs on the discrete objective, and reverts to its
initialisation at half the grid. Ruling out the headroom would need a stronger
search (row-swap hill climbing on the hard mask, or a much lower LR over many
more epochs) before "a learned profile cannot beat variable density" is safe to
claim.

Artifacts: `models_static_profile/profile_s{25,40,50,75}.pth`,
`models_recon_unet/recon_unet_judge.pth`,
`models_mask_gen_fastmri_nested/eval_val/eval_b5_profile.json`.

## B5b — Discrete row-swap hill climb

B5's optimiser oscillated 15% between epochs and reverted to its initialisation
at half the grid, so its 1.5-2.0% shortfall against `acs_vd_gaussian` could not
distinguish "no headroom" from "weak search". This attacks the discrete object
directly: greedy row swaps on the hard mask, gradient used only to *propose*
swaps, exact full-train evaluation to accept them. No STE, no relaxation.

Seeds tried per sparsity: best of 16 fixed `acs_vd_gaussian` draws, the energy
oracle, and the B5 profile. `fixed_vd_best` won the seed selection at every
sparsity.

### B5b results

Val, under the held-out judge U-Net B, against `acs_vd_gaussian` over 5 seeds:

| s | hill climb | acs_vd_gaussian | gain | verdict |
|---|-----------|-----------------|------|---------|
| 0.25 | **0.02355** | 0.02448 ±0.00006 | **+3.79%** | beats (>2 sd) |
| 0.40 | **0.01842** | 0.01882 ±0.00005 | **+2.14%** | beats (>2 sd) |
| 0.50 | **0.01521** | 0.01544 ±0.00007 | **+1.49%** | beats (>2 sd) |
| 0.75 | 0.00763 | 0.00765 ±0.00001 | +0.18% | ties |

**It is the best mask tested at every sparsity**, against every method in the
study:

| s | hill climb | vs acs_vd | vs b5_profile | vs A2_static | vs oracle | vs adaptive |
|---|-----------|-----------|---------------|--------------|-----------|-------------|
| 0.25 | 0.02355 | +3.8% | +5.3% | +19.4% | +12.9% | +15.4% |
| 0.40 | 0.01842 | +2.1% | +3.6% | +4.8% | +5.9% | +5.9% |
| 0.50 | 0.01521 | +1.5% | +3.3% | +2.0% | +3.3% | +5.5% |
| 0.75 | 0.00763 | +0.2% | +2.4% | +0.1% | +2.4% | +4.3% |

**Not a reconstructor artefact.** The climbed mask also beats `acs_vd_gaussian`
under plain zero-filled reconstruction at three of four sparsities (+3.2%,
+2.4%, +2.8%, −0.7%), despite being optimised against a U-Net. A mask that had
merely learned U-Net A's quirks would not survive both a different reconstructor
and no reconstructor.

**Where the gain comes from** — roughly two equal halves, and neither alone
clears the bar everywhere:

| s | per-slice random | -> best fixed draw | -> after hill climb | swaps |
|---|------------------|--------------------|---------------------|-------|
| 0.25 | 0.022949 | 0.022346 (+2.63%) | 0.021684 (+2.96%) | 6 |
| 0.40 | 0.017228 | 0.016929 (+1.74%) | 0.016728 (+1.18%) | 3 |
| 0.50 | 0.013922 | 0.013663 (+1.86%) | 0.013663 (+0.00%) | 0 |
| 0.75 | 0.006805 | 0.006777 (+0.42%) | 0.006694 (+1.22%) | 8 |

(train NMSE against U-Net A; swaps are net row changes)

Per-swap gains decay smoothly (+0.86, +0.83, +0.72, +0.22, +0.22, +0.14% at
s=0.25), which is what a real local search looks like rather than noise-chasing.
Every climb terminated at a genuine local optimum — no improving swap among
4x6 = 24 exactly-evaluated proposals.

### The VD randomisation control

`acs_vd_gaussian` draws a fresh mask per slice, so a fixed learned mask must
beat an average over random draws. Is that randomisation load-bearing?

| s | per-slice random | best of 16 fixed | mean fixed draw |
|---|------------------|------------------|-----------------|
| 0.25 | 0.022949 | 0.022346 (+2.63%) | 0.023017 (−0.30%) |
| 0.40 | 0.017228 | 0.016929 (+1.74%) | 0.017213 (+0.09%) |
| 0.50 | 0.013922 | 0.013663 (+1.86%) | 0.013974 (−0.37%) |
| 0.75 | 0.006805 | 0.006777 (+0.42%) | 0.006798 (+0.10%) |

**Per-slice randomisation is neutral**: the mean fixed draw matches per-slice
random to within ±0.4%. What pays is *selecting* a good fixed draw (+0.4 to
+2.6%), which is a train-side selection effect that partly transfers. An earlier
3-draw reading suggested the mean fixed draw beat per-slice random; 16 draws
refute that.

Artifacts: `models_hill_climb/hillclimb_s{25,40,50,75}.pth`,
`models_mask_gen_fastmri_nested/eval_val/eval_b5b_hillclimb.json`.

## B6 — Write-up

Corrections that must land:

- P4's "50ep does not beat A2 10ep" is **superseded**: budget-matched they are
  equivalent (F5). The practical recommendation (ship the cheap one) stands;
  the stated reason does not.
- All pre-B1 numbers are train-set, single-draw, and density-unmatched — label
  them as such wherever they are quoted.
- Learned-vs-`acs_random` margins shrink by roughly half under budget matching.
- The headline claim becomes whatever B3 supports, which on current evidence is
  a learned static profile rather than instance-adaptive acquisition.
