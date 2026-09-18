# FastMRI row-mask acquisition — corrected results (B6)

Consolidated, corrected write-up of the FastMRI arm. **This document supersedes
the claims in `fastmri_scratch_protocol.md` (P1–P4),
`fastmri_policy_improvements.md` (Stages 1–6) and `fastmri_acs_remainder.md`
(A1–A4).** Those record how the work developed; where they disagree with this
document, this document is right. Provenance, per-stage decision rules and the
full audit trail live in `fastmri_eval_integrity.md`.

Everything below is measured on the held-out `singlecoil_val` split (199 files)
with budget-exact masks, and — except where explicitly labelled — with the
corrected NMSE of finding F9.

---

## Headline

**A learned static Cartesian row mask beats variable-density sampling. Learned
*instance-adaptive* acquisition does not.**

Under a reconstructor that took no part in producing it, a 300-row mask found by
discrete local search is the best mask tested at every sparsity, beating the
strongest heuristic by 3.8% / 2.1% / 1.5% at s = 0.25 / 0.40 / 0.50 (all beyond
2 seed sd) and tying it at s = 0.75.

The information-theoretic apparatus the project was built around — a frozen
D3PM entropy proxy conditioned on a per-slice scout — contributed none of this.
What produced the result was optimising the discrete mask directly against a
reconstructor.

---

## Protocol

| | |
|---|---|
| data | fastMRI singlecoil, one mid-slice per file: 973 train / **199 val** |
| image | 300×300, magnitude from cropped k-space |
| action | Cartesian phase-encode row mask, 300 rows, ACS block of 32 rows locked on |
| budget | exact top-k, `k = round(s·300)`, identical for every method |
| metric | per-slice NMSE = ‖pred−target‖²/‖target‖², mean over slices |
| stochastic baselines | 5 seeds, mean ± sd reported; a win inside 2 sd is not a win |

Two reconstructors, both trained on the train split under *heuristic* masks only
so neither ever saw a learned mask:

- **U-Net A** (base 32, seed 0) — the net masks are optimised against.
- **U-Net B** (base 48, seed 1, independent mask draws) — a **held-out judge**
  used only for scoring. Every headline number is U-Net B.

The judge is not a formality. At s=0.40 the B5 profile reads −0.49% against
U-Net A and −1.97% against U-Net B: a full 1.5 pp of its apparent quality was
overfitting to the reconstructor it was trained against.

---

## Results

Val NMSE under the held-out judge (lower is better; **bold** = best at that s):

| method | s=0.25 | s=0.40 | s=0.50 | s=0.75 |
|--------|--------|--------|--------|--------|
| **hill climb (B5b)** | **0.02355** | **0.01842** | **0.01521** | **0.00763** |
| acs_vd_gaussian | 0.02450 | 0.01873 | 0.01538 | 0.00764 |
| STE profile (B5) | 0.02487 | 0.01910 | 0.01573 | 0.00782 |
| policy consensus mask | 0.02921 | 0.01935 | 0.01553 | 0.00764 |
| energy oracle (F7) | 0.02703 | 0.01958 | 0.01573 | 0.00782 |
| policy, instance-adaptive | 0.02785 | 0.01958 | 0.01610 | 0.00797 |
| acs_equispaced | 0.02591 | 0.01977 | 0.01590 | 0.00880 |
| acs_random | 0.02661 | 0.02072 | 0.01734 | 0.00923 |

Margin of the learned mask over the best heuristic, with the seed spread:

| s | hill climb | acs_vd_gaussian | gain | |
|---|-----------|-----------------|------|--|
| 0.25 | 0.02355 | 0.02450 ± 0.00006 | **+3.79%** | beats, > 2 sd |
| 0.40 | 0.01842 | 0.01873 ± 0.00005 | **+2.14%** | beats, > 2 sd |
| 0.50 | 0.01521 | 0.01538 ± 0.00007 | **+1.49%** | beats, > 2 sd |
| 0.75 | 0.00763 | 0.00764 ± 0.00001 | +0.18% | ties |

**Not an artefact of the reconstructor.** The same mask also beats
`acs_vd_gaussian` under plain zero-filled reconstruction at three of four
sparsities (+3.2%, +2.4%, +2.8%, −0.7%). A mask that had merely learned one
U-Net's quirks would not survive a different U-Net *and* no U-Net at all.

### How the mask was found

Greedy row swaps on the hard mask. The gradient w.r.t. the mask proposes
candidate swaps; each is evaluated exactly on the full train split and accepted
only if it improves. No relaxation, no straight-through estimator, no budget
drift. Every run terminated at a genuine local optimum — no improving swap among
24 exactly-evaluated proposals — after 0–8 accepted swaps.

Contributions are roughly two equal halves, and neither alone clears the bar at
every sparsity (train NMSE vs U-Net A):

| s | per-slice random VD | → best fixed draw | → after hill climb |
|---|---------------------|-------------------|--------------------|
| 0.25 | 0.022949 | 0.022346 (+2.63%) | 0.021684 (+2.96%) |
| 0.40 | 0.017228 | 0.016929 (+1.74%) | 0.016728 (+1.18%) |
| 0.50 | 0.013922 | 0.013663 (+1.86%) | 0.013663 (+0.00%) |
| 0.75 | 0.006805 | 0.006777 (+0.42%) | 0.006694 (+1.22%) |

---

## What the controls establish

### The reconstructor decides which mask is best (B4)

Mask ranking is not a property of the sampling pattern alone. At s=0.25, gain
from replacing zero-filled with the judge:

| mask | zero-filled | U-Net B | gain |
|------|-------------|---------|------|
| acs_equispaced | 0.03820 | 0.02591 | 32.2% |
| acs_random | 0.03819 | 0.02661 | 30.3% |
| hill climb | 0.03274 | 0.02355 | 28.1% |
| acs_vd_gaussian | 0.03382 | 0.02450 | 27.5% |
| policy consensus | 0.03470 | 0.02897 | 16.5% |
| policy, adaptive | 0.03318 | 0.02785 | 16.1% |
| energy oracle | 0.03147 | 0.02703 | 14.1% |

Masks that **spread out** gain most; **centre-concentrated** masks gain least.
A reconstructor carrying a prior can already predict the low-frequency centre,
so budget spent there is redundant. The energy oracle — optimal for raw energy
capture, and the best mask of all under zero-filled at s=0.25 — is the *worst*
performer once a prior exists.

This is why the learned policy underperformed: it was trained on zero-filled
NMSE plus a coarse-entropy term, both of which reward energy capture, which is
precisely the wrong signal once the reconstructor improves.

The B5b mask confirms this constructively: at s=0.25 the B5 profile is 13%
*worse* than the energy oracle under zero-filled while being 8.0% *better* under
the judge. Optimising against a prior-carrying reconstructor moves the mask away
from the energy-optimal one, by design.

### Instance-adaptivity does not pay (B3, corrected)

Give each slice **another slice's** scout (a derangement, zero fixed points) and
re-measure. Penalty relative to correct conditioning, averaged over s:

| policy | reconstructor | mean shuffle penalty |
|--------|---------------|----------------------|
| P3 | zero-filled | −0.40% |
| P3 | U-Net | −0.79% |
| A2 | zero-filled | −0.35% |
| A2 | U-Net | −0.80% |

Negative means the *wrong* scout did better. The effect is small and
sparsity-dependent: at s=0.25 the correct scout genuinely helps (+0.7% U-Net,
+2.3% zero-filled), while at s ≥ 0.40 conditioning is mildly harmful (−0.9% to
−1.7%). Collapsing the policy to a single static mask costs nothing at s ≥ 0.40
(−0.9% to −4.2%, i.e. an improvement) and costs ~4–5% at s=0.25.

Conditioning is worth at most ~2%, only at the tightest budget, and only against
a policy that is itself well behind the heuristics. **The conditional
architecture does not earn its complexity.**

### Variable-density randomisation is neutral

`acs_vd_gaussian` redraws per slice, so a fixed learned mask must beat an
average over draws. Comparing per-slice random against 16 fixed draws (train):

| s | per-slice random | best of 16 fixed | mean fixed draw |
|---|------------------|------------------|-----------------|
| 0.25 | 0.022949 | 0.022346 (+2.63%) | 0.023017 (−0.30%) |
| 0.40 | 0.017228 | 0.016929 (+1.74%) | 0.017213 (+0.09%) |
| 0.50 | 0.013922 | 0.013663 (+1.86%) | 0.013974 (−0.37%) |
| 0.75 | 0.006805 | 0.006777 (+0.42%) | 0.006798 (+0.10%) |

The mean fixed draw matches per-slice random to within ±0.4%. Randomisation is
neither helping nor hurting; what pays is *selecting* a good fixed draw. The
comparison against a fixed learned mask is therefore fair.

### No train/val overfitting

Rankings are identical on both splits, and the ~174K-parameter policy degrades
train→val by the same amount as zero-parameter heuristics (6.2% vs 5.6%/5.0% at
s=0.25 — pre-F9 metric, but a like-for-like ratio). Whatever is wrong with the
learned policy, it is not generalisation.

---

## Negative results worth reporting

1. **The fine D3PM prior is dead as an entropy term.** β=0 survived three
   independent checks (Stage 5, P2, a 100-epoch retrain). Its CE(t) is flat, so
   it supplies no usable gradient.
2. **The coarse-entropy proxy disagrees with reconstruction quality.** At
   s=0.25 the learned policy has the lowest H_c of any method and still loses on
   NMSE. Lower proxy entropy does not imply a better mask.
3. **A 300-number average beats the whole pipeline under zero-filled** (F7).
   Mean PE-row energy, fit on train, beats the learned policy at every sparsity
   under zero-filled reconstruction: +5.1%, +0.5%, +2.9%, +3.6% at
   s=0.25/0.40/0.50/0.75. (F7 originally recorded 17% at s=0.25; that was the
   pre-F9 clamped metric.) The learned policy was
   approximating that profile, slightly worse: selection frequency correlates
   r=0.93 with the VD prior and 0.83 with log row energy, 77–81% row overlap.
4. **More training does not help.** Budget-matched, a 50-epoch from-scratch
   policy on 100-epoch priors is indistinguishable from a 10-epoch warm start at
   every sparsity (F5).
5. **Relaxation-based mask optimisation underperforms discrete search.** The
   STE-gradient profile (B5) lost to `acs_vd_gaussian` at every sparsity; hill
   climbing on the same objective beat it at three of four. B5's negative
   verdict was an artefact of the optimiser, not a property of the task.

---

## Corrections and retractions

| claim, as previously recorded | status |
|-------------------------------|--------|
| P4: "50ep does not beat A2 10ep" | **superseded** — budget-matched they are equivalent (F5). Ship the cheap one, but not for the stated reason. |
| All pre-B1 FastMRI numbers | **train-set, single-draw, density-unmatched.** No val split existed; no seeding outside `lm_deepspeed.py`. Learned-vs-`acs_random` margins roughly halve under budget matching (F4). |
| All pre-F9 NMSE values | **~4× too low.** `nmse` used a 1e-8 absolute denominator floor; median ‖target‖² is 4.2e-9, so the floor was active on 64% of train and 67% of val slices, silently down-weighting low-energy images. Also distorted the policy's *training* loss (`itw/train.py:564`). Reconstructor training was unaffected (`itw/recon.py` uses its own 1e-12 floor). |
| B3: "deranging the scout improves NMSE by 6.4–7.1% at s=0.25" | **retracted** — metric artefact. Corrected, the sign reverses to +0.7…+2.4%: the correct scout helps there. |
| B3: "the policy's own static profile is the best mask in the study" | **retracted** — under the corrected metric `acs_vd_gaussian` wins at s=0.25/0.40/0.50. |
| B0: "the 100ep survival table is non-monotone and corrupts the entropy signal" | **retracted** (F1) — violations are ~3e-4 Monte-Carlo noise; the *winning* table violates more, and the induced s→t maps differ by ≤4 steps in 1000. |
| B5: "a static profile cannot beat variable density" | **superseded by B5b** — it can; the optimiser was the limit. |

---

## Limitations

- **Single-coil, one mid-slice per volume, 973 training images.** Small by
  fastMRI standards; a multi-coil study with full volumes could move these
  numbers.
- **The reconstructors are small** (1.9M / 4.3M parameters, trained on 973
  slices). A diffusion posterior sampler would carry a stronger prior, and B4
  shows the mask ranking depends on the reconstructor — so the *optimal* mask
  would likely shift again.
- **The hill climb reaches a local optimum, not a global one.** It stops when no
  single swap among 24 proposals improves; a wider neighbourhood or simulated
  annealing might do better.
- **The best fixed VD seed is selected on train**, which is a selection effect.
  It transfers (val gains are 1.5–3.8% against train gains of 1.6–5.5%) but is
  partly optimistic.
- **s=0.10 is excluded throughout**: `k = 30 < 32` ACS rows, so the mask is a
  forced centred block and no method has any freedom.
- **The generative prior is not load-bearing in the final result.** The frozen
  D3PM contributes nothing to the winning mask. Framing this work as
  diffusion-prior-driven acquisition would misrepresent it.

---

## Reproduction

```bash
python eval_fastmri_val.py                 # B2: held-out re-decide
python eval_fastmri_adaptivity.py          # B3: shuffle / static / adaptive
python train_fastmri_recon_rerank.py       # B4: reconstructor headroom
python train_fastmri_static_profile.py     # B5: STE profile + judge U-Net
python hill_climb_mask.py                  # B5b: discrete row-swap search
pytest tests/                              # 56 tests, incl. protocol guardrails
```

Artifacts: `models_hill_climb/hillclimb_s{25,40,50,75}.pth` (final masks),
`models_recon_unet/recon_unet{,_judge}.pth`,
`models_mask_gen_fastmri_nested/eval_val/eval_b{2,3,4,5,5b}*.json`.

Guardrail tests worth keeping: NMSE scale-invariance at FastMRI magnitudes
(catches F9), budget-exactness of every mask constructor, and train/val split
isolation.
