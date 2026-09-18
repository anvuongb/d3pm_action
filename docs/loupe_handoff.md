# LOUPE baseline — resume instructions

State as of commit `b55357a`. The port is **built and validated**; **no LOUPE
training run has completed**. Everything below is what a machine with more GPU
time needs to finish the comparison.

Why this exists: `docs/fastmri_results.md` claims our hill-climbed static mask
beats `acs_vd_gaussian`. LOUPE (Bahadir et al. 2020) is the relaxation-based
prior art for learned MRI sampling, so the claim is only meaningful against it.

---

## 1. First, verify the mask actually learns

**Do this before launching anything long.** It is the step that was interrupted.

```bash
uv run python train_loupe_baseline.py --sparsities 0.25 --mask rows \
    --epochs 25 --filt 16 --batch 16 --tag _sanity
```

Watch the `mask churn N/75` column.

| observation | meaning |
|-------------|---------|
| churn > 0 and falling, val NMSE dropping toward ~0.02x | working; proceed to §2 |
| churn stays 0 from epoch 0, NMSE stuck near 0.13 | **gradient through `ThresholdRandomMask` is dead — debug before spending GPU hours** |

A 2-epoch smoke on 64 images gave churn `0/75` and NMSE `0.13634`. That is
consistent with only 8 optimiser steps, but it is *indistinguishable* from a
dead gradient, which is exactly why this check comes first. If churn is 0,
inspect `d(loss)/d(prob.logit)`: the sigmoid at `sample_slope=12` saturates when
`|p - thresh|` is large, and `rescale_prob_map` can push `p` to 0/1.

## 2. Main run: `rows`, all sparsities

`rows` is the like-for-like comparison — whole phase-encode lines, as our masks
are. Their native 2D point mask is not Cartesian-realizable.

```bash
uv run python train_loupe_baseline.py --mask rows --filt 64 --batch 32 --epochs 400
```

Then, for reference only, the native 2D variant:

```bash
uv run python train_loupe_baseline.py --mask native --filt 64 --batch 32 --epochs 400
```

Results land in `models_mask_gen_fastmri_nested/eval_val/eval_loupe.json`,
masks in `models_loupe/`. The script already scores LOUPE against
`ours_hillclimb` and a 5-seed `acs_vd_gaussian` under `zf`, `unetA` and the
held-out judge `unetB`, and ranks by the judge.

### On epoch count — do not use 60

Upstream trains **60 epochs over 60k images ≈ 112k steps**. Our split is **973
images**, so at batch 32 one epoch is ~30 steps and 60 epochs is ~1,800 steps —
roughly **60x less optimisation than LOUPE was published with**. Using 60 here
would understate the baseline and flatter our result, which is the exact failure
mode this project has been correcting for.

Train to convergence instead and let `mask_churn` be the evidence: stop when it
sits at 0 for ~20 consecutive epochs, and record the tail in the write-up.
400 epochs at batch 32 is ~12k steps; raise it if churn has not settled.

### Timings measured on an RTX 4060 (8 GB)

| config | per epoch | peak VRAM |
|--------|-----------|-----------|
| `--filt 64 --batch 4` | 70 s | 3.04 GiB |
| `--filt 32 --batch 8` | 21 s | 2.85 GiB |

`--batch 32` at `--filt 32` needs ~11 GiB, so it needs a bigger card than the
4060. Budget ~8 runs (4 sparsities x 2 mask modes). If GPU time is tight, run
`rows` only and treat `native` as optional.

Worth a capacity control: one `--filt 32` run at s=0.25 to confirm the *mask*
does not depend materially on reconstructor width. If the selected row set is
nearly the same, the cheaper setting is usable for the rest.

## 3. What is already verified

`tests/test_loupe_port.py` (6 tests) pins the port against the **unmodified
upstream layers**, whose reference outputs are committed in `tests/data/`:

| quantity | agreement |
|----------|-----------|
| `ProbMask` | 1.4e-07 |
| `RescaleProbMap` | 2.0e-07 |
| `ThresholdRandomMask` | 2.6e-07 |

The FFT path is **not** pinned, by design. It differs by 1.7e-05, which is TF's
own error, not the port's — an isolated `ifft2(fft2(x))` round-trip containing no
LOUPE code gives:

| N | torch | TF 1.15 |
|---|-------|---------|
| 256 | 3.8e-07 | 2.0e-06 |
| 300 | 4.3e-07 | **2.9e-05** |

TF 1.15's CPU FFT degrades at non-power-of-two sizes (300 = 2^2·3·5^2), so it
cannot be the reference. Do not "fix" the port to match it.

## 4. Caveats that must survive into any write-up

These are forced by LOUPE's design, not choices we made, and each one favours or
penalises a side — so state them:

1. **LOUPE FFTs a real magnitude image.** Its k-space is therefore
   Hermitian-symmetric, i.e. half-redundant — an *easier* problem than our true
   complex k-space. We train it under its own assumption and transfer the mask.
2. **LOUPE trains its reconstructor jointly**; ours is frozen. That is a real
   advantage for LOUPE and is not neutralised: we compare only the *masks*, each
   scored by the same judge.
3. **Its budget holds in expectation** (`RescaleProbMap` sets `mean(p)=s`), not
   exactly. We take a budget-exact top-k of the learned probability map.
4. **It has no ACS constraint**; our masks lock a 32-row ACS block. LOUPE's mask
   is used exactly as learned — it is free to spend budget where it likes.
5. **Unshifted k-space.** `tf.fft2d` puts DC at index 0; our masks are
   fftshift-centred. `probmask_rows_to_centred()` handles this. Getting it wrong
   puts LOUPE's low-frequency preference at the edges of our grid and would make
   it look catastrophically bad — if LOUPE scores ~0.5 NMSE, suspect this first.
6. **Our two optimisers already re-derive both prior-art families** — B5 is
   relaxation-based like LOUPE, B5b is combinatorial like Gozcu et al. 2018. The
   paper (`docs/paper.md` §2) disclaims novelty on the method for this reason.
   A LOUPE number does not change that; it calibrates it.

## 5. Reproducing the TF cross-check (optional)

Only needed to re-derive `tests/data/` or to compare training behaviour against
the original implementation. Two-line image:

```dockerfile
FROM docker.io/tensorflow/tensorflow:1.15.0-py3
RUN pip install --no-cache-dir "keras==2.2.4" "h5py<3" "numpy<1.19"
```

```bash
git clone --depth 1 https://github.com/cagladbahadir/LOUPE.git
podman build -t loupe-tf115 -f Containerfile .
podman run --rm -v "$PWD:/work:z" loupe-tf115 python /work/tf_reference.py
```

CPU only — TF 1.15 needs CUDA 10, and cuDNN 7 has no Ada (sm_89) support. It is
slow: 32 images at `filt=8` took 74 s/epoch, so it is a fidelity check, not a
training route. `tf_reference.py` and `train_loupe_fastmri.py` (the TF-side
runner, which pads 300->304 around their 4-level U-Net since 300 is not
divisible by 16) are not committed; §3's committed reference arrays are their
output.

## 6. When the numbers are in

- Add a LOUPE row to the results table in `docs/fastmri_results.md` §Results.
- Add it to Table `tab:mri-main` in `docs/latex/main.tex` and the matching table
  in `docs/paper.md` §6.4.
- If LOUPE beats our hill-climbed mask, the paper's positive claim must be
  weakened accordingly — §6.4 and the conclusion both assert our mask is the
  best tested. Check that claim against the new number before shipping.
