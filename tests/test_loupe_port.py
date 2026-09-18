"""Pin itw/loupe.py against the upstream LOUPE (Keras/TF1) implementation.

Reference arrays in tests/data/ were produced by the unmodified upstream layers
(ProbMask -> RescaleProbMap -> ThresholdRandomMask) running in a
tensorflow==1.15 + keras==2.2.4 container on fixed inputs. The FFT path is not
pinned here: TF 1.15's CPU FFT is itself inaccurate at non-power-of-two sizes
(measured 2.9e-5 relative round-trip error at N=300, against 4.3e-7 for torch),
so it cannot serve as a reference. The mask mechanism is what the port must get
right, and it is what these tests check.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from itw.loupe import PMASK_SLOPE, SAMPLE_SLOPE, Loupe, rescale_prob_map

DATA = Path(__file__).parent / "data"
SPARSITY = 0.25


def _load(name: str) -> torch.Tensor:
    """tests/data are NHWC as Keras wrote them; return NCHW."""
    a = torch.tensor(np.load(DATA / name), dtype=torch.float64)
    return a.permute(0, 3, 1, 2) if a.ndim == 4 else a.permute(2, 0, 1)[None]


def test_prob_mask_matches_upstream() -> None:
    logit = _load("val_logit.npy")
    got = torch.sigmoid(PMASK_SLOPE * logit)
    assert torch.allclose(got, _load("tfref_p.npy")[:1], atol=1e-6)


def test_rescale_prob_map_matches_upstream() -> None:
    logit = _load("val_logit.npy")
    got = rescale_prob_map(torch.sigmoid(PMASK_SLOPE * logit), SPARSITY)
    assert torch.allclose(got, _load("tfref_ps.npy")[:1], atol=1e-6)
    # The whole point of the layer: mean equals the requested sparsity.
    assert float(got.mean()) == pytest.approx(SPARSITY, abs=1e-6)


def test_threshold_random_mask_matches_upstream() -> None:
    logit, thresh = _load("val_logit.npy"), _load("val_thresh.npy")
    ps = rescale_prob_map(torch.sigmoid(PMASK_SLOPE * logit), SPARSITY)
    got = torch.sigmoid(SAMPLE_SLOPE * (ps - thresh))
    assert torch.allclose(got, _load("tfref_m.npy"), atol=1e-6)


def test_rescale_handles_both_branches() -> None:
    """r<=1 scales x down; r>1 scales (1-x) instead. Both must stay in [0,1]."""
    for mean_p, s in ((0.8, 0.25), (0.1, 0.75)):
        x = torch.full((1, 1, 16, 16), mean_p, dtype=torch.float64)
        out = rescale_prob_map(x, s)
        assert float(out.mean()) == pytest.approx(s, abs=1e-9)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_row_mask_is_constant_along_readout() -> None:
    """The Cartesian variant must select whole PE lines."""
    m = Loupe(64, 48, SPARSITY, rows=True, filt=4)
    p = m.prob()
    assert p.shape == (1, 1, 64, 1)
    hard = m.sampled_mask(2, torch.device("cpu"), hard=True)
    assert hard.shape == (2, 1, 64, 1)


def test_native_mask_is_two_dimensional() -> None:
    m = Loupe(64, 48, SPARSITY, rows=False, filt=4)
    assert m.prob().shape == (1, 1, 64, 48)
