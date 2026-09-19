"""The image-domain H(C|Y) proxy must not reward the choice of observed pixels.

The original objective averaged predictive entropy over the *observed* pixels
only. That is minimised by observing whatever is easiest to predict: on binary
MNIST the learned mask observed 100% background and 0% of the digit at
s <= 0.5, and since background and the absorbing token are both 0 the D3PM
received an all-zero y. ``region="all"`` scores every pixel, so a mask can only
lower the proxy through what it adds to y.
"""

import torch

from itw.entropy import d3pm_cond_entropy_loss
from itw.eval import _proxy_free_metrics


class _FixedLogits:
    """Stand-in D3PM whose prediction ignores y, to isolate the mask's role."""

    def __init__(self, logits):
        self.logits = logits

    def model_predict(self, y, t, cond):
        return self.logits


def _setup():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8, 2)
    # Make the left half near-certain and the right half uncertain.
    logits[..., :4, 0] = 10.0
    logits[..., :4, 1] = -10.0
    logits[..., 4:, :] = 0.0
    left = torch.zeros(2, 1, 8, 8)
    left[..., :4] = 1.0
    return _FixedLogits(logits), left, 1.0 - left


def test_observed_region_rewards_observing_predictable_pixels() -> None:
    d3pm, left, right = _setup()
    y = torch.zeros(2, 1, 8, 8, dtype=torch.long)
    h_left = d3pm_cond_entropy_loss(d3pm, y, None, None, left, region="observed")
    h_right = d3pm_cond_entropy_loss(d3pm, y, None, None, right, region="observed")
    assert h_left < 0.01 < h_right          # the degenerate preference


def test_all_region_ignores_which_pixels_are_observed() -> None:
    d3pm, left, right = _setup()
    y = torch.zeros(2, 1, 8, 8, dtype=torch.long)
    h_left = d3pm_cond_entropy_loss(d3pm, y, None, None, left, region="all")
    h_right = d3pm_cond_entropy_loss(d3pm, y, None, None, right, region="all")
    assert torch.allclose(h_left, h_right)  # only y may move the proxy


def test_proxy_free_metrics_flag_uninformative_observations() -> None:
    d3pm, left, _right = _setup()
    x = torch.zeros(2, 1, 8, 8, dtype=torch.long)
    x[..., 6:] = 1                           # a "digit" on the far right
    y = x * left.long()                      # observes only background
    m = _proxy_free_metrics(d3pm, y, None, None, left, x)
    assert m["informative_frac"] == 0.0      # every observed value is 0
    assert 0.0 <= m["acc_unobserved"] <= 1.0
    # The fixed logits always predict 0 on the left, 0/1 tie (argmax 0) on the
    # right, so every hidden digit pixel is missed.
    assert m["acc_unobserved_fg"] == 0.0
    # Whole reconstruction: observed pixels count as known. The observed left
    # half is right by construction; the hidden right half is predicted 0, so
    # columns 4-5 (true 0) are right and the digit columns 6-7 are wrong.
    assert m["acc_full"] == 0.75
    assert m["recall_fg"] == 0.0
