"""Empty-mask entropy penalty."""

from __future__ import annotations

import math

import torch

from itw.entropy import masked_cond_entropy


def test_empty_mask_is_log_n() -> None:
    n_classes = 8
    logits = torch.zeros(2, 1, 4, 1, n_classes)
    mask = torch.zeros(2, 1, 4, 1)
    h = masked_cond_entropy(logits, mask)
    assert abs(float(h) - math.log(n_classes)) < 1e-5


def test_nonempty_uniform_logits_is_log_n() -> None:
    n_classes = 8
    logits = torch.zeros(2, 1, 4, 1, n_classes)
    mask = torch.ones(2, 1, 4, 1)
    h = masked_cond_entropy(logits, mask)
    assert abs(float(h) - math.log(n_classes)) < 1e-5
