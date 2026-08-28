"""Coarse proxy must use full-profile entropy, not masked-only mean."""

from __future__ import annotations

import torch

from itw.entropy import coarse_cond_entropy_loss, mean_cond_entropy, pixel_entropy


class _ConstD3PM:
    def model_predict(self, y, t, cond):
        del t, cond
        # High entropy everywhere; unused y still needs a grad path in real models.
        n = 8
        logits = torch.zeros(*y.shape, n, device=y.device, dtype=torch.float32)
        return logits + y.float().unsqueeze(-1) * 0.0


def test_coarse_matches_mean_not_masked() -> None:
    d3pm = _ConstD3PM()
    y = torch.zeros(2, 1, 8, 1)
    t = torch.ones(2, dtype=torch.long)
    cond = torch.zeros(2, dtype=torch.long)
    mask = torch.zeros(2, 1, 8, 1)
    mask[:, :, :2, :] = 1
    h = coarse_cond_entropy_loss(d3pm, y, t, cond, mask)
    logits = d3pm.model_predict(y, t, cond)
    assert torch.allclose(h, mean_cond_entropy(logits))
    # Uniform 8-class entropy is log(8), not the 2-row masked mean of the same.
    assert float(pixel_entropy(logits).mean()) == float(h)
