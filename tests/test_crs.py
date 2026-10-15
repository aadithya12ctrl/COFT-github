"""CRS bias: where it lands, how it scales, and that it cannot break causality.

These tests exercise the bias term directly, so they need neither nanoGPT nor a
GPU.
"""

import torch

from coft.crs import crs_bias
from coft.pfcm import thirds


def _qk(B=2, nh=3, T=90, hs=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, nh, T, hs), torch.randn(B, nh, T, hs)


def test_bias_only_touches_middle_keys():
    q, k = _qk()
    bias = crs_bias(q, k, lam=0.5, ft=1.0, layer_id=5, n_layers=6)
    early, middle, late = thirds(q.shape[-2])
    assert torch.all(bias[..., early] == 0)
    assert torch.all(bias[..., late] == 0)
    assert bias[..., middle].abs().sum() > 0


def test_bias_vanishes_as_forgetting_goes_to_zero():
    q, k = _qk()
    assert torch.all(crs_bias(q, k, lam=0.5, ft=0.0, layer_id=5, n_layers=6) == 0)


def test_bias_scales_linearly_with_ft_and_lambda():
    q, k = _qk()
    base = crs_bias(q, k, lam=0.5, ft=0.2, layer_id=3, n_layers=6)
    assert torch.allclose(
        crs_bias(q, k, lam=0.5, ft=0.4, layer_id=3, n_layers=6), 2 * base, atol=1e-6
    )
    assert torch.allclose(
        crs_bias(q, k, lam=1.0, ft=0.2, layer_id=3, n_layers=6), 2 * base, atol=1e-6
    )


def test_deeper_layers_get_more_bias():
    q, k = _qk()
    shallow = crs_bias(q, k, lam=0.5, ft=1.0, layer_id=0, n_layers=12).abs().sum()
    deep = crs_bias(q, k, lam=0.5, ft=1.0, layer_id=11, n_layers=12).abs().sum()
    assert deep > shallow
    assert torch.allclose(deep / shallow, torch.tensor(12.0), rtol=1e-4)


def test_bias_is_content_conditioned():
    """A key aligned with a query gets a positive boost; an anti-aligned one
    gets a negative boost -- CRS promotes relevant middle tokens only."""
    B, nh, T, hs = 1, 1, 9, 4
    q = torch.zeros(B, nh, T, hs)
    k = torch.zeros(B, nh, T, hs)
    q[..., 0] = 1.0  # every query points along dim 0
    k[:, :, 3, 0] = 1.0  # middle key 3 aligned
    k[:, :, 4, 0] = -1.0  # middle key 4 anti-aligned
    k[:, :, 5, 1] = 1.0  # middle key 5 orthogonal

    bias = crs_bias(q, k, lam=1.0, ft=1.0, layer_id=0, n_layers=1)
    assert bias[0, 0, 0, 3] > 0
    assert bias[0, 0, 0, 4] < 0
    assert abs(bias[0, 0, 0, 5].item()) < 1e-6


def test_masked_positions_stay_masked():
    """The bias is finite, so adding it to a -inf causal mask cannot leak
    information from the future."""
    q, k = _qk(T=90)
    T = q.shape[-2]
    att = torch.zeros(q.shape[0], q.shape[1], T, T)
    causal = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    att = att.masked_fill(causal == 0, float("-inf"))

    biased = att + crs_bias(q, k, lam=0.5, ft=1.0, layer_id=5, n_layers=6)
    assert torch.all(torch.isinf(biased[..., :, :][causal.expand_as(att) == 0]))
    weights = torch.softmax(biased, dim=-1)
    assert torch.all(weights[causal.expand_as(att) == 0] == 0)
    assert torch.allclose(weights.sum(-1), torch.ones(weights.shape[:-1]), atol=1e-5)
