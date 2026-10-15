"""PFCM behaviour: sign, magnitude and the per-position tensor PRGG consumes.

CPU-only, no nanoGPT or GPT-2 weights needed.
"""

import torch

from coft.pfcm import compute_pfcm, per_position_nll, positional_curve, thirds


def _logits_with_middle_penalty(B=2, T=90, V=16, penalty=5.0):
    """Logits that are confident everywhere except the middle third, where the
    correct class is penalised -- i.e. a model that forgot the middle."""
    targets = torch.randint(0, V, (B, T))
    logits = torch.zeros(B, T, V)
    logits.scatter_(2, targets.unsqueeze(-1), 10.0)
    _, middle, _ = thirds(T)
    logits[:, middle, :] = logits[:, middle, :].scatter(
        2, targets[:, middle].unsqueeze(-1), 10.0 - penalty
    )
    return logits, targets


def test_thirds_partition_is_contiguous():
    early, middle, late = thirds(99)
    assert (early.start, early.stop) == (0, 33)
    assert (middle.start, middle.stop) == (33, 66)
    assert (late.start, late.stop) == (66, 99)


def test_thirds_rejects_too_short_sequences():
    try:
        thirds(2)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a sequence shorter than 3")


def test_pfcm_positive_when_middle_is_worse():
    logits, targets = _logits_with_middle_penalty()
    result = compute_pfcm(logits, targets)
    assert result.ft > 0
    assert result.nll_middle > result.nll_early
    assert result.nll_middle > result.nll_late


def test_pfcm_near_zero_for_uniform_difficulty():
    torch.manual_seed(0)
    logits = torch.randn(4, 96, 32)
    targets = torch.randint(0, 32, (4, 96))
    assert abs(compute_pfcm(logits, targets).ft) < 0.5


def test_pfcm_negative_when_edges_are_worse():
    logits, targets = _logits_with_middle_penalty()
    early, _, late = thirds(logits.shape[1])
    logits[:, early, :] = 0.0  # make the edges uninformative instead
    logits[:, late, :] = 0.0
    assert compute_pfcm(logits, targets).ft < 0


def test_per_position_nll_shape_and_detachment():
    logits, targets = _logits_with_middle_penalty()
    logits.requires_grad_(True)
    nll = per_position_nll(logits, targets)
    assert nll.shape == targets.shape
    assert compute_pfcm(logits, targets).per_pos_nll.requires_grad is False


def test_positional_curve_binning():
    logits, targets = _logits_with_middle_penalty(T=96)
    curve = positional_curve(compute_pfcm(logits, targets).per_pos_nll, n_bins=12)
    assert len(curve) == 12
    # the middle bins should be the worst ones
    assert max(curve[4:8]) >= max(curve[:4] + curve[8:])
