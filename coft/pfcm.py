"""PFCM -- Positional Forgetting Contrast Metric.

    F_t = PPL_middle - (PPL_early + PPL_late) / 2

The sequence is split into three equal thirds (early / middle / late) and the
mean per-token NLL of each third is compared. F_t > 0 means the model is worse
in the middle of its context than at the edges -- the "lost in the middle"
signature. F_t -> 0 means retention is uniform across positions.

Everything here works on raw per-position NLL, so the same function serves
three purposes in COFT:

* the scalar F_t that drives the delta scheduler (:mod:`coft.scheduler`)
  and the CRS bias (:mod:`coft.crs`),
* the per-position vector F_i that drives PRGG (:mod:`coft.prgg`),
* the reported evaluation metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = ["PFCMResult", "per_position_nll", "thirds", "compute_pfcm", "positional_curve"]


@dataclass
class PFCMResult:
    """Outcome of one PFCM measurement.

    Attributes:
        ft: the PFCM scalar ``F_t`` (in nats; positive = middle is worse).
        per_pos_nll: ``(B, T)`` per-position NLL, detached. Consumed by PRGG.
        nll_early / nll_middle / nll_late: mean NLL of each third (nats).
    """

    ft: float
    per_pos_nll: torch.Tensor
    nll_early: float
    nll_middle: float
    nll_late: float

    @property
    def ppl_middle(self) -> float:
        return float(torch.exp(torch.tensor(self.nll_middle)))

    def as_dict(self) -> dict:
        return {
            "pfcm": self.ft,
            "nll_early": self.nll_early,
            "nll_middle": self.nll_middle,
            "nll_late": self.nll_late,
        }


def per_position_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Un-reduced cross-entropy, reshaped back to ``(B, T)``."""
    B, T, V = logits.shape
    nll = F.cross_entropy(
        logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none"
    )
    return nll.view(B, T)


def thirds(seq_len: int) -> tuple[slice, slice, slice]:
    """Early / middle / late slices. The middle third is the region ``M``
    referenced by both CRS and PRGG, so all components agree on it."""
    third = seq_len // 3
    if third == 0:
        raise ValueError(f"sequence length {seq_len} is too short to split into thirds")
    return slice(0, third), slice(third, 2 * third), slice(2 * third, seq_len)


def compute_pfcm(logits: torch.Tensor, targets: torch.Tensor) -> PFCMResult:
    """Compute PFCM for one batch of logits.

    Args:
        logits: ``(B, T, V)`` model output. Pass ``logits.detach()`` from a
            training step -- PFCM is a measurement, never a loss term.
        targets: ``(B, T)`` next-token ids.
    """
    nll = per_position_nll(logits, targets)
    early, middle, late = thirds(nll.shape[1])

    nll_e = nll[:, early].mean().item()
    nll_m = nll[:, middle].mean().item()
    nll_l = nll[:, late].mean().item()

    return PFCMResult(
        ft=nll_m - (nll_e + nll_l) / 2.0,
        per_pos_nll=nll.detach(),
        nll_early=nll_e,
        nll_middle=nll_m,
        nll_late=nll_l,
    )


def positional_curve(per_pos_nll: torch.Tensor, n_bins: int = 32) -> list[float]:
    """Mean NLL binned along the position axis -- the U-shaped curve that PFCM
    summarises into a single number. Useful for plots and ablation tables."""
    nll = per_pos_nll.mean(0)
    T = nll.shape[0]
    n_bins = min(n_bins, T)
    edges = torch.linspace(0, T, n_bins + 1).long()
    return [nll[edges[i] : edges[i + 1]].mean().item() for i in range(n_bins)]
