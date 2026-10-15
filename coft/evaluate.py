"""Evaluation at inference length.

The point of COFT is that training happens at ``T + delta`` but deployment
happens at ``T``, so evaluation *always* uses ``t_infer`` and *always* runs with
CRS off -- the bias is a training-time intervention and must never flatter the
numbers it is judged by.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import TokenDataset
from .patching import crs_disabled, is_patched
from .pfcm import compute_pfcm, positional_curve

__all__ = ["evaluate", "EvalResult"]


class EvalResult(dict):
    """Plain dict of metrics, with a readable one-line form."""

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"val_loss {self['val_loss']:.4f} | PFCM {self['pfcm']:+.4f} "
            f"(early {self['nll_early']:.3f} / mid {self['nll_middle']:.3f} / "
            f"late {self['nll_late']:.3f})"
        )


@torch.no_grad()
def evaluate(
    model,
    data: TokenDataset,
    *,
    t_infer: int,
    batch_size: int,
    eval_iters: int,
    ctx,
    split: str = "val",
    curve_bins: int = 0,
) -> EvalResult:
    """Mean loss and PFCM over ``eval_iters`` batches at length ``t_infer``.

    Args:
        curve_bins: if > 0, also return the binned positional NLL curve, so the
            U-shape itself can be plotted rather than only its PFCM summary.
    """
    was_training = model.training
    model.eval()

    losses, fts, mids, earlies, lates = [], [], [], [], []
    curves = []

    guard = crs_disabled(model) if is_patched(model) else _nullguard()
    with guard:
        for _ in range(eval_iters):
            x, y = data.get_batch(split, t_infer, batch_size)
            with ctx:
                logits, loss = model(x, y)
            losses.append(loss.item())
            result = compute_pfcm(logits.detach(), y)
            fts.append(result.ft)
            earlies.append(result.nll_early)
            mids.append(result.nll_middle)
            lates.append(result.nll_late)
            if curve_bins:
                curves.append(positional_curve(result.per_pos_nll, curve_bins))

    if was_training:
        model.train()

    out = EvalResult(
        val_loss=float(np.mean(losses)),
        pfcm=float(np.mean(fts)),
        pfcm_std=float(np.std(fts)),
        nll_early=float(np.mean(earlies)),
        nll_middle=float(np.mean(mids)),
        nll_late=float(np.mean(lates)),
    )
    if curve_bins:
        out["positional_curve"] = np.mean(np.array(curves), axis=0).tolist()
    return out


class _nullguard:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
