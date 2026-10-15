"""PRGG -- Positional Recovery Gradient Gating.

Positions the model has forgotten should contribute more to the update than
positions it already predicts well:

    g_i = sigmoid(alpha * F_i / F_bar)

where ``F_i`` is the mean NLL at position ``i`` (from PFCM) and ``F_bar`` its
mean over positions. The gate multiplies the gradient flowing back through the
embedded sequence, so middle-context positions with high ``F_i`` keep more
gradient headroom and well-retained edge positions are damped.

Two attachment modes are provided:

``per_position`` (default)
    A hook on the embedding output ``x`` of shape ``(B, T, C)``. Because the
    position axis is still intact there, each position is scaled by its own
    ``g_i`` -- this is the formulation in the paper.

``scalar``
    A hook on the token-embedding *weight* (``wte.weight``, shape
    ``(V, C)``). The position axis has already been contracted away in that
    gradient, so only the mean gate ``mean_i g_i`` can be applied. This is the
    approximation used by the original Colab notebook; kept for reproducing it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PRGG"]


class PRGG:
    def __init__(
        self,
        alpha: float = 2.0,
        mode: str = "per_position",
        normalize: bool = False,
    ) -> None:
        """
        Args:
            alpha: gate sharpness. 0 disables gating (constant 0.5 gate);
                larger values spread the gate further apart across positions.
            mode: ``"per_position"`` or ``"scalar"`` (see module docstring).
            normalize: divide the gate by its mean so the *average* gradient
                scale is unchanged and only the relative positional weighting
                is affected. Off by default (faithful to the paper, where the
                raw sigmoid also shrinks the overall step by ~2x).
        """
        if mode not in ("per_position", "scalar"):
            raise ValueError(f"unknown PRGG mode: {mode!r}")
        self.alpha = float(alpha)
        self.mode = mode
        self.normalize = bool(normalize)

        self.per_pos_nll: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None
        self._handles: list = []

    # ── wiring ───────────────────────────────────────────────────────────────
    def attach(self, model: nn.Module) -> PRGG:
        """Attach to a nanoGPT-style ``GPT``: ``transformer.{wte, drop}``."""
        if self.mode == "scalar":
            self._handles.append(model.transformer.wte.weight.register_hook(self._weight_hook))
        else:
            self._handles.append(model.transformer.drop.register_forward_hook(self._fwd_hook))
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> PRGG:
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    # ── state ────────────────────────────────────────────────────────────────
    def update_forgetting(self, per_pos_nll: torch.Tensor) -> None:
        """Feed in the newest ``(B, T)`` per-position NLL (from PFCM)."""
        self.per_pos_nll = per_pos_nll.detach()

    def gate(self, seq_len: int | None = None) -> torch.Tensor | None:
        """Current gate ``g_i``, shape ``(T,)``, or ``None`` before the first
        PFCM measurement."""
        if self.per_pos_nll is None:
            return None
        f_i = self.per_pos_nll.mean(0)
        if seq_len is not None and f_i.shape[0] != seq_len:
            return None  # block size changed this step; skip rather than mis-align
        g = torch.sigmoid(self.alpha * f_i / (f_i.mean() + 1e-8))
        if self.normalize:
            g = g / (g.mean() + 1e-8)
        self.last_gate = g
        return g

    # ── hooks ────────────────────────────────────────────────────────────────
    def _fwd_hook(self, module, inputs, output):
        if not torch.is_tensor(output) or not output.requires_grad:
            return output  # eval / no_grad: nothing to gate

        seq_len = output.shape[1]

        def grad_hook(grad):
            g = self.gate(seq_len)
            if g is None:
                return grad
            return grad * g.to(grad.dtype).to(grad.device).view(1, -1, 1)

        output.register_hook(grad_hook)
        return output

    def _weight_hook(self, grad):
        g = self.gate()
        if g is None:
            return grad
        return grad * g.mean().to(grad.dtype).to(grad.device)
