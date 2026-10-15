"""CRS -- Content-conditioned Recency Suppression.

Transformers attend in a U-shape: strongly to the start and to the most recent
tokens, weakly to the middle. CRS counteracts that *during training only* by
adding a bias to the pre-softmax attention logits of keys in the middle third:

    a~[l,h,i,j] = a[l,h,i,j] + 1[j in M] * (l/L) * lambda * F_t * cos(q_i, k_j)

Each factor does one job:

* ``1[j in M]``   -- only middle-context keys are boosted.
* ``(l / L)``     -- deeper layers get more of the bias, where positional
                     routing is decided, leaving early feature extraction alone.
* ``lambda``      -- strength hyperparameter.
* ``F_t``         -- self-annealing: the bias vanishes as forgetting is fixed.
* ``cos(q, k)``   -- *content*-conditioned: only semantically relevant middle
                     keys are promoted, so CRS cannot force attention onto
                     irrelevant tokens.

The bias is removed at inference (``F_t = 0`` or ``crs_enabled = False``), so a
COFT-trained model is architecturally identical to its base model at deploy time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pfcm import thirds

__all__ = ["CRSState", "crs_bias", "apply_crs"]


class CRSState:
    """Per-layer CRS state, mixed into each attention module by
    :func:`coft.patching.patch_model_with_crs`."""

    crs_enabled: bool = True
    crs_lambda: float = 0.5
    crs_ft: float = 0.0
    crs_layer_id: int = 0
    crs_n_layers: int = 1


def crs_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    lam: float,
    ft: float,
    layer_id: int,
    n_layers: int,
) -> torch.Tensor:
    """The additive bias term, shape ``(1, 1, 1, T)`` broadcast times ``(B, nh, T, T)``.

    Args:
        q, k: ``(B, nh, T, hs)`` queries and keys.
        lam: strength ``lambda``.
        ft: current PFCM value ``F_t``.
        layer_id: 0-based layer index ``l``.
        n_layers: total layers ``L``.
    """
    T = q.shape[-2]
    _, middle, _ = thirds(T)

    middle_mask = torch.zeros(T, dtype=q.dtype, device=q.device)
    middle_mask[middle] = 1.0

    cos_sim = torch.einsum(
        "bnqd,bnkd->bnqk", F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    )

    depth_scale = (layer_id + 1) / n_layers
    return middle_mask.view(1, 1, 1, T) * (depth_scale * lam * ft) * cos_sim


def apply_crs(self, att: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Bound as a method on each patched attention module.

    ``att`` is the ``(B, nh, T, T)`` *pre-softmax* logit tensor, already causally
    masked. Masked entries are ``-inf``, so adding a finite bias leaves them masked.
    """
    if not self.crs_enabled or self.crs_ft <= 0:
        return att
    return att + crs_bias(
        q,
        k,
        lam=self.crs_lambda,
        ft=self.crs_ft,
        layer_id=self.crs_layer_id,
        n_layers=self.crs_n_layers,
    )
