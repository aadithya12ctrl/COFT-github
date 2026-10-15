"""Patch nanoGPT's ``CausalSelfAttention`` so CRS can reach the attention logits.

nanoGPT uses ``F.scaled_dot_product_attention`` (Flash Attention) whenever
PyTorch exposes it, which is always on a modern GPU runtime. That kernel never
materialises an explicit attention-logit tensor, so there is nothing for CRS to
add its bias to -- CRS would silently do nothing.

:func:`patch_model_with_crs` therefore forces the manual attention path on the
patched layers (``attn.flash = False``), registers the causal-mask buffer if the
flash path had skipped it, and binds :func:`coft.crs.apply_crs` plus its state
onto each attention module.

Cost: patched layers lose Flash Attention's memory fusion, so attention is
O(T^2) in memory again. That is the main reason ``batch_size`` has to come down
as ``delta`` climbs. Baselines are deliberately left unpatched, so they keep
Flash Attention -- the fair "how you would normally fine-tune GPT-2" comparison.
"""

from __future__ import annotations

import math
import types
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .crs import apply_crs

__all__ = [
    "patch_model_with_crs",
    "set_crs_ft",
    "set_crs_enabled",
    "disable_crs",
    "enable_crs",
    "crs_disabled",
    "is_patched",
]


def _coft_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Replacement ``CausalSelfAttention.forward`` with a CRS injection point.

    Identical to nanoGPT's manual (non-flash) branch, except for the CRS bias
    added to the pre-softmax logits during training.
    """
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    nh, hs = self.n_head, C // self.n_head
    q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
    k = k.view(B, T, nh, hs).transpose(1, 2)
    v = v.view(B, T, nh, hs).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

    # ── CRS: training-time only ──────────────────────────────────────────────
    if getattr(self, "crs_enabled", False) and self.training:
        att = self.apply_crs(att, q, k)

    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_dropout(self.c_proj(y))


def patch_model_with_crs(
    model: nn.Module,
    crs_lambda: float = 0.5,
    layers: list[int] | None = None,
) -> nn.Module:
    """Inject CRS into the attention layers of a nanoGPT ``GPT``.

    Args:
        model: the model to patch, in place.
        crs_lambda: CRS strength ``lambda``.
        layers: optional subset of layer indices. ``None`` patches all layers;
            a subset (e.g. the deeper half) keeps the rest on Flash Attention
            and is a cheap way to trade CRS coverage for speed/memory.
    """
    blocks = model.transformer.h
    n_layers = len(blocks)
    targets = set(range(n_layers) if layers is None else layers)
    block_size = model.config.block_size
    device = next(model.parameters()).device

    for layer_id, block in enumerate(blocks):
        if layer_id not in targets:
            continue
        attn = block.attn
        attn.flash = False  # force the manual path so CRS can touch the logits
        if not hasattr(attn, "bias"):
            causal = torch.tril(torch.ones(block_size, block_size, device=device))
            attn.register_buffer("bias", causal.view(1, 1, block_size, block_size))

        attn.crs_enabled = True
        attn.crs_lambda = float(crs_lambda)
        attn.crs_ft = 0.0
        attn.crs_layer_id = layer_id
        attn.crs_n_layers = n_layers
        attn.apply_crs = types.MethodType(apply_crs, attn)
        attn.forward = types.MethodType(_coft_attention_forward, attn)

    return model


def _patched_attns(model: nn.Module):
    for block in model.transformer.h:
        if hasattr(block.attn, "crs_enabled"):
            yield block.attn


def is_patched(model: nn.Module) -> bool:
    return any(True for _ in _patched_attns(model))


def set_crs_ft(model: nn.Module, ft: float) -> None:
    """Broadcast the current PFCM value to every patched layer."""
    for attn in _patched_attns(model):
        attn.crs_ft = float(ft)


def set_crs_enabled(model: nn.Module, enabled: bool) -> None:
    for attn in _patched_attns(model):
        attn.crs_enabled = bool(enabled)


def disable_crs(model: nn.Module) -> None:
    """Turn CRS off -- used for evaluation, generation and export."""
    set_crs_enabled(model, False)


def enable_crs(model: nn.Module) -> None:
    set_crs_enabled(model, True)


@contextmanager
def crs_disabled(model: nn.Module):
    """Context manager that disables CRS and restores the previous per-layer
    state afterwards."""
    previous = [(attn, attn.crs_enabled) for attn in _patched_attns(model)]
    for attn, _ in previous:
        attn.crs_enabled = False
    try:
        yield model
    finally:
        for attn, was_enabled in previous:
            attn.crs_enabled = was_enabled
