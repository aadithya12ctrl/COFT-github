"""The CRS patch, against a stub that mirrors nanoGPT's ``CausalSelfAttention``
API (``c_attn``, ``c_proj``, ``attn_dropout``, ``resid_dropout``, ``flash``).

This is the test that would have caught the Flash Attention bug: it asserts
that the patched forward actually routes through the manual attention path and
that the CRS bias changes the output only while training.
"""

import torch
import torch.nn as nn

from coft.patching import (
    crs_disabled,
    disable_crs,
    enable_crs,
    is_patched,
    patch_model_with_crs,
    set_crs_ft,
)

N_EMBD, N_HEAD, BLOCK = 16, 2, 64


class StubAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(0.0)
        self.n_head, self.n_embd = N_HEAD, N_EMBD
        self.flash = True  # what nanoGPT sets on a modern PyTorch

    def forward(self, x):  # unpatched: stands in for the flash path
        return self.c_proj(x)


class StubBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = StubAttention()


class StubConfig:
    block_size = BLOCK


class StubGPT(nn.Module):
    def __init__(self, n_layers=4):
        super().__init__()
        blocks = nn.ModuleList(StubBlock() for _ in range(n_layers))
        self.transformer = nn.ModuleDict(dict(h=blocks))
        self.config = StubConfig()


def _model():
    torch.manual_seed(0)
    return patch_model_with_crs(StubGPT(), crs_lambda=0.5)


def test_patch_sets_state_on_every_layer():
    model = _model()
    assert is_patched(model)
    for i, block in enumerate(model.transformer.h):
        attn = block.attn
        assert attn.flash is False  # manual path, or CRS cannot see the logits
        assert attn.crs_layer_id == i
        assert attn.crs_n_layers == 4
        assert attn.bias.shape == (1, 1, BLOCK, BLOCK)


def test_patch_subset_of_layers():
    model = patch_model_with_crs(StubGPT(), layers=[2, 3])
    assert [hasattr(b.attn, "crs_enabled") for b in model.transformer.h] == [
        False,
        False,
        True,
        True,
    ]
    assert model.transformer.h[0].attn.flash is True  # untouched: keeps flash


def test_unpatched_model_reports_not_patched():
    assert is_patched(StubGPT()) is False


def test_patched_forward_runs_and_preserves_shape():
    model = _model()
    x = torch.randn(2, 30, N_EMBD)
    assert model.transformer.h[0].attn(x).shape == x.shape


def test_crs_changes_output_only_when_training_and_ft_positive():
    model = _model()
    attn = model.transformer.h[-1].attn
    x = torch.randn(2, 30, N_EMBD)

    model.train()
    set_crs_ft(model, 0.0)
    baseline = attn(x)

    set_crs_ft(model, 1.0)
    with_crs = attn(x)
    assert not torch.allclose(baseline, with_crs)  # F_t > 0: bias is live

    model.eval()
    assert torch.allclose(attn(x), baseline)  # eval: bias gone


def test_disable_and_enable_crs():
    model = _model()
    attn = model.transformer.h[-1].attn
    x = torch.randn(2, 30, N_EMBD)
    model.train()
    set_crs_ft(model, 1.0)
    biased = attn(x)

    disable_crs(model)
    assert all(b.attn.crs_enabled is False for b in model.transformer.h)
    unbiased = attn(x)
    assert not torch.allclose(biased, unbiased)

    enable_crs(model)
    assert torch.allclose(attn(x), biased)


def test_crs_disabled_context_manager_restores_state():
    model = _model()
    model.transformer.h[0].attn.crs_enabled = False  # a per-layer choice to preserve
    with crs_disabled(model):
        assert all(b.attn.crs_enabled is False for b in model.transformer.h)
    states = [b.attn.crs_enabled for b in model.transformer.h]
    assert states == [False, True, True, True]


def test_attention_stays_causal_after_patching():
    """Changing a late token must not change an earlier token's output."""
    model = _model()
    attn = model.transformer.h[-1].attn
    model.train()
    set_crs_ft(model, 1.0)

    x = torch.randn(1, 30, N_EMBD)
    y = x.clone()
    y[:, -1, :] += 10.0
    assert torch.allclose(attn(x)[:, :-1], attn(y)[:, :-1], atol=1e-5)
