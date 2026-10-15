"""PRGG gating, against a stub that mimics nanoGPT's module layout.

The stub has ``transformer.wte`` and ``transformer.drop``, which is all PRGG
attaches to, so no GPT-2 weights are needed.
"""

import torch
import torch.nn as nn

from coft.prgg import PRGG


class StubGPT(nn.Module):
    """Minimal stand-in for nanoGPT's ``GPT``: embed, dropout, linear head."""

    def __init__(self, vocab=32, n_embd=8):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab, n_embd),
                drop=nn.Dropout(0.0),
            )
        )
        self.head = nn.Linear(n_embd, vocab, bias=False)

    def forward(self, idx):
        x = self.transformer.drop(self.transformer.wte(idx))
        self.hidden = x  # kept so tests can read the gated gradient
        return self.head(x)


def _nll(B=2, T=12, middle_hot=True):
    """Per-position NLL that is high in the middle third."""
    nll = torch.ones(B, T) * 0.5
    if middle_hot:
        nll[:, T // 3 : 2 * T // 3] = 4.0
    return nll


def test_gate_is_higher_for_forgotten_positions():
    prgg = PRGG(alpha=2.0)
    prgg.update_forgetting(_nll())
    gate = prgg.gate()
    assert gate.shape == (12,)
    assert gate[4:8].mean() > gate[:4].mean()
    assert torch.all((gate > 0) & (gate < 1))  # sigmoid range


def test_gate_is_none_before_any_measurement():
    assert PRGG().gate() is None


def test_gate_is_flat_when_alpha_is_zero():
    prgg = PRGG(alpha=0.0)
    prgg.update_forgetting(_nll())
    assert torch.allclose(prgg.gate(), torch.full((12,), 0.5))


def test_normalize_preserves_mean_scale():
    prgg = PRGG(alpha=2.0, normalize=True)
    prgg.update_forgetting(_nll())
    assert abs(prgg.gate().mean().item() - 1.0) < 1e-5


def test_gate_skipped_when_block_size_changed():
    """Delta moves between iterations; a stale gate must not be mis-aligned
    onto a different sequence length."""
    prgg = PRGG()
    prgg.update_forgetting(_nll(T=12))
    assert prgg.gate(seq_len=12) is not None
    assert prgg.gate(seq_len=16) is None


def test_per_position_mode_scales_gradients_by_position():
    torch.manual_seed(0)
    model = StubGPT()
    prgg = PRGG(alpha=2.0, mode="per_position").attach(model)

    idx = torch.randint(0, 32, (2, 12))
    nll = _nll()
    prgg.update_forgetting(nll)

    logits = model(idx)
    model.hidden.retain_grad()
    logits.sum().backward()

    grad_norms = model.hidden.grad.abs().mean(dim=(0, 2))
    # middle positions keep more gradient than the well-retained edges
    assert grad_norms[4:8].mean() > grad_norms[:4].mean()
    prgg.remove()


def test_scalar_mode_applies_one_multiplier_to_the_embedding_weight():
    torch.manual_seed(0)
    model = StubGPT()
    idx = torch.randint(0, 32, (2, 12))

    model(idx).sum().backward()
    ungated = model.transformer.wte.weight.grad.clone()

    model.zero_grad(set_to_none=True)
    prgg = PRGG(alpha=2.0, mode="scalar").attach(model)
    prgg.update_forgetting(_nll())
    model(idx).sum().backward()
    gated = model.transformer.wte.weight.grad

    expected = prgg.gate().mean()
    nonzero = ungated.abs() > 0
    ratio = gated[nonzero] / ungated[nonzero]
    assert torch.allclose(ratio, expected.expand(nonzero.sum()), atol=1e-5)
    prgg.remove()


def test_remove_restores_ungated_gradients():
    torch.manual_seed(0)
    model = StubGPT()
    idx = torch.randint(0, 32, (2, 12))
    with PRGG(alpha=2.0).attach(model) as prgg:
        prgg.update_forgetting(_nll())
        model(idx).sum().backward()
    model.zero_grad(set_to_none=True)
    # hooks gone: a second pass must not raise or gate anything
    model(idx).sum().backward()
    assert model.transformer.wte.weight.grad is not None


def test_unknown_mode_rejected():
    try:
        PRGG(mode="nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown PRGG mode")
