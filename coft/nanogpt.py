"""Thin bridge to Karpathy's nanoGPT.

COFT does not vendor a transformer implementation: it patches nanoGPT's
``model.py`` so the pretrained GPT-2 weights, the optimizer grouping and the
sampling loop are all the reference ones. This module makes ``model.py``
importable and loads a checkpoint.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

NANOGPT_URL = "https://github.com/karpathy/nanoGPT.git"
DEFAULT_PATH = Path(os.environ.get("NANOGPT_PATH", "third_party/nanoGPT"))

__all__ = ["NANOGPT_URL", "DEFAULT_PATH", "ensure_nanogpt", "load_nanogpt", "load_pretrained_gpt2"]


def ensure_nanogpt(path: str | os.PathLike | None = None, clone: bool = True) -> Path:
    """Return a directory containing nanoGPT's ``model.py``, cloning it if needed.

    Looks at ``path``, then ``$NANOGPT_PATH``, then ``./nanoGPT`` and ``.`` (so
    this also works unchanged inside a Colab session that already did
    ``%cd nanoGPT``).
    """
    candidates = [Path(path)] if path else []
    candidates += [DEFAULT_PATH, Path("nanoGPT"), Path(".")]
    for candidate in candidates:
        if (candidate / "model.py").is_file():
            return candidate

    target = Path(path) if path else DEFAULT_PATH
    if not clone:
        raise FileNotFoundError(
            f"nanoGPT not found at {target}. Clone it with "
            f"`python scripts/setup_nanogpt.py` or set NANOGPT_PATH."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", NANOGPT_URL, str(target)], check=True)
    return target


def load_nanogpt(path: str | os.PathLike | None = None):
    """Import nanoGPT and return ``(GPT, GPTConfig, CausalSelfAttention)``."""
    root = ensure_nanogpt(path)
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from model import GPT, CausalSelfAttention, GPTConfig  # noqa: E402  (path set above)

    return GPT, GPTConfig, CausalSelfAttention


def load_pretrained_gpt2(
    model_type: str = "gpt2",
    dropout: float = 0.1,
    device: str = "cuda",
    nanogpt_path: str | os.PathLike | None = None,
):
    """Load pretrained GPT-2 weights through nanoGPT's ``GPT.from_pretrained``.

    ``from_pretrained`` always forces ``vocab_size=50257``, ``block_size=1024``
    and ``bias=True`` to match the original checkpoint, so the usable COFT
    budget is ``T_infer + delta_max <= 1024``.
    """
    GPT, _, _ = load_nanogpt(nanogpt_path)
    model = GPT.from_pretrained(model_type, dict(dropout=dropout))
    return model.to(device)
