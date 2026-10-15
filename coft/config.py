"""Experiment configuration.

One flat dataclass holds every knob, so a run is fully described by a single
YAML file in ``configs/`` plus optional CLI overrides. ``validate()`` catches
the two mistakes that otherwise fail deep inside a training step: exceeding
GPT-2's position-embedding table, and a block size too short to split into
thirds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = ["COFTConfig", "GPT2_BLOCK_SIZE"]

GPT2_BLOCK_SIZE = 1024  # fixed size of GPT-2's learned position embeddings


@dataclass
class COFTConfig:
    # ── run ──────────────────────────────────────────────────────────────────
    run_name: str = "coft-gpt2-shakespeare"
    out_dir: str = "runs"
    seed: int = 1337

    # ── model ────────────────────────────────────────────────────────────────
    model_type: str = "gpt2"  # gpt2 | gpt2-medium | gpt2-large | gpt2-xl
    dropout: float = 0.1
    nanogpt_path: str | None = None  # None -> auto-detect / clone

    # ── data ─────────────────────────────────────────────────────────────────
    dataset: str = "shakespeare"  # directory under data_root with train/val.bin
    data_root: str = "data"

    # ── COFT: context overextension ──────────────────────────────────────────
    t_infer: int = 512  # inference context length T (also the eval length)
    delta_init: int = 128
    delta_step: int = 64
    delta_min: int = 0
    delta_max: int = 256
    delta_tolerance: float = 0.0

    # ── COFT: components (each can be ablated off) ───────────────────────────
    use_overextension: bool = True
    use_prgg: bool = True
    use_crs: bool = True
    prgg_alpha: float = 2.0
    prgg_mode: str = "per_position"  # per_position | scalar
    prgg_normalize: bool = False
    crs_lambda: float = 0.5
    crs_layers: list[int] | None = None  # None -> all layers

    # ── optimisation ─────────────────────────────────────────────────────────
    batch_size: int = 8
    grad_accum: int = 4
    max_iters: int = 300
    eval_interval: int = 25
    eval_iters: int = 20
    learning_rate: float = 3e-5  # fine-tuning pretrained weights, not from scratch
    min_lr: float = 3e-6
    warmup_iters: int = 20
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ── runtime ──────────────────────────────────────────────────────────────
    device: str = "cuda"
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    compile_model: bool = False  # torch.compile; off because CRS re-patches forward
    log_interval: int = 10
    save_checkpoint: bool = True

    extra: dict[str, Any] = field(default_factory=dict)

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def max_block_size(self) -> int:
        return self.t_infer + (self.delta_max if self.use_overextension else 0)

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    def validate(self) -> COFTConfig:
        if self.max_block_size > GPT2_BLOCK_SIZE:
            raise ValueError(
                f"t_infer + delta_max = {self.max_block_size} exceeds GPT-2's "
                f"{GPT2_BLOCK_SIZE} position embeddings"
            )
        if self.t_infer < 3:
            raise ValueError("t_infer must be >= 3 so the context splits into thirds")
        if self.delta_min > self.delta_max:
            raise ValueError("delta_min must be <= delta_max")
        if self.prgg_mode not in ("per_position", "scalar"):
            raise ValueError(f"prgg_mode must be per_position or scalar, got {self.prgg_mode!r}")
        if self.dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError(f"unsupported dtype: {self.dtype!r}")
        return self

    # ── (de)serialisation ────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: dict) -> COFTConfig:
        known = {f.name for f in fields(cls)}
        unknown = {k: v for k, v in data.items() if k not in known}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data).validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> COFTConfig:
        import yaml

        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def merge(self, **overrides) -> COFTConfig:
        """Return a copy with CLI overrides applied (``None`` values ignored)."""
        data = asdict(self)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return COFTConfig.from_dict(data)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
