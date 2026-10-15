"""Seeding, dtype/autocast selection, LR schedule and JSONL logging."""

from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "set_seed",
    "resolve_dtype",
    "autocast_ctx",
    "make_scaler",
    "cosine_lr",
    "RunLogger",
    "Timer",
    "describe_device",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """``"auto"`` picks bf16 on hardware that supports it, else fp16 on CUDA,
    else fp32."""
    if dtype == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            dtype = "float32"
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]


def autocast_ctx(device: str, dtype: torch.dtype):
    if not device.startswith("cuda") or dtype is torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def make_scaler(dtype: torch.dtype, device: str):
    """GradScaler is only needed for fp16; it is a no-op under bf16/fp32."""
    enabled = dtype is torch.float16 and device.startswith("cuda")
    scaler_device = "cuda" if device.startswith("cuda") else "cpu"
    return torch.amp.GradScaler(device=scaler_device, enabled=enabled)


def cosine_lr(it: int, *, lr: float, min_lr: float, warmup_iters: int, max_iters: int) -> float:
    """Linear warmup then cosine decay to ``min_lr``."""
    if warmup_iters > 0 and it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    if it >= max_iters:
        return min_lr
    span = max(max_iters - warmup_iters, 1)
    decay_ratio = (it - warmup_iters) / span
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


class RunLogger:
    """Appends one JSON object per logged step to ``<run_dir>/log.jsonl`` and
    keeps the same records in memory for plotting."""

    def __init__(self, run_dir: str | Path, filename: str = "log.jsonl"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename
        self.records: list[dict] = []
        self.path.write_text("", encoding="utf-8")

    def log(self, **record) -> dict:
        self.records.append(record)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    def series(self, key: str) -> list:
        return [r[key] for r in self.records if key in r]

    def history(self) -> dict[str, list]:
        keys = {k for r in self.records for k in r}
        return {k: self.series(k) for k in sorted(keys)}

    @staticmethod
    def load(path: str | Path) -> list[dict]:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def lap(self) -> float:
        now = time.time()
        dt = now - self.t0
        self.t0 = now
        return dt


def describe_device(device: str) -> str:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return f"{device} (no CUDA device visible)"
    props = torch.cuda.get_device_properties(0)
    return f"{props.name}, {props.total_memory / 1e9:.1f} GB, bf16={torch.cuda.is_bf16_supported()}"
