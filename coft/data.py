"""Token-bin data loading.

Reads the ``train.bin`` / ``val.bin`` ``uint16`` token files produced by
nanoGPT's ``prepare.py`` scripts (and by :mod:`scripts.prepare_data`). The
block size is a *per-call* argument rather than fixed at construction: COFT
changes it every iteration as delta moves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

__all__ = ["TokenDataset"]


class TokenDataset:
    def __init__(self, dataset: str, data_root: str | Path = "data", device: str = "cuda"):
        self.dir = Path(data_root) / dataset
        self.device = device
        missing = [f for f in ("train.bin", "val.bin") if not (self.dir / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{missing} missing from {self.dir}. Run "
                f"`python scripts/prepare_data.py --dataset {dataset}` first."
            )
        # uint16 is what nanoGPT writes: GPT-2's vocab (50257) fits in 16 bits.
        self.splits = {
            "train": np.memmap(self.dir / "train.bin", dtype=np.uint16, mode="r"),
            "val": np.memmap(self.dir / "val.bin", dtype=np.uint16, mode="r"),
        }

    def __len__(self) -> int:
        return len(self.splits["train"])

    def tokens(self, split: str) -> int:
        return len(self.splits[split])

    def get_batch(
        self, split: str, block_size: int, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``batch_size`` random windows of ``block_size`` tokens.

        Returns ``(x, y)`` of shape ``(batch_size, block_size)``, where ``y`` is
        ``x`` shifted one token left.
        """
        data = self.splits[split]
        if len(data) <= block_size + 1:
            raise ValueError(
                f"{split} split has {len(data)} tokens, too few for block_size={block_size}"
            )
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
        )
        if self.device.startswith("cuda"):
            return (
                x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True),
            )
        return x.to(self.device), y.to(self.device)
