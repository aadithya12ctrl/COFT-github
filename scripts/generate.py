"""Sample from a COFT checkpoint.

    python scripts/generate.py --checkpoint runs/coft-gpt2-shakespeare/checkpoint.pt \
        --prompt "The quick brown fox" --max-new-tokens 200

CRS is a training-time intervention, so it is switched off here: generation runs
through plain GPT-2 attention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coft.nanogpt import load_nanogpt  # noqa: E402
from coft.utils import autocast_ctx, resolve_dtype  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--nanogpt-path", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    GPT, GPTConfig, _ = load_nanogpt(args.nanogpt_path)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])  # saved with CRS disabled
    model.eval().to(args.device)

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor(enc.encode(args.prompt), dtype=torch.long, device=args.device)[None, ...]

    ctx = autocast_ctx(args.device, resolve_dtype(args.dtype, args.device))
    with torch.no_grad(), ctx:
        for i in range(args.num_samples):
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(f"\n--- sample {i + 1} ---")
            print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
