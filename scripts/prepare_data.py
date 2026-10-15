"""Tokenise a dataset into ``data/<name>/{train,val}.bin`` with GPT-2 BPE.

    python scripts/prepare_data.py --dataset shakespeare
    python scripts/prepare_data.py --dataset fineweb_edu --docs 25000

Everything is tokenised with the *GPT-2 BPE* (``tiktoken``'s ``gpt2``
encoding), never a character vocabulary: the pretrained embedding table expects
GPT-2 token ids. That is why ``shakespeare`` here maps to nanoGPT's
``data/shakespeare`` and not to ``data/shakespeare_char``.

| dataset       | tokens   | purpose                                             |
|---------------|----------|-----------------------------------------------------|
| shakespeare   | ~300K    | mechanics check; PFCM signal is weak and noisy      |
| fineweb_edu   | ~25-50M  | long coherent documents; real middle-context signal |
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_ROOT = Path("data")


def _write_split(tokens: np.ndarray, out_dir: Path, val_fraction: float = 0.1) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    split = int(len(tokens) * (1.0 - val_fraction))
    tokens[:split].tofile(out_dir / "train.bin")
    tokens[split:].tofile(out_dir / "val.bin")
    print(
        f"{len(tokens):,} tokens -> {out_dir}/train.bin ({split:,}) "
        f"+ val.bin ({len(tokens) - split:,})"
    )


def prepare_shakespeare(out_dir: Path) -> None:
    """Tiny Shakespeare, GPT-2 BPE. Fast sanity check."""
    import requests
    import tiktoken

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw = out_dir / "input.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not raw.is_file():
        print(f"downloading {url}")
        raw.write_text(requests.get(url, timeout=60).text, encoding="utf-8")

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(raw.read_text(encoding="utf-8"))
    _write_split(np.array(ids, dtype=np.uint16), out_dir)


def prepare_fineweb_edu(out_dir: Path, n_docs: int) -> None:
    """A streamed sample of FineWeb-Edu. Long, coherent documents, which is
    what makes middle-context forgetting actually measurable."""
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )

    tokens: list[int] = []
    for i, example in enumerate(stream):
        tokens.extend(enc.encode_ordinary(example["text"]))
        tokens.append(enc.eot_token)  # document separator
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,} docs, {len(tokens):,} tokens")
        if i + 1 >= n_docs:
            break

    _write_split(np.array(tokens, dtype=np.uint16), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="shakespeare", choices=["shakespeare", "fineweb_edu"])
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--docs", type=int, default=25_000, help="fineweb_edu: documents to stream")
    args = parser.parse_args()

    out_dir = Path(args.data_root) / args.dataset
    if args.dataset == "shakespeare":
        prepare_shakespeare(out_dir)
    else:
        prepare_fineweb_edu(out_dir, args.docs)


if __name__ == "__main__":
    main()
