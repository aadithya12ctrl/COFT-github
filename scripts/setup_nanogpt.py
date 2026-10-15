"""Clone nanoGPT, which supplies the GPT-2 model code COFT patches.

    python scripts/setup_nanogpt.py [--path third_party/nanoGPT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coft.nanogpt import DEFAULT_PATH, ensure_nanogpt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_PATH), help="clone destination")
    args = parser.parse_args()

    root = ensure_nanogpt(args.path, clone=True)
    print(f"nanoGPT ready at {root.resolve()}")
    print("Set NANOGPT_PATH or pass --nanogpt-path if you move it.")


if __name__ == "__main__":
    main()
