"""COFT fine-tuning entry point.

    python scripts/train_coft.py --config configs/gpt2_shakespeare.yaml
    python scripts/train_coft.py --config configs/gpt2_fineweb.yaml --max-iters 4000

Writes ``config.json``, ``log.jsonl`` and ``checkpoint.pt`` to
``<out_dir>/<run_name>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coft.cli import add_config_args, config_from_args  # noqa: E402
from coft.trainer import COFTTrainer  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--plot", action="store_true", help="save result plots when done")
    args = parser.parse_args()

    cfg = config_from_args(args)
    history = COFTTrainer(cfg).setup().train()

    if args.plot:
        from coft.plotting import plot_history, plot_positional_curve

        plot_history(history, cfg.run_dir / "history.png", title=f"COFT -- {cfg.run_name}")
        plot_positional_curve(history, cfg.run_dir / "positional_nll.png")


if __name__ == "__main__":
    main()
