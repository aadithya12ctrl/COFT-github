"""Plain fine-tuning baseline: same checkpoint, same steps, no COFT.

    python scripts/train_baseline.py --config configs/gpt2_shakespeare.yaml

Trains with overextension, PRGG and CRS all off (so the model keeps Flash
Attention) while still reporting PFCM, which is the number the COFT run has to
beat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coft.baseline import baseline_config  # noqa: E402
from coft.cli import add_config_args, config_from_args  # noqa: E402
from coft.trainer import COFTTrainer  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cfg = baseline_config(config_from_args(args))
    history = COFTTrainer(cfg).setup().train()

    if args.plot:
        from coft.plotting import plot_history

        plot_history(history, cfg.run_dir / "history.png", title=f"Baseline -- {cfg.run_name}")


if __name__ == "__main__":
    main()
