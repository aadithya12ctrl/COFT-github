"""Re-plot a finished run from its ``log.jsonl`` -- no training required.

    python scripts/plot_results.py runs/coft-gpt2-shakespeare
    python scripts/plot_results.py runs/coft-gpt2-shakespeare-coft-full \
        --compare runs/coft-gpt2-shakespeare-baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coft.utils import RunLogger  # noqa: E402


def load_history(run_dir: Path) -> dict[str, list]:
    records = RunLogger.load(run_dir / "log.jsonl")
    keys = {k for r in records for k in r}
    return {k: [r[k] for r in records if k in r] for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--compare", type=Path, help="a baseline run dir to plot against")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    from coft.plotting import plot_comparison, plot_history, plot_positional_curve

    out_dir = args.out_dir or args.run_dir
    history = load_history(args.run_dir)

    plot_history(history, out_dir / "history.png", title=args.run_dir.name)
    if history.get("positional_curve"):
        plot_positional_curve(history, out_dir / "positional_nll.png")

    if args.compare:
        plot_comparison(history, load_history(args.compare), out_dir / "comparison.png")


if __name__ == "__main__":
    main()
