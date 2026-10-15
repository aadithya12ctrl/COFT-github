"""Train several ablation arms from the same pretrained checkpoint and print the
comparison table.

    python scripts/run_ablation.py --config configs/gpt2_shakespeare.yaml
    python scripts/run_ablation.py --config configs/gpt2_fineweb.yaml \
        --variants baseline coft-full

Each arm starts from a fresh ``GPT.from_pretrained`` and sees the same seed,
steps and learning rate, so the only difference is which COFT components are
on. Results go to ``<out_dir>/<run_name>-<variant>/`` plus an
``ablation.md`` table alongside them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coft.baseline import ABLATIONS, ablation_config, summarise  # noqa: E402
from coft.cli import add_config_args, config_from_args  # noqa: E402
from coft.trainer import COFTTrainer  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "coft-full"],
        choices=sorted(ABLATIONS),
        help="which arms to run (default: baseline and full COFT)",
    )
    parser.add_argument("--plot", action="store_true", help="save a COFT-vs-baseline plot")
    args = parser.parse_args()

    base_cfg = config_from_args(args)
    histories: dict[str, dict[str, list]] = {}

    for name in args.variants:
        print(f"\n{'=' * 72}\nablation arm: {name}\n{'=' * 72}")
        cfg = ablation_config(base_cfg, name)
        histories[name] = COFTTrainer(cfg).setup().train()

    table = summarise(histories)
    out_dir = Path(base_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{base_cfg.run_name}-ablation.md").write_text(table + "\n", encoding="utf-8")
    (out_dir / f"{base_cfg.run_name}-ablation.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8"
    )

    print("\n" + table)
    print("\nPFCM closer to 0 = more uniform retention across context positions.")

    if args.plot and {"baseline", "coft-full"} <= histories.keys():
        from coft.plotting import plot_comparison

        plot_comparison(
            histories["coft-full"],
            histories["baseline"],
            out_dir / f"{base_cfg.run_name}-comparison.png",
        )


if __name__ == "__main__":
    main()
