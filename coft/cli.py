"""Shared CLI plumbing: ``--config a.yaml`` plus flag overrides.

Every script takes a YAML config and the same set of overrides, so a sweep is
one config file and a few flags rather than a new file per variant.
"""

from __future__ import annotations

import argparse

from .config import COFTConfig

__all__ = ["add_config_args", "config_from_args"]


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", help="path to a YAML config in configs/")
    g = parser.add_argument_group("overrides (take precedence over the config file)")
    g.add_argument("--run-name")
    g.add_argument("--out-dir")
    g.add_argument("--seed", type=int)
    g.add_argument("--model-type", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    g.add_argument("--dataset")
    g.add_argument("--data-root")
    g.add_argument("--nanogpt-path")
    g.add_argument("--t-infer", type=int, help="inference context length T")
    g.add_argument("--delta-init", type=int)
    g.add_argument("--delta-step", type=int)
    g.add_argument("--delta-max", type=int)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--grad-accum", type=int)
    g.add_argument("--max-iters", type=int)
    g.add_argument("--eval-interval", type=int)
    g.add_argument("--eval-iters", type=int)
    g.add_argument("--learning-rate", type=float)
    g.add_argument("--crs-lambda", type=float)
    g.add_argument("--prgg-alpha", type=float)
    g.add_argument("--prgg-mode", choices=["per_position", "scalar"])
    g.add_argument("--device")
    g.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"])

    a = parser.add_argument_group("component ablation")
    a.add_argument(
        "--no-overextension", dest="use_overextension", action="store_false", default=None
    )
    a.add_argument("--no-prgg", dest="use_prgg", action="store_false", default=None)
    a.add_argument("--no-crs", dest="use_crs", action="store_false", default=None)
    return parser


_OVERRIDE_KEYS = (
    "run_name",
    "out_dir",
    "seed",
    "model_type",
    "dataset",
    "data_root",
    "nanogpt_path",
    "t_infer",
    "delta_init",
    "delta_step",
    "delta_max",
    "batch_size",
    "grad_accum",
    "max_iters",
    "eval_interval",
    "eval_iters",
    "learning_rate",
    "crs_lambda",
    "prgg_alpha",
    "prgg_mode",
    "device",
    "dtype",
    "use_overextension",
    "use_prgg",
    "use_crs",
)


def config_from_args(args: argparse.Namespace) -> COFTConfig:
    base = COFTConfig.from_yaml(args.config) if args.config else COFTConfig()
    overrides = {k: getattr(args, k, None) for k in _OVERRIDE_KEYS}
    return base.merge(**overrides)
