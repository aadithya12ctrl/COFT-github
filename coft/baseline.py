"""Baseline and ablation helpers.

The baseline is the *same* pretrained checkpoint, fine-tuned for the same number
of steps at the same learning rate, with no overextension, no PRGG and no CRS --
and deliberately left unpatched, so it keeps Flash Attention. That is both the
faster and the fairer comparison: it is how you would fine-tune GPT-2 with
nanoGPT out of the box.
"""

from __future__ import annotations

from dataclasses import replace

from .config import COFTConfig
from .trainer import COFTTrainer

__all__ = ["ABLATIONS", "baseline_config", "ablation_config", "run_variant", "summarise"]


#: Ablation grid: arm name -> which components stay enabled.
ABLATIONS: dict[str, dict] = {
    "baseline": dict(use_overextension=False, use_prgg=False, use_crs=False),
    "overext": dict(use_overextension=True, use_prgg=False, use_crs=False),
    "overext+prgg": dict(use_overextension=True, use_prgg=True, use_crs=False),
    "overext+crs": dict(use_overextension=True, use_prgg=False, use_crs=True),
    "coft-full": dict(use_overextension=True, use_prgg=True, use_crs=True),
}


def baseline_config(cfg: COFTConfig, suffix: str = "-baseline") -> COFTConfig:
    """The same run with every COFT component switched off."""
    return replace(
        cfg,
        run_name=cfg.run_name + suffix,
        use_overextension=False,
        use_prgg=False,
        use_crs=False,
    ).validate()


def ablation_config(cfg: COFTConfig, name: str) -> COFTConfig:
    if name not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; choose from {sorted(ABLATIONS)}")
    return replace(cfg, run_name=f"{cfg.run_name}-{name}", **ABLATIONS[name]).validate()


def run_variant(cfg: COFTConfig, name: str) -> dict[str, list]:
    """Train one ablation arm from a fresh pretrained checkpoint."""
    return COFTTrainer(ablation_config(cfg, name)).setup().train()


def summarise(histories: dict[str, dict[str, list]]) -> str:
    """Markdown ablation table of first/final val loss and PFCM per arm."""
    rows = [
        "| variant | val_loss (final) | PFCM (first eval) | PFCM (final) |",
        "|---|---|---|---|",
    ]
    for name, hist in histories.items():
        val = hist.get("val_loss") or [float("nan")]
        pfcm = hist.get("pfcm") or [float("nan")]
        rows.append(f"| {name} | {val[-1]:.4f} | {pfcm[0]:+.4f} | {pfcm[-1]:+.4f} |")
    return "\n".join(rows)
