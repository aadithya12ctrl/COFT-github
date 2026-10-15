"""COFT -- Context Overextension Fine-Tuning.

Reference implementation for *Train Long, Infer Short: Mitigating
Middle-Context Catastrophic Forgetting in LLMs via Context Overextension
Fine-Tuning*.

Four components, each in its own module and each independently ablatable:

===========================================  ======================
:func:`coft.pfcm.compute_pfcm`               PFCM -- the forgetting metric
:class:`coft.scheduler.DeltaScheduler`       adaptive overextension delta
:class:`coft.prgg.PRGG`                      positional recovery gradient gating
:func:`coft.crs.apply_crs`                   content-conditioned recency suppression
===========================================  ======================

The model itself is nanoGPT's GPT-2 (:mod:`coft.nanogpt`), patched at the
attention layer by :mod:`coft.patching`. :class:`coft.trainer.COFTTrainer` ties
it all together.

Note: ``coft.plotting`` is not imported here, since matplotlib is optional.
"""

from .config import GPT2_BLOCK_SIZE, COFTConfig
from .crs import apply_crs, crs_bias
from .data import TokenDataset
from .evaluate import evaluate
from .patching import (
    crs_disabled,
    disable_crs,
    enable_crs,
    is_patched,
    patch_model_with_crs,
    set_crs_ft,
)
from .pfcm import PFCMResult, compute_pfcm, per_position_nll, positional_curve, thirds
from .prgg import PRGG
from .scheduler import DeltaScheduler
from .trainer import COFTTrainer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "COFTConfig",
    "GPT2_BLOCK_SIZE",
    "COFTTrainer",
    "TokenDataset",
    "DeltaScheduler",
    "PRGG",
    "PFCMResult",
    "compute_pfcm",
    "per_position_nll",
    "positional_curve",
    "thirds",
    "apply_crs",
    "crs_bias",
    "patch_model_with_crs",
    "set_crs_ft",
    "disable_crs",
    "enable_crs",
    "crs_disabled",
    "is_patched",
    "evaluate",
]
