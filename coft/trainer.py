"""The COFT fine-tuning loop.

One iteration:

1. set the LR (warmup + cosine);
2. take the current ``delta`` and build the training block ``T + delta``;
3. broadcast the latest ``F_t`` to the CRS layers;
4. ``grad_accum`` micro-steps at that block size, each one measuring PFCM on
   its own (detached) logits and handing the per-position NLL to PRGG -- no
   extra forward pass is needed for the measurement;
5. clip, step;
6. feed ``F_t`` to the delta scheduler, which picks the next block size;
7. every ``eval_interval`` iterations, evaluate at ``t_infer`` with CRS off.

Setting ``use_overextension`` / ``use_prgg`` / ``use_crs`` to ``False`` turns the
corresponding component off, which is how the ablation table is produced: all
arms run through this same loop. With all three off this is a plain nanoGPT
fine-tune that still reports PFCM.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import COFTConfig
from .data import TokenDataset
from .evaluate import evaluate
from .nanogpt import load_pretrained_gpt2
from .patching import disable_crs, patch_model_with_crs, set_crs_ft
from .pfcm import compute_pfcm
from .prgg import PRGG
from .scheduler import DeltaScheduler
from .utils import (
    RunLogger,
    Timer,
    autocast_ctx,
    cosine_lr,
    describe_device,
    make_scaler,
    resolve_dtype,
    set_seed,
)

__all__ = ["COFTTrainer"]


class COFTTrainer:
    def __init__(self, cfg: COFTConfig, data: TokenDataset | None = None, model=None):
        self.cfg = cfg.validate()
        self.data = data
        self.model = model
        self.optimizer = None
        self.prgg: PRGG | None = None
        self.delta_sched: DeltaScheduler | None = None
        self.logger: RunLogger | None = None
        self.ft_current = 0.0

    # -- setup ---------------------------------------------------------------
    def setup(self) -> COFTTrainer:
        cfg = self.cfg
        set_seed(cfg.seed)

        self.dtype = resolve_dtype(cfg.dtype, cfg.device)
        self.ctx = autocast_ctx(cfg.device, self.dtype)
        self.scaler = make_scaler(self.dtype, cfg.device)

        if self.data is None:
            self.data = TokenDataset(cfg.dataset, cfg.data_root, cfg.device)

        if self.model is None:
            self.model = load_pretrained_gpt2(
                cfg.model_type, cfg.dropout, cfg.device, cfg.nanogpt_path
            )
        if cfg.use_crs:
            patch_model_with_crs(self.model, cfg.crs_lambda, cfg.crs_layers)

        self.optimizer = self.model.configure_optimizers(
            cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), cfg.device
        )

        if cfg.use_prgg:
            self.prgg = PRGG(
                alpha=cfg.prgg_alpha, mode=cfg.prgg_mode, normalize=cfg.prgg_normalize
            ).attach(self.model)

        self.delta_sched = DeltaScheduler(
            delta_init=cfg.delta_init if cfg.use_overextension else 0,
            delta_step=cfg.delta_step if cfg.use_overextension else 0,
            delta_min=cfg.delta_min if cfg.use_overextension else 0,
            delta_max=cfg.delta_max if cfg.use_overextension else 0,
            tolerance=cfg.delta_tolerance,
        )

        self.logger = RunLogger(cfg.run_dir)
        cfg.save(cfg.run_dir / "config.json")

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"run      : {cfg.run_name}  ->  {cfg.run_dir}")
        print(f"device   : {describe_device(cfg.device)}")
        print(f"dtype    : {self.dtype}")
        print(f"model    : {cfg.model_type} ({n_params:.1f}M params)")
        print(f"data     : {cfg.dataset} ({self.data.tokens('train'):,} train tokens)")
        print(
            f"COFT     : T={cfg.t_infer} delta={self.delta_sched.delta} "
            f"(max block {cfg.max_block_size}) | overext={cfg.use_overextension} "
            f"prgg={cfg.use_prgg}({cfg.prgg_mode}) crs={cfg.use_crs}"
        )
        print(f"batch    : {cfg.batch_size} x {cfg.grad_accum} = {cfg.effective_batch}")
        return self

    # -- one iteration -------------------------------------------------------
    def train_step(self, iteration: int) -> dict:
        cfg = self.cfg
        lr = cosine_lr(
            iteration,
            lr=cfg.learning_rate,
            min_lr=cfg.min_lr,
            warmup_iters=cfg.warmup_iters,
            max_iters=cfg.max_iters,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        delta = self.delta_sched.delta if cfg.use_overextension else 0
        train_block = cfg.t_infer + delta

        if cfg.use_crs:
            set_crs_ft(self.model, self.ft_current)

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(cfg.grad_accum):
            x, y = self.data.get_batch("train", train_block, cfg.batch_size)
            with self.ctx:
                logits, loss = self.model(x, y)
                loss = loss / cfg.grad_accum
            total_loss += loss.item()

            # PFCM on this micro-step's detached logits: it feeds PRGG's gate
            # and the next iteration's CRS bias at no extra forward cost.
            result = compute_pfcm(logits.detach(), y)
            self.ft_current = result.ft
            if self.prgg is not None:
                self.prgg.update_forgetting(result.per_pos_nll)

            self.scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        next_delta = self.delta_sched.update(self.ft_current)

        return {
            "iter": iteration,
            "lr": lr,
            "train_loss": total_loss,
            "train_pfcm": self.ft_current,
            "block_size": train_block,
            "delta": next_delta,
        }

    # -- full run ------------------------------------------------------------
    def train(self) -> dict[str, list]:
        cfg = self.cfg
        if self.optimizer is None:
            self.setup()

        self.model.train()
        timer = Timer()

        for iteration in range(1, cfg.max_iters + 1):
            step = self.train_step(iteration)

            if cfg.log_interval and iteration % cfg.log_interval == 0:
                print(
                    f"iter {iteration:5d} | loss {step['train_loss']:.4f} | "
                    f"F_t {step['train_pfcm']:+.4f} | delta {step['delta']:4d} | "
                    f"lr {step['lr']:.2e}"
                )

            if iteration % cfg.eval_interval == 0 or iteration in (1, cfg.max_iters):
                metrics = evaluate(
                    self.model,
                    self.data,
                    t_infer=cfg.t_infer,
                    batch_size=cfg.batch_size,
                    eval_iters=cfg.eval_iters,
                    ctx=self.ctx,
                    curve_bins=32,
                )
                record = self.logger.log(**step, **metrics, elapsed=timer.lap())
                print(
                    f"  eval {iteration:5d} | val_loss {record['val_loss']:.4f} | "
                    f"PFCM {record['pfcm']:+.4f} | delta {record['delta']:4d} | "
                    f"{record['elapsed']:.1f}s"
                )

        if cfg.save_checkpoint:
            self.save_checkpoint()
        return self.logger.history()

    # -- checkpointing -------------------------------------------------------
    def save_checkpoint(self, filename: str = "checkpoint.pt") -> Path:
        """Save weights with CRS off, so the checkpoint restores as a plain
        GPT-2 that happens to have been COFT-trained."""
        if self.cfg.use_crs:
            disable_crs(self.model)
        path = self.cfg.run_dir / filename
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_args": vars(self.model.config),
                "config": self.cfg.to_dict(),
                "delta_history": self.delta_sched.history,
                "log": self.logger.records if self.logger else [],
            },
            path,
        )
        print(f"checkpoint -> {path}")
        return path
