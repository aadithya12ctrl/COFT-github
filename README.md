# COFT — Context Overextension Fine-Tuning

**Train Long, Infer Short: Mitigating Middle-Context Catastrophic Forgetting in LLMs via Context Overextension Fine-Tuning**

Reference implementation on pretrained GPT-2 (124M–774M) via [nanoGPT](https://github.com/karpathy/nanoGPT).

---

## The problem

Transformers do not attend uniformly across their context. Attention mass concentrates on the beginning of the sequence and on the most recent tokens, leaving a trough in the middle — the "lost in the middle" effect. A model with a 512-token window does not have 512 usable tokens: it has strong edges and a soft middle. Standard fine-tuning does not fix this, because the loss is averaged over positions and the middle's contribution is diluted by the edges that the model already handles well.

## The idea

**Train on more context than you intend to use.** If the model is fine-tuned on sequences of length `T + δ` but deployed at length `T`, the region that *was* the fragile middle during training sits comfortably inside the window at inference time. The overextension is not static: it is driven by a live measurement of how much middle-context forgetting is actually happening, and two training-time interventions push the gradient and the attention distribution back toward the middle.

Four components, each implemented in its own module and each independently ablatable:

| Component | What it does | Module |
|---|---|---|
| **PFCM** | Measures middle-context forgetting as a single scalar `F_t` | [coft/pfcm.py](coft/pfcm.py) |
| **δ scheduler** | Adapts the training length `T + δ` to the live `F_t` | [coft/scheduler.py](coft/scheduler.py) |
| **PRGG** | Re-weights gradients toward forgotten positions | [coft/prgg.py](coft/prgg.py) |
| **CRS** | Biases attention off the recency peak and onto the middle, during training only | [coft/crs.py](coft/crs.py) |

Crucially, **all four are training-time only**. CRS is removed at inference, PRGG is a gradient hook, and the δ schedule only ever changes batch shapes. A COFT-trained checkpoint is architecturally a plain GPT-2 — you deploy it with no custom kernels and no inference-time overhead.

---

## Method

### 1. PFCM — Positional Forgetting Contrast Metric

Split the sequence into equal thirds and contrast the middle against the edges:

```
F_t = PPL_middle − (PPL_early + PPL_late) / 2
```

Computed on per-token NLL, so one measurement yields both the scalar `F_t` and the per-position vector `F_i` used downstream.

- `F_t > 0` — the middle is worse than the edges: the forgetting signature.
- `F_t ≈ 0` — retention is uniform across positions. **This is the target.**
- `F_t < 0` — the middle is *easier* than the edges, which usually means the data is degenerate rather than that the model is unusually good.

PFCM is a measurement, never a loss term: it is always computed on detached logits.

### 2. Forgetting-aware δ scheduler

```
δ_{t+1} = min(δ_t + Δ, δ_max)   if F_t > F_{t−1}    # forgetting accelerating → push further
δ_{t+1} = max(δ_t − Δ, δ_min)   if F_t < F_{t−1}    # retention stabilising   → relax
```

The harder the middle gets, the further past the inference length the model is trained; as retention recovers, the overextension is wound back down. `delta_tolerance` adds a dead zone so noisy per-step `F_t` estimates on small corpora do not thrash δ.

### 3. PRGG — Positional Recovery Gradient Gating

Positions the model has forgotten should dominate the update:

```
g_i = σ(α · F_i / F̄)
```

where `F_i` is the mean NLL at position `i` and `F̄` its mean over positions. The gate multiplies the gradient flowing back through the embedded sequence, so high-`F_i` middle positions keep more gradient headroom while well-retained edge positions are damped.

Two modes:

- **`per_position`** (default) — a hook on the embedding output `(B, T, C)`, where the position axis is still intact, so each position gets its own `g_i`. This is the paper's formulation.
- **`scalar`** — a hook on `wte.weight`, whose gradient has already contracted the position axis, so only the mean gate `mean_i g_i` can be applied. This is the approximation used by the original Colab notebook; kept for reproducing it (`prgg_mode: scalar`).

`prgg_normalize: true` divides the gate by its mean, so only the *relative* positional weighting changes and the overall step size is untouched. It is off by default, matching the paper, where the raw sigmoid also shrinks the effective learning rate by roughly 2×.

### 4. CRS — Content-conditioned Recency Suppression

Added to the pre-softmax attention logits during training:

```
ã[l,h,i,j] = a[l,h,i,j] + 1[j ∈ M] · (l/L) · λ · F_t · cos(q_i, k_j)
```

Each factor earns its place:

- `1[j ∈ M]` — only middle-third keys are boosted.
- `l/L` — deeper layers get more of it, where positional routing is decided; early feature extraction is left alone.
- `λ` — strength.
- `F_t` — **self-annealing**: the intervention fades out as the forgetting it corrects disappears.
- `cos(q_i, k_j)` — **content**-conditioned: only semantically relevant middle keys are promoted, so CRS cannot force attention onto irrelevant tokens. Anti-aligned keys are actually pushed down.

The bias is finite and the causal mask is `−inf`, so masked positions stay masked — no information leaks from the future.

---

## Repository layout

```
coft/                       the method, importable as a package
├── pfcm.py                 PFCM: F_t, per-position NLL, positional curve
├── scheduler.py            DeltaScheduler
├── prgg.py                 PRGG gradient gating (per_position | scalar)
├── crs.py                  CRS bias term
├── patching.py             injects CRS into nanoGPT attention; CRS on/off switches
├── nanogpt.py              locates/clones nanoGPT, loads pretrained GPT-2
├── data.py                 uint16 token-bin loader with a per-call block size
├── config.py               COFTConfig dataclass + validation
├── cli.py                  shared --config / override plumbing
├── evaluate.py             evaluation at T_infer, CRS forced off
├── trainer.py              COFTTrainer: the training loop
├── baseline.py             baseline + ablation grid
└── plotting.py             loss / PFCM / δ / positional-NLL figures

configs/                    one YAML per experiment
├── gpt2_shakespeare.yaml   fast mechanics check (~5 min, A100)
├── gpt2_fineweb.yaml       the real experiment (~2-3 h, A100)
├── gpt2_medium_fineweb.yaml  350M variant
└── cpu_smoke.yaml          4 iterations on CPU, wiring check only

scripts/
├── setup_nanogpt.py        clone nanoGPT
├── prepare_data.py         tokenise shakespeare | fineweb_edu with GPT-2 BPE
├── train_coft.py           COFT fine-tuning
├── train_baseline.py       plain fine-tuning baseline
├── run_ablation.py         several arms + comparison table
├── generate.py             sample from a checkpoint (CRS off)
└── plot_results.py         re-plot a finished run from log.jsonl

notebooks/
└── coft_gpt2_a100_colab.ipynb   original end-to-end Colab notebook

tests/                      CPU-only unit tests (no GPT-2 weights needed)
├── test_pfcm.py            F_t sign/magnitude, per-position NLL, positional curve
├── test_scheduler.py       δ direction, clamping, dead zone, state round-trip
├── test_prgg.py            gate shape/ordering, both modes, hook removal
├── test_crs.py             bias placement, scaling, content conditioning, causality
├── test_patching.py        the CRS patch against a nanoGPT-shaped attention stub
└── test_config.py          validation, the block budget, the ablation grid
```

---

## Install

```bash
git clone <this-repo> && cd COFT-github
pip install -r requirements.txt
python scripts/setup_nanogpt.py          # clones nanoGPT to third_party/nanoGPT
```

COFT does not vendor a transformer. It patches nanoGPT's `model.py`, so the GPT-2 weights, the optimizer parameter grouping and the sampling loop are all the reference implementations. If nanoGPT already lives somewhere else, point `NANOGPT_PATH` at it or pass `--nanogpt-path`.

## Quickstart

```bash
# 1. tokenise a corpus (GPT-2 BPE)
python scripts/prepare_data.py --dataset shakespeare

# 2. COFT fine-tune the pretrained 124M checkpoint
python scripts/train_coft.py --config configs/gpt2_shakespeare.yaml --plot

# 3. the comparison that matters
python scripts/run_ablation.py --config configs/gpt2_shakespeare.yaml \
    --variants baseline coft-full --plot
```

Each run writes `config.json`, `log.jsonl`, `checkpoint.pt` and its figures to `runs/<run_name>/`.

Sampling from a trained checkpoint:

```bash
python scripts/generate.py --checkpoint runs/coft-gpt2-shakespeare/checkpoint.pt \
    --prompt "The quick brown fox"
```

Any config value can be overridden from the CLI, so a sweep needs no new files:

```bash
python scripts/train_coft.py --config configs/gpt2_fineweb.yaml \
    --crs-lambda 1.0 --delta-max 384 --run-name coft-lambda1.0
```

### Using the components directly

```python
from coft import COFTConfig, COFTTrainer, compute_pfcm, patch_model_with_crs

cfg = COFTConfig.from_yaml("configs/gpt2_shakespeare.yaml")
history = COFTTrainer(cfg).setup().train()
```

```python
# PFCM on any model's output, COFT or not
result = compute_pfcm(logits.detach(), targets)
print(result.ft, result.nll_early, result.nll_middle, result.nll_late)
```

---

## Datasets

| Dataset | Tokens | Purpose |
|---|---|---|
| `shakespeare` | ~300K | Mechanics check. Short and repetitive, so the PFCM signal is weak and noisy — enough to confirm that PFCM, δ scheduling, PRGG and CRS all run and produce sane numbers, **not** enough to conclude anything about whether COFT helps. |
| `fineweb_edu` | ~25–50M | The real experiment. Long coherent documents, so middle-context forgetting actually appears and PFCM means something. |

Both are tokenised with the GPT-2 BPE (`tiktoken`'s `gpt2`), never a character vocabulary — the pretrained embedding table expects GPT-2 token ids. This is why `shakespeare` maps to nanoGPT's `data/shakespeare` and not `data/shakespeare_char`.

---

## Experimental protocol

- **Evaluation always happens at `T_infer`**, never at `T + δ`. Training long and evaluating long would measure the wrong thing.
- **CRS is always off at evaluation.** It is a training-time intervention and must never flatter the metric that judges it. [coft/evaluate.py](coft/evaluate.py) enforces this with a context manager.
- **The baseline is left unpatched**, so it keeps Flash Attention. Same checkpoint, same steps, same LR, same seed, no overextension/PRGG/CRS. This is both the faster and the fairer comparison: it is how you would fine-tune GPT-2 with nanoGPT out of the box.
- **Checkpoints are saved with CRS disabled**, so they reload as plain GPT-2.

Ablation arms available to `scripts/run_ablation.py`: `baseline`, `overext`, `overext+prgg`, `overext+crs`, `coft-full`.

| variant | val_loss @ T | PFCM final | Δ vs. baseline |
|---|---|---|---|
| baseline | — | — | — |
| overext | — | — | — |
| overext+prgg | — | — | — |
| overext+crs | — | — | — |
| coft-full | — | — | — |

*(Fill from `runs/<run_name>-ablation.md`. The numbers from a 300-iteration tiny-Shakespeare run are not results — use `configs/gpt2_fineweb.yaml` with thousands of iterations.)*

### What to look for

- **PFCM** should trend toward 0 under COFT while staying flat or growing for the baseline. This is the primary claim.
- **Val loss at `T_infer`** should be comparable to or better than the baseline at equal step count. COFT paying for middle-context retention with overall quality would be a failure.
- **δ** should actually move off `delta_init`. Pinned at `delta_min` or `delta_max` for a whole run means `delta_step`/`delta_max` need tuning for the corpus, or the PFCM signal is too noisy (raise `delta_tolerance`).
- **The positional NLL curve** (`positional_nll.png`) shows the U-shape that PFCM collapses into one number — worth checking when PFCM moves unexpectedly.

---

## Implementation notes

### Flash Attention must be disabled on patched layers

nanoGPT uses `F.scaled_dot_product_attention` whenever PyTorch exposes it, which is always on a modern GPU runtime. That kernel never materialises an explicit attention-logit tensor, so **CRS would silently do nothing** — it would have no tensor to add its bias to. [coft/patching.py](coft/patching.py) therefore forces the manual attention path on patched layers (`attn.flash = False`) and registers the causal-mask buffer the flash path had skipped.

This is a real cost: patched layers are back to O(T²) attention memory, and quadratically so as δ climbs. It is the reason `batch_size` has to come down when `delta_max` goes up. `crs_layers` can patch only a subset (e.g. the deeper half, where the `l/L` term is largest anyway) to trade CRS coverage for speed.

### Position-embedding budget

GPT-2's learned position embeddings are fixed at 1024, so **`t_infer + delta_max ≤ 1024`**. `COFTConfig.validate()` rejects configs that exceed it rather than letting them fail deep inside a training step.

### Fine-tuning learning rate

`3e-5`, not the `1e-3` of from-scratch nanoGPT training. A from-scratch LR destroys pretrained representations within a handful of steps, and the resulting PFCM curve measures that destruction rather than COFT.

### Memory budget (gpt2, 124M, A100 40GB)

| `t_infer` | `delta_max` | `batch_size` | peak block | VRAM |
|---|---|---|---|---|
| 512 | 512 | 4–8 | 1024 | ~20–25 GB |
| 384 | 384 | 8–16 | 768 | ~15–18 GB |
| 256 | 256 | 16–32 | 512 | ~10–12 GB |

`gpt2-medium` (350M) and `gpt2-large` (774M) scale these roughly 3× and 6×; drop `batch_size` and raise `grad_accum` to keep the effective batch fixed.

### `torch.compile` is off

CRS replaces `forward` on patched attention modules and the block size changes whenever δ moves, so compilation would re-trigger on every δ change. `compile_model` exists in the config but is not recommended for COFT runs.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

49 tests, CPU-only and fast (~12 s). They cover PFCM's sign and magnitude, the δ scheduler's direction/clamping/dead zone, PRGG's gate under both modes, CRS's placement, scaling, content conditioning and causality preservation, and config validation including every shipped YAML. PRGG and the CRS patch are tested against stubs that mirror nanoGPT's module layout, so no GPT-2 weights, nanoGPT clone or GPU are required.

`test_patching.py` is the test that would have caught the Flash Attention bug: it asserts the patched forward routes through the manual attention path and that the CRS bias changes the output only while training.

---

## Limitations

- **GPT-2's 1024-token window** bounds `T + δ`. Applying COFT at longer contexts needs a base model with a bigger window or positional interpolation.
- **PFCM's equal-thirds split** is a coarse proxy for "the middle". It is cheap and architecture-agnostic, but it says nothing about *which* middle tokens were forgotten, and it conflates positional forgetting with any position-correlated difficulty in the data.
- **PRGG `per_position` gates the embedded sequence**, not individual parameters — exact per-token gradient decomposition would need a per-token backward pass.
- **CRS costs Flash Attention** on patched layers, which is the dominant throughput penalty of the method.
- **A short run on a short corpus proves nothing.** Tiny Shakespeare at 300 iterations is a wiring check; the signal COFT targets barely exists in that data.

## Citation

```bibtex
@misc{coft,
  title  = {Train Long, Infer Short: Mitigating Middle-Context Catastrophic
            Forgetting in LLMs via Context Overextension Fine-Tuning},
  note   = {Reference implementation on pretrained GPT-2 via nanoGPT},
  year   = {2025}
}
```

Built on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT).
