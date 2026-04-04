# ReMDM Planner for MiniHack

PyTorch implementation of **ReMDM** (Remasking Discrete Diffusion Model) for action-sequence planning in [MiniHack](https://github.com/facebookresearch/minihack) navigation environments. A dual-stream transformer generates 192-step action plans by iteratively denoising masked token sequences, conditioned on a 9x9 local crop and the full 21x79 dungeon map.

Trained on BFS oracle demonstrations via behavioural cloning, then refined online with DAgger. Generalises **zero-shot** from 4 in-distribution environments to 3 out-of-distribution environments.

---

## Pipeline

```
[Stage 1]  Offline BC on oracle demos     main.py --mode offline --data dataset.pt
                |
                v  checkpoint
[Stage 2]  DAgger online training          main.py --mode dagger
                |  (collect with model, label with oracle,
                |   efficiency filter, curriculum sampling)
                v  fine-tuned checkpoint
[Stage 3]  Evaluate (ID + OOD)             main.py --mode inference --checkpoint iter8000.pth
```

A `--mode smoke` is provided for quick end-to-end sanity checks (~30 seconds on CPU).

---

## Environments

**In-distribution (training):**

| Environment | Description |
|---|---|
| `MiniHack-Room-Random-5x5-v0` | Small random room |
| `MiniHack-Room-Random-15x15-v0` | Large random room |
| `MiniHack-Corridor-R2-v0` | Two-room corridor |
| `MiniHack-MazeWalk-9x9-v0` | Small maze |

**Out-of-distribution (zero-shot evaluation):**

| Environment | Description |
|---|---|
| `MiniHack-Room-Dark-15x15-v0` | Dark room (limited visibility) |
| `MiniHack-Corridor-R5-v0` | Five-room corridor |
| `MiniHack-MazeWalk-45x19-v0` | Large maze |

---

## Installation

### Prerequisites

**macOS (arm64):** Install cmake via Homebrew (needed to compile `nle` from source):

```bash
brew install cmake
```

**Linux (x86_64):** Pre-built wheels are available, but if building from source:

```bash
sudo apt-get install build-essential cmake bison flex libbz2-dev
```

### Setup

```bash
uv sync
```

This installs all dependencies from the lockfile, including `nle>=1.2.0` (from the maintained [NetHack-LE](https://github.com/NetHack-LE/nle) fork) and `minihack`.

### GPU support (optional)

By default PyTorch runs on CPU. For NVIDIA CUDA 12:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is detected:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

---

## Usage

All modes share a single entry point. Defaults load from `configs/defaults.yaml`; any value can be overridden via `key=value` pairs.

```bash
python main.py --mode <MODE> [--config PATH] [key=value ...]
```

### Smoke test

Collects a few oracle trajectories, trains for 30 iterations, and prints ID evaluation results.

```bash
python main.py --mode smoke
```

### Offline BC

Train the diffusion model on pre-collected oracle demonstrations.

```bash
python main.py --mode offline --data path/to/dataset.pt

# Resume from a checkpoint (restores optimizer, scheduler, epoch, and W&B run)
python main.py --mode offline --data path/to/dataset.pt \
    --checkpoint checkpoints/offline_epoch10.pth
```

Set `offline_checkpoint_every` to save epoch-level checkpoints (default 0 = off):

```bash
python main.py --mode offline --data dataset.pt offline_checkpoint_every=5
```

### DAgger online training

Full DAgger loop: collect with model, label with BFS oracle, filter by efficiency, train on buffer.

```bash
# From scratch (seeds buffer with oracle data automatically)
python main.py --mode dagger

# Resume from local checkpoint
python main.py --mode dagger --checkpoint checkpoints/iter3000.pth

# Resume from a W&B artifact
python main.py --mode dagger \
    --wandb-artifact entity/project/checkpoint-iter3000:latest

# Override hyperparameters
python main.py --mode dagger max_iterations=4000 dagger_lr=0.0001
```

### Inference

Evaluate a checkpoint on specified environments. Accepts either `--checkpoint` (local path) or `--wandb-artifact` (W&B artifact reference).

```bash
# All ID + OOD environments
python main.py --mode inference --checkpoint checkpoints/iter8000.pth

# From a W&B artifact
python main.py --mode inference \
    --wandb-artifact entity/project/checkpoint-iter8000:latest

# Specific environments, save JSON
python main.py --mode inference \
    --checkpoint checkpoints/iter8000.pth \
    --envs MiniHack-Room-Random-5x5-v0 MiniHack-MazeWalk-45x19-v0 \
    --episodes 100 \
    --output results.json

# Use training weights instead of EMA
python main.py --mode inference --checkpoint iter8000.pth --no-ema
```

---

## Architecture

**`LocalDiffusionPlannerWithGlobal`** (~5.2M parameters):

```
Local stream:   9x9 glyphs -> Embedding(6000,64) -> CNN(64->32->64) -> Linear -> 1 token
Global stream:  21x79 glyphs -> Embedding(6000,32) -> CNN(32->32->64) -> Pool(2,4) -> 8 tokens
                Goal head: mean(global) -> MLP -> [B,2] staircase coords (aux loss)
                Gate: sigmoid(learnable scalar, init=-3.0) * global_tokens
Action stream:  Embedding(14, 256) + timestep_emb(100, 256) + position_emb(192, 256)
Transformer:    concat [1 + 8 + 192 = 201 tokens] -> 4-layer encoder (256D, 4 heads, pre-norm)
Output head:    last 192 tokens -> Linear(256, 12) -> action logits
```

The model takes `(local_obs, global_obs, noisy_action_seq, t_discrete)` and returns `{"actions": [B,192,12], "goal_pred": [B,2]}`.

---

## Diffusion

**Forward process (MDLM):** Each action token is independently replaced with `MASK` (token 12) with probability `1 - alpha(t)`, where `alpha(t)` follows a linear or cosine schedule. PAD tokens (13) are never masked.

**Loss:** Cross-entropy on masked positions only, averaged globally across the batch. By default uses a flat average (matching the reference implementation). Optional SUBS importance weighting `w(t) = -alpha'(t) / (1 - alpha(t))`, clipped to `[0, 1000]`, can be enabled via `use_importance_weighting: true`.

**Reverse sampling (ReMDM):** Over `K` denoising steps (default 10):
1. Model predicts logits; apply temperature scaling and top-K filtering.
2. Sample predictions; compute per-token confidence.
3. **MaskGIT unmask:** commit the `n_unmask` highest-confidence masked positions.
4. **ReMDM remask:** stochastically re-mask committed positions to allow refinement.
5. Final step: commit all remaining positions.

**Greedy sampling:** Used during DAgger data collection for deterministic rollouts. Same MaskGIT progressive unmasking loop but with argmax decoding (no temperature, no top-K, no remasking).

### Remasking strategies

| Strategy | Formula | Description |
|---|---|---|
| `rescale` | `p = eta * sigma_max` | Proportional to noise level |
| `cap` | `p = min(eta, sigma_max)` | Fixed upper bound |
| `conf` | `p = eta * sigma_max * (1 - confidence)` | Low-confidence tokens remasked more |

---

## Configuration

### Key hyperparameters

**Model**

| Parameter | Default | Description |
|---|---|---|
| `n_embd` | 256 | Transformer hidden dimension |
| `n_head` | 4 | Attention heads |
| `n_layer` | 4 | Transformer blocks |
| `n_global_tokens` | 8 | Global stream context tokens |
| `seq_len` | 192 | Action plan length |
| `dropout` | 0.0 | Transformer dropout (0.0 — forward masking regularises) |
| `ema_decay` | 0.999 | EMA smoothing for inference weights |

**Diffusion**

| Parameter | Default | Description |
|---|---|---|
| `noise_schedule` | `linear` | `linear` or `cosine` |
| `num_diffusion_steps` | 100 | Discrete timestep resolution |
| `diffusion_steps_eval` | 10 | Denoising iterations at inference |
| `remask_strategy` | `conf` | `rescale`, `cap`, or `conf` |
| `eta` | 0.15 | Remasking strength |
| `temperature` | 0.5 | Sampling temperature |
| `top_k` | 4 | Top-K filtering |
| `replan_every` | 16 | Env steps before replanning |

**Training**

| Parameter | Default | Description |
|---|---|---|
| `offline_lr` | 0.0003 | BC learning rate (cosine-decayed to 10%) |
| `dagger_lr` | 0.00003 | DAgger learning rate |
| `offline_batch_size` | 1024 | Offline BC batch size |
| `dagger_batch_size` | 1024 | DAgger batch size |
| `weight_decay` | 0.0001 | AdamW weight decay (both optimizers) |
| `offline_epochs` | 30 | BC training epochs |
| `max_iterations` | 8000 | DAgger iterations |
| `grad_steps_per_iteration` | 100 | Gradient steps per DAgger iteration |
| `aux_loss_weight` | 0.5 | Weight for auxiliary goal loss |
| `use_importance_weighting` | false | SUBS w(t) in loss (off = flat average) |
| `buffer_capacity` | 10000 | Replay buffer size (windows) |
| `efficiency_multiplier` | 1.5 | DAgger efficiency filter threshold |
| `curriculum_preseed` | true | Pre-seed curriculum with 50/50 prior |
| `physics_aware_sampling` | false | Penalise hazardous actions at inference |

**Evaluation**

| Parameter | Default | Description |
|---|---|---|
| `eval_episodes_per_env` | 50 | Episodes per environment |
| `checkpoint_every` | 1000 | Checkpoint frequency (iterations) |
| `id_eval_every` | 100 | ID evaluation frequency |
| `ood_eval_every` | 500 | OOD evaluation frequency |

### Config presets

| File | Purpose |
|---|---|
| `configs/defaults.yaml` | Base defaults for all modes |
| `configs/smoke.yaml` | Fast smoke test (30 iters, small buffer) |
| `configs/main.yaml` | Full training (inherits defaults) |

---

## DAgger Training Loop

Each DAgger iteration:

1. **Curriculum sampling:** Select an environment weighted by difficulty (low win-rate environments sampled more).
2. **Model rollout:** Generate plans with the EMA model using greedy sampling; execute with replanning every 16 steps.
3. **Oracle rollout:** Run the BFS oracle on the **same seed** for comparison.
4. **Efficiency filter:** Add the oracle trajectory to the buffer if the model failed or took >1.5x the oracle's steps.
5. **Training:** Sample from the replay buffer; run `grad_steps_per_iteration` gradient steps, updating EMA weights after each gradient step.

The BFS oracle uses a 5-tier priority: (1) kick adjacent doors, (2) BFS to staircase, (3) BFS to frontier, (4) BFS to farthest tile, (5) random cardinal.

---

## Reward Shaping

The environment wrapper applies shaped rewards to guide learning:

| Component | Value | Condition |
|---|---|---|
| Win bonus | +20.0 | Episode won |
| BFS progress | +0.5 * (prev_dist - curr_dist) | Closer to staircase |
| Exploration | +0.05 | New tile visited |
| Step penalty | -0.01 | Every step |

---

## Project Structure

```
remdm_minihack/
├── configs/
│   ├── defaults.yaml              Base hyperparameters
│   ├── smoke.yaml                 Smoke test overrides
│   └── main.yaml                  Full training (inherits defaults)
├── src/
│   ├── config.py                  YAML config loader with CLI overrides
│   ├── buffer.py                  ReplayBuffer with offline-protected FIFO
│   ├── curriculum.py              DynamicCurriculum + efficiency_filter
│   ├── diffusion/
│   │   ├── schedules.py           Linear and cosine noise schedules
│   │   ├── forward.py             Forward masking process q(z_t | x_0)
│   │   ├── loss.py                MDLM ELBO + auxiliary goal loss
│   │   └── sampling.py            ReMDM reverse sampling with remasking
│   ├── models/
│   │   └── denoiser.py            LocalDiffusionPlannerWithGlobal + ModelEMA
│   ├── envs/
│   │   └── minihack_env.py        AdvancedObservationEnv + BFS oracle
│   └── planners/
│       ├── collect.py             run_model_episode + DataCollector
│       ├── offline.py             Offline BC trainer
│       ├── online.py              DAgger Trainer + checkpointing
│       ├── inference.py           Evaluator + result formatting
│       ├── smoke.py               Smoke-test runner
│       └── logging.py             Centralised W&B + stdout logging
├── scripts/
│   └── hf_upload.py               HuggingFace Hub upload utility
├── main.py                        CLI entry point (smoke/offline/dagger/inference)
├── pyproject.toml                 PEP 621 project metadata + dependencies
├── uv.lock                        Deterministic lockfile
└── README.md
```

---

## W&B Metric Namespaces

| Namespace | Contents |
|---|---|
| `diffusion/` | Loss, auxiliary loss, total loss |
| `train/` | Buffer size, model win rate, oracle add rate |
| `eval_id/{env}/` | Per-environment win rate, avg steps (in-distribution) |
| `eval_ood/{env}/` | Per-environment win rate, avg steps (out-of-distribution) |

---

## Checkpoint Format

**DAgger checkpoint:**

```python
{
    "model_state_dict":     ...,
    "ema_state_dict":       ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "curriculum_state":     {"env_ids", "queue_size", "queues"},
    "iteration":            int,
    "wandb_run_id":         str | None,
    "rng_states":           {"torch", "numpy", "python"},
}
```

**Offline BC checkpoint** (epoch-level, saved when `offline_checkpoint_every > 0`):

```python
{
    "model_state_dict":     ...,
    "ema_state_dict":       ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "epoch":                int,
    "step":                 int,
    "wandb_run_id":         str | None,
}
```

Inference uses EMA weights by default. Pass `--no-ema` to use training weights.

### W&B Artifacts

Checkpoints are automatically uploaded as versioned W&B artifacts (type `"model"`) at each checkpoint save. Each artifact contains the `.pth` weights and a `config.yaml` snapshot of all hyperparameters used.

To resume from an artifact:

```bash
# DAgger resume
python main.py --mode dagger \
    --wandb-artifact entity/project/checkpoint-iter3000:latest

# Inference
python main.py --mode inference \
    --wandb-artifact entity/project/checkpoint-iter8000:v2
```

The artifact reference format is `entity/project/artifact-name:version` where version is `latest`, `v0`, `v1`, etc.

### W&B Run Resumption

All training loops save the W&B run ID in their checkpoints. When resuming from a checkpoint, the run ID is automatically extracted and passed to `wandb.init(resume="must")`, so metrics continue on the same W&B curves with no gaps.

```bash
# DAgger: automatic — run ID is read from the checkpoint
python main.py --mode dagger --checkpoint checkpoints/iter2000.pth

# Offline BC: automatic
python main.py --mode offline --data dataset.pt \
    --checkpoint checkpoints/offline_epoch10.pth

# Manual override (e.g. checkpoint saved before this feature was added):
python main.py --mode dagger --checkpoint old_checkpoint.pth \
    wandb_resume_id=abc123xyz

# Ablation suite:
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint path/to/ckpt.pth --all --use_wandb \
    --wandb_resume_id abc123xyz
```

The run ID is visible in the W&B dashboard URL: `wandb.ai/.../runs/<run-id>`.

---

## Implementation Notes

- **MDLM loss** returns `0.0` (not NaN) when no masked positions exist in the batch. Uses global averaging by default; SUBS importance weighting is opt-in via `use_importance_weighting: true`.
- **PAD tokens** are never masked during the forward process and are excluded from the loss.
- **Sampling paths:** Evaluation uses stochastic ReMDM sampling (temperature, top-K, remasking). DAgger collection uses greedy argmax sampling (deterministic, no remasking) for reproducible efficiency comparisons.
- **`remdm_sample`** guarantees a fully committed output (no MASK tokens) via a final-step commit and an assertion check. A min-keep 10% safety net prevents degenerate all-masked states.
- **EMA** shadow weights are updated after every gradient step (not per iteration). The `DataCollector` syncs the latest EMA weights before each rollout.
- **Curriculum** initialises with a 50/50 prior per environment (configurable via `curriculum_preseed`) and uses bucket-based weights: low win-rate (0.2), medium (1.0), high (0.1).
- **Replay buffer** pins offline data at the front; only online samples are FIFO-evicted. Returns `None` on empty buffer (callers handle gracefully).
- **Global gate** initialises at `sigmoid(-3.0) ~ 0.047`, starting nearly closed to prevent the global stream from destabilising early training.
- **Dropout** is set to 0.0 by default. The discrete diffusion forward masking already regularises; dropout on top is redundant.
