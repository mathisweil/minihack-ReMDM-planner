# ReMDM Planner for MiniHack

PyTorch implementation of **ReMDM** (Remasking Discrete Diffusion Model) for action-sequence planning in [MiniHack](https://github.com/facebookresearch/minihack) navigation environments. A dual-stream transformer generates 64-step action plans by iteratively denoising masked token sequences, conditioned on a 9x9 local crop and the full 21x79 dungeon map.

The primary training method is **DAgger** with BFS oracle supervision: the buffer is seeded with pure expert trajectories on the first iteration, providing an implicit behavioural cloning warm-start. An optional standalone offline BC mode is available for pre-training on collected datasets. Generalises **zero-shot** from 4 in-distribution environments to 3 out-of-distribution environments.

---

## Pipeline

```
[Primary]  DAgger online training          main.py --mode dagger
               |  (seed buffer with oracle demos on iter 0,
               |   collect with model, label with oracle,
               |   efficiency filter, curriculum sampling)
               v  checkpoint
[Evaluate] ID + OOD evaluation             main.py --mode inference --checkpoint iter8000.pth
```

**Optional standalone modes:**
```
[Collect]     Collect oracle demonstrations main.py --mode collect
[Offline BC]  Train on pre-collected data   main.py --mode offline --data dataset.pt
[Smoke test]  Quick end-to-end check        main.py --mode smoke
```

DAgger with implicit warm-start is the recommended pipeline. The `--mode collect` + `--mode offline` path is available for explicit two-stage pre-training on oracle demonstrations before DAgger.

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

**Python 3.12+** is required.

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

This installs all dependencies from the lockfile, including `nle>=1.2.0` (from the maintained [NetHack-LE](https://github.com/NetHack-LE/nle) fork), `minihack`, `torch>=2.11.0`, `wandb`, `polars`, `orjson`, and `scipy`.

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

Collects a few oracle trajectories, trains under a tiny 5k env-step budget, and prints ID evaluation results.

```bash
python main.py --mode smoke
```

### Collect oracle demonstrations

Run the BFS oracle across all 4 ID environments and save the trajectories as a `.pt` dataset for offline BC training. Uses multiprocessing for parallelism.

```bash
# Default: 5000 episodes per env, output to data/dataset.pt
python main.py --mode collect

# Custom episode count and output
python main.py --mode collect collect_episodes_per_env=2000 \
    collect_output=data/small_dataset.pt

# Fewer workers (default: 8)
python main.py --mode collect collect_num_workers=4

# Reproducible with fixed seed
python main.py --mode collect seed=42
```

The output `.pt` file is directly consumable by `--mode offline`:

```bash
python main.py --mode collect
python main.py --mode offline --data data/dataset.pt
```

### Offline BC (optional)

Train the diffusion model on pre-collected oracle demonstrations. The run length
is controlled by `total_timesteps` — each env-step of the unified budget
corresponds to one dataset sample, so total gradient steps =
`total_timesteps // offline_batch_size`.

```bash
python main.py --mode offline --data path/to/dataset.pt

# Shorter / longer run (the same knob the DAgger and SB3 baselines use):
python main.py --mode offline --data dataset.pt total_timesteps=500000

# Resume from a step-level checkpoint (restores optimizer, scheduler,
# step counter, and W&B run)
python main.py --mode offline --data path/to/dataset.pt \
    --checkpoint checkpoints/offline_step2000.pth
```

Step-level checkpoints are written every `checkpoint_every_timesteps` env-step
equivalents (converted internally to `/ offline_batch_size` grad steps).
Set to `0` to disable:

```bash
python main.py --mode offline --data dataset.pt checkpoint_every_timesteps=0
```

### DAgger online training

Full DAgger loop: seed buffer with oracle data, collect with model, label with BFS oracle, filter by efficiency, train on buffer.

```bash
# From scratch (seeds buffer with oracle data automatically)
python main.py --mode dagger

# Resume from local checkpoint
python main.py --mode dagger --checkpoint checkpoints/iter3000.pth

# Resume from a W&B artifact
python main.py --mode dagger \
    --wandb-artifact entity/project/checkpoint-iter3000:latest

# Skip warm-start from checkpoint (reinitialise model, keep config)
python main.py --mode dagger --checkpoint checkpoints/iter3000.pth --no-warm-start

# Override hyperparameters (total_timesteps is the unified run-length knob)
python main.py --mode dagger total_timesteps=1000000 dagger_lr=0.0001

# Use a GPU-optimised config
python main.py --mode dagger --config configs/qmul_gpu.yaml
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

# Custom .des scenario files
python main.py --mode inference \
    --checkpoint checkpoints/iter8000.pth \
    --des environments/custom_level.des

# Local-only ablation (zero out global map)
python main.py --mode inference \
    --checkpoint checkpoints/iter8000.pth --blind-global

# Use training weights instead of EMA
python main.py --mode inference --checkpoint iter8000.pth --no-ema
```

### CLI flags

| Flag | Description |
|---|---|
| `--mode` | Required. One of `smoke`, `collect`, `offline`, `dagger`, `inference` |
| `--config PATH` | Config file (default: `configs/defaults.yaml`) |
| `--data PATH` | Dataset `.pt` file (offline mode) |
| `--checkpoint PATH` | Checkpoint `.pth` file |
| `--wandb-artifact REF` | W&B artifact reference (e.g. `entity/project/name:latest`) |
| `--no-warm-start` | Skip model warm-start from checkpoint (DAgger) |
| `--no-ema` | Use training weights instead of EMA for inference |
| `--envs ENV [ENV ...]` | Override evaluation environments |
| `--des PATH [PATH ...]` | Custom `.des` scenario files for evaluation |
| `--episodes N` | Episodes per environment (default: 50) |
| `--output PATH` | Save evaluation results to JSON |
| `--blind-global` | Zero out global map observations (local-only ablation) |

---

## Architecture

**`LocalDiffusionPlannerWithGlobal`** (~5.2M parameters):

```
Local stream:   9x9 glyphs -> Embedding(6000,64) -> CNN(64->32->64) -> Linear -> 1 token
Global stream:  21x79 glyphs -> Embedding(6000,32) -> CNN(32->32->64) -> Pool(2,4) -> 8 tokens
                Goal head: mean(global) -> MLP -> [B,2] staircase coords (aux loss)
                Gate: sigmoid(learnable scalar, init=-3.0) * global_tokens
Action stream:  Embedding(14, 256) + timestep_emb(100, 256) + position_emb(64, 256)
Transformer:    concat [1 + 8 + 64 = 73 tokens] -> 4-layer encoder (256D, 4 heads, pre-norm)
Output head:    last 64 tokens -> Linear(256, 12) -> action logits
```

The model takes `(local_obs, global_obs, noisy_action_seq, t_discrete)` and returns `{"actions": [B,64,12], "goal_pred": [B,2]}`.

A `LocalDiffusionPlanner` variant (no global stream, no goal head) is also available for ablation studies.

---

## Diffusion

**Forward process (MDLM):** Each action token is independently replaced with `MASK` (token 12) with probability `1 - alpha(t)`, where `alpha(t)` follows a linear or cosine schedule. PAD tokens (13) are never masked.

**Loss:** Cross-entropy on masked positions only, averaged globally across the batch. By default uses a flat average (matching the reference implementation). Optional SUBS importance weighting `w(t) = -alpha'(t) / (1 - alpha(t))`, clipped to `[0, 1000]`, can be enabled via `use_importance_weighting: true`. Optional label smoothing via `label_smoothing` (default 0.0).

**Reverse sampling (ReMDM):** Over `K` denoising steps (default 10):
1. Model predicts logits; apply temperature scaling and top-K filtering.
2. Sample predictions; compute per-token confidence.
3. **MaskGIT unmask:** commit the `n_unmask` highest-confidence masked positions.
4. **ReMDM remask:** stochastically re-mask committed positions to allow refinement.
5. Final step: commit all remaining positions.

**Greedy sampling:** Used during DAgger data collection for deterministic rollouts. Same MaskGIT progressive unmasking loop but with argmax decoding (no temperature, no top-K, no remasking). Uses fewer denoising steps (`diffusion_steps_collect: 5`) for faster collection.

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
| `seq_len` | 64 | Action plan length |
| `dropout` | 0.0 | Transformer dropout (0.0 -- forward masking regularises) |
| `ema_decay` | 0.999 | EMA smoothing for inference weights |
| `global_gate_init` | -3.0 | Initial value for global gate logit |

**Diffusion**

| Parameter | Default | Description |
|---|---|---|
| `noise_schedule` | `linear` | `linear` or `cosine` |
| `num_diffusion_steps` | 100 | Discrete timestep resolution |
| `diffusion_steps_eval` | 10 | Denoising iterations at inference |
| `diffusion_steps_collect` | 5 | Denoising iterations during DAgger collection |
| `remask_strategy` | `conf` | `rescale`, `cap`, or `conf` |
| `eta` | 0.15 | Remasking strength |
| `temperature` | 0.5 | Sampling temperature |
| `top_k` | 4 | Top-K filtering |
| `replan_every` | 16 | Env steps before replanning |
| `loss_weight_clip` | 1000.0 | SUBS importance weight clip bound |
| `label_smoothing` | 0.0 | Label smoothing for cross-entropy |
| `use_importance_weighting` | false | SUBS w(t) in loss (off = flat average) |
| `physics_aware_sampling` | false | Penalise hazardous actions at inference |

**Training budget (unified)**

Offline BC, DAgger, and the SB3 baselines all share a single env-step budget
expressed in `total_timesteps` (matching the SB3 convention). This is the only
knob that should change to scale a run up or down.

| Parameter | Default | Description |
|---|---|---|
| `total_timesteps` | 2,000,000 | Env-step budget shared across offline / DAgger / SB3 |
| `id_eval_every_timesteps` | 25,000 | ID eval cadence (env-steps) |
| `ood_eval_every_timesteps` | 25,000 | OOD eval cadence (env-steps) |
| `checkpoint_every_timesteps` | 125,000 | Checkpoint cadence (env-steps) |

- **Offline BC:** each dataset sample is one env.step() equivalent, so total
  gradient steps = `total_timesteps // offline_batch_size`. The cosine LR
  schedule's `T_max` derives from the same quantity, so runs of different
  lengths still decay to the 10% floor at their end.
- **DAgger:** the training loop tracks cumulative `env.step()` calls (model +
  oracle rollouts combined) and halts when the running total reaches
  `total_timesteps`. `episodes_per_iteration` and `grad_steps_per_iteration`
  control the collect/train ratio but **must not** scale with the budget.
- **Fairness caveat — `ema_decay`:** this is an absolute-update-count constant
  (half-life ~ `1 / (1 − decay)` steps). If `total_timesteps` shifts by more
  than ~2× from the default, the fraction of training covered by the EMA
  window changes. For very short or very long runs, consider setting a
  matching decay manually.

**Training**

| Parameter | Default | Description |
|---|---|---|
| `offline_lr` | 0.0003 | BC learning rate (cosine-decayed to 10% over `total_grad_steps`) |
| `dagger_lr` | 0.00003 | DAgger learning rate (constant) |
| `offline_batch_size` | 3584 | Offline BC batch size |
| `dagger_batch_size` | 3584 | DAgger batch size |
| `offline_grad_clip` | 1.0 | Gradient norm clip (offline) |
| `dagger_grad_clip` | 1.0 | Gradient norm clip (DAgger) |
| `weight_decay` | 0.0001 | AdamW weight decay (both optimizers) |
| `grad_steps_per_iteration` | 100 | Gradient steps per DAgger iteration |
| `episodes_per_iteration` | 30 | Episodes collected per DAgger iteration |
| `aux_loss_weight` | 0.5 | Weight for auxiliary goal loss |
| `buffer_capacity` | 10000 | Replay buffer size (windows) |
| `efficiency_multiplier` | 1.5 | DAgger efficiency filter threshold |
| `curriculum_preseed` | true | Pre-seed curriculum with 50/50 prior |
| `curriculum_queue_size` | 100 | Curriculum window size per environment |

**Data Collection**

| Parameter | Default | Description |
|---|---|---|
| `collect_episodes_per_env` | 5000 | Oracle episodes per ID environment |
| `collect_num_workers` | 8 | Parallel process workers for collection |
| `collect_output` | `data/dataset.pt` | Output path for collected dataset |

**Evaluation**

| Parameter | Default | Description |
|---|---|---|
| `eval_episodes_per_env` | 50 | Episodes per environment at eval time |
| `checkpoint_eval_episodes` | 50 | Episodes per env at checkpoint eval |

(Eval and checkpoint *cadences* are expressed in env-steps under
**Training budget (unified)** above.)

**Performance**

| Parameter | Default | Description |
|---|---|---|
| `use_amp` | false | Mixed-precision (FP16) training via `torch.amp` |
| `torch_compile` | true | `torch.compile` the model for fused kernels |
| `num_collection_workers` | 8 | Parallel workers for DAgger episode collection |

**Logging**

| Parameter | Default | Description |
|---|---|---|
| `use_wandb` | true | Enable W&B logging |
| `wandb_project` | `remdm-minihack` | W&B project name |
| `wandb_resume_id` | null | W&B run ID for resumption |
| `offline_log_every` | 10 | Stdout/W&B log frequency (offline steps) |
| `seed` | null | RNG seed (null = random) |

### Config presets

| File | Purpose |
|---|---|
| `configs/defaults.yaml` | Base defaults for all modes |
| `configs/smoke.yaml` | Fast smoke test (`total_timesteps=5000`, small buffer, W&B off) |
| `configs/qmul_gpu.yaml` | QMUL GPU cluster (AMP on, 32 workers, B=2048) |
| `configs/ucl_gpu_bigger_model.yaml` | UCL GPU with larger model (384D, 6 heads) |
| `configs/ucl_gpu_learning_behaviour.yaml` | UCL GPU learning behaviour study (eta=0.18, B=6144) |
| `configs/ucl_gpu_no_amp.yaml` | UCL GPU without AMP (B=3584, 32 workers) |

---

## DAgger Training Loop

Each DAgger iteration:

1. **Curriculum sampling:** Select an environment weighted by difficulty (low win-rate environments sampled more).
2. **Model rollout:** Generate plans with the EMA model using greedy sampling; execute with replanning every 16 steps. Collects `episodes_per_iteration` (default 30) episodes per iteration.
3. **Oracle rollout:** Run the BFS oracle on the **same seed** for comparison.
4. **Efficiency filter:** Add the oracle trajectory to the buffer if the model failed or took >1.5x the oracle's steps.
5. **Budget accounting:** Advance `env_steps_total += model_steps + oracle_steps`. The training loop halts when the running total reaches `total_timesteps`.
6. **Training:** Sample from the replay buffer; run `grad_steps_per_iteration` gradient steps, updating EMA weights after each gradient step.

Collection uses GPU-batched rollouts when on CUDA with `episodes_per_iteration > 1`, falling back to threaded CPU collection or sequential collection as appropriate.

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
minihack-ReMDM-planner/
├── configs/
│   ├── defaults.yaml                  Base hyperparameters
│   ├── smoke.yaml                     Smoke test overrides
│   ├── qmul_gpu.yaml                 QMUL GPU cluster config
│   ├── ucl_gpu_bigger_model.yaml      UCL GPU (larger model)
│   ├── ucl_gpu_learning_behaviour.yaml UCL GPU (learning study)
│   └── ucl_gpu_no_amp.yaml           UCL GPU (no AMP)
├── environments/                      Custom .des scenario files
├── src/
│   ├── config.py                      YAML config loader with CLI overrides
│   ├── buffer.py                      ReplayBuffer with offline-protected FIFO
│   ├── curriculum.py                  DynamicCurriculum + efficiency_filter
│   ├── diffusion/
│   │   ├── schedules.py               Linear and cosine noise schedules
│   │   ├── forward.py                 Forward masking process q(z_t | x_0)
│   │   ├── loss.py                    MDLM ELBO + auxiliary goal loss
│   │   └── sampling.py                ReMDM reverse sampling with remasking
│   ├── models/
│   │   └── denoiser.py                LocalDiffusionPlannerWithGlobal + ModelEMA
│   ├── envs/
│   │   ├── minihack_env.py            AdvancedObservationEnv + BFS oracle
│   │   └── discovery.py               Env registry scanner + inference benchmark
│   └── planners/
│       ├── collect.py                 run_model_episode + DataCollector
│       ├── collect_oracle.py          Standalone oracle data collection
│       ├── offline.py                 Offline BC trainer
│       ├── online.py                  DAgger Trainer + checkpointing
│       ├── inference.py               Evaluator + result formatting
│       ├── smoke.py                   Smoke-test runner
│       └── logging.py                 Centralised W&B + stdout logging
├── experiments/
│   └── rl_finetuning/                 RL fine-tuning ablation suite
│       ├── run_ablations.py           CLI entry point
│       ├── configs/                   Ablation config files
│       ├── ablations/                 Loss, optimizer, registry, training
│       ├── diagnostics/               Gradient, representation, timestep metrics
│       └── analysis/                  Plots, tables, reports
├── scripts/
│   ├── hf_upload.py                   HuggingFace Hub upload utility
│   └── profile_dagger.py             DAgger iteration profiler
├── main.py                            CLI entry point (smoke/offline/dagger/inference)
├── pyproject.toml                     PEP 621 project metadata + dependencies
├── uv.lock                            Deterministic lockfile
└── README.md
```

---

## W&B Metric Namespaces

| Namespace | Contents |
|---|---|
| `diffusion/` | `loss`, `loss_diff`, `loss_aux` |
| `train/` | `buffer_size`, `buffer_online_frac`, `model_won`, `added_to_buffer`, `episodes_collected`, `model_steps`, `oracle_steps`, `efficiency_ratio`, `lr`, `grad_norm`, `global_gate`, `env_steps`, `progress` |
| `speed/` | `iter_time_sec`, `collect_time_sec`, `train_step_time_sec`, `samples_per_sec`, `env_steps_per_sec`, `gpu_memory_mb` |
| `perf/` | `iter_time_s`, `collect_time_s`, `train_time_s`, `grad_steps_per_sec` (legacy compat) |
| `model/` | `param_norm`, `param_drift_from_init`, `ema_gate_value` (every 10 iters) |
| `eval_id/{env}/` | Per-environment win rate, avg steps, avg reward (in-distribution) |
| `eval_ood/{env}/` | Per-environment win rate, avg steps, avg reward (out-of-distribution) |
| `eval_id/` | `mean_win_rate` |
| `eval_ood/` | `mean_win_rate` |
| `curriculum/{env}/` | `win_rate` per training environment |
| `ckpt_eval_id/`, `ckpt_eval_ood/` | Per-env metrics at checkpoint time |
| `ckpt_eval/` | `id_winrate`, `ood_winrate` |
| `offline/` | `final_loss`, `total_steps`, `total_timesteps` (summary only) |

---

## Checkpoint Format

**DAgger checkpoint:**

```python
{
    "model_state_dict":     ...,
    "ema_state_dict":       ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "curriculum_state":     {...},
    "iteration":            int,
    "env_steps":            int,   # cumulative env.step() calls so far
    "wandb_run_id":         str | None,
    "rng_states":           {"torch", "numpy", "python"},
}
```

**Offline BC checkpoint** (step-level, file `offline_step{N}.pth`, saved when
`checkpoint_every_timesteps > 0`):

```python
{
    "model_state_dict":     ...,
    "ema_state_dict":       ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "step":                 int,
    "env_steps":            int,   # step * offline_batch_size
    "wandb_run_id":         str | None,
}
```

**Offline final checkpoint** (saved at the end of offline training):

```python
{
    "model_state_dict":     ...,
    "ema_state_dict":       ...,
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
# DAgger: automatic -- run ID is read from the checkpoint
python main.py --mode dagger --checkpoint checkpoints/iter2000.pth

# Offline BC: automatic
python main.py --mode offline --data dataset.pt \
    --checkpoint checkpoints/offline_step2000.pth

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

## Performance Tuning

Three config keys control performance optimisations. Defaults are set for GPU training; override for CPU or different hardware.

### Mixed precision (`use_amp: true`)

Wraps training forward/backward in `torch.amp.autocast("cuda")` with `GradScaler`. Active in both offline BC and DAgger training.

- **Measured speedup:** 2.2x on gradient steps, 1.7x on full smoke test wall-clock
- **Memory:** peak GPU stays ~16 GB at B=3584 (same as FP32 due to embedding-heavy model)
- **Correctness:** loss trajectory and win rates statistically equivalent to FP32
- **When to use:** always on GPU. No effect on CPU (autocast is a no-op)
- **Default:** `false` in `defaults.yaml`; enabled in GPU-specific configs

### torch.compile (`torch_compile: true`)

Applies `torch.compile(model, mode="default")` before training. Falls back gracefully if no C compiler is found (common on managed GPU nodes).

- **Measured speedup:** none beyond AMP alone. Not recommended for primary training.
- **Default:** `true` in `defaults.yaml`
- **When to use:** experimental only. May help on future PyTorch versions with better dynamic shape support.

### Parallel collection (`num_collection_workers: N`)

DAgger episode collection supports three strategies (auto-selected):
1. **GPU-batched** (default on CUDA with `episodes_per_iteration > 1`): all envs in lockstep
2. **Threaded CPU** (fallback when `num_collection_workers > 0`): `ThreadPoolExecutor` with CPU model copies
3. **Sequential** (reference behaviour): one episode at a time

- **Default:** `8` workers in `defaults.yaml`
- **When to use:** GPU-batched is preferred; workers primarily affect the CPU fallback path

### Profiling

Run `python scripts/profile_dagger.py [key=value ...]` to profile DAgger iteration components. Supports all config overrides (e.g., `use_amp=true`).

---

## Implementation Notes

- **MDLM loss** returns `0.0` (not NaN) when no masked positions exist in the batch. Uses global averaging by default; SUBS importance weighting is opt-in via `use_importance_weighting: true`.
- **PAD tokens** are never masked during the forward process and are excluded from the loss.
- **Sampling paths:** Evaluation uses stochastic ReMDM sampling (temperature, top-K, remasking) with `diffusion_steps_eval` (default 10) steps. DAgger collection uses greedy argmax sampling (deterministic, no remasking) with `diffusion_steps_collect` (default 5) steps for faster rollouts.
- **`remdm_sample`** guarantees a fully committed output (no MASK tokens) via a final-step commit and an assertion check. A min-keep 10% safety net prevents degenerate all-masked states.
- **EMA** shadow weights are updated after every gradient step (not per iteration). The `DataCollector` syncs the latest EMA weights before each rollout.
- **Curriculum** initialises with a 50/50 prior per environment (configurable via `curriculum_preseed`) and uses bucket-based weights: low win-rate (0.2), medium (1.0), high (0.1).
- **Replay buffer** pins offline data at the front; only online samples are FIFO-evicted. Returns `None` on empty buffer (callers handle gracefully).
- **Global gate** initialises at `sigmoid(-3.0) ~ 0.047`, starting nearly closed to prevent the global stream from destabilising early training.
- **Dropout** is set to 0.0 by default. The discrete diffusion forward masking already regularises; dropout on top is redundant.
- **DAgger warm-start:** On iteration 0, the buffer is seeded with 3 oracle trajectories per ID environment (12 total), giving the curriculum and training loop data to work with immediately.
