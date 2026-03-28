# ReMDM Planner for MiniHack

PyTorch implementation of **ReMDM** (Remasking Discrete Diffusion Model) for action-sequence planning in [MiniHack](https://github.com/facebookresearch/minihack) navigation environments. A dual-stream transformer generates 64-step action plans by iteratively denoising masked token sequences, conditioned on a 9x9 local crop and the full 21x79 dungeon map.

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

### 1. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate minihack
```

### 2. GPU support (optional)

By default PyTorch runs on CPU. For NVIDIA CUDA 12:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

verify GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

MiniHack requires the [NetHack Learning Environment](https://github.com/facebookresearch/nle). See the NLE docs if the `pip install nle` step fails on your platform.

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
```

### DAgger online training

Full DAgger loop: collect with model, label with BFS oracle, filter by efficiency, train on buffer.

```bash
# From scratch (seeds buffer with oracle data automatically)
python main.py --mode dagger

# Resume from checkpoint
python main.py --mode dagger --checkpoint checkpoints/iter3000.pth

# Override hyperparameters
python main.py --mode dagger max_iterations=4000 dagger_lr=0.0001
```

### Inference

Evaluate a checkpoint on specified environments.

```bash
# All ID + OOD environments
python main.py --mode inference --checkpoint checkpoints/iter8000.pth

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
Action stream:  Embedding(14, 256) + timestep_emb(100, 256) + position_emb(64, 256)
Transformer:    concat [1 + 8 + 64 = 73 tokens] -> 4-layer encoder (256D, 4 heads, pre-norm)
Output head:    last 64 tokens -> Linear(256, 12) -> action logits
```

The model takes `(local_obs, global_obs, noisy_action_seq, t_discrete)` and returns `{"actions": [B,64,12], "goal_pred": [B,2]}`.

---

## Diffusion

**Forward process (MDLM):** Each action token is independently replaced with `MASK` (token 12) with probability `1 - alpha(t)`, where `alpha(t)` follows a linear or cosine schedule. PAD tokens (13) are never masked.

**Loss:** Continuous-time ELBO with SUBS parameterisation. Cross-entropy is computed on masked positions only, weighted by `w(t) = -alpha'(t) / (1 - alpha(t))`, clipped to `[0, 1000]`.

**Reverse sampling (ReMDM):** Over `K` denoising steps:
1. Model predicts logits; apply temperature scaling and top-K filtering.
2. Sample predictions; compute per-token confidence.
3. **MaskGIT unmask:** commit the `n_unmask` highest-confidence masked positions.
4. **ReMDM remask:** stochastically re-mask committed positions to allow refinement.
5. Final step: commit all remaining positions.

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
| `ema_decay` | 0.999 | EMA smoothing for inference weights |

**Diffusion**

| Parameter | Default | Description |
|---|---|---|
| `noise_schedule` | `linear` | `linear` or `cosine` |
| `num_diffusion_steps` | 100 | Discrete timestep resolution |
| `diffusion_steps_eval` | 5 | Denoising iterations at inference |
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
| `offline_batch_size` | 256 | Offline BC batch size |
| `dagger_batch_size` | 256 | DAgger batch size |
| `offline_epochs` | 10 | BC training epochs |
| `max_iterations` | 8000 | DAgger iterations |
| `grad_steps_per_iteration` | 50 | Gradient steps per DAgger iteration |
| `aux_loss_weight` | 0.5 | Weight for auxiliary goal loss |
| `buffer_capacity` | 10000 | Replay buffer size (windows) |
| `efficiency_multiplier` | 1.5 | DAgger efficiency filter threshold |

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
2. **Model rollout:** Generate plans with the EMA model; execute with replanning every 16 steps.
3. **Oracle rollout:** Run the BFS oracle on the **same seed** for comparison.
4. **Efficiency filter:** Add the oracle trajectory to the buffer if the model failed or took >1.5x the oracle's steps.
5. **Training:** Sample from the replay buffer; run `grad_steps_per_iteration` gradient steps.
6. **EMA update:** Blend model weights into the shadow copy.

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
│       ├── train.py               Offline BC trainer
│       ├── online.py              DAgger Trainer + checkpointing
│       ├── inference.py           Evaluator + result formatting
│       └── logging.py             Centralised W&B + stdout logging
├── scripts/
│   └── hf_upload.py               HuggingFace Hub upload utility
├── main.py                        CLI entry point (smoke/offline/dagger/inference)
└── requirements.txt
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

```python
{
    "model_state_dict":     ...,
    "ema_state_dict":       ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "curriculum_state":     {"env_ids", "queue_size", "queues"},
    "iteration":            int,
    "rng_states":           {"torch", "numpy", "python"},
}
```

Inference uses EMA weights by default. Pass `--no-ema` to use training weights.

---

## Implementation Notes

- **MDLM loss** returns `0.0` (not NaN) when no masked positions exist in the batch.
- **PAD tokens** are never masked during the forward process and are excluded from the loss.
- **`remdm_sample`** guarantees a fully committed output (no MASK tokens) via a final-step commit and an assertion check.
- **Curriculum** initialises with a 50/50 prior per environment and uses bucket-based weights: low win-rate (0.2), medium (1.0), high (0.1).
- **Replay buffer** pins offline data at the front; only online samples are FIFO-evicted.
- **Global gate** initialises at `sigmoid(-3.0) ~ 0.047`, starting nearly closed to prevent the global stream from destabilising early training.
