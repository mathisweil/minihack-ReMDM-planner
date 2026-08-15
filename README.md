# ReMDM Planner for MiniHack

PyTorch implementation of **ReMDM** (Remasking Discrete Diffusion Model) for action-sequence planning in [MiniHack](https://github.com/facebookresearch/minihack) navigation environments. A dual-stream transformer generates 64-step action plans by iteratively denoising masked token sequences, conditioned on a 9x9 local crop and the full 21x79 dungeon map. Trained with **DAgger** under BFS oracle supervision, from scratch; generalises zero-shot from 4 in-distribution to 3 out-of-distribution environments.

The sibling repository [`craftax-ReMDM-planner`](../craftax-ReMDM-planner) implements the same method in JAX on Craftax. Both repos share the same CLI, config layout and README structure; commands transfer between them by swapping the repo name and benchmark-specific values.

## Method

The planner starts from a fully-masked action sequence and iteratively unmasks tokens over `K` denoising steps via the ReMDM Algorithm 1 posterior (per-token Bernoulli unmasking), while ReMDM remasking lets committed tokens be re-predicted for plan refinement (a MaskGIT-style greedy decoder is used only for DAgger data collection). Two independent training pipelines are compared head-to-head in the accompanying paper (under submission; citation to follow): **online DAgger** under a BFS oracle (primary) and **offline behavioural cloning** on pre-collected oracle datasets. See [Architecture](#architecture) and [Diffusion](#diffusion) for details.

## Setup

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/). `nle` compiles from source on macOS.
Linux GPU use needs NVIDIA driver >= 580 for CUDA 13, or >= 525 with `--extra cuda12`.

```bash
# macOS (arm64)
brew install cmake

# Linux (x86_64), if building from source
sudo apt-get install build-essential cmake bison flex libbz2-dev

git clone https://github.com/mathisweil/minihack-ReMDM-planner.git
cd minihack-ReMDM-planner

# Default. macOS gets the PyPI build (MPS); Linux gets PyPI's CUDA 13.0 build.
# Installs the dev group (pytest) too.
uv sync

# Linux, explicit CUDA 13.2 (driver >= 580)
uv sync --extra cuda13

# Linux, CUDA 12.6 fallback (driver >= 525, or Maxwell/Pascal cards)
uv sync --extra cuda12
```

Extras: `cuda13` and `cuda12` are mutually exclusive and Linux-only. Neither is needed on a
modern driver: plain `uv sync` already yields a CUDA 13.0 build on Linux. Use `cuda12` only
if `nvidia-smi` reports a driver older than 580.

> **Install path must not contain spaces.** MiniHack's `mh_patch_nhdat.sh` interpolates paths unquoted and fails silently on whitespace, leaving every environment as the same default level with no goal staircase. `src/envs/minihack_env.py` detects this and substitutes a Python implementation, but a space-free path avoids the issue entirely.

## Repo layout

```
minihack-ReMDM-planner/
├── configs/                Experiment configs (defaults.yaml + presets, see Configuration)
├── environments/           Custom .des scenario files (empty; user-supplied)
├── src/                    Model, diffusion, envs, planner pipelines
├── experiments/
│   └── rl_finetuning/      RL fine-tuning ablation suite (run_ablations.py)
├── scripts/                HF upload utilities, DAgger and ablation profilers
├── tests/                  Smoke suite — uv run pytest
├── checkpoints/            Gitignored — offline/, online/ (see Checkpoints)
├── results/inference/      Eval JSONs from --mode inference (published, see Checkpoints)
├── demo_minihack.ipynb     Demo notebook
├── main.py                 CLI entry point
└── pyproject.toml          uv project — deps, cuda extra, dev group
```

## Quickstart

Collects a few oracle trajectories, trains under a 5k env-step budget, prints ID evaluation. A few minutes on CPU.

```bash
python main.py --mode smoke
```

## Training

Two independent training methods; neither depends on the other. An offline BC checkpoint can warm-start DAgger via `--checkpoint`, but this was not used for the paper results.

### Online DAgger (primary)

```bash
python main.py --mode online                                            # full paper recipe (defaults.yaml)
python main.py --mode online --config configs/final_qmul_gpu.yaml       # paper run, QMUL H200
python main.py --mode online --override total_timesteps=1000000 --override dagger_lr=0.0001
python main.py --mode online --checkpoint checkpoints/iter600.pth       # resume
python main.py --mode online --checkpoint checkpoints/iter600.pth --no-warm-start
```

Per iteration: curriculum-sampled model rollouts, BFS oracle labelling on the same seeds, efficiency filtering into the replay buffer, `grad_steps_per_iteration` gradient steps. Halts when cumulative env steps reach `total_timesteps`. See [DAgger training loop](#dagger-training-loop).

### Offline BC

First collect a dataset, then train on it:

```bash
python main.py --mode collect                                    # 5000 eps/env -> data/dataset.pt
python main.py --mode collect --data data/small.pt --override collect_episodes_per_env=2000

python main.py --mode offline --data data/dataset.pt
python main.py --mode offline --data data/dataset.pt --override total_timesteps=500000

# Resume (restores optimizer, scheduler, step counter, W&B run)
python main.py --mode offline --data data/dataset.pt --checkpoint checkpoints/offline_step40000.pth
```

Gradient steps default to `total_timesteps // offline_batch_size`; ID + OOD eval runs on the `id_eval_every_timesteps` / `ood_eval_every_timesteps` cadence. For paper-fair BC-vs-DAgger comparisons the `offline_*_grad_steps` keys pin offline metrics in grad-step units instead, and since `defaults.yaml` is the paper recipe they are set there — so they apply to **every** run unless a preset pins them back to `null`. See the hazard note under [Configuration](#configuration).

## Evaluation from a checkpoint

```bash
python main.py --mode inference --checkpoint checkpoints/iter600.pth    # all ID + OOD
python main.py --mode inference --checkpoint wandb:entity/project/checkpoint-iter600:latest

# Specific environments, save JSON
python main.py --mode inference --checkpoint checkpoints/iter600.pth \
    --envs MiniHack-Room-Random-5x5-v0 MiniHack-MazeWalk-45x19-v0 \
    --episodes 100 --output results/inference/eval.json

python main.py --mode inference --checkpoint checkpoints/iter600.pth \
    --des environments/<your_level>.des        # custom .des scenarios (dir ships empty)
python main.py --mode inference --checkpoint checkpoints/iter600.pth --no-ema
```

`--checkpoint` accepts a local `.pth` path or a `wandb:` artifact reference (`wandb:entity/project/name:version`). Inference uses EMA weights unless `--no-ema` is given; `--episodes` defaults to `eval_episodes_per_env` from the config.

Write eval JSONs into `results/inference/` (created for you): `scripts/hf_upload.py` publishes every JSON it finds there.

Always evaluate with the checkpoint's own config snapshot:

```bash
DIR=checkpoints/online/Minihack-OnlineDiffusion-DAgger-123M
python main.py --mode inference --config $DIR/config_iter600.yaml --checkpoint $DIR/iter600.pth
```

## Baselines and ablations

### RL and imitation baselines

Six algorithms: SB3 discrete-action RL (`ppo`, `a2c`, `dqn`, `ppo-rnn`), Behavioural Cloning (`bc`) on oracle demos, and a causal Decision Transformer (`dt`). All share `total_timesteps`, so numbers are comparable to DAgger and offline BC. Hyperparameters live under the `baselines_*` config namespace; outputs go to `baselines_output_dir`.

```bash
python main.py --mode baselines --algo ppo
python main.py --mode baselines --algo a2c
python main.py --mode baselines --algo dqn --seeds 0 1 2
python main.py --mode baselines --algo ppo-rnn
python main.py --mode baselines --algo bc --num-seeds 3
python main.py --mode baselines --algo dt --seeds 0 1 2
python main.py --mode baselines --algo ppo --output results/ppo.json
python main.py --mode baselines --algo ppo --override total_timesteps=5650000   # match ReMDM online budget
```

### Architecture ablations

```bash
# Local-only planner (no global stream, no goal head), trained from scratch
python main.py --mode online --config configs/ablation_local_only.yaml

# Blind-global: zero the global observation of a trained dual-stream model at eval
python main.py --mode inference --checkpoint checkpoints/iter600.pth --blind-global
```

### RL fine-tuning ablation suite

25 registered ablations (same names as in the craftax repo). See `experiments/README.md`.

```bash
python experiments/rl_finetuning/run_ablations.py --list
python experiments/rl_finetuning/run_ablations.py --checkpoint path/to/ckpt.pth --all
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint wandb:entity/project/checkpoint-iter600:latest \
    --ablations baseline_rl kl_penalty --fast
```

## Configuration

One YAML config holds the experiment; the CLI holds the run.

- **`configs/defaults.yaml`**: the **shared final paper recipe**, not a cheap baseline. Both clusters train exactly this; running with no `--config` trains it too.
- **Config files** (`configs/*.yaml`): any file passed via `--config` is deep-merged onto `defaults.yaml`, so presets contain **only their deltas** — never re-state a default value. Presets are a single layer: they never inherit from one another.
- **CLI flags**: per-invocation values — `--seed`, `--checkpoint`, `--data`, `--output`, `--episodes`, `--envs`, mode switches.
- **`--override KEY=VALUE`** (repeatable): ad hoc config overrides. Keys are validated against `defaults.yaml` and values are cast to the key's type; a typo is an error, not a silent no-op.

Precedence, lowest to highest: `configs/defaults.yaml` < `--config` file < `--override` and run flags.

> **Hazard when writing a preset.** Four keys silently *override* an env-step-derived value when non-null, and `defaults.yaml` now sets all four as part of the recipe: `offline_total_grad_steps`, `offline_eval_every_grad_steps`, `offline_checkpoint_every_grad_steps`, `offline_buffer_capacity`. A preset that wants its own `total_timesteps` to govern the offline budget must pin them back to **explicit `null`** — omitting them inherits the pins. Left unpinned, `smoke.yaml` would train 60,000 offline gradient steps instead of 19. `tests/test_config.py` enforces the pins for every preset that derives its own budget.

| Preset | Purpose |
|---|---|
| `configs/defaults.yaml` | **Shared final paper recipe** — the full run both clusters train |
| `configs/smoke.yaml` | Smoke test (`total_timesteps=5000`, small buffer, W&B off) |
| `configs/ablation_local_only.yaml` | Local-only planner ablation (`use_global_stream: false`) |
| `configs/ucl_gpu_bigger_model.yaml` | UCL GPU, larger model (384D, 6 heads) |
| `configs/ucl_gpu_learning_behaviour.yaml` | UCL GPU learning-behaviour study (eta=0.18, B=6144) |
| `configs/final_qmul_gpu.yaml` | **Paper run, QMUL H200.** Machine values only: worker counts (32) and dataset path |
| `configs/final_ucl_gpu.yaml` | **Paper run, UCL 3090 Ti.** Machine values only: dataset path (workers stay at the default 8) |

Key hyperparameters are documented inline in `configs/defaults.yaml`; the [appendix](#key-hyperparameters) tabulates them.

## Checkpoints

Training writes to a unique run directory under `checkpoint_dir` (default `checkpoints/`), named `{tag}_{YYYYMMDD}_{HHMMSS}_{hex4}`. DAgger saves `iter{N}.pth` on the `checkpoint_every_timesteps` cadence; offline BC saves `offline_step{N}.pth` and `offline_final.pth`. Checkpoints also upload as versioned W&B artifacts (type `model`) when `use_wandb` is on. All checkpoints store the W&B run ID, so passing them back via `--checkpoint` resumes the same W&B curve automatically.

`checkpoints/` is gitignored. Released weights live on the Hugging Face Hub: **[mathisweil/remdm-minihack-checkpoints](https://huggingface.co/mathisweil/remdm-minihack-checkpoints)**

| Directory | Method | Selected at | Sample-equivalents |
|---|---|---|---|
| `checkpoints/online/Minihack-OnlineDiffusion-DAgger-123M` | DAgger (main result) | iteration 600 | 123M |
| `checkpoints/offline/Minihack-OfflineDiffusion-BC-82M` | Offline BC baseline | gradient step 40,000 | 82M |

```bash
# All checkpoints
uv run hf download mathisweil/remdm-minihack-checkpoints --include "checkpoints/**" --local-dir .

# One checkpoint
uv run hf download mathisweil/remdm-minihack-checkpoints \
    --include "checkpoints/online/Minihack-*/**" --local-dir .
```

Each checkpoint directory ships `<step>.pth` (full training state), `model.safetensors` (EMA weights only, no pickle), `config_<step>.yaml` (config snapshot) and `selection.json`. See [Checkpoint format](#checkpoint-format) for the `.pth` schema and programmatic loading.

### Publishing to the Hub

`scripts/hf_upload.py` rediscovers and uploads three things, each keeping its repo-relative path: `checkpoints/` (adding a `model.safetensors` EMA export and `selection.json` per directory), every `experiments/rl_finetuning/outputs/<run>/` holding a `results.json` (with `diagnosis.md`, `tables/`, `figures/`), and the eval JSONs in `results/inference/`. It drops W&B and hub config keys, shortens absolute paths and regenerates the model card.

```bash
HF_TOKEN=hf_xxx uv run python scripts/hf_upload.py --repo-id mathisweil/remdm-minihack-checkpoints \
    --selection-metric "mean ID+OOD win rate" --dry-run
```

`--dry-run` prints the staged tree and card without uploading; drop it to upload. `--selection-metric` records what the best-of-N checkpoints were chosen on. Also `--inference-results <FILE|DIR> ...` (eval JSONs kept elsewhere), `--private`, `--yes`. Publish one model per directory with a single `.pth` and config, since the script takes one row per directory and otherwise picks by sort order.

## Results, citation, licence

Results tables and the full method description are in the accompanying paper (under submission); `demo_minihack.ipynb` reproduces the headline comparison. Citation to be added on publication. Licence: MIT, see `LICENSE`.

---

# Appendix: benchmark-specific detail

## Environments

| In-distribution (training) | Out-of-distribution (zero-shot eval) |
|---|---|
| `MiniHack-Room-Random-5x5-v0` (small random room) | `MiniHack-Room-Dark-15x15-v0` (dark room) |
| `MiniHack-Room-Random-15x15-v0` (large random room) | `MiniHack-Corridor-R5-v0` (five-room corridor) |
| `MiniHack-Corridor-R2-v0` (two-room corridor) | `MiniHack-MazeWalk-45x19-v0` (large maze) |
| `MiniHack-MazeWalk-9x9-v0` (small maze) | |

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

Signature: `(local_obs, global_obs, noisy_action_seq, t_discrete)` -> `{"actions": [B,64,12], "goal_pred": [B,2]}`.

`LocalDiffusionPlanner` (no global stream, no goal head) is the `ablation_local_only` variant. It trains a genuinely local-only model, unlike `--blind-global`, which zeroes the global observation of an already-trained dual-stream model at inference. Supported by `--mode offline` and `--mode online`; the `experiments/` ablation suite assumes the goal head is present.

## Diffusion

- **Forward process (MDLM):** each action token is independently replaced with `MASK` (12) with probability `1 - alpha(t)`, `alpha(t)` linear or cosine. PAD (13) is never masked.
- **Loss:** continuous-time MDLM NELBO: per sample `w(t) * sum_masked(CE) / L` with `w(t) = -alpha'(t) / (1 - alpha(t))` clipped to `[0, 1000]`; optional `label_smoothing`.
- **Greedy sampling:** used for DAgger collection. Same MaskGIT loop, argmax decoding, no temperature/top-K/remasking, `diffusion_steps_collect` steps.

**Reverse sampling (ReMDM Algorithm 1)**, over `K` steps (default 10):

1. Predict logits; apply temperature and top-p (nucleus) filtering; sample predictions and record each committed token's decode probability `psi`.
2. **Unmask:** each masked position commits independently with the posterior probability `(alpha_s - (1 - sigma) alpha_t) / (1 - alpha_t)`.
3. **ReMDM remask:** each committed position re-masks with probability `sigma` from the configured Section-4.1 schedule.
4. Final step: any remaining masked positions are committed by a greedy cleanup pass.

| Strategy | Formula | Description |
|---|---|---|
| `rescale` | `p = eta * sigma_max` | Proportional to noise level |
| `cap` | `p = min(eta, sigma_max)` | Fixed upper bound |
| `conf` | `p = softmax(-confidence) * eta * sigma_max` | Low-confidence tokens remasked more |

## Key hyperparameters

**Model**

| Parameter | Default | Description |
|---|---|---|
| `n_embd` | 256 | Transformer hidden dimension |
| `n_head` | 4 | Attention heads |
| `n_layer` | 4 | Transformer blocks |
| `n_global_tokens` | 8 | Global stream context tokens |
| `seq_len` | 64 | Action plan length |
| `dropout` | 0.0 | Forward masking already regularises |
| `ema_decay` | 0.999 | EMA smoothing for inference weights |
| `global_gate_init` | -3.0 | Initial global gate logit |
| `use_global_stream` | true | `false` builds the local-only ablation variant |

**Diffusion**

| Parameter | Default | Description |
|---|---|---|
| `noise_schedule` | `linear` | `linear`, `cosine`, or `cosine_sq` (MDLM App E.1 naming) |
| `num_diffusion_steps` | 100 | Discrete timestep resolution |
| `diffusion_steps_eval` | 10 | Denoising iterations at inference |
| `diffusion_steps_collect` | 5 | Denoising iterations during collection |
| `remask_strategy` | `conf` | `rescale`, `cap`, or `conf` |
| `eta` | 0.15 | Remasking strength |
| `temperature` | 0.5 | Sampling temperature |
| `top_p` | 0.9 | Nucleus threshold (ReMDM Sec 5) |
| `replan_every` | 16 | Env steps before replanning |
| `loss_weight_clip` | 1000.0 | NELBO weight clip bound |
| `label_smoothing` | 0.0 | Cross-entropy label smoothing |
| `physics_aware_sampling` | false | Penalise hazardous actions at inference |

**Training budget (unified).** Offline BC, DAgger and the SB3 baselines share one env-step budget. This is the only knob that should change to scale a run.

| Parameter | Default | Description |
|---|---|---|
| `total_timesteps` | 5,650,000 | Shared env-step budget |
| `id_eval_every_timesteps` | 470,000 | ID eval cadence |
| `ood_eval_every_timesteps` | 470,000 | OOD eval cadence |
| `checkpoint_every_timesteps` | 940,000 | Checkpoint cadence |

- **Offline BC:** gradient steps = `total_timesteps // offline_batch_size`. The cosine LR `T_max` derives from the same quantity, so any run length decays to the 10% floor at its end.
- **DAgger:** tracks cumulative `env.step()` calls (model + oracle) and halts at `total_timesteps`. `episodes_per_iteration` and `grad_steps_per_iteration` set the collect/train ratio and **must not** scale with the budget.
- **Caveat, `ema_decay`:** an absolute-update-count constant (half-life ~ `1 / (1 - decay)` steps). Shifting `total_timesteps` by more than ~2x changes the fraction of training the EMA window covers; set a matching decay manually for very short or long runs.

**Training**

| Parameter | Default | Description |
|---|---|---|
| `offline_lr` | 0.0003 | BC LR (cosine-decayed to 10%) |
| `dagger_lr` | 0.00003 | DAgger LR (constant) |
| `offline_batch_size` | 2048 | Offline BC batch size |
| `dagger_batch_size` | 2048 | DAgger batch size |
| `offline_grad_clip` | 1.0 | Gradient norm clip (offline) |
| `dagger_grad_clip` | 1.0 | Gradient norm clip (DAgger) |
| `weight_decay` | 0.0001 | AdamW weight decay |
| `grad_steps_per_iteration` | 100 | Gradient steps per DAgger iteration |
| `episodes_per_iteration` | 30 | Episodes per DAgger iteration |
| `aux_loss_weight` | 0.5 | Auxiliary goal loss weight |
| `buffer_capacity` | 10000 | Replay buffer size (windows) |
| `efficiency_multiplier` | 1.5 | DAgger efficiency filter threshold |
| `curriculum_preseed` | true | Pre-seed curriculum with 50/50 prior |
| `curriculum_queue_size` | 100 | Curriculum window size per environment |

**Collection, evaluation, performance, logging**

| Parameter | Default | Description |
|---|---|---|
| `collect_episodes_per_env` | 5000 | Oracle episodes per ID environment |
| `collect_num_workers` | 8 | Process workers for collection |
| `collect_output` | `data/dataset.pt` | Collected dataset path (per-run: `--data`) |
| `eval_episodes_per_env` | 50 | Episodes per env at eval (per-run: `--episodes`) |
| `checkpoint_eval_episodes` | 50 | Episodes per env at checkpoint eval |
| `use_amp` | true | Mixed precision via `torch.amp` |
| `torch_compile` | true | `torch.compile` the model |
| `num_collection_workers` | 8 | Workers for DAgger collection |
| `use_wandb` | true | Enable W&B logging |
| `wandb_project` | `minihack-ReMDM-planner` | W&B project |
| `wandb_resume_id` | null | W&B run ID for resumption |
| `offline_log_every` | 50 | Log frequency (offline steps) |
| `seed` | null | RNG seed (null = random; per-run: `--seed`) |

## DAgger training loop

1. **Curriculum sampling:** pick an environment weighted by difficulty (low win-rate sampled more).
2. **Model rollout:** EMA model, greedy sampling, replanning every 16 steps, `episodes_per_iteration` episodes.
3. **Oracle rollout:** BFS oracle on the **same seed**.
4. **Efficiency filter:** add the oracle trajectory if the model failed or took >1.5x the oracle's steps.
5. **Budget accounting:** `env_steps_total += model_steps + oracle_steps`; halt at `total_timesteps`.
6. **Training:** sample the buffer, run `grad_steps_per_iteration` steps, update EMA after each.

Collection is GPU-batched on CUDA with `episodes_per_iteration > 1`, falling back to threaded CPU or sequential.

BFS oracle priority: (1) kick adjacent doors, (2) BFS to staircase, (3) BFS to frontier, (4) BFS to farthest tile, (5) random cardinal.

## Reward shaping

| Component | Value | Condition |
|---|---|---|
| Win bonus | +20.0 | Episode won |
| BFS progress | +0.5 * (prev_dist - curr_dist) | Closer to staircase |
| Exploration | +0.05 | New tile visited |
| Step penalty | -0.01 | Every step |

## Checkpoint format

```python
# DAgger
{
    "model_state_dict": ..., "ema_state_dict": ...,
    "optimizer_state_dict": ..., "scheduler_state_dict": ...,
    "curriculum_state": {...},
    "iteration": int,
    "env_steps": int,                  # cumulative env.step() calls
    "wandb_run_id": str | None,
    "rng_states": {"torch", "numpy", "python"},
}

# Offline BC, step-level (offline_step{N}.pth, when checkpoint_every_timesteps > 0)
{
    "model_state_dict": ..., "ema_state_dict": ...,
    "optimizer_state_dict": ..., "scheduler_state_dict": ...,
    "step": int,
    "env_steps": int,                  # step * offline_batch_size
    "wandb_run_id": str | None,
}

# Offline BC, final (offline_final.pth)
{"model_state_dict": ..., "ema_state_dict": ..., "wandb_run_id": str | None}
```

### Load programmatically

```python
# Inference, from safetensors (already EMA weights)
from safetensors.torch import load_file
from src.config import load_config
from src.models.denoiser import make_model

DIR = "checkpoints/online/Minihack-OnlineDiffusion-DAgger-123M"
cfg = load_config(f"{DIR}/config_iter600.yaml")
model = make_model(cfg)
model.load_state_dict(load_file(f"{DIR}/model.safetensors"))
model.eval()
```

```python
# From the full .pth, to resume or to pick training vs EMA weights
import torch
from src.config import load_config
from src.models.denoiser import make_model, ModelEMA

DIR = "checkpoints/online/Minihack-OnlineDiffusion-DAgger-123M"
cfg = load_config(f"{DIR}/config_iter600.yaml")
ckpt = torch.load(f"{DIR}/iter600.pth", map_location="cpu", weights_only=False)

model = make_model(cfg)
model.load_state_dict(ckpt["model_state_dict"])

ema = ModelEMA(model, decay=cfg.ema_decay)
ema.load_state_dict(ckpt["ema_state_dict"])
ema.apply_to(model)          # what evaluation uses by default
model.eval()
```

### W&B artifacts and run resumption

Checkpoints upload as versioned W&B artifacts (type `"model"`) at each save, containing the `.pth` and a `config.yaml` snapshot. Reference format is `wandb:entity/project/artifact-name:version`, version being `latest`, `v0`, `v1`.

All training loops store the W&B run ID in their checkpoints. Resuming extracts it and passes it to `wandb.init(resume="must")`, so curves continue with no gaps.

```bash
python main.py --mode online --checkpoint checkpoints/iter600.pth               # automatic
# Manual override (checkpoint predates the feature)
python main.py --mode online --checkpoint old.pth --override wandb_resume_id=abc123xyz
```

## W&B metric namespaces

| Namespace | Contents |
|---|---|
| `diffusion/` | `loss`, `loss_diff`, `loss_aux` |
| `train/` | `buffer_size`, `buffer_online_frac`, `model_won`, `added_to_buffer`, `episodes_collected`, `model_steps`, `oracle_steps`, `efficiency_ratio`, `lr`, `grad_norm`, `global_gate`, `env_steps`, `progress` |
| `speed/` | `iter_time_sec`, `collect_time_sec`, `train_step_time_sec`, `samples_per_sec`, `env_steps_per_sec`, `gpu_memory_mb` |
| `model/` | `param_norm`, `param_drift_from_init`, `ema_gate_value` (every 10 iters) |
| `eval_id/{env}/`, `eval_ood/{env}/` | Per-env win rate, avg steps, avg reward |
| `eval_id/`, `eval_ood/` | `mean_win_rate` |
| `curriculum/{env}/` | `win_rate` per training environment |
| `ckpt_eval_id/`, `ckpt_eval_ood/` | Per-env metrics at checkpoint time |
| `ckpt_eval/` | `id_winrate`, `ood_winrate` |
| `offline/` | `final_loss`, `total_steps`, `total_timesteps` (summary only) |

DAgger and offline BC both emit to `eval_id/` and `eval_ood/`, reusing the same `Evaluator` and EMA-weight path, so curves are directly comparable.

## Performance tuning

| Key | Default | Effect |
|---|---|---|
| `use_amp` | false | `torch.amp.autocast("cuda")` + `GradScaler` in both trainers. **2.2x** on gradient steps, **1.7x** on smoke-test wall-clock. Loss and win rates statistically equivalent to FP32. No-op on CPU. Always enable on GPU |
| `torch_compile` | false | `torch.compile(model, mode="default")`. No measured gain beyond AMP. Experimental only |
| `num_collection_workers` | 8 | Affects the threaded CPU fallback. Collection auto-selects GPU-batched (CUDA, `episodes_per_iteration > 1`) > threaded CPU > sequential |

Profile with `python scripts/profile_dagger.py [--override key=value ...]`.

## Testing

```bash
uv run pytest            # ~35s
uv run pytest -m slow    # slow entry points only (BC + PPO baselines), ~45s
```

`tests/test_smoke_src.py` and `tests/test_smoke_experiments.py` cover both pipelines: modules import, the model builds from `configs/defaults.yaml`, a forward pass returns the expected shape and dtype with no NaNs, one training step gives a finite loss, save/reload reproduces identical output, each entry point runs, and all 25 registry ablations step. They assert things *run*, not that results are good. CPU-only, seeded, synthetic data; nothing written outside `tmp_path`. For a quality signal, use `--mode smoke`.

## Implementation notes

- **MDLM loss** returns `0.0` (not NaN) when no masked positions exist. NELBO-weighted per MDLM eq (10).
- **PAD tokens** are never masked and are excluded from the loss.
- **Sampling paths:** evaluation uses stochastic ReMDM (temperature, top-p, remasking, `diffusion_steps_eval`); DAgger collection uses greedy argmax (`diffusion_steps_collect`).
- **`remdm_sample`** guarantees a fully committed output via a final greedy cleanup of any remaining masked positions (same safety net as the craftax twin).
- **EMA** updates after every gradient step, not per iteration. `DataCollector` syncs EMA weights before each rollout.
- **Curriculum** starts from a 50/50 prior per environment and buckets the rolling win-rate: `[0, 0.15)` -> 0.2, `[0.15, 0.85)` -> 1.0, `[0.85, 1.0]` -> 0.1.
- **Replay buffer** pins offline data at the front; only online samples are FIFO-evicted. Returns `None` when empty.
- **Global gate** starts at `sigmoid(-3.0) ~ 0.047`, nearly closed, so the global stream cannot destabilise early training.
- **DAgger warm-start:** iteration 0 seeds the buffer with 3 oracle trajectories per ID environment (12 total).
- **nhdat patching:** `src/envs/minihack_env.py` substitutes a Python implementation of MiniHack's `mh_patch_nhdat.sh` when the install path contains whitespace, which would otherwise make the script fail silently and yield goalless levels.
