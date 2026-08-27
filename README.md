# ReMDM Planner for MiniHack

PyTorch implementation of **ReMDM** (Remasking Discrete Diffusion Model) for action-sequence planning in [MiniHack](https://github.com/facebookresearch/minihack) navigation environments. A dual-stream transformer generates 64-step action plans by iteratively denoising masked token sequences, conditioned on a 9x9 local crop and the full 21x79 dungeon map. Trained with **DAgger** under BFS oracle supervision, from scratch; generalises zero-shot from 4 in-distribution to 3 out-of-distribution environments.

The sibling repository [`craftax-ReMDM-planner`](../craftax-ReMDM-planner) implements the same method in JAX on Craftax. Both repos share the same CLI, config layout and README structure; commands transfer between them by swapping the repo name and benchmark-specific values.

## Method

The planner starts from a fully-masked action sequence and iteratively unmasks tokens over `K` denoising steps via the ReMDM Algorithm 1 posterior (per-token Bernoulli unmasking), while ReMDM remasking lets committed tokens be re-predicted for plan refinement.

Two independent training pipelines are compared head-to-head, both supervised by the built-in BFS oracle: `--mode online` runs DAgger from scratch (primary), `--mode offline` behaviour-clones a pre-collected oracle dataset. Either output is scored with `--mode inference`. See [Architecture](#architecture) and [Diffusion](#diffusion) for details.

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
modern driver — plain `uv sync` already yields a CUDA 13.0 build on Linux; use `cuda12` only
if `nvidia-smi` reports a driver older than 580.

> **Install path must not contain spaces.** MiniHack's `mh_patch_nhdat.sh` interpolates paths unquoted and fails silently on whitespace, leaving every environment as the same default level with no goal staircase. `src/envs/minihack_env.py` detects this and substitutes a Python implementation, but a space-free path avoids it entirely.

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
├── results/                Gitignored, created on demand — inference/ eval JSONs and
│                           paper_figures/ manuscript PDFs, both published (see Checkpoints)
├── demo_minihack.ipynb     Demo notebook
├── main.py                 CLI entry point
├── RUNS.md                 Measurement runs and what they found
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
python main.py --mode online --config configs/final_minihack_gpu_24gb.yaml      
python main.py --mode online --override total_timesteps=1000000 --override dagger_lr=0.0001
python main.py --mode online --checkpoint checkpoints/iter600.pth       # resume
python main.py --mode online --checkpoint checkpoints/iter600.pth --no-warm-start
```

Per iteration: curriculum-sampled model rollouts, BFS oracle labelling on the same seeds, efficiency filtering into the replay buffer, then `grad_steps_per_iteration` gradient steps — see [DAgger training loop](#dagger-training-loop).

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

Gradient steps default to `total_timesteps // offline_batch_size`; ID + OOD eval runs on the `id_eval_every_timesteps` / `ood_eval_every_timesteps` cadence. The `offline_*_grad_steps` keys override that in grad-step units — see the hazard note under [Configuration](#configuration).

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

`--checkpoint` accepts a local `.pth` path or a `wandb:` artifact reference (`wandb:entity/project/name:version`). Inference uses EMA weights unless `--no-ema` is given.

Write eval JSONs into `results/inference/` (created for you): `scripts/hf_upload.py` publishes every JSON it finds there.

**Match the config to the checkpoint.** The model is built from the config, not the checkpoint, and a mismatch raises at load. Always evaluate with the checkpoint's own config snapshot:

```bash
DIR=checkpoints/online/Minihack-Online-Diffusion-DAgger-100M
python main.py --mode inference --config $DIR/config.yaml --checkpoint $DIR/iter563.pth
```

## Baselines and ablations

### RL and imitation baselines

Six algorithms: SB3 discrete-action RL (`ppo`, `a2c`, `dqn`, `ppo-rnn`), Behavioural Cloning (`bc`) on oracle demos, and a causal Decision Transformer (`dt`). All share `total_timesteps`, so numbers are comparable to DAgger and offline BC. Hyperparameters live under the `baselines_*` config namespace; outputs go to `baselines_output_dir`.

```bash
python main.py --mode baselines --algo ppo                        # any of the six
python main.py --mode baselines --algo dqn --seeds 0 1 2          # explicit seeds
python main.py --mode baselines --algo bc --num-seeds 3           # or a seed count
python main.py --mode baselines --algo ppo --output results/ppo.json
```

### Architecture ablations

```bash
# Local-only planner (no global stream, no goal head), trained from scratch
python main.py --mode online --config configs/ablation_local_only.yaml

# Blind-global: zero the global observation of a trained dual-stream model at eval
python main.py --mode inference --checkpoint checkpoints/iter600.pth --blind-global
```

### RL fine-tuning ablation suite

26 registered ablations (same names as in the craftax repo). See `experiments/README.md`.

```bash
python experiments/rl_finetuning/run_ablations.py --list
python experiments/rl_finetuning/run_ablations.py --checkpoint path/to/ckpt.pth --all
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint wandb:entity/project/checkpoint-iter600:latest \
    --ablations baseline_rl kl_penalty --fast
```

Pass `--emit-tex-macros` to also write `tables/results.tex`, one `\newcommand` per
headline quantity, so the manuscript cites generated numbers instead of retyping them.
Macros from this repository are prefixed `mh` and the sibling suite's `rw`, so both
files can be `\input` together; the name-mangling rule is shared between the two.

`--measure-gdelta` splits the return-weighted ELBO gradient into its imitation and return
terms at one parameter point. No training, no GPU needed; results land in `gdelta/` beside
the run's `results.json` — see `experiments/README.md`.

```bash
python experiments/rl_finetuning/run_ablations.py --measure-gdelta --gdelta-seeds 0 1 2 \
    --checkpoint path/to/ckpt.pth \
    --results-path outputs/minihack_ablations/results.json \
    --output-dir outputs/minihack_ablations
```

## Configuration

One YAML config holds the experiment; the CLI holds the run.

Precedence, lowest to highest: `configs/defaults.yaml` < `--config` preset < `--override` and run flags. Exactly two config layers — a preset never inherits from another preset.

- **`configs/defaults.yaml`**: the **shared final paper recipe**, not a cheap baseline. Both machines train exactly this; running with no `--config` trains it too.
- **Config files** (`configs/*.yaml`): deep-merged onto `defaults.yaml`, so presets contain **only their deltas** — never re-state a default value.
- **Run flags**: `--seed`, `--checkpoint`, `--data`, `--output`, `--episodes`, `--envs`, mode switches.
- **`--override KEY=VALUE`** (repeatable): keys are validated against `defaults.yaml` and cast to the key's type, so a typo is an error, not a silent no-op.

> **Hazard when writing a preset.** Four keys silently *override* an env-step-derived value when non-null, and `defaults.yaml` now sets all four as part of the recipe: `offline_total_grad_steps`, `offline_eval_every_grad_steps`, `offline_checkpoint_every_grad_steps`, `offline_buffer_capacity`. A preset that wants its own `total_timesteps` to govern the offline budget must pin them back to **explicit `null`** — omitting them inherits the pins. Left unpinned, `smoke.yaml` would train 60,000 offline gradient steps instead of 19. `tests/test_config.py` enforces the pins for every preset that derives its own budget.

| Preset | Purpose |
|---|---|
| `configs/defaults.yaml` | **Shared final paper recipe** — the full run both clusters train |
| `configs/smoke.yaml` | Smoke test (`total_timesteps=5000`, small buffer, W&B off) |
| `configs/ablation_local_only.yaml` | Local-only planner ablation (`use_global_stream: false`) |
| `configs/gpu_24gb_bigger_model.yaml` | GPU-24GB, larger model (384D, 6 heads) |
| `configs/gpu_24gb_learning_behaviour.yaml` | GPU-24GB learning-behaviour study (eta=0.18, B=6144) |
| `configs/final_minihack_gpu_h200.yaml` | **Paper run, H200.** Machine values only: worker counts (32) and dataset path |
| `configs/final_minihack_gpu_24gb.yaml` | **Paper run, RTX 3090 Ti.** Machine values only: dataset path (workers stay at the default 8) |

Key hyperparameters are documented inline in `configs/defaults.yaml`; the [appendix](#key-hyperparameters) tabulates them.

## Checkpoints

Training writes to a unique run directory under `checkpoint_dir` (default `checkpoints/`), named `{tag}_{YYYYMMDD}_{HHMMSS}_{hex4}`. DAgger saves `iter{N}.pth` on the `checkpoint_every_timesteps` cadence; offline BC saves `offline_step{N}.pth` and `offline_final.pth`. With `use_wandb` on they also upload as versioned W&B artifacts (type `model`). Every checkpoint stores its W&B run ID, so passing it back via `--checkpoint` resumes the same curve.

`checkpoints/` is gitignored. Released weights live on the Hugging Face Hub: **[mathisweil/remdm-minihack-checkpoints](https://huggingface.co/mathisweil/remdm-minihack-checkpoints)**

| Directory | Method | Selected at | Sample-equivalents |
|---|---|---|---|
| `checkpoints/online/Minihack-Online-Diffusion-DAgger-100M` | DAgger (main result) | `iter563` | 100M |
| `checkpoints/offline/Minihack-Offline-Diffusion-BC-100M` | Offline BC baseline | `offline_step50000` | 100M |

```bash
# All checkpoints
uv run hf download mathisweil/remdm-minihack-checkpoints --include "checkpoints/**" --local-dir .

# One checkpoint
uv run hf download mathisweil/remdm-minihack-checkpoints \
    --include "checkpoints/online/Minihack-*/**" --local-dir .
```

**Keep the `--include`.** The Hub repo carries its own `README.md` (the generated model card), `LICENSE` and `.gitattributes`; dropping the glob and pulling into `--local-dir .` overwrites this repository's copies of all three. To fetch everything, add `--exclude "README.md" "LICENSE" ".gitattributes"`, or use a separate `--local-dir`. Publishing is safe either way — `hf_upload.py` stages `LICENSE` and the demo `README.md` from git, not the working tree.

Each released directory ships `<step>.pth` (full training state), `model.safetensors` (EMA weights only, no pickle), `config.yaml` and `selection.json`. The `-100M` suffix counts **sample-equivalents, not env steps** — the runs behind these train 5,650,000 env steps. See [Checkpoint format](#checkpoint-format) for the `.pth` schema and programmatic loading.

Historical note: the released DAgger `selection.json` records `"every": null, "configured_max": null` and `"unit": "dagger_iterations"`, written by a `selection()` that read two since-renamed config keys. It is **historical and noncanonical** and stays as published (author decision 2026-08-17); the checkpoint's own `config_<step>.yaml` carries the real cadence and budget. Current code records the candidate set in env steps — `"every": 940000, "configured_max": 5650000` for the shipped recipe — and raises rather than writing a null.

### Publishing to the Hub

`scripts/hf_upload.py` rediscovers and uploads four things, each keeping its repo-relative path: `checkpoints/` (adding a `model.safetensors` EMA export and `selection.json` per directory), every `experiments/rl_finetuning/outputs/<run>/` holding a `results.json` (with `diagnosis.md`, `tables/`, `figures/`, `gdelta/`), the eval JSONs in `results/inference/`, and the manuscript figure PDFs in `results/paper_figures/`. It drops W&B and hub config keys, shortens absolute paths and regenerates the model card.

```bash
HF_TOKEN=hf_xxx uv run python scripts/hf_upload.py --repo-id mathisweil/remdm-minihack-checkpoints \
    --selection-metric "mean ID+OOD win rate" --dry-run
```

`--dry-run` prints the staged tree and card without uploading; drop it to upload. `--selection-metric` records what the best-of-N checkpoints were chosen on. Also `--inference-results <FILE|DIR> ...` (eval JSONs kept elsewhere), `--private`, `--yes`. Publish one model per directory, with a single `.pth` and config.

**The manuscript figures are built by the sibling repo.** Each one puts Craftax Classic and MiniHack side by side, so `../craftax-ReMDM-planner/scripts/paper_figures.py` reads *both* repositories' ablation `results.json` and neither can build them alone. Copy the PDFs it emits into `results/paper_figures/` here; both Hub repos publish the same set, and the upload warns when they are absent rather than passing over them silently.

**A `hf download --local-dir .` overwrites `README.md` and `LICENSE` in the working tree.** Publishing is unaffected — `hf_upload.py` stages `LICENSE` from `git cat-file blob HEAD:LICENSE`, and `hf_upload_demo.py` its bundle's `README.md`, warning if git cannot be consulted — but restore your own files with `git checkout -- README.md LICENSE`, or avoid the clobber with the download flags above.

**Checkpoint discovery expects the released layout**, `checkpoints/<role>/<name>/*.pth`. A training run writes to its own `checkpoints/dagger_<timestamp>/`, so copy what you mean to release into `checkpoints/{offline,online}/<name>/` first, or nothing is staged. `checkpoints/hf/` is skipped — that is where a Hub *download* lands, and publishing from it would nest already-published artefacts under `checkpoints/hf/checkpoints/...`.

## Results, citation, licence

Results tables and the full method description are in *Return-Weighted ELBO Fine-Tuning Degrades Masked Diffusion Planners* (under submission); `demo_minihack.ipynb` reproduces the headline comparison. Citation to be added on publication. Licence: MIT, see `LICENSE`.

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

`LocalDiffusionPlanner` (no global stream, no goal head) is the `ablation_local_only` variant. Supported by `--mode offline` and `--mode online`; the `experiments/` ablation suite assumes the goal head is present.

## Diffusion

- **Forward process (MDLM):** each action token is independently replaced with `MASK` (12) with probability `1 - alpha(t)`, `alpha(t)` linear or cosine. PAD (13) is never masked.
- **Loss:** continuous-time MDLM NELBO: per sample `w(t) * sum_masked(CE) / L` with `w(t) = -alpha'(t) / (1 - alpha(t))` clipped to `[0, 1000]`; optional `label_smoothing`.
- **Greedy sampling:** used for DAgger collection. Same MaskGIT loop, argmax decoding, no temperature/top-K/remasking, `diffusion_steps_collect` steps.

**Reverse sampling (ReMDM Algorithm 1)**, over `K` steps (default 10). Per step: predict logits, apply temperature and top-p filtering, sample, and record each committed token's decode probability `psi`; **unmask** each masked position independently with posterior probability `(alpha_s - (1 - sigma) alpha_t) / (1 - alpha_t)`; **remask** each committed position with probability `sigma` from the configured Section-4.1 schedule. A final greedy cleanup commits anything still masked.

| Strategy | Formula | Description |
|---|---|---|
| `rescale` | `p = eta * sigma_max` | Proportional to noise level |
| `cap` | `p = min(eta, sigma_max)` | Fixed upper bound |
| `conf` | `p = softmax(-confidence) * eta * sigma_max` | Low-confidence tokens remasked more |

## Key hyperparameters

`configs/defaults.yaml` is authoritative and commented inline. Tabulated here are the
keys that change a result, carry a hazard, or are named elsewhere in this README.

**Model.** `n_embd` 256, `n_head` 4, `n_layer` 4, `n_global_tokens` 8, `seq_len` 64,
`dropout` 0.0, `global_gate_init` -3.0 — the shape every released checkpoint carries
(see [Architecture](#architecture)); a checkpoint restores only against a matching
config. Two model keys are result-affecting in their own right:

| Parameter | Default | Description |
|---|---|---|
| `ema_decay` | 0.999 | EMA smoothing for inference weights; an absolute update count, see the budget caveat below |
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
| `replan_every` | 16 | Env steps before replanning; the actions already executed in the current plan window are locked into the new plan (inpainting) |
| `loss_weight_clip` | 1000.0 | NELBO weight clip bound |
| `label_smoothing` | 0.0 | Cross-entropy label smoothing (0 = exact ELBO) |

**Training budget (unified).** Offline BC, DAgger and the SB3 baselines share one env-step budget. This is the only knob that should change to scale a run.

| Parameter | Default | Description |
|---|---|---|
| `total_timesteps` | 5,650,000 | Shared env-step budget |
| `id_eval_every_timesteps` | 470,000 | ID eval cadence |
| `ood_eval_every_timesteps` | 470,000 | OOD eval cadence |
| `checkpoint_every_timesteps` | 940,000 | Checkpoint cadence |

- **Offline BC:** gradient steps = `total_timesteps // offline_batch_size`, and the cosine LR `T_max` derives from the same quantity, so any run length decays to the 10% floor at its end.
- **DAgger:** tracks cumulative `env.step()` calls (model + oracle) and halts at `total_timesteps`. `episodes_per_iteration` and `grad_steps_per_iteration` set the collect/train ratio and **must not** scale with the budget.
- **Caveat, `ema_decay`:** an absolute-update-count constant (half-life ~ `1 / (1 - decay)` steps). Shifting `total_timesteps` by more than ~2x changes the fraction of training the EMA window covers; set a matching decay by hand for very short or long runs.

**Offline grad-step pins.** These four override the env-step-derived budget whenever
non-null, and `defaults.yaml` sets all four. A preset whose own `total_timesteps` should
govern must pin them back to explicit `null` — see the hazard note under
[Configuration](#configuration).

| Parameter | Default | Description |
|---|---|---|
| `offline_total_grad_steps` | 60000 | Total gradient steps, overriding `total_timesteps // offline_batch_size` |
| `offline_eval_every_grad_steps` | 5000 | Eval cadence in grad steps |
| `offline_checkpoint_every_grad_steps` | 10000 | Checkpoint cadence in grad steps |
| `offline_buffer_capacity` | 1500000 | Offline replay capacity |

**Training**

| Parameter | Default | Description |
|---|---|---|
| `offline_lr` / `dagger_lr` | 0.0003 / 0.00003 | BC LR (cosine-decayed to 10%) and DAgger LR (constant) |
| `offline_batch_size` / `dagger_batch_size` | 2048 / 2048 | Batch size per pipeline |
| `offline_grad_clip` / `dagger_grad_clip` | 1.0 / 1.0 | Gradient norm clip per pipeline |
| `weight_decay` | 0.0 | AdamW weight decay (core training; the ablation suite keeps 1e-4) |
| `grad_steps_per_iteration` | 100 | Gradient steps per DAgger iteration |
| `episodes_per_iteration` | 30 | Episodes per DAgger iteration |
| `aux_loss_weight` | 0.5 | Auxiliary goal loss weight |
| `buffer_capacity` | 10000 | Replay buffer size (windows) |
| `efficiency_multiplier` | 1.5 | DAgger efficiency filter threshold |

**Collection, evaluation, performance**

| Parameter | Default | Description |
|---|---|---|
| `collect_episodes_per_env` | 5000 | Oracle episodes per ID environment |
| `eval_episodes_per_env` | 50 | Episodes per env at eval (per-run: `--episodes`) |
| `use_amp` | true | Mixed precision via `torch.amp`; see [Performance tuning](#performance-tuning) |
| `torch_compile` | true | `torch.compile` the model |
| `checkpoint_dir` | `checkpoints` | Root for per-run checkpoint directories |
| `seed` | null | RNG seed (null = random; per-run: `--seed`) |

Worker counts (`collect_num_workers`, `num_collection_workers`, both 8) are machine values.
The `collect_output`, `use_wandb`, `wandb_*` and `offline_log_every` keys mirror the run
flags under [Configuration](#configuration), the `curriculum_*` keys the behaviour under
[DAgger training loop](#dagger-training-loop), and the 21 `baselines_*` keys hold the
SB3/BC/DT hyperparameters; all are commented where they are declared.

## DAgger training loop

1. **Curriculum sampling:** pick an environment weighted by difficulty (low win-rate sampled more).
2. **Model rollout:** EMA model, greedy sampling, replanning every 16 steps with the executed prefix locked, `episodes_per_iteration` episodes.
3. **Oracle rollout:** BFS oracle on the **same seed**.
4. **Efficiency filter:** add the oracle trajectory if the model failed or took >1.5x the oracle's steps.
5. **Budget accounting:** `env_steps_total += model_steps + oracle_steps`; halt at `total_timesteps`.
6. **Training:** sample the buffer, run `grad_steps_per_iteration` steps, update EMA after each.

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
# DAgger (iter{N}.pth)
{
    "model_state_dict": ..., "ema_state_dict": ...,
    "optimizer_state_dict": ..., "scheduler_state_dict": ...,
    "curriculum_state": {...},
    "iteration": int,
    "env_steps": int,                  # cumulative env.step() calls
    "wandb_run_id": str | None,
    "rng_states": {"torch", "numpy", "python"},
}
```

Offline BC step-level (`offline_step{N}.pth`, when `checkpoint_every_timesteps > 0`) is the
same minus `curriculum_state`, with `step` for `iteration` and `env_steps = step *
offline_batch_size`. Its `rng_states` is **required**: resume raises without it. The final
`offline_final.pth` carries only `model_state_dict`, `ema_state_dict` and `wandb_run_id`.

### Load programmatically

```python
# Inference, from safetensors (already EMA weights)
from safetensors.torch import load_file
from src.config import load_config
from src.models.denoiser import make_model

DIR = "checkpoints/online/Minihack-Online-Diffusion-DAgger-100M"
cfg = load_config(f"{DIR}/config.yaml")
model = make_model(cfg)
model.load_state_dict(load_file(f"{DIR}/model.safetensors"))
model.eval()
```

From the full `.pth` instead, to resume or to pick training over EMA weights: `torch.load(..., weights_only=False)`, then `model.load_state_dict(ckpt["model_state_dict"])` and, for what evaluation uses by default, `ModelEMA(model, decay=cfg.ema_decay)` with `load_state_dict(ckpt["ema_state_dict"])` and `apply_to(model)`.

### W&B artifacts and run resumption

W&B model artifacts contain the `.pth` and a `config.yaml` snapshot; the reference format is `wandb:entity/project/artifact-name:version`, version being `latest`, `v0`, `v1`. Resuming reads the run ID out of the checkpoint and passes it to `wandb.init(resume="must")`, so curves continue with no gaps.

```bash
# Automatic. A checkpoint predating the feature needs the ID passing by hand:
python main.py --mode online --checkpoint old.pth --override wandb_resume_id=abc123xyz
```

## W&B metric namespaces

Declared in `src/planners/logging.py`; the key lists there are authoritative.

| Namespace | Contents |
|---|---|
| `diffusion/` | `loss`, `loss_diff`, `loss_aux` |
| `train/` | Buffer, collection and optimiser state — 13 keys including `model_steps`, `oracle_steps`, `efficiency_ratio`, `global_gate`, `env_steps` |
| `speed/` | Per-iteration timings, throughput and `gpu_memory_mb` |
| `model/` | `param_norm`, `param_drift_from_init`, `ema_gate_value` (every 10 iters) |
| `eval_id/{env}/`, `eval_ood/{env}/` | Per-env `win_rate`, `wins`, `avg_reward`, `avg_steps`, `n_episodes` |
| `eval_id/`, `eval_ood/` | `mean_win_rate` |
| `ckpt_eval_id/`, `ckpt_eval_ood/`, `ckpt_eval/` | The same, at checkpoint time |
| `curriculum/{env}/` | `win_rate` per training environment |
| `offline/` | `final_loss`, `total_steps`, `total_timesteps` (summary only) |
| `inference/{env}/` | Per-env metrics from `--mode inference` |

DAgger and offline BC both emit to `eval_id/` and `eval_ood/`, through the same `Evaluator` and EMA-weight path.

## Performance tuning

`use_amp` (default true) puts `torch.amp.autocast("cuda")` + `GradScaler` in both trainers —
roughly 2x on gradient steps, with loss and win rates statistically equivalent to FP32, and a
no-op on CPU. `torch_compile` (default true) shows no measured gain beyond AMP.
`num_collection_workers` affects only the threaded CPU fallback: collection auto-selects
GPU-batched (CUDA, `episodes_per_iteration > 1`) > threaded CPU > sequential.

Profile with `python scripts/profile_dagger.py [--override key=value ...]`.

## Testing

```bash
uv run pytest            # the default suite
uv run pytest -m slow    # slow entry points only (BC + PPO baselines)
```

A CPU-only suite, 17 modules. Tiny synthetic data and a shrunken model throughout — no real checkpoints, datasets or network calls, and nothing written outside `tmp_path`. `conftest.py` forces CPU and disables W&B; `slow` marks the multi-second CLI smokes and is deselected by default. For a quality signal, use `--mode smoke`.

| File | Covers |
|---|---|
| `test_smoke_src.py`, `test_smoke_experiments.py` | that things **run**: imports, model from the real config, a forward pass of the expected shape and dtype with no NaNs, a finite training step, save/reload identity, every CLI entry point, and all 26 registry ablations |
| `test_spec_*.py`, `test_method_spec*.py` | that things are **correct**: each canonical statement of the parent workspace's `research/spec-*.md` pinned against the implementation |
| `test_config.py`, `test_recipe_values.py` | the preset, delta-only and poolability rules, and the shipped recipe values |
| `test_gdelta.py`, `test_tex_macros.py` | the `--measure-gdelta` decomposition, and the `--emit-tex-macros` output: definitions only, uniquely named, letters only |
| `test_ablation_perf.py`, `test_gpu_step_perf.py` | measured throughput expectations |
| `test_env_reuse.py`, `test_failure_behaviour.py` | MiniHack env pooling, and failures that must raise rather than be swallowed |
| `test_gpu_agreement.py` | CPU/GPU agreement, skipped without a device |

## Implementation notes

- **MDLM loss** returns `0.0` (not NaN) when no masked positions exist. NELBO-weighted per MDLM eq (10).
- **PAD tokens** are never masked and are excluded from the loss.
- **EMA** updates after every gradient step, not per iteration. `DataCollector` syncs EMA weights before each rollout.
- **Curriculum** starts from a 50/50 prior per environment and buckets the rolling win-rate: `[0, 0.15)` -> 0.2, `[0.15, 0.85)` -> 1.0, `[0.85, 1.0]` -> 0.1.
- **Replay buffer** pins offline data at the front; only online samples are FIFO-evicted. Returns `None` when empty.
- **Global gate** starts at `sigmoid(-3.0) ~ 0.047`, nearly closed, so the global stream cannot destabilise early training.
- **DAgger warm-start:** iteration 0 seeds the buffer with 3 oracle trajectories per ID environment (12 total).
