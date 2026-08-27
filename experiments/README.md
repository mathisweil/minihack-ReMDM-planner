# ReMDM Experiments

Research and diagnostic scripts for investigating RL fine-tuning of the ReMDM
diffusion planner. These scripts are **standalone research code** -- they
import from `src/` (model, sampling, env wrapper, evaluator) but never modify
the core training pipeline. They start from a pretrained DAgger checkpoint.

---

## `rl_finetuning/` -- RL Fine-Tuning Ablation Suite

Diagnoses why RL fine-tuning of the diffusion model collapses and which interventions fix it.
Implements **25 ablations**: a baseline plus four groups (A: Regularisation, B: Training Signal, C: Architecture, D: Data Quality), with a comprehensive diagnostic and analysis pipeline.

**Training data is on-policy, and returns are per window.** Each iteration rolls the *current* model out under its EMA weights and trains on the resulting windows; a window's return is the reward sum over exactly the actions it trains on, not the episode total broadcast to every window.

### Directory structure

```
rl_finetuning/
├── run_ablations.py          # CLI entry point
├── ablations/
│   ├── losses.py             # 15 loss/objective factory functions + LossContext
│   ├── optimizers.py         # AdamW, LLRD, LoRA, frozen params, PCGrad helpers
│   ├── registry.py           # AblationSpec dataclass + REGISTRY (25 ablations)
│   └── training.py           # run_ablation() loop, MixedReplayBuffer, RewardModel,
│                             #   AblationHistory dataclass
├── diagnostics/
│   ├── gradient.py           # Grad alignment, per-layer norms, surgery metrics
│   ├── representation.py     # KL drift, CKA similarity, activation norms
│   └── timestep.py           # t-bin gradient norms, per-t loss decomposition
├── analysis/
│   ├── gdelta.py             # Return term g_delta of the decomposition (no training)
│   ├── plots.py              # 12 matplotlib figure generators
│   ├── tables.py             # Summary tables as polars DataFrames + LaTeX export
│   ├── report.py             # diagnosis.md + decision tree figure
│   └── action_distribution.py  # Pre/post-RL action distribution analysis
└── configs/
    ├── ablations_default.yaml   # Base: all ablation hyperparameters
    ├── ablations_fast.yaml      # Smoke-test overlay (50 iterations)
    ├── ablations_final_minihack_gpu_h200.yaml  # H200 overrides only
    └── ablations_final_minihack_gpu_24gb.yaml   # RTX 3090 Ti overrides only (reference)
```

### Config layering

Two layers, always: `ablations_default.yaml` carries every ablation
hyperparameter, a machine config carries only what that machine changes, and
configs never inherit from one another.

Merge order, later wins:

```
configs/defaults.yaml         # architecture, env IDs, token IDs
  -> ablations_default.yaml     # base: all ablation hyperparameters
  -> --ablations-config FILE    # machine overrides only
  -> ablations_fast.yaml        # --fast only, raw overlay
  -> CLI flags                  # --max-iter --batch-size --eval-every --lr --seed
```

Machine configs carry only the keys they change:

```yaml
# ablations_final_minihack_gpu_24gb.yaml
batch_size: 4608
cka_batch_size: 128
```

| Rule | Behaviour |
|---|---|
| Base | `ablations_default.yaml`, applied automatically; there is no `extends` key |
| `--fast` | Applied raw on top, so machine keys (`use_amp`, `batch_size`, `diffusion_steps_collect`) survive |
| Unknown key | `KeyError`, not a silent no-op. Valid keys are those in `configs/defaults.yaml` plus `ablations_default.yaml` |
| Restated default | Rejected by `tests/test_config.py`: a key whose value equals what it would inherit is redundant and must be deleted |

Key validation matters here because every ablation reads config through
`getattr(cfg, key, fallback)`, so an unrejected typo such as `batch_sze: 512`
leaves the real `batch_size` at its inherited value with no error.

A bare `--fast` run (no `--ablations-config`) inherits `num_seeds: 3` and AMP
from the base. Pass `--num-seeds 1` for a single-seed smoke run.

### Usage

**List all ablations:**
```bash
python experiments/rl_finetuning/run_ablations.py --list
```

**Smoke test (2 ablations, fast config):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint path/to/dagger_checkpoint.pth \
    --ablations baseline_rl kl_penalty \
    --fast
```

**Using a W&B artifact as checkpoint:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint wandb:entity/project/checkpoint-iter8000:latest \
    --ablations baseline_rl kl_penalty \
    --fast
```

**Full suite (all 25 ablations):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint path/to/dagger_checkpoint.pth \
    --all \
    --num-seeds 3 \
    --use-wandb
```

**Full suite on a specific machine:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint path/to/dagger_checkpoint.pth \
    --ablations-config experiments/rl_finetuning/configs/ablations_final_minihack_gpu_24gb.yaml \
    --all \
    --use-wandb
```

**Specific ablations:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint path/to/dagger_checkpoint.pth \
    --ablations ewc lora gradient_surgery trust_region_kl
```

**Re-plot from saved results (no training):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --analyze-only --output-dir outputs/run_20260331_120000
```

**Re-plot a subset of ablations:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --analyze-only --output-dir outputs/run_20260331_120000 \
    --ablations baseline_rl kl_penalty ewc
```

**Spread across GPUs, then merge:**

Run independent subsets on different machines or GPUs, then combine:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python experiments/rl_finetuning/run_ablations.py \
    --checkpoint ckpt.pth \
    --ablations baseline_rl kl_penalty ewc llrd lora mixed_replay \
    --output-dir outputs/gpu0

# GPU 1
CUDA_VISIBLE_DEVICES=1 python experiments/rl_finetuning/run_ablations.py \
    --checkpoint ckpt.pth \
    --ablations trust_region_kl low_t t_curriculum entropy_bonus \
    --output-dir outputs/gpu1

# Merge and regenerate all analysis
python experiments/rl_finetuning/run_ablations.py \
    --merge outputs/gpu0/results.json outputs/gpu1/results.json \
    --output-dir outputs/combined
```

`--merge` accepts any number of `results.json` files. Where the same ablation
appears in more than one, the per-seed scores are concatenated and mean/std
recomputed over the union:

```bash
# Seed 0 on GPU 0
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint ckpt.pth --ablations baseline_rl --seed 0 \
    --output-dir outputs/seed0

# Seed 1000 on GPU 1
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint ckpt.pth --ablations baseline_rl --seed 1000 \
    --output-dir outputs/seed1

# Merge: aggregates both seeds into one entry
python experiments/rl_finetuning/run_ablations.py \
    --merge outputs/seed0/results.json outputs/seed1/results.json \
    --output-dir outputs/merged
# -> <ablation>: <mean> +/- <std> (2 seeds)
```

The merged file is a `results.json` like any other, so `--analyze-only` works on it.

**`--merge` only pools runs from configs that agree on result-affecting keys.**
It compares the configs the results files recorded and refuses, naming every
diverging key with both values; a file that records no config is refused too.
All published
MiniHack ablation results were produced on the **RTX 3090 Ti**
(`ablations_final_minihack_gpu_24gb.yaml`), which is the reference config.

`ablations_final_minihack_gpu_h200.yaml` is **not poolable** with it. It diverges on four
keys that change the result, not just the wall-clock:

| Key | GPU-24GB | GPU-H200 | Effect |
|---|---|---|---|
| `batch_size` | 4608 | 512 | ~9x per-update SNR |
| `episodes_per_iter` | 30 | 20 | 15k vs 10k total episodes |
| `diffusion_steps_collect` | 5 | 3 | different collection policy |
| `eval_episodes` | 20 | 10 | noisier score |

Differences in diagnostic cadence (`eval_every`, `cka_every`, `cka_batch_size`,
`per_layer_every`, `repr_drift_every`, `grad_align_every`, `t_analysis_every`)
are wall-clock only and do not affect poolability.

`tests/test_config.py` enforces this: every `ablations_final_*.yaml` must be
declared poolable or not, configs declared poolable must match the reference on
the result-affecting keys, and the recorded GPU-H200 divergence must stay accurate.
Aligning GPU-H200 later fails the test until it is moved to the poolable set. The
key set itself is declared once, in `run_ablations._RESULT_AFFECTING`, so the
classification the tests check and the refusal `--merge` performs are the same
policy.

### Ablations

| Group | Name | Tests |
|---|---|---|
| Baseline | `baseline_rl` | Standard return-weighted ELBO |
| **A: Regularisation** | `kl_penalty` | Soft KL constraint vs pretrained |
| | `ewc` | Elastic Weight Consolidation (Fisher diagonal) |
| | `llrd` | Layer-wise Learning Rate Decay |
| | `lora` | Low-Rank Adaptation of attention projections |
| | `mixed_replay` | Self-replay: the run's own past online windows resampled into each batch |
| | `trust_region_kl` | Hard KL trust region via quadratic barrier |
| **B: Training Signal** | `t_curriculum` | Anneal t range high-to-low over training |
| | `entropy_bonus` | Entropy regularisation for action diversity |
| | `gradient_surgery` | PCGrad: project conflicting RL/BC gradients |
| | `advantage_clip` | PPO-style advantage clipping [1-eps, 1+eps] |
| | `normalized_adv` | Std-normalised advantages (per-minibatch) |
| | `bc_wins` | Uniform ELBO on win windows (no advantage weighting) |
| | `low_t` | ELBO restricted to low-t (fine-detail) regime |
| **C: Architecture** | `frozen_backbone` | Train the action head + token embeddings (backbone frozen) |
| | `head_only` | Train only the final action projection |
| | `attention_only` | Train only the attention projections (Q/K/V/O) |
| | `ffn_only` | Train only the per-block FFN layers |
| | `layer_ablation_top1` | Train only the top-1 transformer block + head |
| | `layer_ablation_top2` | Train only the top-2 transformer blocks + head |
| | `layer_ablation_top3` | Train only the top-3 transformer blocks + head |
| **D: Data Quality** | `reward_filtering` | Top-75th-percentile return windows only |
| | `running_stats` | EMA running mean/std for advantage normalisation |
| | `action_diversity` | Discard degenerate (all-same-action) plans |
| | `reward_model` | MLP reward model soft-weighting of advantages |

Group C freezes parameters exactly: under an adversarial gradient on every
tensor, each frozen tensor of each arm measures a parameter delta of exactly
0.0 (`tests/test_spec_ablations.py`, at the production architecture). The
weights the suite *evaluates* are the EMA shadow, which drifts even when its
parameter does not, because `decay * x + (1 - decay) * x` is not exactly `x` in
float32: **45 of 72 tensors, by up to 4.6e-05 over 500 updates at decay 0.999**.
A group-C arm's frozen parameters are therefore bit-exact in the trained weights
and approximate to that magnitude in the evaluated ones — far below the
resolution of any reported win rate, and left as it is (`ModelEMA` in
`src/models/denoiser.py`).

### Output structure

```
experiments/rl_finetuning/outputs/{run_id}/
├── results.json                       # All histories + final scores (machine-readable)
├── diagnosis.md                       # Human-readable verdict + evidence + recommendations
├── checkpoint_{name}.pth              # Per-ablation fine-tuned model state dict
├── figures/
│   ├── curves_{name}.png              # Per-ablation 2x3 curves (eval, loss, env score,
│   │                                  #   KL drift, grad alignment, grad norms)
│   ├── per_layer_grad_heatmap_{name}.png  # Per-layer gradient norms over training
│   ├── t_bin_grad_norms_{name}.png    # Per-t-bin gradient norms over training
│   ├── per_env_collapse_{name}.png    # Per-env win rate over eval checkpoints
│   ├── final_score_comparison.png     # Bar chart of final scores across ablations
│   ├── eval_scores_over_training.png  # All ablation eval curves overlaid
│   ├── score_delta_over_baseline_rl.png  # Sorted bar chart of improvement over baseline_rl
│   ├── gradient_alignment.png         # Gradient cosine similarity over training
│   ├── gradient_conflict_map.png      # Binary heatmap of gradient conflicts (cos_sim < 0)
│   ├── representation_drift.png       # KL divergence drift by t-range
│   ├── cka_similarity.png             # CKA similarity vs pretrained over training
│   ├── t_distribution_analysis.png    # High/low-t norm ratio + low-high cosine alignment
│   ├── t_bin_norms_heatmap.png        # Heatmap of per-t-bin gradient norms (final iter)
│   ├── win_rate_and_effective_batch_size.png  # Online win rate + effective batch size
│   ├── group_comparison.png           # Boxplot of scores by ablation group
│   ├── per_env_delta.png              # Heatmap of per-env win rate change (end - start)
│   ├── diagnosis_decision_tree.png    # Hypothesis evidence bar chart
│   └── action_dist/                   # Only with --action-dist (see below)
│       ├── action_dist_comparison_{name}.png   # Pre/post action frequency bars
│       ├── probability_change_{name}.png       # Per-action delta and log-ratio
│       ├── distribution_metrics_{name}.png     # Entropy (nats), effective actions, Gini
│       ├── episode_analysis_{name}.png         # Return and length histograms
│       ├── cumulative_distribution_{name}.png  # Cumulative sorted probability
│       ├── action_transitions_{name}.png       # Pre, post, diff transition matrices
│       ├── action_distribution_results_{name}.json  # Metrics + statistical tests
│       └── js_divergence_comparison.png        # JS divergence across ablations
├── gdelta/                            # --measure-gdelta only
│   ├── gdelta_seed{n}.json            # Per rollout seed; +/- within is across that seed's draws
│   └── gdelta_aggregate.json          # Across seeds; the dispersion the paper's table prints
└── tables/
    ├── results.tex                    # --emit-tex-macros only: \newcommand per headline number
    ├── main_results.{csv,tex}         # Per-condition table: score, seed sd, deltas, verdict
    ├── significance_test.txt          # Max-statistic permutation test + p floor + bootstrap CI
    ├── group_summary.{csv,tex}        # Group-level summary table
    ├── gradient_analysis.{csv,tex}    # Grad alignment (mean/final/trend) + KL drift
    ├── t_distribution.{csv,tex}       # High/low-t ratio, alignment, dominant regime
    ├── repr_drift.{csv,tex}           # KL drift values at final iteration
    ├── per_env.{csv,tex}              # Per-environment win rates
    ├── forgetting_analysis.{csv,tex}  # First collapse iter, min score, recovery
    ├── hypothesis_verdict.{csv,tex}   # Per-ablation hypothesis verdict + conclusion
    └── gdelta.{csv,tex}               # --measure-gdelta only: the decomposition per weight transform
```

**Action distribution analysis** is opt-in via `--action-dist`. It costs roughly
`len(id_envs) * --action-dist-episodes * (1 + n_ablations)` episodes. The
pretrained baseline is rolled out once and reused across ablations. It reads
the per-ablation `checkpoint_{name}.pth` files, so it only covers ablations
whose checkpoint was saved.

### Measuring the return term (`--measure-gdelta`)

Splits the return-weighted ELBO gradient into an imitation term and a return
term at a single parameter point:

```
grad L_RW  =  Abar * ( grad L_BC + g_delta ),
g_delta    =  (1/B) sum_i delta_i grad l_i,    delta_i = A_i/Abar - 1.
```

It loads the pretrained checkpoint, collects one on-policy batch from it, and
evaluates `grad L_BC`, `g_delta` and `grad L_RW` on that batch at those
parameters under a shared `(z_t, t)` draw, so the only difference between the
three is the weight vector. It repeats for the four weighting ablations
(`baseline_rl`, `advantage_clip`, `normalized_adv`, `bc_wins`). No training and
no optimiser step occur; it runs on a laptop CPU in minutes.

Results land in `gdelta/` under the run's own output directory, beside
`results.json`, and the aggregate additionally produces
`tables/gdelta.{csv,tex}`. With `--emit-tex-macros`, the analysis pass picks
the aggregate up and emits the measured quantities as `\mhGdelta*` macros.
Those are kept separate from the `\mhCvA*` macros, which recover `CV_A` from
the ESS logged during training: the two are measured on different batches and
do not agree.

Config comes from `--results-path`, so the weight transforms measured are the
ones that run trained under; without it the standard layering applies.

**Reproduction (three rollout seeds, aggregated in one pass):**
```bash
uv run python experiments/rl_finetuning/run_ablations.py \
    --measure-gdelta --gdelta-seeds 0 1 2 \
    --checkpoint checkpoints/online/<run>/iterNNN.pth \
    --results-path experiments/rl_finetuning/outputs/minihack_ablations/results.json \
    --output-dir experiments/rl_finetuning/outputs/minihack_ablations
```

Seeds run on separate machines are aggregated afterwards with
`--gdelta-inputs`, the counterpart to `--merge`:
```bash
uv run python experiments/rl_finetuning/run_ablations.py --run-id gdelta \
    --gdelta-inputs experiments/rl_finetuning/outputs/gdelta/gdelta_seed{0,1,2}.json
```

A single seed's `ratio_std_draws` / `cos_std_draws` are dispersions over that
seed's eight `(z_t, t)` draws. The aggregate averages the per-seed means and
reports the standard deviation **across seeds**, which is what the paper's
table prints.

Reported per weight transform: `CV_A`, `Abar`, `Abar` relative to the
baseline's, ESS as a fraction of the batch, `|g_delta| / |grad L_BC|`, the
cosine between them, and the same two against a **shuffled-delta null** --
delta permuted across the batch, which preserves `CV_A` and destroys the
association between a window's weight and its own gradient. Anything that
survives the shuffle is batch heterogeneity, not return signal.

Four points on scope:

- **The objective is the ELBO term alone.** The trainer also adds an
  unweighted auxiliary goal loss; including it would break the identity above
  for reasons unrelated to the return, so it is excluded. The goal head
  therefore carries no gradient here.
- **The collection size is `episodes_per_iter`.** MiniHack rollouts are
  sequential rather than vectorised, so the sibling suite's `NUM_ENVS` has no
  counterpart; `measure()` takes the override under the sibling's `num_envs`
  name and applies it to `episodes_per_iter`.
- **`--results-path` layers over `configs/defaults.yaml`.** `run_ablations.py`
  records only scalar keys in `results.json`, so a recorded config carries no
  `id_envs`; defaults supplies the structural keys and the recorded config
  wins everywhere it speaks.
- **`--device` has no sibling counterpart.** JAX picks its backend from the
  environment; torch needs to be told. It defaults to CUDA where available.

The sibling `craftax-ReMDM-planner` carries the same module with the same
driver flags and the same output JSON schema. Its version needs an explicit
Orbax sharding to restore a GPU-written checkpoint on CPU;
`torch.load(map_location=...)` has no such problem, which is the only
structural difference between the two.

**`results.json` schema:**
```json
{
  "pretrained_score": 0.1234,
  "config": {"max_iter": 1000, "batch_size": 512, ...},
  "ablations": {
    "kl_penalty": {
      "score": 0.1456,
      "score_std": 0.008,
      "all_scores": [0.14, 0.15, 0.14],
      "history": { ... }
    }
  }
}
```

`results.json` is written incrementally after each ablation completes -- a partial file
with N of 25 ablations is fully valid and loadable by `--analyze-only` or `--merge`.

### CLI reference

| Flag | Description |
|---|---|
| `--checkpoint PATH` | Pretrained DAgger checkpoint (`.pth` or `wandb:` artifact) |
| `--config PATH` | Main config override (default: `configs/defaults.yaml`) |
| `--ablations-config PATH` | Ablations config, layered on `ablations_default.yaml` (default: `ablations_default.yaml`) |
| `--all` | Run all 25 ablations |
| `--ablations NAME [NAME ...]` | Run specific ablations by name |
| `--list` | Print registered ablations and exit |
| `--fast` | Smoke-test mode (50 iterations, 20 eval episodes) |
| `--num-seeds N` | Number of seeds per ablation (overrides `num_seeds`, default 3) |
| `--seed N` | Base random seed |
| `--output-dir DIR` | Output directory (default: auto-timestamped) |
| `--run-id ID` | Custom run ID for output directory naming |
| `--analyze-only` | Skip training, regenerate analysis from existing results |
| `--results-path PATH` | Explicit path to `results.json` (with `--analyze-only` or `--measure-gdelta`) |
| `--merge JSON [JSON ...]` | Merge multiple `results.json` files and regenerate analysis |
| `--measure-gdelta` | Measure the return term at the pretrained checkpoint; no training |
| `--gdelta-seeds N [N ...]` | Rollout seeds to measure (default `0`); the reported +/- is across these |
| `--gdelta-draws N` | Independent `(z_t, t)` draws per seed (default 8) |
| `--gdelta-inputs PATH [PATH ...]` | Aggregate per-seed gdelta JSONs from separate machines |
| `--use-wandb` / `--no-use-wandb` | Enable/disable W&B logging (overrides `use_wandb`, default `false`) |
| `--wandb-project NAME` | W&B project (overrides `wandb_project`, default `minihack-ReMDM-planner-ablations`) |
| `--wandb-entity NAME` | W&B entity (overrides `wandb_entity`) |
| `--wandb-resume-id ID` | W&B run ID for curve continuity |
| `--max-iter N` | Override max training iterations |
| `--batch-size N` | Override batch size |
| `--eval-every N` | Override evaluation frequency |
| `--lr FLOAT` | Override learning rate |
| `--device DEVICE` | Torch device (default: auto-detect) |

### W&B logging

When `--use-wandb` is passed, all training dynamics are logged in real time:

| Namespace | Metrics | Frequency |
|---|---|---|
| `train/` | `loss`, `learning_rate`, `grad_norm`, `effective_batch_size`, `ablation_local_iter` | Every iteration |
| `online/` | `win_rate`, `mean_return` | Every iteration |
| `speed/` | `iter_time_sec`, `collect_time_sec`, `train_step_time_sec`, `gpu_memory_mb` | Every iteration |
| `model/` | `param_norm`, `param_drift_from_init`, `ema_gate_value` | Every 10 iterations |
| `eval/` | `id_win_rate`, `per_env/{env}/win_rate` | Every `eval_every` |
| `diag/` | `grad_alignment_cos`, `repr_drift_kl`, `cka_similarity`, `t_grad_norm_low/high` | At diagnostic intervals |

Final scores per ablation are written to `wandb.summary`.

### Diagnostic metrics collected

| Metric | Frequency | What it answers |
|---|---|---|
| Eval score (ID win rate) | every `eval_every` iters | Primary performance |
| Training loss | every iteration | Optimisation signal |
| Win rate (online) | every iteration | Online rollout quality |
| Gradient alignment (cos sim) | every `grad_align_every` | Is the RL gradient useful? |
| Per-layer gradient norms | every `per_layer_every` | Which layers collapse? |
| KL drift from pretrained | every `repr_drift_every` | How much has the model changed? |
| CKA similarity | every `cka_every` | Representational drift (activation level) |
| t-bin gradient norms | every `t_analysis_every` | Is high-t gradient biased? |
| Gradient surgery fraction | every `grad_align_every` | PCGrad projected mass |
