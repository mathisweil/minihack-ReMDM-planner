# ReMDM Experiments

Research and diagnostic scripts for investigating RL fine-tuning of the ReMDM
diffusion planner. These scripts are **standalone research code** -- they
import from `src/` (model, sampling, env wrapper, evaluator) but never modify
the core training pipeline. They start from a pretrained DAgger checkpoint
and answer the question: *which intervention prevents RL fine-tuning collapse?*

---

## `rl_finetuning/` -- RL Fine-Tuning Ablation Suite

Diagnoses why RL fine-tuning of the diffusion model collapses and which interventions fix it.
Implements **25 ablations** across five groups, plus a comprehensive diagnostic and analysis pipeline.

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
│   ├── plots.py              # 12 matplotlib figure generators
│   ├── tables.py             # Summary tables as polars DataFrames + LaTeX export
│   ├── report.py             # diagnosis.md + decision tree figure
│   ├── action_distribution.py  # Pre/post-RL action distribution analysis
│   └── mixing_experiment.py    # Data quality degradation curve experiment
└── configs/
    ├── ablations_default.yaml   # Base: all ablation hyperparameters
    ├── ablations_fast.yaml      # Smoke-test overlay (50 iterations)
    ├── final_ablations_qmul.yaml  # QMUL H200 overrides only (extends base)
    └── final_ablations_ucl.yaml   # UCL 3090 Ti overrides only (extends base)
```

### Config inheritance

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
# final_ablations_ucl.yaml
extends: ablations_default.yaml
batch_size: 4608
use_amp: true
cka_batch_size: 128
num_seeds: 3
```

| Rule | Behaviour |
|---|---|
| `extends: <path>` | Resolved relative to the declaring file; absolute paths accepted |
| No `extends` key | Implicitly extends `ablations_default.yaml` |
| `extends:` (empty) | No inheritance |
| Chain depth | Any; cycles raise `ValueError` |
| `--fast` | Applied raw after the chain, so machine keys (`use_amp`, `batch_size`, `diffusion_steps_collect`) survive |

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
    --ablations-config experiments/rl_finetuning/configs/final_ablations_ucl.yaml \
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

`--merge` accepts any number of `results.json` files. When the same ablation
appears in multiple files (e.g. different seeds on different GPUs), the
per-seed scores are concatenated and mean/std are recomputed over the union:

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
# -> baseline_rl: 0.6250 +/- 0.0250 (2 seeds)
```

The merged `results.json` is identical in format to a single-run file and can
be used with `--analyze-only` for further filtering or re-plotting.

### Ablations

| Group | Name | Tests |
|---|---|---|
| Baseline | `baseline_rl` | Standard return-weighted ELBO |
| **A: Regularisation** | `kl_penalty` | Soft KL constraint vs pretrained |
| | `ewc` | Elastic Weight Consolidation (Fisher diagonal) |
| | `llrd` | Layer-wise Learning Rate Decay |
| | `lora` | Low-Rank Adaptation of attention projections |
| | `mixed_replay` | Offline data mixed into online batches |
| | `trust_region_kl` | Hard KL trust region via quadratic barrier |
| **B: Training Signal** | `t_curriculum` | Anneal t range high-to-low over training |
| | `entropy_bonus` | Entropy regularisation for action diversity |
| | `gradient_surgery` | PCGrad: project conflicting RL/BC gradients |
| | `advantage_clip` | PPO-style advantage clipping [1-eps, 1+eps] |
| | `normalized_adv` | Std-normalised advantages (GRPO-style) |
| | `bc_wins` | Uniform ELBO on win windows (no advantage weighting) |
| | `low_t` | ELBO restricted to low-t (fine-detail) regime |
| **C: Architecture** | `frozen_backbone` | Only train the output head |
| | `head_only` | Only train the final linear projection |
| | `attention_only` | Only train attention weights (Q/K/V/O) |
| | `ffn_only` | Only train FFN layers |
| | `layer_ablation_top1` | Only train top-1 transformer block |
| | `layer_ablation_top2` | Only train top-2 transformer blocks |
| | `layer_ablation_top3` | Only train top-3 transformer blocks |
| **D: Data Quality** | `reward_filtering` | Top-75th-percentile return windows only |
| | `running_stats` | EMA running mean/std for advantage normalisation |
| | `action_diversity` | Discard degenerate (all-same-action) plans |
| | `reward_model` | MLP reward model soft-weighting of advantages |

### Output structure

```
experiments/rl_finetuning/outputs/{run_id}/
├── results.json                    # All histories + final scores (machine-readable)
├── diagnosis.md                    # Human-readable verdict + evidence + recommendations
├── diagnosis_decision_tree.png     # Hypothesis evidence bar chart
├── checkpoint_{name}.pth           # Per-ablation fine-tuned model state dict
├── train_{name}.png                # Per-ablation training curves (loss + win rate)
├── score_comparison.png            # Bar chart of final scores across ablations
├── grad_alignment.png              # Gradient cosine similarity over training
├── repr_drift.png                  # KL divergence drift by t-range
├── cka_similarity.png              # CKA similarity vs pretrained over training
├── t_bin_norms.png                 # Heatmap of per-t-bin gradient norms
├── t_ratio.png                     # High-t / low-t gradient norm ratio over training
├── win_rate.png                    # Online win rate over training
├── group_comparison.png            # Boxplot of scores by ablation group
├── gradient_conflict_map.png       # Binary heatmap of gradient conflicts (cos_sim < 0)
├── score_delta.png                 # Sorted bar chart of improvement over baseline_rl
├── per_env_delta.png               # Heatmap of per-env win rate change (end - start)
├── main_results.{csv,tex}          # Main results table
├── group_summary.{csv,tex}         # Group-level summary table
├── gradient_diagnostics.{csv,tex}  # Gradient alignment at final iteration
├── repr_drift.{csv,tex}            # KL drift values at final iteration
├── per_env_win_rates.{csv,tex}     # Per-environment win rates
├── forgetting_analysis.{csv,tex}   # First collapse iter, min score, recovery
└── hypothesis_verdicts.{csv,tex}   # Per-ablation hypothesis verdict + conclusion
```

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
| `--num-seeds N` | Number of seeds per ablation (default: 1) |
| `--seed N` | Base random seed |
| `--output-dir DIR` | Output directory (default: auto-timestamped) |
| `--run-id ID` | Custom run ID for output directory naming |
| `--analyze-only` | Skip training, regenerate analysis from existing results |
| `--results-path PATH` | Explicit path to `results.json` (with `--analyze-only`) |
| `--merge JSON [JSON ...]` | Merge multiple `results.json` files and regenerate analysis |
| `--use-wandb` | Enable W&B logging |
| `--wandb-project NAME` | W&B project name (default: `remdm-minihack-ablations`) |
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
