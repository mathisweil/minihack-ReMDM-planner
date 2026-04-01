# ReMDM Experiments

Research and diagnostic scripts for investigating RL fine-tuning of the ReMDM diffusion planner.
These scripts are **standalone research code** -- they import from `src/` but do not modify it.

---

## `rl_finetuning/` -- RL Fine-Tuning Ablation Suite

Diagnoses why RL fine-tuning of the diffusion model collapses and which interventions fix it.
Implements **25 ablations** across five groups, plus a comprehensive diagnostic and analysis pipeline.

### Directory structure

```
rl_finetuning/
├── run_ablations.py          # CLI entry point
├── ablations/
│   ├── losses.py             # 16 loss/objective variants as factory functions
│   ├── optimizers.py         # AdamW, LLRD, LoRA, gradient surgery, frozen params
│   ├── registry.py           # AblationSpec dataclass + REGISTRY (25 ablations)
│   └── training.py           # run_ablation() loop + AblationHistory dataclass
├── diagnostics/
│   ├── gradient.py           # Grad alignment, per-layer norms, surgery metrics
│   ├── representation.py     # KL drift, CKA similarity, activation norms
│   └── timestep.py           # t-bin gradient norms, per-t loss decomposition
├── analysis/
│   ├── plots.py              # 8 matplotlib figure generators
│   ├── tables.py             # Summary tables as polars DataFrames + LaTeX export
│   └── report.py             # diagnosis.md + decision tree figure
└── configs/
    ├── ablations_default.yaml   # Full-run hyperparameters
    └── ablations_fast.yaml      # Smoke-test overrides (50 iterations)
```

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

**Full suite (all 25 ablations):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint path/to/dagger_checkpoint.pth \
    --all \
    --num_seeds 3 \
    --use_wandb
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
    --analyze_only --output_dir outputs/run_20260331_120000
```

**Re-plot a subset of ablations:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --analyze_only --output_dir outputs/run_20260331_120000 \
    --ablations baseline_rl kl_penalty ewc
```

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
| **B: Training Signal** | `low_t` | ELBO restricted to low-t (fine-detail) regime |
| | `t_curriculum` | Anneal t range high-to-low over training |
| | `entropy_bonus` | Entropy regularisation for action diversity |
| | `gradient_surgery` | PCGrad: project conflicting RL/BC gradients |
| | `advantage_clip` | PPO-style advantage clipping [1-eps, 1+eps] |
| | `normalized_adv` | Std-normalised advantages (GRPO-style) |
| | `bc_wins` | Uniform ELBO on win windows (no advantage weighting) |
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
├── win_rate.png                    # Online win rate over training
├── group_comparison.png            # Boxplot of scores by ablation group
├── main_results.{csv,tex}          # Main results table
├── group_summary.{csv,tex}         # Group-level summary table
├── gradient_diagnostics.{csv,tex}  # Gradient alignment at final iteration
├── repr_drift.{csv,tex}            # KL drift values at final iteration
└── per_env_win_rates.{csv,tex}     # Per-environment win rates
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
      "history": { ... }
    }
  }
}
```

`results.json` is written incrementally after each ablation completes -- a partial file
with N of 25 ablations is fully valid and loadable by `--analyze_only`.

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
