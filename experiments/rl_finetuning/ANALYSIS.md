# Phase 1: Deep Analysis of the Craftax RL Fine-Tuning Ablation Suite

---

## 1.1 Registry & Ablation Definitions (`ablations/registry.py`)

### AblationSpec Dataclass Schema

```python
@dataclass
class AblationSpec:
    name: str                           # Short CLI identifier (e.g. "kl_penalty")
    group: str                          # "Baseline", "A", "B", "C", or "D"
    description: str                    # One-line human-readable description
    hypothesis: str                     # What failure mode this ablation tests
    loss_factory: LossFactory           # Callable(ctx, **extra) -> LossFn
    optimizer_factory: OptimizerFactory  # Callable(config, params) -> GradientTransformation
    frozen_path_fragments: list[str]    # Param path substrings to freeze (zero gradient)
    wins_only: bool                     # Pre-filter batch to winning windows only
    gradient_surgery: bool              # Apply PCGrad to RL vs BC gradients
    mixed_replay: bool                  # Mix offline buffer into each batch
    t_curriculum: bool                  # Anneal t range over training
    reward_filtering: bool              # Discard windows with return < percentile
    running_stats: bool                 # EMA running mean/std for advantage normalisation
    action_diversity_filter: bool       # Discard degenerate all-same-action plans
    reward_model_weighting: bool        # Weight advantages with a learned reward model
    extra_loss_kwargs: dict             # Forwarded to loss_factory
```

Default for most boolean flags is `False`; default `optimizer_factory` is `make_optimizer_standard`.

### Full Ablation Listing (25 ablations)

| # | Name | Group | Hypothesis | What it modifies | Key hyperparameters |
|---|---|---|---|---|---|
| 1 | `baseline_rl` | Baseline | Diagnoses whether RL signal alone causes collapse | Nothing -- pure return-weighted ELBO | -- |
| 2 | `kl_penalty` | A | Catastrophic forgetting; soft regularisation suffices | Loss: + kl_coef * KL(current \|\| pretrained) | `kl_coef=0.1` |
| 3 | `ewc` | A | Forgetting pretrained representations | Loss: + ewc_lambda * Fisher-weighted L2 penalty | `ewc_lambda=100.0`, `ewc_fisher_batches=20` |
| 4 | `llrd` | A | Deep gradient flow into early layers corrupts representations | Optimizer: layer-wise LR decay | `llrd_decay=0.9` |
| 5 | `lora` | A | Too many unconstrained degrees of freedom cause collapse | Optimizer: only LoRA A/B matrices trainable | `lora_rank=8`, `lora_alpha=16.0` |
| 6 | `mixed_replay` | A | Online data distribution alone is too corrupted | Data: blend offline buffer into each batch | `mixed_replay_ratio=0.25`, `mixed_replay_buffer_size=10000` |
| 7 | `trust_region_kl` | A | Soft KL insufficient; hard boundary needed | Loss: quadratic barrier when KL > threshold | `trust_region_kl=0.05` |
| 8 | `t_curriculum` | B | Ordering of learning signals matters | Loss: anneal t range from high-t to low-t | `t_curriculum_start=0.8`, `t_curriculum_end=0.2`, `t_curriculum_steps=200` |
| 9 | `entropy_bonus` | B | Collapse is mode-collapse, not a gradient problem | Loss: - entropy_coef * entropy of p_theta | `entropy_coef=0.01` |
| 10 | `gradient_surgery` | B | Gradients are conflicting and resolvable by projection | Training loop: PCGrad on RL vs BC grads | -- (applied externally) |
| 11 | `advantage_clip` | B | Large advantage magnitudes destabilise training | Loss: clip advantages to [1-eps, 1+eps] | `adv_clip_eps=0.2` |
| 12 | `normalized_adv` | B | Simple mean normalisation is too loose | Loss: (A - mean) / (std + eps) GRPO-style | -- |
| 13 | `bc_wins` | B | The return weighting is the specific cause | Loss: uniform ELBO on win windows only (no advantages) | `wins_only=True`, `win_threshold=0.1` |
| 14 | `low_t` | B | High-t gradients are biased | Loss: restrict t to [eps, t_max_low] | `t_max_low=0.2` |
| 15 | `frozen_backbone` | C | Deep gradient flow into backbone causes collapse | Optimizer: freeze everything except output head | Frozen paths: TransformerBlock, obs encoder, etc. |
| 16 | `head_only` | C | Backbone representations fine; only decision boundary needs updating | Optimizer: only final linear projection trainable | Frozen paths: all except head |
| 17 | `attention_only` | C | Model needs routing updates, not feature updates | Optimizer: only Q/K/V/O trainable; FFN frozen | -- |
| 18 | `ffn_only` | C | Stored knowledge (FFN as memory) needs updating | Optimizer: only FFN trainable; attention frozen | -- |
| 19 | `layer_ablation_top1` | C | Minimal unfrozen depth; collapse depth correlates with gradient flow depth | Optimizer: only top-1 transformer block trainable | -- |
| 20 | `layer_ablation_top2` | C | Same as top1 with more capacity | Optimizer: only top-2 transformer blocks trainable | -- |
| 21 | `layer_ablation_top3` | C | Same as top1/2 with more capacity | Optimizer: only top-3 transformer blocks trainable | -- |
| 22 | `reward_filtering` | D | Noisy/low-return data poisons gradients | Data: train only on top-75th-percentile return windows | `reward_filter_percentile=75` |
| 23 | `running_stats` | D | Batch normalisation too noisy for small batches | Data: EMA running mean/std for advantage normalisation | `running_stats_ema_decay=0.99` |
| 24 | `action_diversity` | D | Degenerate PPO plans corrupt training | Data: discard all-same-action plans | -- |
| 25 | `reward_model` | D | Raw returns too sparse; learned model smooths signal | Data: re-weight advantages with an MLP reward model | `reward_model_width=64`, `reward_model_depth=2`, `reward_model_lr=1e-3` |

### Registry Pattern

- `REGISTRY` is a module-level `dict[str, AblationSpec]` keyed by ablation name.
- Lookup is a simple dict access: `REGISTRY["ewc"]`.
- Instantiation happens in `run_ablation()` which reads the `loss_factory` and `optimizer_factory` from the spec, constructs a `LossContext`, and calls the factory functions.
- Helper lambdas (`_std_opt`, `_llrd_opt`, `_frozen_backbone_opt`, etc.) wrap the optimizer factory functions for uniform signature `(config, params) -> GradientTransformation`.
- The `_layer_ablation_top_n_opt(n)` is a factory-of-factories that returns an optimizer factory for a given number of unfrozen top blocks.

---

## 1.2 Loss Variants (`ablations/losses.py`)

### Base Loss Function Signature

```python
LossFn = Callable[
    [params, acts[B,H], obs[B,obs_dim], valid[B], rng, advantages[B]],
    scalar
]
```

All loss functions are closures created by factory functions that take a `LossContext` and optional extra kwargs.

### LossContext Dataclass

Bundles shared context: `apply_fn` (training forward pass), `ref_params` (frozen pretrained), `schedule_fn`, `schedule_deriv_fn`, `num_actions`, and `config` dict.

### Core Loss Helper: `_core_loss`

Wraps `compute_loss` from `src/diffusion/loss.py`. Accepts optional `t_min`/`t_max` to restrict the uniform t sampling range. Passes through `advantages`, `sigma_t`, and `label_smoothing` from config.

### Loss Variant Details

| Variant | Mathematical Objective | Additional State | Factory Signature |
|---|---|---|---|
| **`make_loss_baseline`** | Return-weighted ELBO: `compute_loss(..., advantages=advantages)` | None | `(ctx) -> LossFn` |
| **`make_loss_kl_penalty`** | `RL_ELBO + kl_coef * KL(current \|\| pretrained)` on masked positions. KL = `sum(p_cur * (log p_cur - log p_ref))` per position, averaged over masked positions and batch. | Frozen `ref_params` (via ctx) | `(ctx) -> LossFn` |
| **`make_loss_ewc`** | `RL_ELBO + ewc_lambda * sum(F_i * (theta_i - theta_ref_i)^2)`. Summation over all parameter leaves. | Pre-computed Fisher diagonal pytree + ref_params | `(ctx, fisher) -> LossFn` |
| **`make_loss_trust_region_kl`** | `RL_ELBO + 1e4 * max(KL - threshold, 0)^2`. Quadratic barrier method. | ref_params via ctx | `(ctx) -> LossFn` |
| **`make_loss_mixed_replay`** | Identical to baseline (data mixing handled externally in training loop) | None | `(ctx) -> LossFn` |
| **`make_loss_bc_wins`** | `compute_loss(..., advantages=None)` -- uniform ELBO, ignoring advantages. Caller pre-filters to win windows. | None | `(ctx) -> LossFn` |
| **`make_loss_low_t`** | `compute_loss(..., t_min=eps, t_max=t_max_low)`. Restricts to low-t regime (default t_max=0.2). | None | `(ctx) -> LossFn` |
| **`make_loss_t_curriculum`** | t range anneals over training: `[t_start, 1.0] -> [eps, t_end]`. Uses mutable `current_iter` list for non-JIT context. JIT variant `make_loss_t_curriculum_jit` takes `step_idx` as JAX array. | Mutable iter counter (or step_idx) | `(ctx, current_iter) -> LossFn` |
| **`make_loss_entropy_bonus`** | `RL_ELBO - entropy_coef * H(p_theta)`. Entropy = `-sum(p * log p)` over action vocab, averaged over masked positions. | None | `(ctx) -> LossFn` |
| **`make_loss_gradient_surgery`** | Identical to baseline (PCGrad applied externally in training loop) | None | `(ctx) -> LossFn` |
| **`make_loss_advantage_clip`** | `compute_loss(..., advantages=clip(advantages, 1-eps, 1+eps))` | None | `(ctx) -> LossFn` |
| **`make_loss_normalized_adv`** | `compute_loss(..., advantages=(A-mean)/(std+eps))` | None | `(ctx) -> LossFn` |
| **`make_loss_frozen_backbone`** | Identical to baseline (freezing handled by optimizer) | None | `(ctx) -> LossFn` |
| **`make_loss_param_isolation`** | Identical to baseline (masking handled by optimizer). Shared by head_only, attention_only, ffn_only, layer_ablation variants. | None | `(ctx) -> LossFn` |
| **`make_loss_reward_quality`** | Identical to baseline (data quality filtering handled externally). Shared by reward_filtering, running_stats, action_diversity, reward_model. | None | `(ctx) -> LossFn` |

### Fisher Diagonal Estimation (`estimate_fisher_diagonal`)

- Computes `F_i = E[(d log p / d theta_i)^2]` by iterating over held-out batches.
- For each batch: compute gradient of BC loss (advantages=1.0), square element-wise, accumulate.
- Average over all batches. Returns a pytree matching `ref_params` structure.

---

## 1.3 Optimizer Variants (`ablations/optimizers.py`)

### Standard Optimizer (`make_optimizer_standard`)

`optax.chain(clip_by_global_norm(max_grad_norm), adam(lr, eps=1e-5))`.

### LLRD (`make_optimizer_llrd`)

- **Layer identification**: parses parameter path strings to classify into groups:
  - `head`: final output projection (not inside TransformerBlock)
  - `block_{i}`: TransformerBlock at index i
  - `obs_enc`: observation encoder layers
- **LR assignment**: `lr_group = base_lr * decay^(depth_from_top)` where:
  - Head: depth 0 (fastest LR)
  - TransformerBlock_{N-1}: depth 1
  - TransformerBlock_{0}: depth N
  - Obs encoder: depth N+1 (slowest LR)
- Uses `optax.multi_transform` with a label tree built by `jax.tree_util.tree_map_with_path`.

### LoRA

- **Target modules**: attention kernels identified by `"MultiHeadDotProductAttention"` path fragment + `"kernel"` in path.
- **Rank**: configurable (`lora_rank=8`), alpha = 16.0.
- **Initialisation**: A matrices ~ N(0, 0.02), B matrices = zeros (so initial delta = 0).
- **Application**: `apply_fn_with_lora` injects effective weights `W_eff = W_frozen + (alpha/rank) * A @ B` into the param tree via `tree_map_with_path`.
- **Optimizer**: `make_optimizer_lora_only` creates a masked optimizer where `base` params get zero gradient and only `lora` params are updated. Combined tree is `{"base": base_params, "lora": lora_params}`.

### Gradient Surgery (PCGrad)

- `gradient_surgery(g_rl, g_bc)`: per-leaf projection.
- For each leaf: if `dot(g_rl, g_bc) < 0`, project RL gradient onto the plane orthogonal to BC gradient: `g_projected = g_rl - (dot/norm_sq) * g_bc`.
- Otherwise, keep `g_rl` unchanged.
- Applied at the training loop level, not inside the loss function.

### Parameter Masking / Freezing (`make_optimizer_frozen_paths`)

- Takes a list of `frozen_path_fragments` (substrings).
- Builds a boolean mask tree: `True` = trainable, `False` = frozen.
- Uses `optax.masked(adam, mask_tree)` to zero gradients for frozen params.
- Specific ablation configurations:
  - **`frozen_backbone`**: freezes `TransformerBlock_`, `SinusoidalPosEmbed_`, `Dense_0`, `Dense_1`, `LayerNorm_0`, `LayerNorm_1`
  - **`head_only`**: freezes everything (transformer, obs encoder, embeddings); only final Dense trainable
  - **`attention_only`**: freezes FFN Dense layers inside TransformerBlocks and obs encoder; keeps MultiHeadDotProductAttention trainable
  - **`ffn_only`**: freezes attention, embeddings, obs encoder; keeps FFN Dense layers trainable
  - **`layer_ablation_top_n`**: freezes all transformer blocks except top N, plus obs encoder and embeddings

---

## 1.4 Training Loop (`ablations/training.py`)

### `make_run_ablation()` Factory

**Signature:**
```python
make_run_ablation(
    spec, config, pretrained_params, apply_train, apply_eval,
    env, env_params, ppo, schedule_fn, schedule_deriv_fn,
    num_actions, obs_dim, fisher=None
) -> Callable  # run(rng) -> (AblationCarry, StepMetrics)
```

**Returns** a compiled closure `run(rng)` that executes the full ablation training loop via `jax.lax.scan` (no Python-level loop).

### Training Loop Structure

The loop runs entirely inside `jax.lax.scan` for `max_iter` iterations. Each `_update_step` does:

1. **Collect rollout**: PPO agent runs in the environment to generate `(obs, acts, valid, returns)` windows.
2. **Action diversity filter** (optional): discard all-same-action plans.
3. **Reward filtering** (optional): discard windows below return percentile.
4. **Replay buffer update** (for mixed_replay): push new samples into a ring buffer.
5. **Reward model training** (optional): train MLP reward model on obs -> returns, then re-weight returns.
6. **Compute advantages**: normalise returns to advantage weights with floor/cap clipping. Three modes: standard (return/mean), wins_only (binary), running_stats (EMA normalisation).
7. **Shuffle and batch**: permute and take `batch_size` samples.
8. **Mixed replay blending** (optional): concatenate offline buffer samples into the batch.
9. **Gradient step**:
   - If `gradient_surgery`: compute RL loss grad and BC loss grad separately, apply PCGrad, then `apply_gradients`.
   - Otherwise: standard `value_and_grad` + `apply_gradients`.
   - If `t_curriculum`: uses JIT-compatible t-curriculum loss with `step_idx`.
10. **Diagnostics** (conditional on step frequencies):
    - Gradient alignment: cosine similarity between RL and BC gradients
    - Representation drift: KL(ref || current) at overall/low-t/mid-t/high-t
    - CKA: linear CKA between current and ref output representations
    - t-analysis: per-bin gradient norms and low/high-t alignment
    - Per-layer gradient norms
    - Surgery metrics (if gradient_surgery)
11. **Eval** (conditional on `eval_every`): run diffusion sampling policy in environment.
12. **Build StepMetrics** with all values + boolean flags for which diagnostics ran.

### AblationCarry (scan carry)

Contains: `TrainState`, `env_state`, `obs`, `done`, `hstate`, `rng`, `step_idx`, `running_mean`, `running_std`, `replay_buf`, `reward_model_state`.

### StepMetrics (per-step output)

Contains: `loss`, `env_score`, `win_rate`, `eff_batch_size`, `cos_sim`, `rl_grad_norm`, `bc_grad_norm`, `kl_mean`, `kl_low_t`, `kl_mid_t`, `kl_high_t`, `cka`, `t_bin_norms[n_bins]`, `low_high_cos`, `t_norm_low`, `t_norm_high`, `surgery_frac`, `surgery_n_conflict`, `per_layer_norms[num_leaves]`, `eval_score`, and 8 boolean `did_*` flags.

### AblationHistory Dataclass

Lists tracked over time at various frequencies:
- **Every 10 iters**: `iters`, `loss`, `env_score_iters`, `env_score`, `win_rate`, `effective_batch_size`
- **Every eval_every**: `eval_iters`, `eval_score`, `per_achievement_rates`
- **Every grad_align_every**: `grad_align_iters`, `grad_align`, `rl_grad_norm`, `bc_grad_norm`
- **Every per_layer_every**: `per_layer_iters`, `per_layer_norms` (dict per layer)
- **Every repr_drift_every**: `repr_drift_iters`, `repr_drift_kl`, `repr_drift_kl_low_t/mid_t/high_t`
- **Every cka_every**: `cka_iters`, `cka_similarity`
- **Every t_analysis_every**: `t_analysis_iters`, `norm_low_t`, `norm_high_t`, `lowhigh_cos`, `t_bin_norms`
- **Gradient surgery**: `surgery_iters`, `surgery_fraction`, `surgery_n_conflicting`

Supports `to_dict()` / `from_dict()` for JSON serialisation.

### Rollout Collection

Uses the existing Craftax project's `PPOAgent` for rollout collection. The ablation suite does NOT use DAgger with oracle -- it uses PPO to collect trajectories, then applies return-weighted ELBO on those trajectories. This is a significant difference from the main MiniHack DAgger pipeline.

The `build_rollout_fn` creates a JIT-compiled function that:
1. Runs `num_steps` PPO environment steps in parallel across `num_envs`.
2. Extracts sliding windows of length `plan_horizon` from the trajectory.
3. Flattens to `[N, ...]` arrays where `N = valid_per_rollout * num_envs`.

### Multi-seed Handling

The outer `run_ablation()` function is called once per seed in `run_ablations.py`. Each seed gets a different `jax.random.PRNGKey`. Results are aggregated: first seed's history is used for plots, mean/std of scores reported.

### `run_ablation()` High-level Entry Point

Python-level wrapper that:
1. Estimates Fisher diagonal (for EWC only) before JIT.
2. Calls `make_run_ablation()` to build the compiled closure.
3. JIT-compiles and runs it.
4. Converts scan output to `AblationHistory` via `metrics_to_history`.
5. Runs final eval to extract achievement rates (can't be done inside `jax.lax.scan` due to variable-key dicts).
6. Logs to W&B if enabled.

---

## 1.5 Diagnostics (`diagnostics/`)

### Gradient Diagnostics (`gradient.py`)

#### Gradient Alignment (cosine similarity)

- **`make_grad_alignment_fn`**: builds a JIT-compiled function that computes:
  - `g_rl`: gradient of return-weighted ELBO w.r.t. current params
  - `g_bc`: gradient of unweighted ELBO w.r.t. pretrained ref params
  - Flatten both to 1D vectors, compute `cos_sim = dot(g_rl, g_bc) / (||g_rl|| * ||g_bc|| + eps)`
  - Returns: `(cos_sim, rl_norm, bc_norm)` as JAX scalars.
- Both RL and BC losses call `compute_loss` from `src/diffusion/loss.py` with different advantage weights (actual advantages vs None).
- **Collection frequency**: every `grad_align_every` iterations (default 25).

#### Per-layer Gradient Norms

- **`compute_per_layer_grad_norms_jax`**: computes L2 norm per pytree leaf of the gradient. Returns `[num_leaves]` array.
- Layer names recovered outside JIT via `jax.tree.structure(params)`.
- **Collection frequency**: every `per_layer_every` iterations (default 25).

#### Gradient Surgery Metrics

- **`compute_surgery_metrics_jax`**: compares RL gradients before and after PCGrad projection.
- Returns: `(projected_mass_fraction, n_conflicting_params)`.
  - `projected_mass_fraction = (||g_before||^2 - ||g_after||^2) / ||g_before||^2`
  - `n_conflicting_params` = count of leaves where projection was applied.
- Only collected when `gradient_surgery=True`.

### Representation Diagnostics (`representation.py`)

#### KL Drift from Pretrained

- **`make_repr_drift_fn`**: builds a JIT-compiled function that computes `KL(ref || current)` (note: KL from reference distribution, not from current).
- Samples t uniformly in a specified range, applies forward masking, computes logits from both models on the same noisy input.
- Returns 4 scalars: `kl_mean` (full range), `kl_low` (t in [eps, 0.2]), `kl_mid` (t in [0.3, 0.7]), `kl_high` (t in [0.8, 1.0]).
- **Collection frequency**: every `repr_drift_every` iterations (default 25).

#### CKA Similarity

- **`_linear_cka`**: Linear CKA with HSIC estimator: `CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)`.
- **`make_cka_fn`**: uses the first `cka_batch_size` (default 64) samples from the batch, samples t in [0.3, 0.7], computes logits from both models, takes mean over sequence positions to get `[B, V]` representations, then applies `_linear_cka`.
- Returns: scalar CKA value in [0, 1] (1 = identical representations).
- **Collection frequency**: every `cka_every` iterations (default 50, more expensive).

#### Activation Norm Statistics

- **`make_activation_norm_fn`**: computes mean, std, p50, p90 of per-sample logit L2 norms. Available but not heavily used in the main training loop or analysis.

### Timestep Diagnostics (`timestep.py`)

#### t-bin Analysis

- **`N_BINS = 10`** equal bins over t in [0, 1].
- **`make_t_analysis_fn`**: builds JIT-compiled function that:
  1. For each of the 10 bins: computes gradient of the loss restricted to that t-range, flattens, computes L2 norm. Uses `jax.lax.scan` over bin indices.
  2. Separately computes full gradient vectors for low-t [eps, 0.2] and high-t [0.8, 1.0].
  3. Computes cosine similarity between low-t and high-t gradients.
  4. Returns: `(bin_norms[10], low_high_cos, norm_low_t, norm_high_t)`.

#### Per-t-bin Loss Decomposition

- **`make_t_bin_loss_fn`**: similar structure but computes per-bin *loss values* (not gradient norms). Available for deeper analysis.

- **Collection frequency**: every `t_analysis_every` iterations (default 25).

---

## 1.6 Analysis & Reporting (`analysis/`)

### Plots (`plots.py`)

Style: Agg backend, 150 DPI, white background, grid alpha=0.3. Group colours: Baseline=grey, A=blue, B=orange, C=green, D=red.

**Per-ablation plots** (one per ablation):

| Figure | File | Data Required | What it Visualises |
|---|---|---|---|
| Training curves 2x3 grid | `curves_{name}.png` | loss, eval_score, env_score, grad_align, repr_drift_kl, rl/bc_grad_norm | Comprehensive per-ablation training progression |
| Per-layer gradient heatmap | `per_layer_grad_heatmap_{name}.png` | per_layer_norms over time | Rows=layers, cols=iterations, colour=L2 norm |
| Per-t-bin gradient norms | `t_bin_grad_norms_{name}.png` | t_bin_norms over time | One line per t-bin, coloured by plasma colormap |
| Achievement collapse heatmap | `achievement_collapse_{name}.png` | per_achievement_rates over eval checkpoints | Rows=achievements, cols=eval iters, colour=unlock rate |

**Aggregate plots** (one total):

| Figure | File | Data Required | What it Visualises |
|---|---|---|---|
| Final score bar chart | `final_score_comparison.png` | Final scores per ablation | Bars coloured by group with pretrained baseline line |
| Eval scores over training | `eval_scores_over_training.png` | eval_iters, eval_score per ablation | All ablation curves overlaid |
| Score delta vs baseline-RL | `score_delta_over_baseline_rl.png` | Final scores | Sorted bar chart of improvement over baseline_rl |
| Gradient alignment overlay | `gradient_alignment.png` | grad_align per ablation | All cos_sim curves overlaid |
| Gradient conflict map | `gradient_conflict_map.png` | grad_align per ablation | Binary heatmap: rows=ablations, cols=steps, red if cos_sim < 0 |
| Representation drift overlay | `representation_drift.png` | repr_drift_kl per ablation | All KL curves overlaid |
| CKA similarity overlay | `cka_similarity.png` | cka_similarity per ablation | All CKA curves overlaid |
| t-distribution analysis | `t_distribution_analysis.png` | norm_high_t, norm_low_t, lowhigh_cos | 2 subplots: high-t vs low-t norms; low/high-t cosine similarity |
| Win rate and eff batch size | `win_rate_and_effective_batch_size.png` | win_rate, effective_batch_size | 2 subplots over training |
| Achievement breakdown | `achievement_breakdown.png` | per_achievement_rates (start + end) | Stacked bar: start vs end per ablation |

**Total: ~14 distinct figure types**, with per-ablation figures multiplied by the number of ablations.

### Tables (`tables.py`)

Uses polars DataFrames. Exports both CSV and LaTeX (manual `_df_to_latex` since polars has no built-in).

| Table | Columns | What it Summarises |
|---|---|---|
| **`main_results`** | Method, Group, Final_Score, Delta_vs_Pretrained, Delta_vs_Baseline_RL, Verdict (IMPROVEMENT/COLLAPSE/NEUTRAL) | Primary results ranking |
| **`gradient_analysis`** | Method, Mean_Grad_Align, Final_Grad_Align, Trend (up/down/flat), Mean_KL_Drift, Final_KL_Drift | Gradient and representation health |
| **`t_distribution`** | Method, HighLow_Ratio, LowHigh_Cos_Sim, Dominant_Regime (high-t/low-t/balanced) | Timestep regime analysis |
| **`forgetting_analysis`** | Method, First_Collapse_Iter, Min_Score, Recovery_Score, Recovered (Y/N) | Catastrophic forgetting timeline |
| **`hypothesis_verdict`** | Ablation, Group, Hypothesis, Result (IMPROVEMENT/COLLAPSE/NEUTRAL), Conclusion | Per-ablation hypothesis testing |
| **`achievement_summary`** | Achievement, Pretrained, {ablation}_rate, delta_{ablation} | Per-achievement breakdown (when available) |

### Report (`report.py`)

Generates `diagnosis.md` and `diagnosis_decision_tree.png`.

**Hypothesis Groups** (6 failure mode hypotheses):

1. **Catastrophic Forgetting**: supporting ablations = kl_penalty, ewc, llrd, frozen_backbone, head_only
2. **Gradient Conflict**: supporting = gradient_surgery, kl_penalty, low_t
3. **Signal Sparsity**: supporting = bc_wins, reward_filtering, running_stats, reward_model
4. **Distributional Shift**: supporting = mixed_replay, action_diversity
5. **Mode Collapse**: supporting = entropy_bonus, advantage_clip, normalized_adv
6. **t-Bias**: supporting = low_t, t_curriculum

**Scoring logic**: For each hypothesis, count how many of its supporting ablations achieved scores near or above pretrained (score > pretrained - 0.005). `evidence_score = n_supporting / n_tested`.

**Decision tree logic**:
1. If ALL ablations collapsed: "fundamental incompatibility".
2. Otherwise: rank hypotheses by evidence_score.
3. Primary failure mode = highest-scoring hypothesis.
4. Per-ablation verdicts: IMPROVEMENT (delta > 0.005), COLLAPSE (score < pretrained - 0.005), NEUTRAL.
5. Recommendations based on top-2 hypotheses.
6. Suggested next experiments: deep dive on top hypothesis, combine top-2, increase seeds.

**Decision tree figure**: horizontal tree with colour-coded hypothesis boxes (red-yellow-green by evidence strength), listing supporting ablation names.

---

## 1.7 Configs (`configs/`)

### `ablations_default.yaml`

Self-contained config (does NOT require `defaults.yaml`). Keys converted to UPPERCASE at load time.

**Key knob groups:**

| Group | Parameters | Defaults |
|---|---|---|
| Architecture | d_model=256, n_heads=4, n_layers=4, d_ff=512, obs_encoder_layers=2, dropout_rate=0.1 | Must match pretrained checkpoint |
| Diffusion | plan_horizon=40, diffusion_schedule=cosine, train_sigma=0.0 | -- |
| Sampling (eval) | val_diffusion_steps=50, remask_strategy=rescale, eta=0.5, temperature=0.5, top_p=0.95 | -- |
| Training | max_iter=1000, num_envs=64, num_steps=128, batch_size=512, lr=3e-4, max_grad_norm=1.0 | Main training knobs |
| Evaluation | eval_every=50, eval_steps=512, eval_replan=8 | -- |
| Diagnostics | grad_align_every=25, repr_drift_every=25, t_analysis_every=25, cka_every=50, per_layer_every=25 | Frequencies |
| Advantages | win_threshold=0.1, return_weight_floor=0.1, return_weight_cap=5.0 | -- |
| Group A | ewc_lambda=100.0, ewc_fisher_batches=20, llrd_decay=0.9, lora_rank=8, lora_alpha=16.0, mixed_replay_ratio=0.25, mixed_replay_buffer_size=10000, trust_region_kl=0.05, kl_coef=0.1 | Per-ablation hyperparams |
| Group B | t_curriculum_start=0.8, t_curriculum_end=0.2, t_curriculum_steps=200, t_max_low=0.2, entropy_coef=0.01, adv_clip_eps=0.2 | -- |
| Group D | reward_filter_percentile=75, running_stats_ema_decay=0.99, reward_model_width=64, reward_model_depth=2, reward_model_lr=1e-3, reward_model_train_steps=50 | -- |
| Output | use_wandb=false, save_checkpoints=false | -- |
| Multi-seed | num_seeds=1 | -- |

### `ablations_fast.yaml`

Overrides for smoke testing:
- `max_iter=50`, `num_envs=16`, `num_steps=64`, `batch_size=128`
- `eval_every=10`, `eval_steps=128`
- All diagnostic frequencies = 10 (except cka_every=25)
- `ewc_fisher_batches=5`, `reward_model_train_steps=10`, `mixed_replay_buffer_size=500`

### Config Interaction with Main Project

The ablation configs are **self-contained** -- they do NOT inherit from the main project's `configs/defaults.yaml`. The main config can optionally be loaded via `--config` and merged (main config values are overridden by ablation config values). All keys are uppercased before use.

---

## 1.8 CLI Entry Point (`run_ablations.py`)

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | str | None | Main pipeline config (optional) |
| `--ablations_config` | str | `experiments/.../ablations_default.yaml` | Ablation-specific config |
| `--offline_checkpoint_path` | str | required | Pretrained offline diffusion checkpoint |
| `--ppo_checkpoint_path` | str | required | PPO checkpoint for rollout collection |
| `--all` | flag | -- | Run all registered ablations |
| `--ablations` | str+ | -- | Names of specific ablations to run |
| `--list` | flag | -- | List all registered ablation names and exit |
| `--fast` | flag | -- | Apply smoke-test overrides |
| `--analyze_only` | flag | -- | Skip training; regenerate analysis from results.json |
| `--results_path` | str | -- | Path to results.json for analyze_only |
| `--output_dir` | str | auto | Root output directory |
| `--run_id` | str | auto (timestamp) | Run identifier |
| `--num_seeds` | int | from config | Seeds per ablation |
| `--use_wandb` | flag | from config | Enable W&B |
| `--wandb_project` | str | from config | W&B project name |
| `--wandb_entity` | str | from config | W&B entity |
| `--max_iter`, `--num_envs`, `--batch_size`, `--eval_every`, `--lr`, `--seed` | various | from config | Direct config overrides |

### `--analyze_only` Reloading

1. Loads `results.json` via `_results_from_json()`.
2. Deserialises `AblationHistory.from_dict()` for each ablation.
3. Regenerates all plots, tables, and diagnosis report using the loaded data.
4. Writes outputs to the specified `--output_dir`.

### Incremental `results.json` Writing

After each ablation completes, the current results dict is serialised to `results.json` via `orjson` with `OPT_INDENT_2`. The file is valid JSON at all times, so a crash mid-run doesn't lose completed results. The file is directly loadable by `--analyze_only`.

Serialised structure:
```json
{
    "pretrained_score": float,
    "pretrained_ach_rates": {str: float},
    "config": {str: scalar},
    "ablations": {
        "name": {
            "score": float,
            "history": {AblationHistory.to_dict()}
        }
    }
}
```

### W&B Integration

- Initialised with `wandb.init()` using project/entity from config.
- Tags include "ablations" + selected ablation names.
- After each ablation, history metrics are logged retroactively (not live during JIT scan).
- Namespaced as `ablations/{spec.name}/train_loss`, `ablations/{spec.name}/eval_score`, etc.

### Execution Flow

1. Parse CLI args.
2. If `--list`: print registry and exit.
3. Set up output directory.
4. If `--analyze_only`: load results, regenerate analysis, exit.
5. Load and merge configs (main + ablation + fast overrides + CLI overrides).
6. Validate required checkpoint paths.
7. Set up environment, PPO, model, schedules.
8. Load pretrained checkpoint.
9. Evaluate pretrained baseline (score + achievement rates).
10. Init W&B if enabled.
11. For each selected ablation, for each seed:
    - Call `run_ablation()` -> (history, score, params).
    - Aggregate multi-seed results.
    - Write incremental `results.json`.
12. Generate all plots, tables, and diagnosis report.
13. Finish W&B run.

---

## Key Architectural Notes for Phase 2

### Framework: JAX/Flax (Craftax) vs PyTorch (MiniHack)

The entire Craftax ablation suite is written in JAX/Flax:
- Parameters are pytrees, not `nn.Module.state_dict()`.
- Optimizers are `optax.GradientTransformation`, not `torch.optim.Optimizer`.
- Training loop uses `jax.lax.scan` (no Python loop).
- Gradients computed via `jax.grad`/`jax.value_and_grad`.
- Random state uses `jax.random.PRNGKey`, not `torch.manual_seed`.
- `TrainState` is Flax's `flax.training.train_state.TrainState`.
- Model apply is functional: `apply_fn(params, obs, z_t, t, rng) -> logits`.

All of this must be rewritten in PyTorch idiom.

### Rollout Collection: PPO (Craftax) vs DAgger (MiniHack)

Craftax uses a separate PPO agent for rollout collection -- the diffusion model is NOT used for collecting episodes. The PPO agent generates trajectories, and the diffusion model is trained on those trajectories with return-weighted ELBO.

MiniHack uses the diffusion model itself (via `run_model_episode` and `DataCollector`) for rollouts, with a BFS oracle providing the training signal. This is a fundamental architectural difference that needs careful adaptation.

### Environment Interface

- Craftax uses Gymnax (functional, vmapped, JIT-compatible).
- MiniHack uses gym-style environments (Python objects, single-threaded).

### Ablation Count

Despite the task description saying 25 ablations, the actual registry has exactly **25 entries** (1 Baseline + 6 Group A + 7 Group B + 7 Group C + 4 Group D = 25).
