# RL Fine-Tuning Ablation Suite -- Environment-Agnostic Spec

> **Purpose:** A developer on a different codebase (e.g. Craftax) can use this
> document to verify their `experiments/rl_finetuning/` implements the same
> algorithmic logic. Check off each item once parity is confirmed.
>
> **Scope:** Environment-agnostic algorithmic logic only. MiniHack-specific
> details (glyph embeddings, BFS oracle, `.des` files, NLE wrappers,
> observation shapes) are omitted.

---

## 1. Ablation Registry (`ablations/registry.py`)

25 ablations in 5 groups. Each is an `AblationSpec` dataclass with: name,
group, description, hypothesis, loss_factory, optimizer_factory, and boolean
flags (use_lora, wins_only, gradient_surgery, mixed_replay, t_curriculum,
reward_filtering, running_stats, action_diversity_filter,
reward_model_weighting).

### Baseline

- [ ] **baseline_rl** -- Return-weighted ELBO, standard AdamW, no modifications.

### Group A: Regularisation / Constraint Methods

- [ ] **kl_penalty** -- Return-weighted ELBO + soft KL(current || pretrained) penalty. Coef: `kl_coef=0.1`. Standard AdamW.
- [ ] **ewc** -- ELBO + Elastic Weight Consolidation (Fisher diagonal penalty). Lambda: `ewc_lambda=100.0`. Fisher estimated from `ewc_fisher_batches=20` held-out batches. Standard AdamW.
- [ ] **llrd** -- Baseline ELBO with Layer-wise Learning Rate Decay. Decay per transformer depth: `llrd_decay=0.9`. Head=base_lr, block_i=base_lr*decay^(N-i), obs_enc=base_lr*decay^(N+1).
- [ ] **lora** -- Baseline ELBO with LoRA on attention projections (in_proj_weight, out_proj.weight). `lora_rank=8`, `lora_alpha=16.0`. All base params frozen; only LoRA A/B trainable. B zero-init, A Gaussian(0.02). Effective delta = (alpha/rank)*B@A.
- [ ] **mixed_replay** -- Baseline ELBO; each batch is `mixed_replay_ratio=0.25` offline + 0.75 online from a ring buffer (`mixed_replay_buffer_size=10000`).
- [ ] **trust_region_kl** -- ELBO + hard KL trust region via quadratic barrier: penalty = `1e4 * max(KL - threshold, 0)^2`. Threshold: `trust_region_kl=0.05`.

### Group B: Training Signal Modifications

- [ ] **t_curriculum** -- ELBO with t range annealed from `[t_start=0.8, 1.0]` to `[eps, t_end=0.2]` over `t_curriculum_steps=200` iterations. Linear interpolation of both endpoints.
- [ ] **entropy_bonus** -- ELBO minus `entropy_coef=0.01` * mean entropy of p_theta on masked positions.
- [ ] **gradient_surgery** -- PCGrad: compute RL grad and BC grad separately; project RL grad to remove component conflicting with BC grad (if dot < 0). Applied per-parameter tensor.
- [ ] **advantage_clip** -- ELBO with advantages clamped to `[1 - adv_clip_eps, 1 + adv_clip_eps]`. `adv_clip_eps=0.2`.
- [ ] **normalized_adv** -- ELBO with advantages normalised: `(A - mean) / (std + 1e-8)` (GRPO-style group normalisation).
- [ ] **bc_wins** -- Uniform ELBO (advantages=None) on win-only windows. `wins_only=True` flag filters to windows with return > `win_threshold=0.5`.
- [ ] **low_t** -- Return-weighted ELBO restricted to `t in [eps, t_max_low=0.2]`.

### Group C: Architecture / Parameter Isolation

All use baseline loss; isolation is via optimizer frozen-param masks.

- [ ] **frozen_backbone** -- Freeze all except action output head. Frozen: embeddings, CNNs, global stream, goal head, action/timestep/position embeddings, transformer.
- [ ] **head_only** -- Identical to frozen_backbone (freeze all except head).
- [ ] **attention_only** -- Freeze everything except attention sublayers (self_attn.*). FFN, norms, head, all encoders frozen.
- [ ] **ffn_only** -- Freeze everything except FFN sublayers (linear1.*, linear2.*). Attention, norms, head, all encoders frozen.
- [ ] **layer_ablation_top1** -- Freeze all except top-1 transformer block + head.
- [ ] **layer_ablation_top2** -- Freeze all except top-2 transformer blocks + head.
- [ ] **layer_ablation_top3** -- Freeze all except top-3 transformer blocks + head.

### Group D: Reward / Data Quality

All use baseline loss; data transforms applied externally in training loop.

- [ ] **reward_filtering** -- Train only on windows with return >= 75th percentile of batch. `reward_filter_percentile=75`.
- [ ] **running_stats** -- EMA running mean/std for advantage normalisation. `running_stats_ema_decay=0.99`. Advantages = `((clipped - ema_mean) / ema_std + 1.0).clamp(floor, cap)`.
- [ ] **action_diversity** -- Discard degenerate plans where all actions are identical (all-same-action filter).
- [ ] **reward_model** -- Train a lightweight MLP reward model on collected returns; re-weight advantages with its predictions. MLP: `reward_model_width=64`, `reward_model_depth=2`, `reward_model_lr=1e-3`, `reward_model_train_steps=50` per iter. Input: flattened observation features.

---

## 2. Loss Variants (`ablations/losses.py`)

### Core mechanism

All losses share `_core_loss`:

1. Sample continuous `t ~ Uniform[t_min, t_max]` per sample. Default `[eps, 1.0]`.
2. Forward-mask: `zt = q_sample(x0, t, mask_token, pad_token, schedule_fn)`.
3. Convert to discrete timestep: `t_discrete = (t * T).long().clamp(0, T-1)`.
4. Model forward: `out = model(obs, zt, t_discrete)` -> logits `[B,H,V]`, goal_pred `[B,2]`.
5. Per-sample masked cross-entropy on positions where `zt == mask_token` and `x0 != pad_token`.
6. If advantages provided: `loss = (per_sample * advantages).mean()`, else `per_sample.mean()`.
7. Add auxiliary loss: `total = loss + aux_loss_weight * aux_loss`.

### Loss-specific additions

- [ ] **make_loss_baseline** -- Pure `_core_loss`. No additions.
- [ ] **make_loss_kl_penalty** -- `_core_loss + kl_coef * KL(current || ref)`. KL computed on masked positions via `log_softmax` difference. `kl_coef=0.1`.
- [ ] **make_loss_ewc** -- `_core_loss + ewc_lambda * sum(F_i * (theta_i - theta_i*)^2)`. Fisher diagonal pre-computed by squaring gradients of baseline loss over held-out batches. `ewc_lambda=100.0`.
- [ ] **make_loss_trust_region_kl** -- `_core_loss + 1e4 * max(KL - threshold, 0)^2`. `threshold=0.05`.
- [ ] **make_loss_mixed_replay** -- Delegates to baseline (batching handled externally).
- [ ] **make_loss_bc_wins** -- `_core_loss` with `advantages=None` (ignores advantages).
- [ ] **make_loss_low_t** -- `_core_loss` with `t_max=0.2`.
- [ ] **make_loss_t_curriculum** -- `_core_loss` with iteration-dependent `[t_min, t_max]`. At iter i: `frac = min(i/steps, 1)`, `t_min = t_start - frac*(t_start - eps)`, `t_max = 1.0 - frac*(1.0 - t_end)`. Min gap enforced: `t_max >= t_min + 0.05`.
- [ ] **make_loss_entropy_bonus** -- `_core_loss - entropy_coef * H(p_theta)`. Entropy = `-sum(p * log(p))` over action dim, averaged over masked positions. `entropy_coef=0.01`.
- [ ] **make_loss_gradient_surgery** -- Delegates to baseline (PCGrad handled externally in training loop).
- [ ] **make_loss_advantage_clip** -- `_core_loss` with `advantages.clamp(1-eps, 1+eps)`. `eps=0.2`.
- [ ] **make_loss_normalized_adv** -- `_core_loss` with `advantages = (A - mean) / (std + 1e-8)`.
- [ ] **make_loss_frozen_backbone** -- Delegates to baseline.
- [ ] **make_loss_param_isolation** -- Delegates to baseline.
- [ ] **make_loss_reward_quality** -- Delegates to baseline.

### Fisher diagonal estimation

- [ ] `estimate_fisher_diagonal`: For each batch, compute baseline loss, backward, accumulate `grad^2` per parameter. Average over N batches. Returns `dict[param_name -> Fisher tensor]`.

---

## 3. Optimizer Variants (`ablations/optimizers.py`)

### Standard

- [ ] **make_optimizer_standard** -- AdamW over all `requires_grad` params. `lr=3e-4`, `weight_decay=1e-4`, `eps=1e-5`.

### LLRD (Layer-wise Learning Rate Decay)

- [ ] **make_optimizer_llrd** -- AdamW with per-group LRs. Groups: `head` (base_lr), `block_{i}` (base_lr * decay^(N-i)), `obs_enc` (base_lr * decay^(N+1)). `llrd_decay=0.9`, 4 layers default.

### Frozen parameters

- [ ] **make_optimizer_frozen** -- Set `requires_grad=False` for params matching any substring in `frozen_fragments`. AdamW on remaining trainable params.
- [ ] **FROZEN_BACKBONE** -- Freeze: embeddings, CNNs, global stream, goal head, action/timestep/position embeddings, transformer.
- [ ] **FROZEN_EXCEPT_ATTENTION** -- Freeze everything except self_attn.*
- [ ] **FROZEN_EXCEPT_FFN** -- Freeze everything except linear1.*, linear2.*
- [ ] **FROZEN_EXCEPT_LAST_LAYER** -- Freeze everything except last transformer layer + head.
- [ ] **_layer_ablation_top_n_opt(n)** -- Factory: freeze all transformer layers except top n.

### LoRA

- [ ] **_LoRAParametrization** -- Low-rank: `W_eff = W + (alpha/rank) * B @ A`. A: Gaussian(0.02), B: zeros. Applied via `torch.nn.utils.parametrize`.
- [ ] **apply_lora_to_model** -- Targets: `self_attn.in_proj_weight` (fused QKV) and `self_attn.out_proj.weight` per transformer layer. All base params frozen; only LoRA A/B trainable.
- [ ] **remove_lora_from_model** -- Bakes LoRA deltas into weights permanently.
- [ ] **make_optimizer_lora** -- AdamW on LoRA params only.

### Gradient surgery (PCGrad)

- [ ] **collect_gradients** -- Snapshot `.grad` for all params into a dict.
- [ ] **apply_gradients** -- Write dict back into `.grad` attributes.
- [ ] **gradient_surgery** -- Per-parameter: if `dot(g_rl, g_bc) < 0`, project `g_rl` onto plane orthogonal to `g_bc`: `g_rl - (dot/||g_bc||^2) * g_bc`. Otherwise keep unchanged.

---

## 4. Training Loop (`ablations/training.py`)

### One iteration of `run_ablation`:

1. **Collect episodes**: Run EMA model in eval mode for `episodes_per_iter=30` episodes across training envs. GPU-batched on CUDA (stochastic ReMDM sampling, `diffusion_steps_collect=5`), sequential fallback on CPU.
2. **Extract windows**: Sliding windows of length `seq_len` from each episode. Short episodes padded with `pad_token`. Each window inherits the episode's total return.
3. **Action diversity filter** (if enabled): Discard windows where all action tokens are identical.
4. **Reward filtering** (if enabled): Keep only windows with return >= `reward_filter_percentile`th percentile.
5. **Push to mixed replay buffer** (if enabled): Ring buffer push.
6. **Reward model training** (if enabled): Train MLP on (flattened_obs -> return) for `reward_model_train_steps` steps, then replace returns with model predictions.
7. **Compute advantages**: `clipped = max(returns, 0)`. Default: `weights = clipped / (batch_mean + eps)`, clamped to `[return_weight_floor=0.1, return_weight_cap=5.0]`. Variants: `wins_only` -> binary; `running_stats` -> EMA normalisation.
8. **Shuffle and batch**: Random permutation, take first `batch_size` samples.
9. **Mixed replay splice** (if enabled): Replace `mixed_replay_ratio=0.25` of batch with samples from replay buffer.
10. **Gradient step(s)** (`grad_steps_per_iter=1`):
    - If gradient_surgery: compute RL loss backward, snapshot RL grads; compute BC loss backward, snapshot BC grads; project RL grads via PCGrad; apply projected grads; clip and step.
    - Otherwise: standard forward-backward-clip-step.
    - AMP (`torch.amp.autocast` + `GradScaler`) supported.
    - Gradient clipping: `max_grad_norm=1.0`.
11. **EMA update**: After each gradient step.
12. **Diagnostics** (at configured frequencies): gradient alignment, per-layer norms, KL drift, CKA, t-bin analysis.
13. **Evaluation** (every `eval_every=25` iters): Run EMA model on all ID envs, `eval_episodes=20` per env. Score = mean ID win rate.

### Effective batch size metric

- [ ] `(sum(w))^2 / sum(w^2)` where w = advantage weights. Measures how many samples effectively contribute.

### MixedReplayBuffer

- [ ] Fixed-size ring buffer storing `(local_obs, global_obs, x0, returns)`. FIFO eviction. Uniform sampling with replacement.

### RewardModel

- [ ] MLP: `Linear(obs_dim, width) -> ReLU -> ... -> Linear(width, 1)`. Trained via MSE on collected returns.

### Checkpoint loading

- [ ] Load pretrained via `ema_state_dict` from DAgger checkpoint.
- [ ] Deepcopy frozen reference model for KL/drift diagnostics.
- [ ] Seed `torch`, `numpy`, `random` before each run.

---

## 5. Diagnostics (`diagnostics/`)

### Gradient diagnostics (`gradient.py`)

- [ ] **compute_grad_alignment** -- Cosine similarity between RL and BC gradient vectors. RL grad = backward of advantage-weighted loss. BC grad = backward of unweighted loss. Returns `(cos_sim, rl_norm, bc_norm)`. Frequency: `grad_align_every=25`.
- [ ] **compute_per_layer_grad_norms** -- L2 norm of `.grad` per named parameter. Frequency: `per_layer_every=25`.
- [ ] **compute_surgery_metrics** -- Gradient mass fraction removed by PCGrad: `(||g_before||^2 - ||g_after||^2) / ||g_before||^2`. Also counts number of conflicting parameters.

### Representation diagnostics (`representation.py`)

- [ ] **compute_repr_drift** -- KL(ref || current) at 4 t-ranges: full `[eps, 1.0]`, low `[eps, 0.2]`, mid `[0.3, 0.7]`, high `[0.8, 1.0]`. Direction: KL(ref || cur). Frequency: `repr_drift_every=25`.
- [ ] **compute_cka** -- Linear CKA between current and reference model output representations (mean-pooled logits). `CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)` using centred kernels. t sampled in `[0.3, 0.7]`. `cka_batch_size=64`. Frequency: `cka_every=50`.
- [ ] **compute_activation_norms** -- Mean, std, p50, p90 of per-sample logit L2 norms. t in `[0.3, 0.7]`.

### Timestep diagnostics (`timestep.py`)

- [ ] **compute_t_analysis** -- Partition `[0, 1]` into `n_bins=10` equal bins. For each bin, compute RL loss restricted to that t-range, backward, measure gradient L2 norm. Also compute cosine similarity between low-t `[eps, 0.2]` and high-t `[0.8, 1.0]` gradient vectors. Frequency: `t_analysis_every=25`.
- [ ] **compute_t_bin_losses** -- Mean loss per t-bin (no gradients). `n_bins=10`.

---

## 6. Analysis Pipeline (`analysis/`)

### Plots (`plots.py`) -- 9 figure generators

- [ ] **plot_training_curve** -- Per-ablation: loss (EMA-smoothed) and ID win rate over iterations. One plot per ablation.
- [ ] **plot_score_comparison** -- Bar chart of final scores across all ablations. Pretrained baseline as dashed horizontal line.
- [ ] **plot_grad_alignment** -- Cosine similarity (RL vs BC) curves for all ablations over training.
- [ ] **plot_repr_drift** -- 4-panel: KL drift at full, low-t, mid-t, high-t ranges over training.
- [ ] **plot_cka** -- CKA similarity vs pretrained curves for all ablations.
- [ ] **plot_t_bin_norms** -- Heatmap: rows = ablations, cols = t-bins, values = gradient L2 norms at final iteration.
- [ ] **plot_t_ratio** -- High-t / low-t gradient norm ratio over training. Ratio >> 1 indicates t-bias.
- [ ] **plot_win_rate** -- Online win rate (EMA-smoothed) for all ablations.
- [ ] **plot_group_comparison** -- Boxplot of final scores grouped by ablation category (Baseline, A, B, C, D).

Style: colorblind-safe (Wong 2011), EMA smoothing (alpha=0.3), unique color+linestyle per ablation within group, pretrained baseline as dashed line.

### Tables (`tables.py`) -- 5 summary tables (CSV + LaTeX)

- [ ] **make_main_results_table** -- Method | Group | Score | Delta_Pretrained | Delta_Baseline | Verdict. Verdict: IMPROVEMENT (>+0.05), COLLAPSE (<-0.1), NEUTRAL.
- [ ] **make_group_summary_table** -- Group | N | Mean | Best | Worst | StdDev.
- [ ] **make_gradient_diagnostics_table** -- Method | Cos_Sim | RL_Norm | BC_Norm (final values).
- [ ] **make_repr_drift_table** -- Method | KL_mean | KL_low_t | KL_mid_t | KL_high_t (final values).
- [ ] **make_per_env_table** -- Method + per-environment win rates at final eval.

### Diagnosis report (`report.py`)

- [ ] **generate_diagnosis_report** -- Produces `diagnosis.md` + `diagnosis_decision_tree.png`.

Six hypothesis groups, each with supporting ablations:

| Hypothesis | Supporting Ablations |
|---|---|
| Catastrophic Forgetting | kl_penalty, ewc, llrd, frozen_backbone, head_only |
| Gradient Conflict | gradient_surgery, kl_penalty, low_t |
| Signal Sparsity | bc_wins, reward_filtering, running_stats, reward_model |
| Distributional Shift | mixed_replay, action_diversity |
| Mode Collapse | entropy_bonus, advantage_clip, normalized_adv |
| t-Bias | low_t, t_curriculum |

Scoring: an ablation "supports" a hypothesis if its score exceeds `max(pretrained_score, baseline_score) + 0.01`. Evidence score = `n_supporting / n_tested`.

Aggregate verdict logic:
- If ALL RL ablations score < pretrained - 0.1 -> "ALL collapse, self-generated data is root cause".
- Gradient alignment interpretation: `< -0.01` = actively wrong, `|x| < 0.05` = noise, else = useful signal.

Decision tree: horizontal bar chart of evidence scores (0-1), coloured by RdYlGn.

### Action distribution analysis (`action_distribution.py`)

- [ ] **collect_action_statistics** -- Roll out model, collect per-action counts, episode returns/lengths/wins.
- [ ] **compute_all_metrics** -- Entropy, KL(post||pre), KL(pre||post), JS divergence, TV distance, effective actions (>1% prob), Gini coefficient, mode probability, win rate, mean return.
- [ ] **run_statistical_tests** -- Chi-squared on action counts (with pseudocount), Mann-Whitney U on episode returns. p < 0.05 = significant.
- [ ] **action_transitions** -- Row-normalised P(next | current) transition matrix.
- [ ] **interpret_results** -- JS < 0.05 = representation drift (not behavioural), 0.05-0.15 = mixed, >= 0.15 = mode collapse.
- [ ] 6 diagnostic plots: side-by-side action bars, delta+log-ratio bars, 2x2 metrics dashboard, episode histograms, cumulative distribution curve (80%/95% thresholds), transition matrix heatmaps (pre, post, diff).

### Mixing experiment (`mixing_experiment.py`)

- [ ] **run_mixing_experiment** -- Tests performance degradation across oracle/self-generated data mixing ratios: `[1.0, 0.9, 0.7, 0.5, 0.0]`.
- [ ] 100% oracle = pretrained eval (known endpoint). 0% = pure RL (no oracle data). Intermediate points trained with `MixedReplayBuffer` pre-filled with oracle fraction.
- [ ] Training per mixing point: `max_iter=500`, collect self-generated episode each iteration, gradient step with baseline loss, periodic eval.
- [ ] **check_monotonicity** -- Verify win rates are non-increasing as oracle fraction decreases.
- [ ] 3-panel plot: degradation curve (win rate vs oracle fraction), ID win rate over training per fraction, final bar chart.

---

## 7. Config Schema (`configs/`)

### `ablations_default.yaml` -- all hyperparameters with defaults

| Key | Default | Description |
|---|---|---|
| `max_iter` | 500 | Iterations per ablation |
| `batch_size` | 3072 | Gradient batch size |
| `lr` | 3e-4 | Learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `grad_steps_per_iter` | 1 | Gradient steps per iteration |
| `episodes_per_iter` | 30 | Episodes collected per iteration |
| `diffusion_steps_collect` | 5 | Denoising steps during collection |
| `eval_every` | 25 | Evaluation frequency |
| `eval_episodes` | 20 | Episodes per env during eval |
| `grad_align_every` | 25 | Gradient alignment diagnostic frequency |
| `repr_drift_every` | 25 | KL drift diagnostic frequency |
| `t_analysis_every` | 25 | t-bin analysis frequency |
| `cka_every` | 50 | CKA diagnostic frequency |
| `per_layer_every` | 25 | Per-layer gradient norm frequency |
| `t_analysis_n_bins` | 10 | Number of t bins |
| `cka_batch_size` | 64 | CKA batch size |
| `win_threshold` | 0.5 | Minimum return for win |
| `return_weight_floor` | 0.1 | Advantage lower clip |
| `return_weight_cap` | 5.0 | Advantage upper clip |
| `ewc_lambda` | 100.0 | EWC penalty coefficient |
| `ewc_fisher_batches` | 20 | Fisher estimation batches |
| `llrd_decay` | 0.9 | LLRD decay factor |
| `lora_rank` | 8 | LoRA rank |
| `lora_alpha` | 16.0 | LoRA alpha |
| `mixed_replay_ratio` | 0.25 | Offline fraction of batch |
| `mixed_replay_buffer_size` | 10000 | Replay buffer capacity |
| `trust_region_kl` | 0.05 | KL trust region threshold |
| `kl_coef` | 0.1 | Soft KL penalty coef |
| `t_curriculum_start` | 0.8 | t-curriculum start |
| `t_curriculum_end` | 0.2 | t-curriculum end |
| `t_curriculum_steps` | 200 | t-curriculum annealing steps |
| `t_max_low` | 0.2 | Low-t upper bound |
| `entropy_coef` | 0.01 | Entropy bonus coef |
| `adv_clip_eps` | 0.2 | Advantage clip epsilon |
| `reward_filter_percentile` | 75 | Reward filter percentile |
| `running_stats_ema_decay` | 0.99 | Running stats EMA decay |
| `reward_model_width` | 64 | Reward model hidden width |
| `reward_model_depth` | 2 | Reward model hidden layers |
| `reward_model_lr` | 1e-3 | Reward model learning rate |
| `reward_model_train_steps` | 50 | Reward model steps per iter |
| `ablation_use_wandb` | true | W&B logging |
| `ablation_wandb_project` | "remdm-minihack-ablations" | W&B project name |
| `num_seeds` | 1 | Seeds per ablation |

### `ablations_fast.yaml` -- overrides for smoke testing

| Key | Fast Value | Default |
|---|---|---|
| `max_iter` | 50 | 500 |
| `batch_size` | 128 | 3072 |
| `episodes_per_iter` | 2 | 30 |
| `eval_every` | 10 | 25 |
| `eval_episodes` | 5 | 20 |
| `grad_align_every` | 10 | 25 |
| `repr_drift_every` | 10 | 25 |
| `t_analysis_every` | 10 | 25 |
| `cka_every` | 25 | 50 |
| `per_layer_every` | 10 | 25 |
| `ewc_fisher_batches` | 5 | 20 |
| `reward_model_train_steps` | 10 | 50 |
| `mixed_replay_buffer_size` | 500 | 10000 |

### `ablations_qmul_gpu.yaml` -- GPU cluster overrides

Key changes: `use_amp=true`, `batch_size=1024`, `episodes_per_iter=20`, `diffusion_steps_collect=3`, `eval_every=50`, `eval_episodes=10`, all diagnostics at 50-iter intervals, `cka_every=100`, `cka_batch_size=32`.

### `ablations_ucl_gpu.yaml`

Identical to default (full copy).

---

## 8. CLI Entry Point (`run_ablations.py`)

### Modes

- [ ] **Training** -- `--checkpoint PATH --all` or `--ablations NAME [NAME ...]`. Loads pretrained checkpoint, evaluates pretrained baseline, runs selected ablations sequentially, saves incremental results + per-ablation model checkpoints. Generates plots/tables/report after each ablation.
- [ ] **Fast** -- `--fast` overlays `ablations_fast.yaml`.
- [ ] **List** -- `--list` prints all registered ablations and exits.
- [ ] **Analyze only** -- `--analyze_only --results_path PATH` loads existing results, regenerates analysis. Supports `--ablations` to filter subset.
- [ ] **Merge** -- `--merge FILE [FILE ...]` merges results.json files from independent GPU runs. Same-ablation scores are concatenated, mean/std recomputed. Regenerates analysis.
- [ ] **W&B artifact** -- `--checkpoint wandb://entity/project/artifact:version` resolves W&B artifact to local path.
- [ ] **W&B resumption** -- `--wandb_resume_id ID` resumes an existing W&B run for curve continuity.

### Config precedence

`defaults.yaml` -> `ablations_default.yaml` -> `ablations_fast.yaml` (if --fast) -> CLI overrides.

### Multi-seed

Each ablation runs `num_seeds` times (default 1). Seeds: `base_seed + seed_idx * 1000`. Mean and std reported. wandb_step_offset monotonically increases across ablations/seeds.

### Result serialisation

JSON via `orjson` with `OPT_INDENT_2`. Structure: `{pretrained_score, config, ablations: {name: {score, score_std, all_scores, history}}}`. AblationHistory serialised via `to_dict()`/`from_dict()`.
