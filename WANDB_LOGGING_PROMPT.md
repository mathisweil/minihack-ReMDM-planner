# WandB Logging Audit & Implementation

Make wandb logging comprehensive across both codebases. Every metric that matters for debugging a training run or writing a paper should be logged. Nothing useless.

## Scope

Two codebases to cover:

1. **`src/`** — the main DAgger training pipeline (`trainer.py`, `evaluator.py`, `collector.py`, `utils.py`)
2. **`experiments/rl_finetuning/`** — the 25-ablation RL fine-tuning suite

## Phase 1: Audit what's currently logged

For each codebase, list every `wandb.log()` call that exists today. Note:
- What key is logged
- What value it contains
- At what frequency (every iter, every N iters, every eval)
- Whether `step=` is set correctly

Then list everything that is **computed but not logged** — metrics that exist as local variables or print statements but never reach wandb.

## Phase 2: Plan what should be logged

Design the logging schema for each codebase. Use these categories:

### Training dynamics (every iteration or every N iterations)
- `train/loss` — total loss
- `train/loss_diffusion` — CE component
- `train/loss_aux` — auxiliary goal loss (if applicable)
- `train/loss_kl` — KL penalty component (if applicable)
- `train/loss_ewc` — EWC penalty component (if applicable)
- `train/learning_rate` — current LR (captures scheduler effects)
- `train/grad_norm` — global gradient norm **before** clipping
- `train/grad_norm_clipped` — global gradient norm **after** clipping (shows how often clipping activates)
- `train/buffer_size` — current replay buffer size
- `train/buffer_online_frac` — fraction of buffer that is online data (for DAgger)
- `train/episodes_added` — number of new trajectories added this iteration

### Training speed (every iteration)
- `speed/iter_time_sec` — wall-clock time for one full iteration (collect + train)
- `speed/collect_time_sec` — wall-clock time for episode collection only
- `speed/train_step_time_sec` — wall-clock time for gradient steps only
- `speed/samples_per_sec` — training samples processed per second
- `speed/env_steps_per_sec` — environment steps per second during collection
- `speed/gpu_memory_mb` — peak GPU memory usage (`torch.cuda.max_memory_allocated`)
- `speed/gpu_util_pct` — GPU utilisation if available (skip if not trivially queryable)

### Online rollout quality (every iteration)
- `online/win_rate` — rolling win rate from collection episodes
- `online/mean_return` — rolling mean episode return
- `online/mean_steps` — rolling mean episode length
- `online/curriculum_weights` — per-env sampling weights (as a wandb.Table or dict)

### Evaluation (every eval interval)
- `eval/id_win_rate` — mean win rate across ID envs
- `eval/ood_win_rate` — mean win rate across OOD envs
- `eval/per_env/{env_name}/win_rate` — per-environment win rate
- `eval/per_env/{env_name}/avg_steps` — per-environment average steps
- `eval/per_env/{env_name}/avg_reward` — per-environment average reward

### Model health (every N iterations, not every iter — these are expensive)
- `model/ema_gate_value` — the global gate sigmoid value (architecture-specific, important for this model)
- `model/param_norm` — total parameter L2 norm
- `model/param_drift_from_init` — L2 distance from initial/pretrained weights

### Diagnostics — ablation suite only (at configured intervals)
- `diag/grad_alignment_cos` — cosine similarity between RL and BC gradients
- `diag/grad_alignment_rl_norm` — RL gradient norm
- `diag/grad_alignment_bc_norm` — BC gradient norm
- `diag/repr_drift_kl` — KL divergence from pretrained
- `diag/cka_similarity` — CKA similarity to pretrained (if computed)
- `diag/t_grad_norm_low` — gradient norm for low-t range
- `diag/t_grad_norm_high` — gradient norm for high-t range
- `diag/t_grad_cos_lohi` — cosine similarity between low-t and high-t gradients
- `diag/surgery_frac` — fraction of gradient projections in PCGrad (if applicable)

### Run metadata (logged once at start via `wandb.config`)
- All hyperparameters from config
- Model parameter count (total and trainable)
- Buffer capacity
- ID/OOD env lists
- Git commit hash if available
- Ablation name and group (for experiments)

## Phase 3: Implement

### Rules
- **Guard every wandb call** with `if wandb.run is not None:` — wandb must be optional, never crash if not initialised
- **Use `step=iteration`** consistently — never rely on wandb's auto-incrementing step counter
- **Don't log large objects every iteration** — tables, histograms, and images should be logged at eval intervals only
- **Use `wandb.define_metric()`** at init to set proper x-axes (e.g., all `train/*` metrics use `iteration` as x-axis)
- **Time measurements**: use `time.perf_counter()`, not `time.time()`. Wrap the minimal code span — don't time logging itself.
- **GPU memory**: call `torch.cuda.reset_peak_memory_stats()` at the start of each iteration, read `torch.cuda.max_memory_allocated()` at the end
- **Grad norm**: capture the return value of `clip_grad_norm_()` — it returns the total norm before clipping. Then compute the norm again after clipping if needed, or just log the pre-clip value and the clip threshold.

### For `src/trainer.py`
- Add timing around `run_iteration()` (total), `collector.collect_episode()` (collect), and the gradient step loop (train)
- Log all training dynamics and speed metrics
- Log evaluation metrics at checkpoint/eval intervals
- Log curriculum weights when they change
- Make sure `wandb.init()` is called in `run()` with the full config, and `wandb.finish()` at the end

### For `experiments/rl_finetuning/`
- Add timing around each iteration in the training loop
- Log all training dynamics, speed, online rollout, and diagnostic metrics
- Each ablation run should be a separate wandb run with `name=ablation_name, group=run_id`
- Log the ablation spec (name, group, description) to `wandb.config`
- At the end of each ablation, log a `wandb.summary` with final scores

### Don't log
- Raw observation arrays or action sequences (too large, not useful)
- Per-sample losses (just the mean)
- Individual episode returns (just the rolling stats)
- Anything that's only useful for debugging and would be a `print()` statement

## Phase 4: Verify

After implementing, do a quick check:
1. `grep -rn "wandb.log" src/ experiments/` — list all log calls, verify step= is set
2. `grep -rn "wandb" src/ experiments/ | grep -v "wandb.run is not None" | grep -v "import wandb" | grep -v "#"` — find any unguarded wandb calls
3. Confirm both codebases still run without wandb installed (the guards should handle it)
