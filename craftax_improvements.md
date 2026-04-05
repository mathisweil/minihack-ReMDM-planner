# Craftax RL Fine-Tuning Ablation Suite -- Improvements Over Spec

This document records where the Craftax implementation is genuinely better
than the environment-agnostic spec, and what (if anything) should be
backported to the MiniHack version.

---

## 1. JIT-Compatible t-Curriculum Loss

**What's different:**
`make_loss_t_curriculum_jit` accepts `step_idx` as a JAX array argument
rather than closing over a mutable Python list. This allows the t-curriculum
loss to run inside `jax.lax.scan` without recompilation.

**Why it's better:**
The mutable-list approach (`make_loss_t_curriculum`) forces the training
loop to run in Python, preventing full JIT compilation. The JIT-compatible
variant enables the entire training loop to be a single compiled scan.

**Backport recommendation:**
Add a PyTorch equivalent by accepting `step_idx` as a tensor argument
instead of using a `current_iter` container. The mutable-list version can
remain as a convenience wrapper for non-compiled call sites.

---

## 2. Full `jax.lax.scan` Training Loop

**What's different:**
`make_run_ablation` returns a closure where the entire training loop
(rollout, filtering, advantage computation, gradient step, diagnostics,
eval) runs inside a single `jax.lax.scan`. There are zero Python-level
iteration loops. The carry (`AblationCarry`) tracks all mutable state as
a NamedTuple of JAX arrays.

**Why it's better:**
- Single JIT compilation for the entire ablation (no per-iteration
  recompilation overhead).
- XLA can fuse operations across iterations.
- Eliminates host-device transfer overhead between iterations.
- Enables `jax.vmap` over seeds for free parallelism.

**Backport recommendation:**
Not directly portable to PyTorch, but the design pattern (pre-allocate
all state, express the loop as a functional scan) is worth documenting.
PyTorch users could adopt `torch.compile` with similar carry-state patterns.

---

## 3. Pure JAX Ring Buffer for Mixed Replay

**What's different:**
`ReplayBuffer` is a `NamedTuple` of pre-allocated JAX arrays with
`_push_to_buffer` and `_sample_from_buffer` implemented as pure
functions. The buffer lives in the `AblationCarry` and participates
in the JIT-compiled scan.

**Why it's better:**
No Python-level data structures (lists, deques) cross the JIT boundary.
The buffer has O(1) push and O(1) sample (with replacement) via
index-scatter and `jax.random.randint`.

**Backport recommendation:**
The PyTorch version can use pre-allocated tensors with a write pointer
(same algorithm). The key insight is avoiding Python collections inside
the training loop.

---

## 4. Conditional Diagnostics via `jax.lax.cond`

**What's different:**
Each diagnostic (grad alignment, repr drift, CKA, t-analysis, per-layer
norms, eval) is wrapped in `jax.lax.cond(step_idx % frequency == 0, ...)`.
The "did run" flags are recorded in `StepMetrics` and used during
post-JIT history extraction (`metrics_to_history`).

**Why it's better:**
Diagnostics run conditionally *inside* the compiled loop without
breaking the scan. The alternative -- Python-level `if` statements --
would require exiting JIT every N iterations.

**Backport recommendation:**
PyTorch equivalent: wrap diagnostics in `if iteration % freq == 0`
(already the standard approach). The key Craftax insight is structuring
diagnostics as pure functions that return fixed-shape arrays (padding
with zeros when not run), so the scan output shape is uniform.

---

## 5. Achievement Tracking (Craftax-Specific)

**What's different:**
`AblationHistory` includes `per_achievement_rates` (list of dicts mapping
achievement name to unlock rate at each eval checkpoint). Two dedicated
plots are generated:
- `plot_achievement_breakdown`: stacked bar (start vs end) per ablation
- `plot_achievement_collapse_heatmap`: rows=achievements, cols=eval iters

An `make_achievement_table` in `tables.py` reports per-achievement final
unlock rates with delta-vs-pretrained columns.

**Why it's better:**
Craftax has 22 distinct achievements. A "neutral" overall score can mask
shifts in *which* achievements are unlocked. The achievement breakdown
reveals whether RL fine-tuning trades one capability for another.

**Backport recommendation:**
MiniHack has per-task success rates; similar per-task tracking could be
added. The stacked bar and heatmap visualisations are generic.

---

## 6. Extra Analysis Features

### Forgetting Analysis Table
`make_forgetting_analysis_table` tracks: first collapse iteration, minimum
score, recovery score, and whether the ablation recovered. This timeline
view is not in the spec.

### Hypothesis Verdict Table
`make_hypothesis_verdict_table` maps each ablation to its hypothesis and
renders IMPROVEMENT/COLLAPSE/NEUTRAL verdicts with one-line conclusions.

### Gradient Conflict Map
`plot_gradient_conflict_map` is a binary heatmap (rows=ablations,
cols=diagnostic steps) showing where `cos_sim(RL, BC) < 0`. This gives a
quick visual of conflict hotspots across the suite.

### Score Delta Plot
`plot_score_delta` shows improvement over baseline-RL as a sorted bar
chart, making it easy to rank ablations by marginal value.

**Backport recommendation:**
All four are environment-agnostic and should be added to the MiniHack
suite.

---

## Remaining Gaps (Not Implemented in Craftax)

The following spec features are absent from the Craftax implementation.
They are noted here for completeness but were not addressed during
alignment:

- `analysis/action_distribution.py` -- full action-distribution analysis
  module (6 diagnostic plots, JS/KL/TV metrics, transition matrices)
- `analysis/mixing_experiment.py` -- oracle/self-generated mixing ratio
  sweep experiment
- `--merge` CLI mode for combining results from independent GPU runs
- `--wandb_resume_id` for W&B run continuation
- `remove_lora_from_model` (bake LoRA deltas into weights permanently)
- EMA model update after each gradient step (spec uses EMA for eval;
  Craftax evaluates current params directly)
