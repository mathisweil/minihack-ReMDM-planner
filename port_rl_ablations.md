# Task: Port the RL Fine-Tuning Ablation Suite from Craftax to MiniHack

## Context

I have two projects that both use a **ReMDM discrete diffusion planner** (Remasking Discrete Diffusion Model). The Craftax project has a fully implemented RL fine-tuning ablation suite at `reference/craftax_rl_finetuning/`. The MiniHack project does **not** have this yet. Your job is to deeply understand the Craftax ablation suite, then recreate it for MiniHack — adapting to the MiniHack project's architecture, environment interface, model, and training pipeline.

The reference Craftax code lives at `reference/craftax_rl_finetuning/` (temporary — will be deleted after this task). The MiniHack project is the main codebase in this repo.

---

## Phase 1: Deep Analysis of the Craftax Ablation Suite

Before writing any code, read and analyze **every single file** in `reference/craftax_rl_finetuning/`. For each file, document:

### 1.1 — Registry & Ablation Definitions (`ablations/registry.py`)
- What is the `AblationSpec` dataclass schema? What fields does each ablation define?
- List all 25 ablations. For each one, note: the group it belongs to (A/B/C/D), what hypothesis it tests, what it modifies (loss, optimizer, architecture, data), and the key hyperparameters.
- How does the registry pattern work — how are ablations looked up and instantiated?

### 1.2 — Loss Variants (`ablations/losses.py`)
- What is the base loss function signature? What inputs does it expect (logits, targets, timesteps, masks, advantages, etc.)?
- For each loss variant (kl_penalty, ewc, trust_region_kl, entropy_bonus, bc_wins, low_t, advantage_clip, normalized_adv, reward_filtering, etc.), document:
  - Exactly what mathematical objective it implements
  - How it wraps or modifies the base ELBO loss
  - What additional state it needs (e.g., Fisher diagonal for EWC, reference model for KL)
  - The factory function signature and return type

### 1.3 — Optimizer Variants (`ablations/optimizers.py`)
- How does LLRD (layer-wise learning rate decay) identify and group layers?
- How is LoRA implemented — which modules get adapted, what rank, how are weights merged/unmerged?
- How does gradient surgery (PCGrad) work — what are the two gradient sources, how is projection done?
- How do parameter masking ablations work (frozen_backbone, head_only, attention_only, ffn_only, layer_ablation_top1/2/3)?

### 1.4 — Training Loop (`ablations/training.py`)
- What does `make_run_ablation()` return? What is its signature?
- How does the training loop differ from standard DAgger? What hooks exist for:
  - Custom loss computation
  - Custom optimizer setup
  - Diagnostic metric collection
  - Evaluation scheduling
- What is the `AblationHistory` dataclass? What metrics does it track over time?
- How are rollouts collected — does it reuse the existing project's rollout infrastructure or have its own?
- How does it handle multi-seed runs?

### 1.5 — Diagnostics (`diagnostics/`)
- **gradient.py**: How is gradient alignment (cosine similarity) computed? Between which two gradients? How are per-layer norms extracted? What is the gradient surgery metric?
- **representation.py**: How is KL drift from pretrained computed? What activations are compared for CKA similarity? How are activation norms tracked?
- **timestep.py**: How are t-bins defined? How are per-t gradient norms computed? What does per-t loss decomposition measure?
- For each diagnostic: what model hooks or forward passes are needed? What is the collection frequency?

### 1.6 — Analysis & Reporting (`analysis/`)
- **plots.py**: List every figure generated. For each, note what data it requires and what it visualizes.
- **tables.py**: List every table generated. Note the columns and what they summarize.
- **report.py**: How is `diagnosis.md` generated? What is the decision tree logic? What verdicts can it reach?

### 1.7 — Configs (`configs/`)
- What hyperparameters does `ablations_default.yaml` define? What are the key knobs?
- How does `ablations_fast.yaml` differ (for smoke testing)?
- How do ablation configs interact with the main project config system?

### 1.8 — CLI Entry Point (`run_ablations.py`)
- What CLI arguments are supported?
- How does `--analyze_only` reloading work?
- How does incremental `results.json` writing work?
- How is W&B integration handled?

**Write up your full analysis as `experiments/rl_finetuning/ANALYSIS.md` before proceeding to Phase 2.** I will review this before you start coding.

---

## Phase 2: Mapping Craftax → MiniHack

After the analysis, identify every adaptation point. Create a mapping document at `experiments/rl_finetuning/ADAPTATION_MAP.md` covering:

### 2.1 — Model Interface Differences
- The MiniHack model is `LocalDiffusionPlannerWithGlobal` (~5.2M params) in `src/models/denoiser.py`. It takes `(local_obs, global_obs, noisy_action_seq, t_discrete)` and returns `{"actions": [B,192,12], "goal_pred": [B,2]}`.
- Map every point where the Craftax ablation code touches model internals (layer names, module paths, output format) to the MiniHack equivalents.
- Identify the transformer layer naming convention in `denoiser.py` — this is critical for LLRD, LoRA, parameter masking, per-layer diagnostics, and layer ablations.
- The MiniHack model has auxiliary goal prediction (`goal_pred`). Decide how this interacts with each ablation (should EWC protect goal head weights? Should LoRA adapt goal head? etc.).

### 2.2 — Diffusion Interface Differences
- MiniHack diffusion code lives in `src/diffusion/` (schedules.py, forward.py, loss.py, sampling.py).
- Map the Craftax loss function interface to MiniHack's MDLM loss. Identify: what the base loss expects, how masked positions are handled, how PAD tokens are excluded, how importance weighting works.
- Map the sampling interface — MiniHack has both stochastic ReMDM sampling (eval) and greedy argmax sampling (DAgger collection).

### 2.3 — Environment & Rollout Differences
- MiniHack environments are in `src/envs/minihack_env.py` with BFS oracle, shaped rewards, and AdvancedObservationEnv wrapper.
- MiniHack uses 7 environments (4 ID + 3 OOD). Map the Craftax evaluation protocol to MiniHack's environment set.
- MiniHack has `run_model_episode` and `DataCollector` in `src/planners/collect.py`. Identify how the ablation suite should collect rollout data.

### 2.4 — Training Infrastructure Differences
- MiniHack training: offline BC in `src/planners/offline.py`, DAgger in `src/planners/online.py`, buffer in `src/buffer.py`, curriculum in `src/curriculum.py`.
- Identify which MiniHack training components the ablation suite should reuse vs. wrap vs. replace.
- MiniHack uses EMA weights (updated every gradient step). How does this interact with LoRA, frozen layers, etc.?

### 2.5 — Config System Differences
- MiniHack configs are in `configs/` with YAML loading + CLI overrides via `src/config.py`.
- Design how ablation configs will integrate (separate yaml files under `experiments/rl_finetuning/configs/`).

### 2.6 — Metric & Logging Differences
- MiniHack logs to W&B under specific namespaces (`diffusion/`, `train/`, `eval_id/`, `eval_ood/`).
- MiniHack tracks win rate and avg steps per environment. Map Craftax metrics to MiniHack equivalents. Note: MiniHack doesn't have "achievements" — it has binary win/loss per episode plus shaped reward components. Adapt the achievement_breakdown and achievement_collapse figures to something meaningful for MiniHack (e.g., per-environment win rate breakdown, or reward component analysis).

---

## Phase 3: Implementation

Recreate the full ablation suite under `experiments/rl_finetuning/` with the **same directory structure** as the Craftax version. The code should:

### Ground Rules
- **Import from `src/` — never modify `src/`.** The ablation suite is standalone research code.
- Reuse MiniHack infrastructure wherever possible (model, diffusion, envs, buffer, config, logging).
- Maintain the same ablation registry pattern, the same 25 ablations (adapted), the same diagnostic metrics, the same output structure.
- Every file should have clear docstrings explaining what it does and how it differs from the Craftax version.

### 3.1 — `ablations/registry.py`
- Reproduce the `AblationSpec` dataclass and the full `REGISTRY` of 25 ablations, adapted for MiniHack's model and diffusion setup.

### 3.2 — `ablations/losses.py`
- Implement all loss variants. The base loss wraps MiniHack's MDLM loss from `src/diffusion/loss.py`.
- EWC needs Fisher diagonal computation using MiniHack's loss interface.
- KL penalty and trust region need a frozen reference model (clone of pretrained MiniHack model).
- Entropy bonus operates on the 12-class action logits (MiniHack has 12 actions, not whatever Craftax has).
- `low_t` restricts to MiniHack's timestep range (0–99 for 100 discrete steps).
- Handle the auxiliary goal loss: ablation losses should preserve `aux_loss_weight * goal_loss` unless specifically testing without it.

### 3.3 — `ablations/optimizers.py`
- LLRD: identify transformer blocks in `LocalDiffusionPlannerWithGlobal` by name. Group: embeddings, CNN streams, transformer layers 0–3, output head.
- LoRA: target attention projections (Q/K/V/O) in the 4-layer transformer. Use the same rank as Craftax.
- Parameter masking: `frozen_backbone` = freeze everything except output head. `head_only` = only the final Linear(256,12). `attention_only` = Q/K/V/O projections. `ffn_only` = feedforward layers. `layer_ablation_top1/2/3` = top N of the 4 transformer blocks.
- Gradient surgery: RL gradient vs BC/diffusion gradient, projected via PCGrad.

### 3.4 — `ablations/training.py`
- `make_run_ablation()` factory that takes an `AblationSpec`, the pretrained MiniHack checkpoint, and config, and returns a callable that runs the full ablation.
- The training loop should:
  - Load pretrained weights from the provided offline/DAgger checkpoint.
  - Collect rollouts using MiniHack's `DataCollector` with the model's policy.
  - Compute RL objectives (return-weighted ELBO as baseline).
  - Apply the ablation-specific loss, optimizer, and data modifications.
  - Collect all diagnostic metrics at the specified frequencies.
  - Run eval on MiniHack ID+OOD environments at `eval_every` intervals.
  - Track everything in `AblationHistory`.
- Support multi-seed runs.
- Write incremental `results.json` after each ablation.

### 3.5 — `diagnostics/`
- Port all three diagnostic modules. Adapt model hook points to MiniHack's `LocalDiffusionPlannerWithGlobal` architecture.
- KL drift: compare current model's output distribution to the frozen pretrained model on a held-out batch.
- CKA: hook into transformer layer outputs.
- Per-layer gradient norms: iterate `model.named_parameters()` with MiniHack's parameter naming.
- t-bin analysis: use MiniHack's 100 discrete timesteps, bin into ~10 bins.

### 3.6 — `analysis/`
- Port all plot generators. Replace any Craftax-specific metrics (achievements) with MiniHack equivalents (per-environment win rates for 7 envs, avg episode steps, shaped reward components).
- Port all table generators. Adapt columns for MiniHack metrics.
- Port the diagnosis report generator. Adapt the decision tree logic.
- Add a new figure: per-environment heatmap showing win rate over training for each of the 7 MiniHack environments, per ablation.

### 3.7 — Configs
- `configs/ablations_default.yaml`: full-run hyperparameters for MiniHack (adapt iteration counts, eval frequencies, buffer sizes to MiniHack's scale).
- `configs/ablations_fast.yaml`: smoke-test config (~50 iterations, small buffer, 2 envs only).
- Configs should inherit from or reference the main MiniHack `configs/defaults.yaml` for shared params.

### 3.8 — `run_ablations.py`
- CLI entry point with the same interface as Craftax's.
- Arguments: `--ablations`, `--all`, `--fast`, `--list`, `--analyze_only`, `--results_path`, `--offline_checkpoint_path`, `--ppo_checkpoint_path` (rename to `--dagger_checkpoint_path` if more appropriate for MiniHack), `--num_seeds`, `--use_wandb`, `--config`, `--ablations_config`.
- Handle MiniHack's checkpoint format (see main README for schema).

### 3.9 — Output Structure
Reproduce the same output structure:
```
experiments/rl_finetuning/outputs/{run_id}/
├── results.json
├── diagnosis.md
├── figures/    (all adapted plots)
└── tables/     (all adapted tables as csv + tex)
```

---

## Phase 4: Verification

After implementation:
1. Run the fast config with 2 ablations (`baseline_rl`, `kl_penalty`) and verify:
   - No import errors, no crashes
   - `results.json` is written and valid
   - At least one figure and one table are generated
   - `diagnosis.md` is produced
2. Run `--list` and confirm all 25 ablations are registered.
3. Run `--analyze_only` on the results from step 1 and confirm re-plotting works.
4. Verify that `src/` has not been modified (check with `git diff src/`).

---

## Important Notes

- **Do NOT blindly copy-paste from Craftax.** The Craftax code uses a different model, different env interface, different action space, and possibly a different framework (JAX vs PyTorch). Every line needs to be understood and adapted.
- **The MiniHack model has 4 transformer layers, 256D hidden, 4 heads, 12 action classes, 192-step sequence length, dual-stream (local + global) architecture with an auxiliary goal head.** These specifics must be reflected everywhere.
- **MiniHack uses PyTorch.** If the Craftax version uses JAX/Flax, all code must be rewritten in PyTorch idiom.
- **Ask me before making any design decisions that aren't clearly determined by the Craftax reference or MiniHack architecture.** For example: how to handle the goal head in RL fine-tuning, whether to use DAgger checkpoints or offline BC checkpoints as the pretrained baseline, etc.
