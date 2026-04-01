# Phase 2: Mapping Craftax -> MiniHack

---

## 2.1 Model Interface Differences

### Architecture Comparison

| Aspect | Craftax (Flax) | MiniHack (PyTorch) |
|---|---|---|
| Framework | JAX/Flax (`@nn.compact`) | PyTorch (`nn.Module`) |
| Forward signature | `apply_fn(params, obs, z_t, t, rng) -> logits [B,H,V]` | `model(local_obs, global_obs, action_seq, t_discrete) -> {"actions": [B,H,12], "goal_pred": [B,2]}` |
| Obs input | Single flat `obs [B, obs_dim]` | Dual: `local_obs [B,9,9]` int + `global_obs [B,21,79]` int |
| Action vocab | `num_actions` (variable) | 12 actions + MASK(12) + PAD(13) = 14 in embedding, 12 in output |
| Plan horizon | 40 | 128 (cfg.seq_len) |
| Hidden dim | 256 | 256 |
| Heads | 4 | 4 |
| Layers | 4 | 4 |
| Diffusion steps | 50 (eval) | 100 (training), 10 (eval) |
| Aux output | None | `goal_pred [B,2]` (staircase coordinates) |
| Parameters | Functional pytree | `nn.Module.state_dict()` |
| EMA | Not in Craftax ablation suite | `ModelEMA` class in `denoiser.py` |

### Module Naming Convention in MiniHack

From `LocalDiffusionPlannerWithGlobal`, the PyTorch named parameters follow this naming:

```
# Local stream
embedding.weight                           # Embedding(6000, 64)
cnn.0.weight, cnn.0.bias                   # Conv2d(64, 32, 3)
cnn.2.weight, cnn.2.bias                   # Conv2d(32, 64, 3)
cnn.5.weight, cnn.5.bias                   # Linear(64*9*9, 256)

# Action stream
action_emb.weight                          # Embedding(14, 256)
timestep_emb.weight                        # Embedding(100, 256)
pos_emb.weight                             # Embedding(128, 256)

# Transformer (4 layers, PyTorch TransformerEncoder)
transformer.layers.{0-3}.self_attn.in_proj_weight    # [768, 256] (Q+K+V fused)
transformer.layers.{0-3}.self_attn.in_proj_bias      # [768]
transformer.layers.{0-3}.self_attn.out_proj.weight    # [256, 256]
transformer.layers.{0-3}.self_attn.out_proj.bias      # [256]
transformer.layers.{0-3}.linear1.weight               # [1024, 256] (FFN up)
transformer.layers.{0-3}.linear1.bias                  # [1024]
transformer.layers.{0-3}.linear2.weight               # [256, 1024] (FFN down)
transformer.layers.{0-3}.linear2.bias                  # [256]
transformer.layers.{0-3}.norm1.weight                  # [256] (pre-LN for attn)
transformer.layers.{0-3}.norm1.bias                    # [256]
transformer.layers.{0-3}.norm2.weight                  # [256] (pre-LN for FFN)
transformer.layers.{0-3}.norm2.bias                    # [256]

# Output head
head.weight                                # Linear(256, 12)
head.bias                                  # [12]

# Global stream
global_embedding.weight                    # Embedding(6000, 32)
global_cnn.0.weight, global_cnn.0.bias     # Conv2d(32, 32, 5)
global_cnn.2.weight, global_cnn.2.bias     # Conv2d(32, 64, 3)
global_proj.weight, global_proj.bias       # Linear(64, 256)
global_gate                                # scalar Parameter

# Goal head
goal_head.0.weight, goal_head.0.bias       # Linear(256, 128)
goal_head.2.weight, goal_head.2.bias       # Linear(128, 2)
```

### Craftax -> MiniHack Module Name Mapping

| Craftax Path Fragment | MiniHack Parameter Name Pattern | Used By |
|---|---|---|
| `TransformerBlock_{i}` | `transformer.layers.{i}` | LLRD, layer ablation, per-layer diagnostics |
| `MultiHeadDotProductAttention` | `transformer.layers.{i}.self_attn` | LoRA, attention_only |
| `Dense_0`, `Dense_1` (inside block FFN) | `transformer.layers.{i}.linear1`, `transformer.layers.{i}.linear2` | ffn_only |
| `LayerNorm_` | `transformer.layers.{i}.norm1`, `transformer.layers.{i}.norm2` | frozen_backbone |
| `SinusoidalPosEmbed_` | `timestep_emb`, `pos_emb` | frozen_backbone |
| `Embed_` | `embedding`, `action_emb`, `global_embedding` | head_only, frozen_backbone |
| Final Dense (head) | `head` | head_only output |

### Goal Head Interaction with Ablations

Decision needed for each ablation:

| Ablation | Goal head treatment | Rationale |
|---|---|---|
| `ewc` | **Protect**: include in Fisher diagonal | Goal head helps learn global-map features; losing it degrades OOD |
| `lora` | **Do NOT adapt**: freeze goal head | LoRA targets attention only; goal head is a 2-layer MLP, not attention |
| `frozen_backbone` | **Trainable**: goal head is NOT backbone | Goal head should remain trainable when backbone is frozen |
| `head_only` | **Trainable**: include goal head with action head | Both are "heads" (output projections) |
| `attention_only` | **Frozen** | Goal head is not attention |
| `ffn_only` | **Frozen** | Goal head is not FFN |
| `layer_ablation_*` | **Frozen** | These ablations only unfreeze transformer blocks |
| All loss ablations | **Preserve** `aux_loss_weight * goal_loss` | Always add auxiliary loss unless explicitly testing without it |

---

## 2.2 Diffusion Interface Differences

### Base Loss Interface

| Aspect | Craftax | MiniHack |
|---|---|---|
| Function | `compute_loss(apply_fn, params, rng, acts, obs, valid, num_actions, schedule_fn, schedule_deriv_fn, sigma_t, label_smoothing, advantages, t_min, t_max) -> (loss, aux)` | `mdlm_loss(logits, x0, zt, t, mask_token, pad_token, schedule_fn, weight_clip, label_smoothing, use_importance_weighting) -> loss` |
| t sampling | Inside `compute_loss` | Outside: caller samples `t`, runs `q_sample`, then `model(...)`, then `mdlm_loss(...)` |
| Advantage weighting | Built into `compute_loss` via `advantages` param | **Not built in** -- must be applied externally by multiplying per-sample loss |
| PAD handling | Via `valid` mask passed to `compute_loss` | Built into `mdlm_loss`: `(zt == mask_token) & (x0 != pad_token)` |
| Importance weighting | Via `schedule_deriv_fn` | Via `use_importance_weighting` flag |
| Auxiliary goal loss | Not in diffusion loss | Separate `auxiliary_goal_loss()` function |

### Key Adaptation: Wrapping MiniHack's Loss for Ablations

The ablation `_core_loss` must:
1. Sample `t` uniformly in `[t_min, t_max]` (caller controls range for low_t, t_curriculum).
2. Run `q_sample(x0, t, mask_token, pad_token, schedule_fn)` to get `zt`.
3. Convert `t` to `t_discrete = (t * num_diffusion_steps).long().clamp(0, num_diffusion_steps - 1)`.
4. Run `model(local_obs, global_obs, zt, t_discrete)` to get `logits` and `goal_pred`.
5. Compute `mdlm_loss(logits, x0, zt, t, ...)` to get per-position cross-entropy.
6. For advantage-weighted variants: compute per-sample loss (not global average), multiply by advantages, then average.
7. Add `aux_loss_weight * auxiliary_goal_loss(goal_pred, global_obs)`.

**Critical**: MiniHack's `mdlm_loss` currently returns a scalar global average. For advantage weighting, we need a **per-sample** loss variant. This means the ablation loss wrapper must replicate the masked cross-entropy logic with per-sample reduction, then apply advantage weights. We can do this without modifying `src/diffusion/loss.py` by implementing the per-sample variant in the ablation loss module.

### Sampling Interface

| Aspect | Craftax | MiniHack |
|---|---|---|
| Eval sampling | `sample_plan()` inside `build_eval_fn` (JAX, vmapped) | `remdm_sample()` in `src/diffusion/sampling.py` (PyTorch, batch loop) |
| DAgger sampling | N/A (uses PPO) | `greedy_sample()` in `src/diffusion/sampling.py` |
| Steps | `val_diffusion_steps=50` | `diffusion_steps_eval=10` |
| Remasking | rescale, eta=0.5 | conf, eta=0.15 |

For the ablation suite evaluation, we reuse MiniHack's `remdm_sample()` via `run_model_episode()` with `stochastic=True`.

---

## 2.3 Environment & Rollout Differences

### Environment Comparison

| Aspect | Craftax | MiniHack |
|---|---|---|
| Interface | Gymnax (functional, vmapped, JIT) | Gym (Python objects, single-threaded) |
| Parallelism | `jax.vmap` over `num_envs=64` | Sequential or `ThreadPoolExecutor` |
| Obs format | Flat vector `[obs_dim]` | Tuple: `(local [9,9], global [21,79])` as int16 arrays |
| Action space | Variable | 12 discrete actions |
| Reward | Sparse returns | Shaped: win(+20), BFS progress, exploration, step penalty |
| Achievements | Yes (Craftax-specific) | No -- binary win/loss per episode |
| Environments | 1 (Craftax) | 7 (4 ID + 3 OOD) |

### Evaluation Protocol Mapping

| Craftax | MiniHack |
|---|---|
| Single env, `eval_steps=512` env steps | 7 envs, `eval_episodes_per_env=50` episodes each |
| Returns `returned_episode_returns` (scalar) | Returns `{env_id: {win_rate, avg_steps, avg_reward}}` |
| Per-achievement unlock rates | Per-environment win rates |

For the ablation suite, the "score" used for verdict determination should be the **mean ID win rate** (average across 4 ID environments). OOD win rate is tracked separately.

### Rollout Collection for RL Fine-Tuning

**Craftax**: PPO agent collects trajectories. The diffusion model is trained on PPO-collected data with return-weighted ELBO. The PPO agent is a separate network loaded from a checkpoint.

**MiniHack**: The diffusion model itself collects trajectories (via `run_model_episode`), and a BFS oracle provides the training signal via DAgger.

**Adaptation decision**: For the RL fine-tuning ablation suite in MiniHack, we collect rollouts using the **diffusion model itself** (via `run_model_episode`), compute **shaped returns** from the environment's reward signal, and apply return-weighted ELBO. The oracle is NOT used for data collection in the ablation suite -- we're testing how to fine-tune with RL signal, not DAgger signal.

Specifically:
1. Run the diffusion model (EMA weights) on an environment for one episode.
2. Record the shaped reward at each step.
3. Compute windowed returns (sum of rewards over each seq_len window).
4. Use these returns as advantages for the return-weighted ELBO.

This matches the Craftax pattern: PPO generates data, returns weight the loss. The difference is that in MiniHack, the policy IS the diffusion model, not a separate PPO agent.

---

## 2.4 Training Infrastructure Differences

### Training Loop Comparison

| Aspect | Craftax | MiniHack (Existing DAgger) | MiniHack (Ablation Suite) |
|---|---|---|---|
| Loop type | `jax.lax.scan` (compiled) | Python `for` loop | Python `for` loop (PyTorch) |
| Optimizer | `optax` (functional) | `torch.optim.AdamW` | `torch.optim.AdamW` + custom wrappers |
| Gradient step | `state.apply_gradients(grads=g)` | `loss.backward(); optimizer.step()` | Same PyTorch pattern |
| Data source | PPO rollouts | DAgger (model + oracle) | Model rollouts with shaped returns |
| EMA | Not in ablation suite | `ModelEMA` updated every grad step | Reuse `ModelEMA` |

### What to Reuse vs Wrap vs Replace

| Component | Decision | Rationale |
|---|---|---|
| `src/models/denoiser.py` | **Reuse as-is** | Import model class, no modifications |
| `src/diffusion/loss.py` | **Wrap** | Need per-sample loss variant for advantage weighting; implement in ablation code |
| `src/diffusion/forward.py` | **Reuse** `q_sample` directly | Same interface works |
| `src/diffusion/sampling.py` | **Reuse** `remdm_sample`, `greedy_sample` | For eval rollouts |
| `src/diffusion/schedules.py` | **Reuse** `get_schedule`, `alpha_prime` | Same interface |
| `src/envs/minihack_env.py` | **Reuse** `make_env`, `collect_oracle_trajectory` | For env setup and oracle baseline |
| `src/planners/collect.py` | **Reuse** `run_model_episode` | For rollout collection |
| `src/planners/inference.py` | **Reuse** `Evaluator` class | For ID+OOD evaluation |
| `src/buffer.py` | **Replace** | Ablation suite needs its own simpler buffer for rollout windows |
| `src/curriculum.py` | **Not used** | Ablation suite samples envs uniformly (or uses all 4 ID envs) |
| `src/config.py` | **Reuse** `load_config` for loading main defaults | Ablation config loaded separately |
| `src/planners/logging.py` | **Wrap** | Ablation suite has its own logging; can optionally emit to the existing Logger |
| `ModelEMA` | **Reuse** from `denoiser.py` | Updated after each grad step in ablation training |

### EMA Interaction with LoRA, Frozen Layers

- **LoRA**: EMA should track the LoRA parameters only (base weights are frozen). Implement by calling `ema.update(model)` which iterates `named_parameters()` -- LoRA params will have `requires_grad=True` and base params `requires_grad=False`. However, `ModelEMA` copies ALL parameters, not just trainable ones. For LoRA: create EMA of the full model (base + LoRA delta). At eval time, apply EMA weights and the LoRA delta is baked in.
- **Frozen layers**: EMA still tracks all parameters. Since frozen params don't change, EMA shadow = original values. No special handling needed.

---

## 2.5 Config System Differences

### Craftax Config System

- YAML loaded as flat dict, all keys uppercased (`LR`, `MAX_ITER`, etc.).
- Ablation configs are self-contained (don't inherit from main config).
- CLI overrides via explicit `argparse` arguments.

### MiniHack Config System

- YAML loaded via `src/config.py`, deep-merged, returned as `SimpleNamespace`.
- Keys are lowercase (`lr`, `max_iterations`, etc.).
- CLI overrides via `key=value` pairs.

### Ablation Config Integration

The ablation configs will live in `experiments/rl_finetuning/configs/` and:
1. Load MiniHack's `configs/defaults.yaml` as the base (for model architecture, env IDs, token IDs, etc.).
2. Deep-merge ablation-specific overrides on top.
3. Return a `SimpleNamespace` (matching MiniHack convention, NOT uppercased).

Key mapping from Craftax ablation config to MiniHack:

| Craftax Key | MiniHack Key | Default Value |
|---|---|---|
| `MAX_ITER` | `max_iter` | 1000 |
| `NUM_ENVS` | (not applicable -- MiniHack is single-threaded) | -- |
| `NUM_STEPS` | (not applicable) | -- |
| `BATCH_SIZE` | `batch_size` | 512 |
| `LR` | `lr` | 3e-4 |
| `MAX_GRAD_NORM` | `max_grad_norm` | 1.0 |
| `EVAL_EVERY` | `eval_every` | 50 |
| `PLAN_HORIZON` | `seq_len` | 128 |
| `N_LAYERS` | `n_layer` | 4 |
| `D_MODEL` | `n_embd` | 256 |
| All ablation-specific keys | Same names, lowercase | Same defaults |

---

## 2.6 Metric & Logging Differences

### Metric Mapping

| Craftax Metric | MiniHack Equivalent | Notes |
|---|---|---|
| `returned_episode_returns` (scalar) | `mean_id_win_rate` (float, 0-1) | Aggregated across 4 ID envs |
| Per-achievement unlock rates | Per-environment win rates | 7 environments instead of N achievements |
| `env_score` (online rollout score) | `model_win_rate` during collection | Binary win/loss per episode |
| -- | `mean_ood_win_rate` | New: OOD generalisation metric |
| -- | Per-env `avg_steps`, `avg_reward` | MiniHack-specific metrics |

### AblationHistory Adaptation

Replace Craftax-specific fields:

| Craftax Field | MiniHack Field |
|---|---|
| `env_score` / `env_score_iters` | `model_win_rate` / `model_wr_iters` |
| `eval_score` / `eval_iters` | `id_win_rate`, `ood_win_rate` / `eval_iters` |
| `per_achievement_rates` | `per_env_win_rates: list[dict[str, float]]` (7 envs) |

### W&B Namespaces

Ablation suite metrics will be namespaced under `ablations/{ablation_name}/`:

```
ablations/{name}/train_loss
ablations/{name}/train_loss_diff
ablations/{name}/train_loss_aux
ablations/{name}/model_win_rate
ablations/{name}/id_win_rate
ablations/{name}/ood_win_rate
ablations/{name}/grad_align_cos_sim
ablations/{name}/kl_drift
ablations/{name}/cka_similarity
```

This avoids collisions with the main training pipeline's namespaces (`diffusion/`, `train/`, `eval_id/`, `eval_ood/`).

### Figure Adaptations

| Craftax Figure | MiniHack Adaptation |
|---|---|
| Achievement breakdown (stacked bar) | **Per-environment win rate breakdown**: stacked or grouped bar showing win rates for each of the 7 envs at start vs end of training |
| Achievement collapse heatmap | **Per-environment win rate heatmap**: rows=7 envs, cols=eval iterations, colour=win rate. Shows which environments degrade during fine-tuning |
| -- (new) | **Per-environment heatmap over training**: rows=7 envs, cols=ablations, colour=final win rate. New summary view |

---

## Design Decisions Requiring Confirmation

Before proceeding to Phase 3, the following decisions need your input:

### 1. Rollout Collection Strategy

**Proposed**: The diffusion model itself collects rollouts (via `run_model_episode`), and we compute shaped returns from the environment rewards. No oracle involved.

**Alternative**: Use DAgger-style collection where both model and oracle run, but weight loss by return instead of using oracle actions.

**Recommendation**: Use the model's own rollouts with shaped returns. This tests RL fine-tuning in the purest sense.

### 2. Checkpoint Baseline

**Proposed**: Use a DAgger-trained checkpoint as the pretrained baseline (not offline BC only). The DAgger checkpoint represents the best pre-RL model.

**Alternative**: Use offline BC checkpoint only.

**Recommendation**: DAgger checkpoint -- it's the more realistic starting point for RL fine-tuning.

### 3. Goal Head in RL Fine-Tuning

**Proposed**: Always preserve `aux_loss_weight * goal_loss` in every ablation loss (unless specifically ablating it). The goal head helps the model utilise the global map.

### 4. "Score" for Verdict Determination

**Proposed**: Use mean ID win rate across the 4 ID environments as the primary score. This is the closest analogue to Craftax's `returned_episode_returns`.

### 5. Number of Episodes Per Iteration

**Proposed**: Collect `episodes_per_iteration` (default 10, from main config) rollout episodes per ablation iteration, compute returns, train on them. This matches the DAgger collection cadence.

### 6. EMA During Ablation Training

**Proposed**: Maintain EMA and update after every gradient step, same as DAgger training. Use EMA weights for evaluation and rollout collection.
