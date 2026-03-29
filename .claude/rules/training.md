---
paths: src/planners/**
---

# Training Pipeline Rules — src/planners/

> Also load @.claude/rules/python-idioms.md when editing this directory.

---

## Module responsibilities — do not blur these

| File | Mode | Responsibility |
|---|---|---|
| `train.py` | `offline` | Offline BC trainer: dataset → batches → gradient steps |
| `online.py` | `dagger` | DAgger loop: collect → filter → train → checkpoint |
| `inference.py` | `inference` | Evaluator: rollouts → win rate / steps / reward |
| `collect.py` | shared | `run_model_episode`, `DataCollector` (model + oracle rollouts) |
| `logging.py` | shared | All W&B + stdout metric logging |

- **MUST NOT** add a new training mode by appending logic to an existing mode file. Each mode gets its own file or a cleanly separated branch in `main.py`.
- **MUST NOT** duplicate environment construction logic. Always use `AdvancedObservationEnv` from `src/envs/minihack_env.py`.
- **MUST NOT** duplicate checkpoint loading. Always use the checkpoint utilities in `online.py`.
- **MUST NOT** call `wandb.log(...)` directly from any file in this directory. All metric emission goes through `logging.py`.

---

## W&B logging conventions

Metric namespaces — **MUST NOT** rename or merge without updating `logging.py` docstring and `README.md`:

| Namespace | Contents |
|---|---|
| `diffusion/` | `loss`, `aux_loss`, `total_loss` |
| `train/` | `buffer_size`, `model_win_rate`, `oracle_add_rate` |
| `eval_id/{env}/` | `win_rate`, `avg_steps` (in-distribution) |
| `eval_ood/{env}/` | `win_rate`, `avg_steps` (out-of-distribution) |

- `train/` metrics are only logged in modes that perform live environment interaction (`dagger`). Not in `inference`.
- Logging **MUST** gracefully no-op when W&B is not initialised — never crash if `wandb.run is None`.

---

## Offline BC (`train.py`)

- Optimiser: AdamW with `lr = cfg.offline_lr` (default `3e-4`).
- LR schedule: cosine decay from `offline_lr` to `10%` of initial LR over `cfg.offline_epochs` epochs. **MUST** preserve the 10% floor — do not decay to zero.
- EMA is updated after each gradient step (not each epoch).
- Batch size: `cfg.offline_batch_size` (default 256). Shuffle each epoch.
- **MUST** apply gradient clipping if it is present in the reference implementation.

---

## DAgger loop (`online.py`)

Each iteration follows this exact order — **MUST NOT** reorder steps:

1. **Curriculum sampling:** select env weighted by `DynamicCurriculum` difficulty.
2. **Model rollout:** `DataCollector.collect` runs EMA model on a seeded env.
3. **Oracle rollout:** BFS oracle runs on the **same seed**.
4. **Efficiency filter:** `efficiency_filter(model_steps, oracle_steps, model_won)` — add oracle trajectory to buffer only if model failed or `model_steps > cfg.efficiency_multiplier * oracle_steps`.
5. **Training:** `cfg.grad_steps_per_iteration` gradient steps from the replay buffer.
6. **EMA update:** blend live model weights into shadow copy after each gradient step.
7. **Logging:** emit `train/` metrics.
8. **Periodic eval:** ID eval every `cfg.id_eval_every` iterations; OOD eval every `cfg.ood_eval_every` iterations.
9. **Checkpointing:** save every `cfg.checkpoint_every` iterations.

### Checkpointing

Checkpoint dict **MUST** contain exactly these keys:

```python
{
    "model_state_dict":     model.state_dict(),
    "ema_state_dict":       ema.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "curriculum_state":     curriculum.state_dict(),
    "iteration":            int,
    "rng_states": {
        "torch":   torch.get_rng_state(),
        "numpy":   numpy.random.get_state(),
        "python":  random.getstate(),
    },
}
```

- **MUST** save and restore all three RNG states for deterministic resume.
- **MUST** save and restore `curriculum_state` — losing curriculum queues corrupts difficulty weighting.
- **MUST** use `torch.save` / `torch.load` with `weights_only=False` (until a migration to safetensors is decided).

### DAgger LR

- Optimiser: AdamW with `lr = cfg.dagger_lr` (default `3e-5`).
- If a cosine schedule is added, document it here and in `README.md`. The reference uses a constant rate.

---

## Data collection (`collect.py`)

- **MUST** set the environment seed identically before both model and oracle rollouts. The efficiency filter is only meaningful if both see the same episode.
  ```python
  env.reset(seed=seed)   # model rollout
  env.reset(seed=seed)   # oracle rollout — same seed
  ```
- **MUST** label trajectories with **oracle actions**, not model actions, when adding to the replay buffer.
- Trajectory windows of length `H = seq_len = 64`. Shorter episodes are padded with `PAD_TOKEN` (13).
- **MUST** exclude PAD-padded positions from the loss (handled downstream in `loss.py`, but verify windows are correctly constructed here).
- `run_model_episode` replans every `cfg.replan_every` steps (default 16). Only `seq[0]` (the first token) of each plan is executed.

---

## Evaluation (`inference.py`)

- **MUST** use EMA weights by default. Pass `use_ema=False` only when `--no-ema` is set.
- **MUST NOT** modify the replay buffer or curriculum during evaluation.
- **MUST** use `torch.no_grad()` for all inference forward passes.
- OOD environments **MUST NOT** be added to the curriculum or replay buffer, even when evaluated.
- Results dict **MUST** include `win_rate`, `avg_steps`, and `avg_reward` per environment, plus aggregated `id_win_rate` and `ood_win_rate`.

---

## Replay buffer (`src/buffer.py`)

- Offline (BC) data is pinned at the front of the buffer and **MUST NOT** be FIFO-evicted.
- Only online DAgger samples are subject to FIFO eviction when `buffer_capacity` is exceeded.
- Sampling is uniform over the full buffer (offline + online).
- **MUST** track the offline/online split boundary explicitly — do not infer it from insertion order alone.

---

## Curriculum (`src/curriculum.py`)

- `DynamicCurriculum` maintains a rolling win-rate queue of size `cfg.curriculum_queue_size` (default 100) per environment.
- Bucket weights: low `[0, 0.3)` → 0.2; medium `[0.3, 0.7)` → 1.0; high `[0.7, 1.0]` → 0.1.
- Initialised with a 50/50 prior so early sampling is roughly uniform across environments.
- The 4 in-distribution environments are the only entries. **MUST NOT** add OOD environments to the curriculum.
- `state_dict()` / `load_state_dict()` **MUST** correctly serialise and restore all per-environment queues.
