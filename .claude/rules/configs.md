---
paths: configs/**
---

# Config Rules — configs/

---

## Config file inventory

| File | Purpose | When to use |
|---|---|---|
| `defaults.yaml` | Base hyperparameters for all modes | Always loaded first |
| `smoke.yaml` | Fast smoke-test overrides (~30 s on CPU) | `--mode smoke` |
| `main.yaml` | Full training run (inherits defaults) | `--mode dagger` full run |

## Loading order and precedence

Config is applied in this order — later entries override earlier ones:

1. `configs/defaults.yaml` (base)
2. Mode-specific override file (`smoke.yaml`, `main.yaml`) if applicable
3. CLI `key=value` pairs

- **MUST** implement this precedence in `src/config.py`. Do not reverse or skip any step.
- **MUST NOT** make a parameter config-file-only. Every key **MUST** be overridable from the CLI via `key=value` syntax.

## Adding new hyperparameters

- **MUST** add every new hyperparameter to `defaults.yaml` with a safe default that preserves backward-compatible behaviour.
- **MUST** add a comment explaining the parameter's meaning and valid range:
  ```yaml
  # Remasking strength. Higher = more stochastic revision of committed tokens.
  # Range: [0, 1]. 0 disables remasking entirely. Default: 0.15.
  eta: 0.15
  ```
- **MUST NOT** add training-mode-specific parameters to `defaults.yaml` without ensuring they are safely ignored when not in the relevant mode.
- **SHOULD** group related parameters with a YAML comment header:
  ```yaml
  # ── Diffusion sampling ───────────────────────────────────────────────────
  diffusion_steps_eval: 5
  remask_strategy: conf
  eta: 0.15
  temperature: 0.5
  top_k: 4
  ```

## Naming conventions

- Use `snake_case` for all keys.
- Boolean flags: use bare `true` / `false` (not `True`, `False`, `yes`, `no`).
- Paths: use relative paths from the project root. Absolute paths **MUST NOT** be committed.
- Scientific notation is acceptable for very small/large floats: `offline_lr: 3e-4`.

## Preset / override files

Preset files (`smoke.yaml`, `main.yaml`) contain **only** keys that differ from `defaults.yaml`:

- **MUST** document what each preset changes and why, in a comment at the top of that file.
- **MUST NOT** copy-paste the entire `defaults.yaml` into a preset. Override only what changes.
- **MUST** ensure smoke test overrides keep the run to ≤ 30 iterations and a small buffer so it completes on CPU in ~30 seconds.

## Key hyperparameters reference

The following values are load-bearing. Changing them is a significant experiment decision, not a routine edit:

| Key | Default | Why it matters |
|---|---|---|
| `mask_token` | 12 | Global token vocabulary — changing breaks all saved datasets |
| `pad_token` | 13 | Global token vocabulary — same |
| `seq_len` | 64 | Plan horizon — changes model input/output shapes |
| `n_embd` | 256 | Transformer hidden dim — changes model parameter count |
| `global_gate_init` | -3.0 | Controls how aggressively global stream is used early in training |
| `efficiency_multiplier` | 1.5 | Controls DAgger data quality vs. coverage tradeoff |
| `replan_every` | 16 | Controls how often the model re-runs diffusion during evaluation |

If any of these are changed, **MUST** document the change in `README.md` and verify that `--mode smoke` still passes.
