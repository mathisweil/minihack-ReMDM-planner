---
paths: src/**
---

# Python / PyTorch Idioms — src/

> Loaded for any file under `src/`. Also load the more specific rule file for your subdirectory.

---

## Tensor discipline

Tensor shape bugs are the most common and hardest-to-debug error class in this codebase. Prevent them up front.

- **MUST** add a shape comment on every tensor that is not self-evident. Use the `[B, ...]` convention with uppercase dimension names.
  ```python
  logits = model(local_obs, global_obs, seq, t)["actions"]  # [B, H, A]
  ```
- **MUST** assert shapes at the entry point of any function called from outside its own module:
  ```python
  assert local_obs.shape[-2:] == (9, 9), f"Expected (9,9) crop, got {local_obs.shape}"
  ```
- **SHOULD** name batch dimensions consistently across the codebase:
  - `B` — minibatch size
  - `H` — plan horizon / sequence length (= `seq_len = 64`)
  - `D` — feature / hidden dimension
  - `A` — action vocabulary size (= `num_actions = 12`; MASK and PAD are inputs only)

## Device discipline

- **MUST** pass `cfg.device` through to every tensor creation. Never hardcode `"cuda"` or `"cpu"`.
  ```python
  # CORRECT
  seq = torch.full((B, cfg.seq_len), cfg.mask_token, dtype=torch.long, device=cfg.device)

  # WRONG
  seq = torch.full((B, 64), 12, dtype=torch.long, device="cuda")
  ```
- **MUST** call `.to(cfg.device)` on model parameters and all input tensors before the first forward pass.
- **SHOULD** use `torch.no_grad()` context for all inference and evaluation code. Never rely on the caller to handle this.

## Gradient hygiene

- **MUST** call `optimizer.zero_grad()` before every backward pass.
- **MUST** clip gradients if the reference implementation does: `torch.nn.utils.clip_grad_norm_(params, max_norm)`.
- **MUST NOT** call `.item()` inside a training loop on quantities that are used in further computation — it breaks the autograd graph.
- **SHOULD** use `torch.no_grad()` or `@torch.inference_mode()` for EMA updates and evaluation, to avoid unnecessary graph construction.

## Randomness and reproducibility

- **MUST** thread seeds explicitly. Set `torch.manual_seed`, `numpy.random.seed`, and `random.seed` together at the start of smoke tests and evaluation runs.
- **MUST NOT** use a hard-coded integer seed inside a reusable function. Seeds belong in config (`cfg.seed`) or at call sites.
- **MUST** save and restore all three RNG states in checkpoints: `torch.get_rng_state()`, `numpy.random.get_state()`, `random.getstate()`.

## Module boundaries

- **MUST NOT** import from `minihack_reference/` anywhere in `src/`. That directory is read-only reference material.
- **MUST NOT** add environment interaction (gym calls, BFS, oracle) to `src/diffusion/` or `src/models/`. Those directories are pure math and model definitions.
- **MUST NOT** add training logic (optimiser steps, gradient computation) to `src/models/`. Models define `forward()` only.
- **MUST NOT** add checkpoint I/O or W&B logging to `src/diffusion/` or `src/models/`. All I/O stays in `src/planners/`.

## Numerical hygiene

- **SHOULD** clip loss weights before reduction (the MDLM SUBS weight `-alpha'(t) / (1 - alpha_t)` is already clipped at 1000 — follow this pattern for any new weighting scheme).
- **SHOULD** use a safe denominator when dividing by a quantity that could be zero:
  ```python
  safe_denom = denom.clamp(min=1e-8)
  result = numerator / safe_denom
  ```
- **MUST NOT** use Python `float('inf')` where `torch.inf` is available and more consistent.

## Debugging

- **MUST NOT** leave `torch.autograd.set_detect_anomaly(True)` enabled in production code — it has significant overhead. Use it only in debugging sessions.
- **SHOULD** use `assert` statements for shape checks in development; they can be disabled with `python -O` in production if needed.
- Use `print(tensor.shape, tensor.dtype, tensor.device)` sparingly for quick debugging; clean up before committing.
