---
paths: src/diffusion/**
---

# Diffusion Rules — src/diffusion/

> Also load @.claude/rules/python-idioms.md when editing this directory.

---

## Module responsibilities — do not cross them

| File | Single responsibility |
|---|---|
| `schedules.py` | Noise schedule functions (`linear`, `cosine`) and `SCHEDULE_MAP` |
| `forward.py` | Forward masking process `q(z_t \| x_0)` only |
| `loss.py` | Continuous-time MDLM ELBO loss + auxiliary goal loss |
| `sampling.py` | ReMDM reverse denoising loop with remasking strategies |

- **MUST NOT** add environment interaction, checkpoint I/O, or W&B logging inside any file in `src/diffusion/`. This directory is pure math.
- **MUST NOT** add new loss variants directly to `loss.py` unless they are core MDLM variants. Experimental losses belong in a dedicated file.

---

## Mathematical conventions

Always use these variable names to match the paper and the rest of the codebase:

| Symbol | Meaning |
|---|---|
| `x0` | Clean action sequence (integer tokens), shape `[B, H]` |
| `zt` | Noisy / partially masked sequence at time `t`, shape `[B, H]` |
| `alpha_t` | Fraction of tokens that are *unmasked* at time `t` (absorption schedule value), scalar or `[B]` |
| `sigma_t` | `1 - alpha_t` — masking probability at time `t` |
| `t` | Discrete timestep index ∈ `[1, num_diffusion_steps]` during training |
| `ratio` | Continuous unmasking fraction ∈ `[0, 1]` at inference (`ratio = k/K`) |
| `K` | Number of denoising steps at inference (default `diffusion_steps_eval = 5`) |
| `mask_id` | Integer token ID for the MASK token (= `cfg.mask_token = 12`) |
| `pad_id` | Integer token ID for the PAD token (= `cfg.pad_token = 13`) |

- **MUST NOT** confuse `K` (number of denoising steps, integer) with `t` (discrete training timestep index, integer).
- **MUST** keep timestep indices consistent: training uses `t ∈ [1, T]`; inference uses step index `k ∈ [1, K]` mapped to `ratio = k/K`.

---

## Schedules

- **MUST** add new noise schedule functions to `schedules.py` and register them in `SCHEDULE_MAP`.
- Schedule functions **MUST** be vectorisable (accept a batch of `t` values as a tensor).
- **MUST** ensure `alpha(t=0) = 1.0` (fully unmasked) and `alpha(t=T) = 0.0` (fully masked) for any new schedule.
  ```python
  # Boundary check (add to any new schedule):
  assert torch.isclose(schedule(torch.tensor(0.0)), torch.tensor(1.0))
  assert torch.isclose(schedule(torch.tensor(1.0)), torch.tensor(0.0))
  ```

---

## Forward process

```python
# Correct: bernoulli mask where sigma_t = 1 - alpha_t
mask = torch.bernoulli(sigma_t.expand_as(x0).float()).bool()
mask = mask & (x0 != pad_id)   # ← PAD tokens are NEVER masked — mandatory guard
zt = torch.where(mask, torch.full_like(x0, mask_id), x0)
```

- **MUST** use the existing `forward_process` function from `forward.py` rather than re-implementing masking inline elsewhere.
- **MUST NOT** apply the forward process to observation tokens — only to action tokens. Observations are always conditioned on, never masked.
- **MUST** include the PAD guard (`x0 != pad_id`) in any masking operation. This is the single most important correctness invariant in the codebase.

---

## Loss function

The MDLM ELBO in `loss.py` uses the SUBS parameterisation. Key invariants to preserve:

- The weight `w(t) = -alpha'(t) / (1 - alpha_t)` is clipped to `_MAX_WEIGHT = 1000` to prevent numerical instability near `alpha_t ≈ 1`. **MUST** preserve this clip in any extension.
- Cross-entropy is computed **only on masked positions** (`zt == mask_id`). Non-masked and PAD positions contribute zero loss.
- **MUST** guard against zero masked positions in the batch:
  ```python
  n_masked = mask.sum()
  if n_masked == 0:
      return torch.tensor(0.0, device=x0.device)
  loss = cross_entropy_on_masked / n_masked
  ```
- Auxiliary goal loss is MSE between `goal_pred [B, 2]` and normalised staircase coordinates: row divided by map height (21), col divided by map width (79).
- Total loss: `diffusion_loss + cfg.aux_loss_weight * aux_loss`. Default `aux_loss_weight = 0.5`.
- Loss is averaged over the batch dimension, not summed.

---

## Sampling — ReMDM reverse loop

The three remasking strategies share a common interface. **MUST** preserve it when adding new ones:

```python
def remask_<strategy>(
    committed: torch.Tensor,    # [B, H]  bool — positions already committed
    confidence: torch.Tensor,   # [B, H]  float — per-token confidence
    sigma_max: float,           # scalar  maximum remasking probability
    eta: float,                 # scalar  remasking strength hyperparameter
) -> torch.Tensor:              # [B, H]  bool — positions to re-mask
```

- **MUST** implement new remasking strategies in `sampling.py` and register them in the strategy dispatch dict.
- **MUST** verify that `remdm_sample` returns a fully committed sequence — assert `(seq == mask_id).sum() == 0` after the final step.

### Denoising step indexing

```
step k goes from 1 to K:
    ratio   = k / K              # unmasking fraction (0 → 1)
    t_val   = int(T * (1 - ratio))   # discrete timestep passed to model
```

- **MUST** use this indexing consistently. Do not invert the direction without updating all downstream consumers.
- At `k = K` (final step): force-commit ALL remaining masked tokens. No stochastic remasking on the final step.

### Remasking strategies

| Strategy | Re-mask probability |
|---|---|
| `rescale` | `p = eta * sigma_max` |
| `cap` | `p = min(eta, sigma_max)` |
| `conf` | `p = eta * sigma_max * (1 - confidence)` |

Default: `conf`, `eta = 0.15`.

### Inference-time action masking

- **MUST** apply action masking (`logits[:, :, env.action_space.n:] = -inf`) **before** temperature scaling and top-K filtering.
- Temperature scaling: divide logits by `cfg.temperature` (default 0.5).
- Top-K filtering: keep only the `cfg.top_k` (default 4) highest-logit actions per position; set others to `-inf`.
- Sample stochastically from the filtered distribution — do not use `argmax`.
