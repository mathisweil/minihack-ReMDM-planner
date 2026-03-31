---
paths: src/models/**
---

# Model Architecture Rules — src/models/

> Also load @.claude/rules/python-idioms.md when editing this directory.

---

## Module responsibilities

| File | Responsibility |
|---|---|
| `denoiser.py` | `LocalDiffusionPlannerWithGlobal` — dual-stream obs encoder + transformer backbone + action head + `ModelEMA` |

- **MUST NOT** add training logic (optimiser steps, loss computation, gradient clipping) to model files. Models define `forward()` only.
- **MUST NOT** add environment interaction, checkpoint I/O, or W&B logging to model files.
- **MUST NOT** add new model variants by monkey-patching `LocalDiffusionPlannerWithGlobal` at runtime. New variants are new classes.

---

## `LocalDiffusionPlannerWithGlobal` architecture

Named dimensions used in all comments and assertions:

| Name | Meaning | Default |
|---|---|---|
| `d_model` (`n_embd`) | Transformer hidden dim | 256 |
| `n_heads` (`n_head`) | Attention heads | 4 |
| `n_layers` (`n_layer`) | Transformer blocks | 4 |
| `n_global_tokens` | Global stream context tokens | 8 |
| `H` (`seq_len`) | Action sequence length | 64 |
| `num_actions` | Valid action vocabulary | 12 |
| `vocab_size` | Embedding vocab (actions + MASK + PAD) | 14 |

The model takes `(local_obs [B,9,9], global_obs [B,21,79], noisy_seq [B,H], t [B])` and returns `{"actions": [B,H,12], "goal_pred": [B,2]}`.

### Local stream

```
Embedding(6000, 64) → Conv2d(64→32, 3×3) → Conv2d(32→64, 3×3) → Flatten → Linear(→ d_model)
→ 1 token  [B, 1, d_model]
```

- **MUST** use exactly two Conv2d layers with kernel 3×3 for the local CNN. Changing depth requires updating this rule.

### Global stream

```
Embedding(6000, 32) → Conv2d(32→32, 5×5, stride=2) → Conv2d(32→64, 3×3, stride=2)
→ AdaptiveAvgPool2d((2, 4)) → Linear(64, d_model) → 8 tokens  [B, 8, d_model]
```

- **MUST** use `AdaptiveAvgPool2d((2, 4))` to produce exactly `n_global_tokens = 8` spatial tokens (2 × 4 = 8).

### Global gate

```python
self.global_gate = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3.0) ≈ 0.047
...
gated_global = torch.sigmoid(self.global_gate) * global_tokens  # [B, 8, d_model]
```

- **MUST** initialise `global_gate` as a scalar `nn.Parameter` with value `-3.0` (not `0.0`). This keeps the gate nearly closed early in training, preventing the global stream from destabilising BC.
- **MUST** apply the gate **after** computing `goal_pred` (see below). The aux loss needs an unattenuated gradient path to the global CNN.

### Auxiliary goal head

```python
# Applied to mean of UNGATTED global tokens:
goal_pred = self.goal_mlp(global_tokens.mean(dim=1))  # [B, 2]  — normalised (row, col)
# THEN apply gate:
gated_global = torch.sigmoid(self.global_gate) * global_tokens
```

- **MUST** compute `goal_pred` from the **ungatted** global tokens. Computing it after gating blocks gradients to the global CNN early in training.
- **MUST** use a two-layer MLP: `Linear(d_model, 128) → GELU → Linear(128, 2)`.
- Output represents normalised staircase coordinates: `(row / 21, col / 79)`.

### Action stream

```python
action_emb   = self.action_embedding(noisy_seq)   # [B, H, d_model]
timestep_emb = self.timestep_embedding(t)          # [B, d_model] → unsqueeze → [B, 1, d_model]
pos_emb      = self.position_embedding(positions)  # [H, d_model]
seq_tokens   = action_emb + timestep_emb + pos_emb  # [B, H, d_model]
```

- **MUST** use `Embedding(14, d_model)` for actions (14 = 12 actions + MASK + PAD).
- **MUST** use `Embedding(100, d_model)` for timesteps (covers `[0, num_diffusion_steps]`).
- **MUST** use `Embedding(seq_len, d_model)` for positions (covers `[0, H-1]`).
- **MUST** sum all three embeddings — do not concatenate them.

### Transformer

```
concat [local(1) + global(8) + action(64) = 73 tokens] → 4-layer TransformerEncoder → 73 tokens
→ discard first 9 → last 64 tokens → Linear(d_model, 12) → action logits [B, H, 12]
```

- **MUST** use bidirectional (full) attention — `nn.TransformerEncoderLayer` with `norm_first=True` (Pre-LN), GELU activation, no causal mask.
- **MUST** assert `d_model % n_head == 0` at model construction time.
- **MUST** discard the first `1 + n_global_tokens = 9` output tokens. Only the last `H = 64` tokens feed the action head.
- **MUST NOT** include the MASK token in the output vocabulary. The action head projects to `num_actions = 12`, not 13 or 14.

---

## `ModelEMA`

```python
class ModelEMA:
    def update(self, model: nn.Module, decay: float = 0.999) -> None:
        with torch.no_grad():
            for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
                ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)
```

- **MUST** update EMA after every gradient step, not every epoch.
- **MUST** use `torch.no_grad()` during EMA updates.
- `state_dict()` on `ModelEMA` **MUST** return the shadow copy's weights, not the live model's.
- **SHOULD** apply EMA only to model parameters, not buffers (e.g., BatchNorm running stats), unless explicitly decided otherwise.

---

## Parameter initialisation

- All `nn.Linear` layers: use PyTorch default (`kaiming_uniform_` for kernel, `zeros` for bias).
- All `nn.Embedding` layers: use `nn.init.normal_(emb.weight, std=0.02)` (GPT-style).
- Action output head: **MUST** initialise weight to near-zero (`std=0.02`) to prevent large initial logits that destabilise early MDLM training.
- `global_gate`: initialise to `-3.0` as specified above — **MUST NOT** use the default `nn.Parameter(torch.zeros(1))`.

---

## Adding new architecture components

1. Implement as a standalone `nn.Module` subclass.
2. Write a shape sanity check (even 3 lines) alongside the implementation.
3. Add it to the docstring of `denoiser.py`.
4. **MUST NOT** add it by patching `LocalDiffusionPlannerWithGlobal` at runtime.
