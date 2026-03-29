# Post-Audit Fix & Improve — minihack-ReMDM-planner

You have just completed a full reimplementation audit. The report is in the conversation above. This prompt tells you exactly what to fix, what to keep, and what to investigate before deciding.

Work through each section **in order**. Do not skip ahead. Commit after each section with a message that references the section number.

---

## Section 1 — Fix the 4 BUGs (non-negotiable)

These are correctness errors. Fix all four before doing anything else.

### BUG 1: EMA update outside gradient loop (online.py:106)

Move `ema_model.update(self.model)` **inside** the `for _ in range(cfg.grad_steps_per_iteration)` loop so it fires after every optimizer step, not once per iteration. The reference does this in `_train_step()` — each call to `_train_step` ends with the EMA update. Skipping 99/100 updates makes the EMA shadow weights lag behind by thousands of gradient steps.

### BUG 2: DataCollector uses frozen EMA snapshot (online.py:457-458)

Replace the deep-copied `eval_model` with a **live reference** to the EMA shadow model. The collector must always use the latest EMA weights. Two acceptable patterns:

**Option A (simplest):** Pass the EMA object itself to the collector. Before each `collect_episode`, call `ema.copy_to(eval_model)` to sync weights. This is what the reference does implicitly — `self.collector = DataCollector(self.ema_model, ...)` where `self.ema_model` is the live shadow.

**Option B:** Store a reference to `ema._shadow` and use it directly for forward passes. No copy needed, but you must ensure `eval()` mode and `torch.no_grad()` are used.

Pick the option that requires fewer changes. Document your choice in a code comment.

### BUG 3: buffer.sample() crashes on empty buffer (buffer.py:172)

Add an early return when the buffer is empty:
```python
if len(self) == 0:
    return None
```
Then in the training step, handle `None` gracefully:
```python
batch = self.buffer.sample(batch_size)
if batch is None:
    return 0.0, 0.0, 0.0  # no-op
```
This matches the reference's `_train_step` behaviour. The DAgger seeding mitigates this in practice, but the crash path must not exist.

### BUG 4: Missing weight_decay=1e-4 (online.py:444, offline.py:62)

Add `weight_decay=1e-4` to both AdamW constructors. The PyTorch default is `0.01` — a 100x difference from the reference. This silently changes regularisation strength and can cause underfitting or instability.

```python
# online.py — DAgger optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.dagger_lr, weight_decay=1e-4)

# offline.py — BC optimizer  
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.offline_lr, weight_decay=1e-4)
```

Make `weight_decay` configurable via `cfg.weight_decay` in `defaults.yaml` with default `1e-4`.

**After fixing all 4 bugs, run the smoke test to verify nothing is broken:**
```bash
python main.py --mode smoke
```

---

## Section 2 — Evaluate deviations: KEEP, REVERT, or INVESTIGATE

The audit found ~13 deviations from the reference. For each one below, I've marked the recommended action. Follow these recommendations unless you find a strong reason not to — and if you deviate, leave a comment explaining why.

### KEEP (improvements over reference)

These are deliberate upgrades. Leave them as-is.

1. **DEVIATION 12: Dedicated offline BC phase (offline.py).** The reference skips this entirely — it loads a pre-trained checkpoint or starts from random init. A proper BC phase is strictly better: it gives the model a warm start from your own data pipeline rather than depending on an external checkpoint. **Keep.**

2. **DEVIATION 10: Queues pre-seeded with 50/50 prior (curriculum.py:42-46).** The reference starts empty and returns 0.5 as a neutral prior, which gives weight 1.0 to all envs. Pre-seeding achieves the same initial weighting but also means the curriculum is immediately responsive to real outcomes (they start diluting the prior from episode 1) rather than having undefined behaviour until the first outcome arrives. **Keep**, but add a config option `curriculum_preseed: true` so it can be toggled.

3. **DEVIATION 3: dropout=0.0 (denoiser.py:78).** Discrete diffusion already has a built-in regularisation mechanism — the random masking in the forward process. Dropout on top of masking noise is redundant and can hurt sample quality. The reference inherits PyTorch's default 0.1 by accident, not by design. **Keep at 0.0.**

4. **Post-sample assertion (NOTE 3, sampling.py:263).** Good addition — catches silent failures. **Keep.**

5. **Min-keep 10% safety net (NOTE 4, sampling.py:145).** Prevents degenerate all-masked states during sampling. The reference's Path A lacks this, which is a deficiency. **Keep.**

### REVERT (regressions or risky without evidence)

6. **DEVIATION 4: MDLM SUBS importance weighting w(t) (loss.py:52-56).** This fundamentally changes the loss landscape. The reference uses a flat average. SUBS weighting upweights early timesteps (high mask ratio) where the model has the least information — this can destabilise training, especially with DAgger's non-stationary data distribution. **Revert to simple masked average.** If you want to experiment with SUBS weighting later, make it a config flag `use_importance_weighting: false` (default off) so it can be toggled cleanly.

7. **DEVIATION 5: Per-sample loss averaging (loss.py:80-81).** The reference does global averaging: `total_CE_on_masked / (total_n_masked + 1e-6)`. Per-sample averaging gives equal weight to every sample regardless of mask count, which biases toward low-mask-ratio samples (where fewer tokens are masked but each one counts as much as a sample with many masked tokens). **Revert to global averaging** to match the reference. This is the safer default for DAgger.

8. **DEVIATION 7: efficiency_multiplier=2.0 (defaults.yaml:57).** The reference uses 1.5. A multiplier of 2.0 means the model must be 2x worse than oracle before its trajectory triggers data collection — this is much more lenient and will result in fewer oracle demonstrations entering the buffer. With DAgger, you want the buffer to be enriched aggressively early on. **Revert to 1.5.** If you want to sweep this later, make it a named config override.

9. **DEVIATION 13: Timestep scaling t*(num_steps-1) vs t*100 (online.py:222).** This causes timestep embedding index 99 to never be used during training, creating a dead embedding. **Fix to match reference:** use `t_discrete = (mask_ratios * 100).long()` which produces values in [0, 99]. Alternatively, if `num_steps` is configurable and differs from 100, use `t_discrete = (mask_ratios * cfg.num_timestep_embeddings).long()` and set `num_timestep_embeddings: 100`.

### INVESTIGATE before deciding

10. **DEVIATION 1: seq_len=192 (defaults.yaml:23).** This is the single most impactful deviation — it changes the transformer's token count from 73 to 201, triples memory cost, and extends the planning horizon from 64 to 192 steps. **Run a quick experiment before committing:** train a smoke test with seq_len=64 and seq_len=192, compare loss curves over 30 iterations. If 192 doesn't converge noticeably differently in smoke, keep it but add a comment explaining the rationale (longer plans for larger mazes). If it diverges or is much slower to converge, revert to 64.

11. **DEVIATION 2: dim_feedforward=1024 (denoiser.py:77).** Halving the feedforward capacity is significant. The reference uses the PyTorch default of 2048. With seq_len=192, you might need the extra capacity. **Investigate:** if you keep seq_len=192, keep dim_feedforward=2048 to avoid a bottleneck. If you revert seq_len to 64, dim_feedforward=1024 is reasonable to save memory.

12. **DEVIATION 6: diffusion_steps_eval=30 (defaults.yaml:38).** 6x more denoising steps at inference. More steps generally improve sample quality but at linear inference cost. The reference uses 5 steps which is very aggressive. **Compromise:** set default to `10` and make it configurable. Add a comment noting the reference uses 5 and you can sweep this.

13. **DEVIATION 8: batch_size=1024 (defaults.yaml:45,52).** 4x larger batches. This is fine if your GPU can handle it, but changes the effective learning rate (larger batch → needs higher LR for same convergence speed, per linear scaling rule). **Keep 1024 if GPU memory allows**, but verify the learning rate is scaled appropriately. The reference uses batch=256 with lr=3e-5 for DAgger. With batch=1024, consider lr=1.2e-4 (linear scaling) or keep 3e-5 and accept slower convergence per iteration but more stable gradients.

14. **DEVIATION 9: grad_steps_per_iteration=100 (defaults.yaml:56).** 2x more gradient steps per iteration. Combined with 4x batch size, this means 8x more data processed per DAgger iteration. This might cause the model to overfit to the current buffer before new data arrives. **Keep for now** but monitor: if the loss drops rapidly then spikes when new data enters, reduce to 50.

15. **DEVIATION 11: max_steps=500 (minihack_env.py:398, collect.py:31).** 2.5x more steps per episode. Larger mazes (MazeWalk-45x19) may genuinely need more steps. **Keep 500** but add a per-environment config option if possible.

---

## Section 3 — Fix the MISSING features

### MISSING 1: Greedy decoding path for data collection

This is the most impactful missing feature. The reference deliberately uses greedy decoding for DAgger collection — this makes the model's rollout deterministic and reproducible, which is important for the efficiency filter comparison (model steps vs oracle steps on the same seed).

**Add a `greedy_sample` function** to `src/diffusion/sampling.py`:

```python
def greedy_sample(model, local_obs, global_obs, cfg, valid_actions=None):
    """Greedy (argmax) sampling — no temperature, no top-K, no remasking.
    Used by DataCollector during DAgger (deterministic rollouts).
    """
    # Same MaskGIT loop as remdm_sample but:
    # - probs = softmax(logits)  (no temperature)
    # - preds = argmax(probs)    (no sampling)
    # - No stochastic remasking
    # - Confidence = max prob
```

Then in `collect.py`, use `greedy_sample` instead of `remdm_sample`. Keep `remdm_sample` for evaluation (`inference.py`).

Also add a `replan_every` config option (default 16) and execute plan tokens sequentially in the collector, matching the reference's `run_model_episode` behaviour.

### MISSING 2: Observation key "pixel"

In `minihack_env.py:87`, change:
```python
observation_keys=("glyphs", "chars", "blstats")
```
to:
```python
observation_keys=("glyphs", "chars", "pixel")
```
The reference uses `"pixel"` — some MiniHack environments require it to be present even if we don't use it directly. Using `"blstats"` instead may silently break specific environments.

**Also verify:** the reimplementation uses `chars == ord('@')` for agent centering (matching the reference), NOT `blstats`. If `blstats` was being used for agent position, switch to `chars`.

### MISSING 3: Codebase map accuracy

Update `CLAUDE.md` to reflect the actual file structure:
- `src/planners/train.py` → `src/planners/offline.py`
- Add `src/planners/smoke.py` to the file listing

---

## Section 4 — Physics-aware sampling (NOTE 5)

The reimplementation merged the physics-aware confidence override from the reference's `run_inference_test` (Path B) into the main sampling path. This is a reasonable enhancement but it was never active in the reference's evaluation pipeline. 

**Make it configurable:**
```yaml
# defaults.yaml
physics_aware_sampling: false  # Enable to penalise hazardous actions during sampling
```

Default to `false` so baseline results match the reference. This can be toggled on for experiments.

---

## Section 5 — Verification checklist

After completing Sections 1-4, verify each item:

- [ ] Smoke test passes: `python main.py --mode smoke`
- [ ] EMA updates inside gradient loop (grep for `ema.*update` — should be inside the inner loop)
- [ ] DataCollector uses live EMA weights (no `deepcopy` at init)
- [ ] `buffer.sample()` returns `None` on empty buffer
- [ ] `weight_decay=1e-4` in both optimizers
- [ ] Loss uses global averaging (not per-sample)
- [ ] No importance weighting by default (`use_importance_weighting: false`)
- [ ] `efficiency_multiplier: 1.5` in defaults.yaml
- [ ] Timestep scaling produces values in [0, 99]
- [ ] `greedy_sample` function exists and is used by collector
- [ ] Observation keys include `"pixel"`, not `"blstats"`
- [ ] `CLAUDE.md` file listing matches actual files
- [ ] All new config keys have defaults in `defaults.yaml`

---

## Section 6 — Final: update defaults.yaml

After all changes, `defaults.yaml` should have these values (update any that were changed):

```yaml
# Tokens
mask_token: 12
pad_token: 13
action_dim: 12

# Model
n_embd: 256
n_head: 4
n_layer: 4
n_global_tokens: 8
global_gate_init: -3.0
# dim_feedforward: decision depends on seq_len investigation (Section 2, item 11)
dropout: 0.0

# Diffusion
num_timestep_embeddings: 100
diffusion_steps_eval: 10        # compromise (ref=5, was 30)
temperature: 0.5
top_k: 4

# Training — offline BC
offline_lr: 3e-4
offline_batch_size: 1024        # keep if GPU allows, else 256
offline_grad_clip: 1.0

# Training — DAgger
dagger_lr: 3e-5
dagger_batch_size: 1024         # keep if GPU allows, else 256
weight_decay: 1e-4
ema_decay: 0.999
grad_steps_per_iteration: 100   # monitor for overfitting
episodes_per_iteration: 10
efficiency_multiplier: 1.5      # reverted from 2.0
replan_every: 16

# Buffer
buffer_capacity: 10000

# Curriculum
curriculum_queue_size: 100
curriculum_preseed: true

# Evaluation
eval_episodes_per_env: 50

# Loss
aux_loss_weight: 0.5
use_importance_weighting: false  # SUBS w(t) — off by default

# Sampling
physics_aware_sampling: false

# Environment
max_steps: 500
crop_size: 9

# Smoke test overrides (smoke.yaml)
# buffer_capacity: 50
# max_iterations: 30
# eval_episodes: 5
```