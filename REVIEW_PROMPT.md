# Full Review & Functional Test: `experiments/rl_finetuning/`

You are auditing the entire `experiments/rl_finetuning/` codebase for correctness, cleanliness, and completeness.
The reference implementations live in `minihack_reference/experiments/` (4 Python scripts: `rl_shared.py`, `rl_ablations.py`, `action_distribution_analysis.py`, `mixing_experiment.py`).

Work through every section below **in order**. Do not skip sections. Print a status line after each section so I can track progress.

---

## 1. Structural Inventory

Read the entire `experiments/rl_finetuning/` tree. For every `.py` file, list:
- File path
- Number of lines
- Every top-level function, class, and constant defined in it
- Every import (group by: stdlib, third-party, project-internal)

Then verify the directory layout matches the README:
```
rl_finetuning/
├── run_ablations.py
├── ablations/{losses,optimizers,registry,training}.py
├── diagnostics/{gradient,representation,timestep}.py
├── analysis/{plots,tables,report}.py
└── configs/{ablations_default,ablations_fast}.yaml
```
Flag any **extra files** not in the README (may be fine, but note them) and any **missing files** promised by the README.

Check every `__init__.py` — does each subpackage have one? Are the `__all__` exports correct and complete? Can you `from ablations import ...` the things that `training.py` needs?

---

## 2. Import Graph & Circular Dependencies

Build the full import graph across all files in `experiments/rl_finetuning/`.

- Draw it out (textually is fine: `A -> B -> C`).
- Flag any **circular imports** (A imports B which imports A).
- Flag any imports that **will fail at runtime** because:
  - The imported name doesn't exist in the target module
  - The target module itself has a top-level error
  - A conditional import (`try/except ImportError`) silently swallows something critical
- Flag imports from `src/` that assume a specific `sys.path` — will they work when invoked as `python experiments/rl_finetuning/run_ablations.py` from the repo root?

---

## 3. Dead Code & Unused Symbols

For **every** file in the codebase:

1. List every function, class, method, and module-level variable that is **never called/referenced** by any other code in the project. Check:
   - Not called within its own module
   - Not imported by any other module
   - Not referenced in the registry or config
   - Not used in tests
   - Not a public API documented in the README
2. List every import that is **unused** in the file that imports it.
3. List every local variable inside a function that is **assigned but never read**.
4. List every function parameter that is **accepted but never used** inside the function body.
5. List any **duplicate code** — two functions or blocks that do nearly the same thing and could be merged.

For each item found, recommend: **delete**, **merge**, or **keep (with justification)**.

---

## 4. Code Quality & Style

Check every file for:

### 4a. Type safety
- Functions missing type hints on parameters or return values (list the top 10 worst offenders)
- Any use of `Any` that could be made more specific
- Tensor shape mismatches that would only surface at runtime (e.g., wrong `dim=` in gather/softmax)
- Places where `int` vs `float` vs `torch.Tensor` confusion could cause silent bugs

### 4b. Error handling
- Bare `except Exception` or `except:` that swallows errors silently — list each one and say whether it's justified
- Places where a function returns `None` on failure but the caller doesn't check for `None`
- Missing `with torch.no_grad()` in evaluation/diagnostic code that should not track gradients
- GPU/CPU device mismatches — are tensors consistently moved to the right device?

### 4c. Naming & readability
- Single-letter variable names outside of loop indices or well-known conventions (B, H, N, T)
- Functions longer than 80 lines that should be split
- Magic numbers not defined as named constants
- Inconsistent naming conventions (camelCase vs snake_case, etc.)

### 4d. Modularity
- Any file longer than 500 lines that should be split
- Any function that does more than one conceptual thing (e.g., trains AND evaluates AND plots)
- Any place where `training.py` directly imports from `analysis/` or vice versa (wrong dependency direction)
- Shared constants/config that are hardcoded in multiple places instead of living in one config

### 4e. Docstrings
- Every public function and class should have a docstring. List any that are missing.
- Check that docstrings match actual behaviour (especially parameter names and return types).

---

## 5. Registry & Config Consistency

### 5a. Registry completeness
Read `ablations/registry.py`. For every ablation in the REGISTRY:
- Does the loss function it references actually exist in `losses.py`?
- Does the optimizer factory it references actually exist in `optimizers.py`?
- Are the hyperparameter keys it passes (`kl_coef`, `t_max`, `ewc_lambda`, etc.) actually consumed by the loss/optimizer function?
- Are there loss functions or optimizer factories in the code that are **not referenced** by any registry entry?

### 5b. Config files
Read both YAML configs. Verify:
- Every key referenced in `training.py` or `run_ablations.py` exists in the default config
- The fast config only **overrides** keys from the default (doesn't introduce new keys)
- Numeric values are reasonable (e.g., `max_iter: 50` for fast, not `max_iter: 0`)
- No config key is read in code but missing from both configs (would cause a KeyError at runtime)

### 5c. The 25 ablations
Verify all 25 ablations from the README are registered:
```
baseline_rl, kl_penalty, ewc, llrd, lora, mixed_replay, trust_region_kl,
low_t, t_curriculum, entropy_bonus, gradient_surgery, advantage_clip,
normalized_adv, bc_wins, frozen_backbone, head_only, attention_only,
ffn_only, layer_ablation_top1, layer_ablation_top2, layer_ablation_top3,
reward_filtering, running_stats, action_diversity, reward_model
```
For each one, confirm it has: a registered name, group label, loss factory, optimizer factory, and description.

---

## 6. Logic Correctness Deep-Dive

### 6a. Masked diffusion forward pass
Find the core forward pass (equivalent to `_forward_and_ce` in the reference). Verify **line by line**:
1. `t_int = torch.randint(1, 100, (B,))` — NOT `randint(0, 100)` or `randint(1, 101)`
2. `mask_prob = t_int.float() / 100.0`
3. `is_masked = torch.bernoulli(mask_prob.unsqueeze(1).expand(B, H)).bool()`
4. `z_t = action_batch.clone(); z_t[is_masked] = MASK_TOKEN`
5. Model forward: `model(local, global, z_t, t_int)` — correct argument order
6. CE loss: `log_softmax → gather → negate`, applied only where `is_masked` is True
7. Normalisation: `per_sample = (ce * is_masked.float()).sum(1) / is_masked.sum(1).clamp(min=1.0)`
8. The function returns enough values for downstream use (KL penalty needs action_batch, local, global, t_int, is_masked)

### 6b. KL penalty loss
Find the KL penalty implementation. Verify:
1. Uses a **frozen** reference model (separate from the training model)
2. Computes `KL(current || ref)` not `KL(ref || current)` — i.e., `cur_probs * (cur_log - ref_log)`
3. KL is computed only on **masked positions**
4. KL is normalised by number of masked tokens per sample
5. Final loss = `rl_loss + kl_coef * kl.mean()`
6. The reference model's forward pass is inside `torch.no_grad()`

### 6c. Low-t loss
Verify: identical to baseline except `t_int = torch.randint(1, t_max + 1, ...)` where `t_max` defaults to 20.

### 6d. BC-on-wins loss
Verify: `per_sample.mean()` with no return weighting. The `wins_only` filtering happens at the **buffer sampling** level, not inside the loss function.

### 6e. Frozen backbone / head_only / attention_only etc.
For every architecture ablation, verify:
- The correct parameters are frozen (`requires_grad = False`) based on **parameter name patterns**
- The optimizer only receives `[p for p in model.parameters() if p.requires_grad]`
- After freezing, print or log the trainable parameter count as a sanity check

### 6f. EWC loss
If implemented, verify:
- Fisher diagonal is computed **once** from the pretrained model on a held-out batch
- EWC penalty = `sum(fisher_i * (theta_i - theta_star_i)^2)` summed over all parameters
- Fisher is detached and not updated during training

### 6g. Gradient surgery (PCGrad)
If implemented, verify:
- Two separate backward passes (RL loss and BC/regularisation loss)
- If gradients conflict (negative cosine similarity), project the RL gradient onto the normal plane of the BC gradient
- The projected gradient is then used for the optimizer step
- Fraction of projections is logged as a diagnostic

### 6h. Advantage clipping / normalised advantages
Verify:
- `advantage_clip`: weights are `clamp(w, 1-eps, 1+eps)` before multiplying per-sample loss
- `normalized_adv`: weights are `(w - w.mean()) / (w.std() + 1e-8)` then optionally clamped

### 6i. Reward model
If implemented, verify:
- A small MLP (not the main model) that maps (state, action_seq) → scalar reward estimate
- Trained on collected (return, trajectory) pairs
- Used to soft-weight training samples (not hard filter)

---

## 7. Diagnostics Deep-Dive

### 7a. Gradient alignment
Find the gradient alignment computation. Verify:
1. RL gradient: backward on `(per_sample * weight).mean()` using the **training model**
2. BC gradient: backward on `per_sample.mean()` using a **separate trainable copy of the reference model** (not the frozen ref — it needs gradients)
3. Cosine similarity: `F.cosine_similarity(rl_flat, bc_flat)`
4. Both gradient vectors are **detached** and **flattened** across all parameters
5. `model.zero_grad()` is called before each backward to avoid accumulation
6. Returns `(cos_sim, rl_norm, bc_norm)`

### 7b. Representation drift
Find the KL drift computation. Verify:
1. Uses `torch.no_grad()` — this is a diagnostic, not a training signal
2. Computes `KL(ref || current)` on masked positions — note the direction is **opposite** to the KL penalty loss
3. Uses mid-range t (30-70) for a balanced signal
4. Returns a single float

### 7c. t-gradient analysis
Find the t-bin gradient analysis. Verify:
1. Computes loss separately for low-t (1-20) and high-t (80-100) ranges
2. Takes backward pass for each, extracts gradient norms
3. Computes cosine similarity between the two gradient vectors
4. Returns `(norm_low, norm_high, cos_sim_lohi)`

### 7d. CKA similarity
If implemented, verify:
- Linear CKA or kernel CKA between activations of current model vs pretrained model
- Computed on a held-out batch (not the training batch)
- Returns a scalar in [0, 1] where 1 = identical representations

### 7e. Per-layer gradient norms
If implemented, verify:
- Iterates over `model.named_parameters()`
- For each parameter with a gradient, records `param.grad.norm().item()`
- Groups by layer name prefix (e.g., `transformer.layers.0`, `transformer.layers.1`)

---

## 8. Analysis Pipeline Deep-Dive

### 8a. Plots
For every plot function in `analysis/plots.py`, verify:
- It can handle **partial results** (e.g., only 3 of 25 ablations completed) without crashing
- It uses `plt.savefig()` with `dpi=150, bbox_inches='tight'`
- It calls `plt.close()` after saving (not `plt.show()` in batch mode)
- Axis labels, titles, and legends are readable
- Colour scheme is consistent across related plots

Verify these specific plots exist and are correct:
1. **Score comparison bar chart** — all ablations, sorted or grouped
2. **Training curves** — loss and/or win rate over iterations, per ablation
3. **Gradient alignment** — cos_sim over iterations, zero line, per ablation
4. **Representation drift** — KL over iterations, per ablation
5. **t-bin gradient norms** — heatmap or line plot
6. **Win rate over training** — per ablation
7. **Group comparison** — boxplot or grouped bar chart by ablation group
8. **Decision tree / diagnosis chart** — hypothesis evidence summary

### 8b. Tables
For every table function in `analysis/tables.py`, verify:
- Works with partial results
- Produces valid LaTeX (escaped underscores, correct column alignment)
- CSV output is loadable by pandas without errors
- Numbers are formatted consistently (e.g., `:.2%` for win rates, `:.4f` for loss values)

### 8c. Report / diagnosis
For `analysis/report.py`, verify:
- Generates a valid `.md` file with correct markdown syntax
- Verdict logic is sound:
  - "COLLAPSE" threshold is clearly defined
  - "IMPROVEMENT" threshold is clearly defined
  - Gradient alignment interpretation is correct
- References specific ablation results by name
- Includes concrete recommendations

---

## 9. CLI & Entry Point

Read `run_ablations.py` end to end. Verify:

### 9a. Argument parsing
- `--list` prints all 25 ablation names and exits
- `--ablations name1 name2` runs only those ablations
- `--all` runs all 25
- `--fast` loads `ablations_fast.yaml` overrides
- `--checkpoint` is required unless `--analyze_only`
- `--analyze_only --output_dir` skips training, loads results.json, regenerates plots/tables/report
- `--num_seeds` runs each ablation N times with different seeds
- `--use_wandb` is optional and doesn't crash if wandb is not installed
- Unknown ablation names produce a clear error message listing valid names

### 9b. Execution flow
- Loads config from YAML
- Loads pretrained checkpoint
- Runs pretrained baseline evaluation **once** (not per-ablation)
- Iterates over selected ablations
- After each ablation: writes to results.json incrementally
- After all ablations: generates plots, tables, report
- `--analyze_only` path correctly loads and re-generates everything

### 9c. Error resilience
- If one ablation crashes, does execution continue to the next? (it should)
- Is the partial results.json still valid after a crash?
- Is there a try/except around each ablation with proper logging?

---

## 10. Functional Smoke Tests

**This is the most important section.** Run actual code to verify nothing is broken.

### 10a. Import test
```python
# Must not crash
import experiments.rl_finetuning.run_ablations
from experiments.rl_finetuning.ablations.losses import *
from experiments.rl_finetuning.ablations.optimizers import *
from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import run_ablation
from experiments.rl_finetuning.diagnostics.gradient import *
from experiments.rl_finetuning.diagnostics.representation import *
from experiments.rl_finetuning.diagnostics.timestep import *
from experiments.rl_finetuning.analysis.plots import *
from experiments.rl_finetuning.analysis.tables import *
from experiments.rl_finetuning.analysis.report import *
```
Run this. Report every ImportError with full traceback.

### 10b. Registry sanity check
```python
from experiments.rl_finetuning.ablations.registry import REGISTRY
assert len(REGISTRY) == 25, f"Expected 25 ablations, got {len(REGISTRY)}"
for name, spec in REGISTRY.items():
    assert callable(spec.loss_factory), f"{name}: loss_factory not callable"
    assert callable(spec.optimizer_factory), f"{name}: optimizer_factory not callable"
    assert spec.group in ('baseline', 'regularisation', 'training_signal', 'architecture', 'data_quality'), \
        f"{name}: unknown group {spec.group}"
    print(f"  OK: {name} (group={spec.group})")
```

### 10c. Loss function unit tests
For **each** loss factory in the registry, create a synthetic batch and verify the loss function runs without error and returns a scalar:
```python
import torch
from src.config import ACTION_DIM, MASK_TOKEN, HYPERPARAMS

device = HYPERPARAMS['device']
B, H = 4, 10  # small batch, short horizon
local = torch.randint(0, 5999, (B, 9, 9), device=device)
global_obs = torch.randint(0, 5999, (B, 21, 79), device=device)
actions = torch.randint(0, ACTION_DIM, (B, H), device=device)
weights = torch.ones(B, device=device)
samples_batch = [((local[i].cpu().numpy(), global_obs[i].cpu().numpy()),
                   actions[i].cpu().tolist()) for i in range(B)]

# For each ablation, build the loss function and call it
for name, spec in REGISTRY.items():
    model = ...  # fresh model on device
    loss_fn = spec.loss_factory(model, ...)  # pass whatever kwargs the factory needs
    loss = loss_fn(model, samples_batch, weights, device)
    assert loss.ndim == 0, f"{name}: loss is not scalar, shape={loss.shape}"
    assert not torch.isnan(loss), f"{name}: loss is NaN"
    assert not torch.isinf(loss), f"{name}: loss is Inf"
    loss.backward()  # must not crash
    print(f"  OK: {name} loss={loss.item():.4f}")
```
Adapt the above to the actual factory signatures. **Set a 4-minute timeout** for the entire test. If any loss function hangs or takes too long, kill it and report it.

### 10d. Optimizer factory unit tests
For each optimizer factory, verify it returns a valid `torch.optim.Optimizer`:
```python
for name, spec in REGISTRY.items():
    model = ...  # fresh model
    optimizer = spec.optimizer_factory(model, lr=3e-4, ...)
    assert isinstance(optimizer, torch.optim.Optimizer), f"{name}: not an Optimizer"
    assert len(optimizer.param_groups) > 0, f"{name}: empty param_groups"
    total_params = sum(len(g['params']) for g in optimizer.param_groups)
    print(f"  OK: {name} optimizer, {total_params} params in {len(optimizer.param_groups)} groups")
```

### 10e. Single-iteration smoke test per ablation
This is the full end-to-end test. For **each of the 25 ablations**, run the training loop for **exactly 1 iteration** using:
- A randomly initialised model (no checkpoint needed — just `model = LocalDiffusionPlannerWithGlobal(...)`)
- A synthetic buffer pre-filled with 20 dummy samples
- `max_iter=1, eval_every=9999` (skip eval), `grad_align_every=1, repr_drift_every=1, t_analysis_every=1` (run all diagnostics)

The test passes if:
1. No exception is raised
2. The returned history dict has the expected keys
3. Loss is a finite number
4. The model's parameters have changed (at least one param differs from initial state) — unless it's a frozen-everything ablation

**Set a 4-minute wall-clock timeout** for **each** ablation. If it exceeds 4 minutes, kill it, report it as TIMEOUT, and move to the next.

Print a summary table at the end:
```
Ablation                 | Status  | Loss   | Time(s) | Notes
-------------------------|---------|--------|---------|------
baseline_rl              | PASS    | 2.3451 | 12.3    |
kl_penalty               | PASS    | 2.5123 | 14.1    |
ewc                      | FAIL    | NaN    | 8.2     | Fisher computation returned NaN
lora                     | TIMEOUT | —      | 240.0   | Killed after 4 min
...
```

### 10f. Analysis pipeline smoke test
Create a fake `results.json` with 3 dummy ablation entries (baseline_rl, kl_penalty, frozen_backbone) containing synthetic history data (random numbers with correct shapes). Then:
1. Call every plot function — it must produce a `.png` file without crashing
2. Call every table function — it must produce `.csv` and `.tex` files
3. Call the report function — it must produce a `.md` file
4. Verify all output files exist and are non-empty

### 10g. CLI test
Run these commands and verify they don't crash:
```bash
python experiments/rl_finetuning/run_ablations.py --list
python experiments/rl_finetuning/run_ablations.py --help
```

---

## 11. Reference Cross-Check

Read the 4 reference scripts in `minihack_reference/experiments/`. For each piece of logic below, confirm it exists somewhere in the new codebase (not necessarily in the same file):

### 11a. From `rl_shared.py`
- [ ] `_forward_and_ce` with exact 6-return-value interface
- [ ] `_forward_and_ce_low_t` with `t_max` parameter
- [ ] `loss_baseline_rl`, `loss_kl_penalty`, `loss_bc_wins`, `loss_low_t`
- [ ] `compute_grad_alignment` returning `(cos_sim, rl_norm, bc_norm)`
- [ ] `compute_repr_drift` returning float
- [ ] `compute_t_gradient_analysis` returning `(norm_low, norm_high, cos_lohi)`
- [ ] `RLReplayBuffer` with `add_episode`, `sample(wins_only=)`, `n_wins()`
- [ ] `MixedReplayBuffer` with fixed oracle portion (weight 1.0) and ring-buffer self-gen
- [ ] `SimpleCurriculum` (round-robin env sampler)
- [ ] `rollout_episode` returning `(samples, ep_return, won)`
- [ ] `make_empty_history(extended=True)` with all diagnostic fields

### 11b. From `rl_ablations.py`
- [ ] Baseline evaluation before any training
- [ ] 5 original ablations: baseline_rl, kl_penalty, frozen_backbone, bc_wins, low_t
- [ ] Synthetic sanity check (BC on oracle winning episodes)
- [ ] Summary table with COLLAPSE/IMPROVEMENT/NEUTRAL verdicts
- [ ] Paper verdict logic (all-collapse check, synthetic-works check, gradient interpretation)
- [ ] 6-panel analysis plot

### 11c. From `action_distribution_analysis.py`
- [ ] `collect_action_statistics()` from real rollouts
- [ ] Entropy, KL, JS, TV, Gini metrics
- [ ] Chi-squared test on action counts
- [ ] Mann-Whitney U test on episode returns
- [ ] Action transition matrices
- [ ] Cumulative distribution with 80%/95% thresholds
- [ ] JS divergence interpretation (< 0.05, 0.05-0.15, > 0.15)

### 11d. From `mixing_experiment.py`
- [ ] `MixedReplayBuffer` with oracle:self-gen ratio
- [ ] `dataset_to_samples()` conversion
- [ ] Training at multiple oracle fractions
- [ ] Monotonicity check
- [ ] Degradation curve plot (inverted x-axis)

For any unchecked box, note whether it's **missing**, **partially implemented**, or **intentionally dropped** (with justification).

---

## 12. Final Summary & Action Items

After completing all sections, produce:

1. **A scorecard** — one line per file, rating it GREEN (clean, correct), YELLOW (minor issues), or RED (broken or seriously flawed)
2. **A prioritised fix list** — every issue found, ordered by severity:
   - P0 (Blocker): code that crashes or produces wrong results
   - P1 (Important): missing functionality, dead code that causes confusion, broken tests
   - P2 (Cleanup): style issues, missing docstrings, minor naming inconsistencies
3. **Apply all P0 and P1 fixes** — make the edits, run syntax checks after each.
4. **Re-run the smoke tests from Section 10** after fixes to confirm they now pass.
