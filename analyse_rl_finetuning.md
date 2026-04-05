# Task

Read every file in `experiments/rl_finetuning/` and produce a single **spec document** (`rl_finetuning_spec.md`) that a developer on a *different* codebase (Craftax) can use to verify their own `experiments/rl_finetuning/` does the same thing.

## What to extract

For each file, capture the **what** and **why**, not the MiniHack-specific implementation details (observation shapes, env wrappers, action vocab, etc.). Specifically:

### 1. Ablation registry (`ablations/registry.py`)
- List every ablation name, its group, and a one-sentence description of what it changes.
- Note any dependencies between ablations (e.g. does `lora` also change the optimizer?).

### 2. Loss variants (`ablations/losses.py`)
- For each loss factory function: what mathematical objective does it implement? What are the key hyperparameters and their defaults?
- Which losses use advantage weighting, which don't? How are advantages computed?

### 3. Optimizer variants (`ablations/optimizers.py`)
- What optimizer configs exist (LLRD, LoRA, frozen params, gradient surgery, etc.)?
- For each: which parameters are trainable, what LR schedule is used, any special gradient manipulation?

### 4. Training loop (`ablations/training.py`)
- Step-by-step: what happens in one training iteration? (data sampling → forward → loss → backward → step → diagnostics)
- How is online data collected? How are episodes scored?
- What is the evaluation protocol (how many episodes, which envs, what metric = "score")?

### 5. Diagnostics (`diagnostics/`)
- List every diagnostic metric computed, when it's computed, and what it measures.
- Note the exact computation (e.g. "CKA between layer N activations at init vs current").

### 6. Analysis pipeline (`analysis/`)
- What plots/tables are generated? What is `diagnosis.md`?
- How does the decision-tree / hypothesis-testing report work?

### 7. Config schema (`configs/`)
- List all hyperparameters with defaults from `ablations_default.yaml`.
- Note which are overridden in `ablations_fast.yaml`.

## Output format

Write `rl_finetuning_spec.md` to the project root. Structure it as a flat checklist so the Craftax developer can go section by section and tick off parity. Use this template per item:

```
- [ ] **ablation_name** — description of what it does, key hyperparams (default values)
```

Keep it under 800 lines. Omit MiniHack env-specific code (BFS oracle, glyph embeddings, .des files, NLE wrappers). Focus on the environment-agnostic algorithmic logic.
