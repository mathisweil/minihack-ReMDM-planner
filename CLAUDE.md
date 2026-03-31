# CLAUDE.md — minihack-ReMDM-planner

> **Backbone file.** Read this first, always. Then follow the `@import` pointers for the area you are working in.

---

## Project snapshot

PyTorch implementation of **ReMDM** (Remasking Discrete Diffusion Model) — a dual-stream transformer that generates 64-step action-sequence plans for [MiniHack](https://github.com/facebookresearch/minihack) navigation environments by iteratively denoising masked token sequences, conditioned on a 9×9 local crop and a full 21×79 dungeon map.

**Three-stage pipeline:**
```
[Stage 1]  Offline BC on oracle demos     main.py --mode offline --data dataset.pt
                |
                v  checkpoint
[Stage 2]  DAgger online training          main.py --mode dagger
                |
                v  fine-tuned checkpoint
[Stage 3]  Evaluate (ID + OOD)             main.py --mode inference --checkpoint iter8000.pth
```

**Primary research question:** Does a planner trained on 4 in-distribution MiniHack environments generalise zero-shot to 3 held-out OOD environments?

---

## Codebase map

```
minihack-ReMDM-planner/
├── CLAUDE.md                          ← you are here
├── .claude/rules/                     ← path-scoped rule files
│   ├── python-idioms.md               # paths: src/**
│   ├── diffusion.md                   # paths: src/diffusion/**
│   ├── models.md                      # paths: src/models/**
│   ├── training.md                    # paths: src/planners/**
│   ├── envs.md                        # paths: src/envs/**
│   └── configs.md                     # paths: configs/**
├── minihack_reference/                ← READ-ONLY reference prototype — never modify
├── configs/
│   ├── defaults.yaml                  # source of truth for ALL hyperparameters
│   ├── smoke.yaml                     # fast smoke-test overrides
│   └── main.yaml                      # full training (inherits defaults)
├── src/
│   ├── config.py                      # YAML loader + CLI key=value override
│   ├── buffer.py                      # ReplayBuffer: offline-pinned FIFO
│   ├── curriculum.py                  # DynamicCurriculum + efficiency_filter
│   ├── diffusion/
│   │   ├── schedules.py               # linear / cosine alpha schedules
│   │   ├── forward.py                 # forward masking q(z_t | x_0)
│   │   ├── loss.py                    # MDLM ELBO + auxiliary goal loss
│   │   └── sampling.py               # ReMDM reverse sampling + remasking strategies
│   ├── models/
│   │   └── denoiser.py               # LocalDiffusionPlannerWithGlobal + ModelEMA
│   ├── envs/
│   │   └── minihack_env.py           # AdvancedObservationEnv + BFS oracle
│   └── planners/
│       ├── collect.py                 # run_model_episode, DataCollector
│       ├── offline.py                 # offline BC trainer
│       ├── online.py                  # DAgger trainer + checkpointing
│       ├── inference.py               # Evaluator + result formatting
│       ├── smoke.py                   # smoke-test runner
│       └── logging.py                 # centralised W&B + stdout logging
├── scripts/
│   ├── hf_upload.py                   # HuggingFace Hub upload utility
│   └── profile_dagger.py             # DAgger iteration profiler (Phase 1)
├── main.py                            # unified CLI entry point
├── environment.yaml                   # conda environment spec
└── README.md                          # full project documentation
```

---

## Universal constraints — apply everywhere

### Hard stops (MUST NOT)

- **MUST NOT** modify any file inside `minihack_reference/`. It is a read-only reference prototype for cross-checking only. Wrap or reimplement in `src/` instead.
- **MUST NOT** hardcode hyperparameters (learning rates, sequence lengths, token IDs, buffer sizes, temperatures, etc.) in `src/`. All such values **MUST** live in `configs/defaults.yaml` and be accessed via the config object `cfg`.
- **MUST NOT** break the existing CLI surface in `main.py`. New parameters **MUST** be added as optional arguments with safe defaults so every invocation in `README.md` continues to work unchanged.
- **MUST NOT** call `wandb.log(...)` directly from `src/diffusion/`, `src/models/`, or `src/envs/`. All metric emission goes through `src/planners/logging.py`.
- **MUST NOT** commit checkpoint binaries (`*.pth`, `*.pt`), dataset files, or W&B run directories.
- **MUST NOT** commit absolute filesystem paths anywhere in source or config.
- **MUST NOT** mask PAD tokens (id = 13) in the forward diffusion process. This is a load-bearing correctness invariant.
- **MUST NOT** let the MDLM loss return `NaN` when no tokens are masked in a batch. Return `0.0` instead.

### Hard requirements (MUST)

- **MUST** run `python main.py --mode smoke` to verify end-to-end correctness after any non-trivial change.
- **MUST** write docstrings for every public function in `src/`. Minimum: one-line summary, parameter types, return type.
- **MUST** add type annotations to every new function signature. Use `torch.Tensor` for tensor arguments, not `Any`.
- **MUST** use RFC-2119 language (MUST / MUST NOT / SHOULD / MAY) for any new rule added to a `.claude/rules/` file.
- **MUST** update this `CLAUDE.md` codebase map if new source files or directories are added.
- **MUST** use EMA weights by default at inference. Only `--no-ema` bypasses this.
- **MUST** move all tensors to `cfg.device`. Never hardcode `"cuda"` or `"cpu"`.

### Code style

- Python 3.10+. Use `match`/`case` only where it genuinely improves clarity.
- Line length: 100 characters.
- Imports: stdlib → third-party → local, separated by blank lines. No wildcard imports.
- No `print()` in `src/` outside `__main__` blocks. Use `src/planners/logging.py`.
- Prefer explicit over implicit. Avoid `**kwargs` in new public functions — it hides tensor shapes.

---

## Token vocabulary — global constants, never change

| Token | ID | Role |
|---|---|---|
| Actions | 0–11 | 12 discrete movement/interaction actions |
| `MASK_TOKEN` | 12 | Masked position during diffusion |
| `PAD_TOKEN` | 13 | Padding beyond episode end |

Defined in `configs/defaults.yaml` as `mask_token: 12` and `pad_token: 13`. Always read from `cfg`, never hardcoded.

---

## Technology stack

| Layer | Library |
|---|---|
| Array computation + autograd | PyTorch |
| Neural networks | `torch.nn` |
| Optimiser | AdamW (`torch.optim.AdamW`) |
| LR scheduling | `torch.optim.lr_scheduler` (CosineAnnealingLR) |
| Environment | MiniHack + NLE |
| Experiment logging | W&B (`wandb`) |
| Config parsing | PyYAML |
| Checkpointing | `torch.save` / `torch.load` |
| Hub uploads | `huggingface_hub` |

**Device:** CUDA 12 recommended for full training; CPU sufficient for smoke tests. Always set via `cfg.device`.

---

## Path-scoped rule index

Load the relevant file for your current task:

| Working in… | Load |
|---|---|
| Any `src/` code | @.claude/rules/python-idioms.md |
| `src/diffusion/` | @.claude/rules/diffusion.md |
| `src/models/` | @.claude/rules/models.md |
| `src/planners/` | @.claude/rules/training.md |
| `src/envs/` | @.claude/rules/envs.md |
| `configs/*.yaml` | @.claude/rules/configs.md |

---

## Maintenance checklist

Before considering any task complete:

- [ ] New public functions have docstrings and type annotations
- [ ] No hyperparameters hardcoded in `src/` — all read from `cfg`
- [ ] `python main.py --mode smoke` passes end-to-end
- [ ] Existing CLI invocations from `README.md` still work
- [ ] W&B metric namespaces respected (see `src/planners/logging.py`)
- [ ] `minihack_reference/` untouched
- [ ] `CLAUDE.md` codebase map updated if files were added or removed
- [ ] Relevant `.claude/rules/` file updated if a new convention was established

---

## Performance tuning

Three config keys control performance optimisations. All default to off (safe, reference-matching behaviour).

### Mixed precision (`use_amp: true`)

Wraps training forward/backward in `torch.amp.autocast("cuda")` with `GradScaler`. Active in both offline BC and DAgger training.

- **Measured speedup:** 2.2x on gradient steps, 1.7x on full smoke test wall-clock
- **Memory:** peak GPU stays ~16 GB at B=1024 (same as FP32 due to embedding-heavy model)
- **Correctness:** loss trajectory and win rates statistically equivalent to FP32
- **When to use:** always on GPU. No effect on CPU (autocast is a no-op)

### torch.compile (`torch_compile: true`)

Applies `torch.compile(model, mode="default")` before training.

- **Measured speedup:** none beyond AMP alone. Not recommended.
- **When to use:** experimental only. May help on future PyTorch versions with better dynamic shape support.

### Parallel collection (`num_collection_workers: N`)

Runs DAgger episode collection in parallel using `ThreadPoolExecutor`. Each thread gets a CPU copy of the model. NLE and PyTorch CPU release the GIL.

- **Measured speedup:** marginal. CPU model inference is slower than GPU, offsetting parallelism gains. Best avoided unless CPU-only.
- **When to use:** set to 0 (default) for GPU training. May help on CPU-only machines with many cores.

### Profiling

Run `python scripts/profile_dagger.py [key=value ...]` to profile DAgger iteration components. Supports all config overrides (e.g., `use_amp=true`).
