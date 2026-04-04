# CLAUDE.md — minihack-ReMDM-planner

> **Backbone file.** Read this first, always. Then follow the `@import` pointers for the area you are working in.

---

## Project snapshot

PyTorch implementation of **ReMDM** (Remasking Discrete Diffusion Model) — a dual-stream transformer that generates 64-step action-sequence plans for [MiniHack](https://github.com/facebookresearch/minihack) navigation environments by iteratively denoising masked token sequences, conditioned on a 9×9 local crop and a full 21×79 dungeon map.

**Primary pipeline — DAgger with implicit warm-start:**
```
[Primary]  DAgger online training          main.py --mode dagger
               |  (seed buffer with oracle demos on iter 0,
               |   collect with model, label with oracle,
               |   efficiency filter, curriculum sampling)
               v  checkpoint
[Evaluate] ID + OOD evaluation             main.py --mode inference --checkpoint iter8000.pth
```

**Optional standalone modes:**
```
[Collect]     Collect oracle demonstrations main.py --mode collect
[Offline BC]  Train on pre-collected data   main.py --mode offline --data dataset.pt
[Smoke test]  Quick end-to-end check        main.py --mode smoke
```

DAgger with implicit warm-start is the recommended pipeline. The `--mode collect` + `--mode offline` path is available for explicit two-stage pre-training on oracle demonstrations before DAgger.

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
│   ├── main.yaml                      # full training (inherits defaults)
│   ├── qmul_gpu.yaml                 # QMUL GPU cluster config
│   ├── ucl_gpu_bigger_model.yaml      # UCL GPU (larger model: 384D, 6 heads)
│   ├── ucl_gpu_learning_behaviour.yaml # UCL GPU (learning study: eta=0.18, B=6144)
│   └── ucl_gpu_no_amp.yaml           # UCL GPU (no AMP: B=3584, 32 workers)
├── environments/                      # Custom .des scenario files
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
│   │   └── denoiser.py               # LocalDiffusionPlannerWithGlobal + LocalDiffusionPlanner + ModelEMA
│   ├── envs/
│   │   ├── minihack_env.py           # AdvancedObservationEnv + BFS oracle
│   │   └── discovery.py              # Env registry scanner + inference benchmark
│   └── planners/
│       ├── collect.py                 # run_model_episode, DataCollector
│       ├── collect_oracle.py          # Standalone oracle data collection (multiprocessing)
│       ├── offline.py                 # offline BC trainer
│       ├── online.py                  # DAgger trainer + checkpointing
│       ├── inference.py               # Evaluator + result formatting
│       ├── smoke.py                   # smoke-test runner
│       └── logging.py                 # centralised W&B + stdout logging
├── experiments/
│   └── rl_finetuning/                 # Stage 3: RL fine-tuning ablation suite
│       ├── run_ablations.py           # CLI entry point (--list, --all, --fast, --analyze_only, --merge)
│       ├── configs/
│       │   ├── ablations_default.yaml # ablation-specific hyperparameters
│       │   ├── ablations_fast.yaml    # fast smoke-test overrides (50 iters)
│       │   ├── ablations_qmul_gpu.yaml # QMUL GPU cluster overrides
│       │   └── ablations_ucl_gpu.yaml  # UCL GPU cluster overrides
│       ├── ablations/
│       │   ├── losses.py              # 16 loss factory functions (return-weighted ELBO variants)
│       │   ├── optimizers.py          # AdamW, LLRD, LoRA, frozen params, PCGrad
│       │   ├── registry.py            # REGISTRY: 25 AblationSpec entries (5 groups)
│       │   └── training.py            # run_ablation loop, data collection, AblationHistory
│       ├── diagnostics/
│       │   ├── gradient.py            # grad alignment, per-layer norms, surgery metrics
│       │   ├── representation.py      # KL drift, CKA, activation norms
│       │   └── timestep.py            # t-bin gradient norms, per-bin losses
│       └── analysis/
│           ├── plots.py               # 9 matplotlib figure generators
│           ├── tables.py              # polars summary tables (CSV + LaTeX)
│           ├── report.py              # hypothesis attribution + diagnosis.md
│           ├── action_distribution.py # pre/post-RL action distribution analysis
│           └── mixing_experiment.py   # data quality degradation curve experiment
├── scripts/
│   ├── hf_upload.py                   # HuggingFace Hub upload utility
│   └── profile_dagger.py             # DAgger iteration profiler (Phase 1)
├── main.py                            # unified CLI entry point
├── pyproject.toml                     # uv/PEP 621 project metadata + dependencies
├── uv.lock                            # deterministic lockfile (committed)
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

- Python 3.12+. Use `match`/`case` only where it genuinely improves clarity.
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
| Array computation + autograd | PyTorch (≥ 2.11.0) |
| Neural networks | `torch.nn` |
| Optimiser | AdamW (`torch.optim.AdamW`) |
| LR scheduling | `torch.optim.lr_scheduler` (CosineAnnealingLR) |
| Environment | MiniHack + NLE (≥ 1.2.0, NetHack-LE fork) |
| Experiment logging | W&B (`wandb`) |
| Config parsing | PyYAML |
| Checkpointing | `torch.save` / `torch.load` |
| Hub uploads | `huggingface_hub` |
| Data analysis | `polars`, `orjson`, `scipy` |
| Package management | `uv` (PEP 621 `pyproject.toml` + `uv.lock`) |

**Device:** CUDA 12 recommended for full training; CPU sufficient for smoke tests. Always set via `cfg.device`.

---

## CLI surface

| Flag | Description |
|---|---|
| `--mode` | Required. One of `smoke`, `collect`, `offline`, `dagger`, `inference` |
| `--config PATH` | Config file (default: `configs/defaults.yaml`) |
| `--data PATH` | Dataset `.pt` file (offline mode) |
| `--checkpoint PATH` | Checkpoint `.pth` file |
| `--wandb-artifact REF` | W&B artifact reference (e.g. `entity/project/name:latest`) |
| `--no-warm-start` | Skip model warm-start from checkpoint (DAgger) |
| `--no-ema` | Use training weights instead of EMA for inference |
| `--envs ENV [ENV ...]` | Override evaluation environments |
| `--des PATH [PATH ...]` | Custom `.des` scenario files for evaluation |
| `--episodes N` | Episodes per environment (default: 50) |
| `--output PATH` | Save evaluation results to JSON |
| `--blind-global` | Zero out global map observations (local-only ablation) |

Any config key can also be overridden via `key=value` pairs on the command line.

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

Three config keys control performance optimisations. Defaults are set for GPU training; override for CPU or different hardware.

### Mixed precision (`use_amp: true`)

Wraps training forward/backward in `torch.amp.autocast("cuda")` with `GradScaler`. Active in both offline BC and DAgger training.

- **Measured speedup:** 2.2x on gradient steps, 1.7x on full smoke test wall-clock
- **Memory:** peak GPU stays ~16 GB at B=3584 (same as FP32 due to embedding-heavy model)
- **Correctness:** loss trajectory and win rates statistically equivalent to FP32
- **When to use:** always on GPU. No effect on CPU (autocast is a no-op)
- **Default:** `false` in `defaults.yaml`; enabled in GPU-specific configs

### torch.compile (`torch_compile: true`)

Applies `torch.compile(model, mode="default")` before training. Falls back gracefully if no C compiler is found (common on managed GPU nodes).

- **Measured speedup:** none beyond AMP alone. Not recommended for primary training.
- **Default:** `true` in `defaults.yaml`
- **When to use:** experimental only. May help on future PyTorch versions with better dynamic shape support.

### Parallel collection (`num_collection_workers: N`)

DAgger episode collection supports three strategies (auto-selected):
1. **GPU-batched** (default on CUDA with `episodes_per_iteration > 1`): all envs in lockstep
2. **Threaded CPU** (fallback when `num_collection_workers > 0`): `ThreadPoolExecutor` with CPU model copies
3. **Sequential** (reference behaviour): one episode at a time

- **Default:** `8` workers in `defaults.yaml`
- **When to use:** GPU-batched is preferred; workers primarily affect the CPU fallback path

### Profiling

Run `python scripts/profile_dagger.py [key=value ...]` to profile DAgger iteration components. Supports all config overrides (e.g., `use_amp=true`).
