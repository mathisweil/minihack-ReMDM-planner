# CLAUDE.md — minihack-ReMDM-planner

PyTorch ReMDM planner on MiniHack; supervision by the built-in BFS oracle (no expert checkpoint, no submodule). Sibling repo `craftax-ReMDM-planner` (JAX) implements the same method with deliberately shared scaffolding: keep structure, configs, CLIs, docs and tests aligned unless the divergence is environment/framework-forced. When present, the parent workspace `CLAUDE.md` (one level up) governs.

## Configuration

- Exactly two config layers: a `--config` preset merges onto `configs/defaults.yaml`; presets never inherit from presets (there is no `extends` key). `--override KEY=VALUE` keys are validated against `defaults.yaml`.
- `configs/defaults.yaml` IS the final paper recipe, not a neutral baseline, and is authoritative for hyperparameter values; README tables may lag behind it. Presets are delta-only; restating a defaults value silently pins it (enforced by `tests/test_config.py`).
- Silent-pin hazard: the four `offline_*` grad-step/capacity keys override env-step-derived values whenever non-null; a preset deriving its own budget must pin all four to explicit `null`. Read README.md §Configuration before touching any YAML.
- Machine configs (`final_qmul_gpu.yaml`, `final_ucl_gpu.yaml`, `final_ablations_{qmul,ucl}.yaml`) carry machine values only; UCL is the ablation reference and QMUL results are not poolable with it (experiments/README.md). Never edit `final_*` or ablation machine configs to fit local hardware.
- The ablation suite has its own config precedence chain and no `--override`; read experiments/README.md before touching `experiments/rl_finetuning/configs/`.

## Checkpoints

- The model is built from the config: match config to checkpoint (local `.pth` or `wandb:` refs).
- EMA vs raw weights is a result-affecting choice (`--no-ema`); see README.md §Checkpoints.

## Tests

- Run `uv run pytest` after any change. `tests/conftest.py` forces CPU and disables W&B; `slow`-marked tests are excluded by default (`-m 'not slow'`).
- `tests/test_config.py` guards the preset/pin/poolability rules above — a config change that breaks it is wrong until proven otherwise.
- Perf-pin tests (`test_ablation_perf.py`, `test_gpu_step_perf.py`) encode measured expectations; never adjust a pin to make a test pass without investigation.

## Environment and dependencies

- The install path must not contain spaces: MiniHack's `mh_patch_nhdat.sh` fails silently (Python fallback in `src/envs/minihack_env.py`).
- Custom `.des` scenarios belong in `environments/` (ships empty).

## Cautions

- Comments/docstrings cite documents and tracker IDs not present in the repo (e.g. `METHOD_PARITY`, `CHANGES.md`, `CLEANUP_PLAN.md`, FIX-*/PERF-*/CH-* IDs). Treat these as unresolved references, not as sources.
