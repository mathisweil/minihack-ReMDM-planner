# CLI migration (branch chore/align-cli)

Interface alignment with `craftax-ReMDM-planner`. Old names are removed outright; there are no aliases.

## main.py

| Old | New | Layer change |
|---|---|---|
| `--mode dagger` | `--mode online` | none (CLI) |
| `KEY=VALUE` (bare positional override, any key silently accepted) | `--override KEY=VALUE` (repeatable; unknown key or wrong type is an error) | none (CLI) |
| `--wandb-artifact REF` | `--checkpoint wandb:REF` | none (CLI) |
| `--n-seeds N` | `--num-seeds N` | none (CLI) |
| `--episodes N` (argparse default 50) | `--episodes N` (default now taken from `eval_episodes_per_env`) | duplicated default removed |
| `collect_output=PATH` (override) | `--data PATH` (or `--override collect_output=PATH`); config key kept as the default | config default + CLI override |
| n/a | `--seed N` (new; overrides the config `seed`) | config -> CLI |

Unchanged: `--config`, `--checkpoint`, `--data` (offline input), `--output`, `--envs`, `--des`, `--algo`, `--seeds`, `--no-warm-start`, `--no-ema`, `--blind-global`.

Config semantics: unchanged (`--config` was already deep-merged onto `configs/defaults.yaml`). New: unknown keys in a `--config` file are now rejected instead of silently carried; `device` remains valid (appears in checkpoint config snapshots).

## experiments/rl_finetuning/run_ablations.py

| Old | New |
|---|---|
| `--ablations_config` | `--ablations-config` |
| `--analyze_only` | `--analyze-only` |
| `--results_path` | `--results-path` |
| `--output_dir` | `--output-dir` |
| `--run_id` | `--run-id` |
| `--num_seeds` | `--num-seeds` |
| `--use_wandb` | `--use-wandb` |
| `--wandb_project` | `--wandb-project` |
| `--wandb_resume_id` | `--wandb-resume-id` |
| `--max_iter` | `--max-iter` |
| `--batch_size` | `--batch-size` |
| `--eval_every` | `--eval-every` |
| `--checkpoint wandb://REF` | `--checkpoint wandb:REF` |

Config keys (`experiments/rl_finetuning/configs/*.yaml`):

| Old | New | Note |
|---|---|---|
| `ablation_use_wandb` | `use_wandb` | still not read; W&B stays opt-in via `--use-wandb` (wiring the config value would silently enable W&B, a behaviour change) |
| `ablation_wandb_project` | `wandb_project` | now read as a fallback when `--wandb-project` is not given (shipped value identical to the old hardcoded fallback) |

## scripts/

| Old | New |
|---|---|
| `python scripts/profile_dagger.py key=value ...` | `python scripts/profile_dagger.py --override key=value ...` |

`hf_release.py`, `hf_upload_demo.py` were already kebab-case; unchanged.

## pyproject.toml

- New extra: `uv sync --extra cuda` installs torch 2.11+cu126 (Linux). Replaces the README's manual `uv pip install torch --index-url .../whl/cu121`, which could no longer satisfy `torch>=2.11.0` (that index stops at 2.5.1). Same extra name as the craftax repo.
- `pytest` removed from base `dependencies` (kept in the `dev` group, which `uv sync` installs by default).

## Defaults deliberately different from craftax-ReMDM-planner

Benchmark-tuned values, unchanged by this alignment:

| Key | minihack | craftax |
|---|---|---|
| `eta` | 0.15 | 0.5 |
| `remask_strategy` | `conf` | `rescale` |
| noise schedule | `noise_schedule: linear` | `diffusion_schedule: cosine` |
| dropout | `dropout: 0.0` (masking already regularises) | `dropout_rate: 0.1` |
| plan length | `seq_len: 64` | `plan_horizon: 32` |
| training budget | `total_timesteps: 2e6` (unified, env steps) | `offline/online_total_timesteps` (per mode, env frames) |
| LR | per-mode `offline_lr`/`dagger_lr` | single `lr` |
| top-K vs top-p | `top_k: 4` | `top_p: 0.95` |

Model-architecture key names (`n_embd`/`n_head`/`n_layer` vs `d_model`/`n_heads`/`n_layers`, etc.) are deliberately **not** renamed: every released HF checkpoint ships a config snapshot under the existing names and the documented workflow evaluates with that snapshot.

## Noticed but not touched

- `pyproject.toml` `description` is still the uv placeholder.
- Offline-BC resume reuses `--checkpoint` while craftax uses `--resume` + metadata sidecar; the mechanisms differ (internals, out of scope).
- In-code `getattr(cfg, key, default)` fallbacks duplicating `defaults.yaml` values remain: the ablation suite builds partial configs that rely on them, so removing them would change behaviour there.
- `.idea/`, `__pycache__/`, `.DS_Store` present in the working tree.
