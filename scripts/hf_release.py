"""Publish the trained MiniHack ReMDM planner checkpoints to the Hugging Face Hub.

Stages every checkpoint under ``checkpoints/`` with the repo-relative layout
preserved, exports safetensors inference weights alongside each full training
state, scrubs cluster paths and W&B identifiers out of the config snapshots,
generates a model card from the checkpoints' own metadata, and uploads.

    HF_TOKEN=hf_xxx uv run python scripts/hf_release.py \\
        --repo-id MathisW78/remdm-minihack-checkpoints [--dry-run] [--private]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CKPTS = ROOT / "checkpoints"
RESULTS = ROOT / "experiments" / "rl_finetuning" / "outputs" / "minihack_final_results"

PAPER = (
    "The Double Intractability of Reinforcement Learning "
    "for Discrete Diffusion Planners"
)
CODE_URL = "https://github.com/mathisweil/minihack-ReMDM-planner"

ROLES = {
    "offline": "Diffusion planner (offline BC)",
    "online": "Diffusion planner (online DAgger)",
    "imported": "Imported baseline",
}

# Cluster paths, W&B identities and hub tokens are environment provenance and
# are never needed to restore a checkpoint.
DROP_PREFIXES = ("wandb_", "hub_")
HUB_IGNORE = ["**/.DS_Store", "**/__pycache__/**"]


def discover() -> dict[Path, list[Path]]:
    """Map each checkpoint directory to its ``.pth`` files, oldest first."""
    models: dict[Path, list[Path]] = {}
    for pth in sorted(CKPTS.rglob("*.pth")):
        models.setdefault(pth.parent, []).append(pth)
    return models


def scrub(cfg: dict) -> dict:
    """Drop W&B/hub keys and shorten absolute cluster paths."""
    clean = {}
    for key, value in cfg.items():
        if key.startswith(DROP_PREFIXES):
            continue
        if isinstance(value, str) and value.startswith("/"):
            value = "/".join(Path(value).parts[-2:])
        clean[key] = value
    return clean


def export_weights(pth: Path, out: Path) -> dict:
    """Write EMA inference weights as safetensors; return checkpoint stats."""
    import torch
    from safetensors.torch import save_file

    ckpt = torch.load(pth, map_location="cpu", weights_only=False)
    model_sd = ckpt["model_state_dict"]
    ema_sd = ckpt.get("ema_state_dict") or {}
    source = "ema_state_dict" if set(ema_sd) >= set(model_sd) else "model_state_dict"
    tensors = {k: v.contiguous() for k, v in (ema_sd if source == "ema_state_dict" else model_sd).items()}
    save_file(tensors, str(out), metadata={"format": "pt", "source": source})

    # The offline trainer stores step x batch under `env_steps`; those are
    # sample-equivalents, not env.step() calls. Only DAgger's are real.
    if "iteration" in ckpt:
        counter, value = "dagger_iteration", ckpt["iteration"]
        trained, unit = f"DAgger iteration {value:,}", "env steps"
    elif "step" in ckpt:
        counter, value = "gradient_step", ckpt["step"]
        trained, unit = f"{value:,} gradient steps", "sample-equivalents"
    else:
        counter, value = None, None
        trained, unit = "final", "sample-equivalents"
    if ckpt.get("env_steps"):
        trained += f", {ckpt['env_steps']:,} {unit}"

    return {
        "params": sum(v.numel() for v in tensors.values()),
        "source": source,
        "counter": counter,
        "value": value,
        "trained": trained,
        "keys": sorted(k for k in ckpt if k.endswith("state_dict") or k in {"rng_states", "curriculum_state"}),
    }


def describe(ckpt_dir: Path, pth: Path, stats: dict, cfg: dict) -> dict[str, str]:
    """Build one model-card row from a checkpoint and its config snapshot."""
    return {
        "path": str(ckpt_dir.relative_to(ROOT)),
        "role": ROLES.get(ckpt_dir.parent.name, ckpt_dir.parent.name),
        "arch": (
            f"{cfg['n_layer']}L, d_model {cfg['n_embd']}, {cfg['n_head']} heads, "
            f"horizon {cfg['seq_len']}"
        ),
        "params": f"{stats['params'] / 1e6:.1f}M",
        "trained": stats["trained"],
        "size": f"{pth.stat().st_size / 1_048_576:.0f} MB",
        "restores": ", ".join(f"`{k}`" for k in stats["keys"]),
    }


def selection(cfg: dict, stats: dict, metric: str | None) -> dict:
    """Record the best-of-N selection a published checkpoint came from."""
    if stats["counter"] == "gradient_step":
        candidates = {
            "unit": "gradient_steps",
            "every": cfg.get("offline_checkpoint_every_grad_steps"),
            "configured_budget": cfg.get("offline_total_grad_steps"),
        }
    else:
        candidates = {
            "unit": "dagger_iterations",
            "every": cfg.get("checkpoint_every"),
            "configured_max": cfg.get("max_iterations"),
        }
    return {
        "policy": "best-of-N over periodic checkpoints",
        "selected": {stats["counter"]: stats["value"]},
        "selection_metric": metric,
        "candidates": candidates,
        "eval_protocol": {
            "episodes_per_env": cfg.get("checkpoint_eval_episodes"),
            "weights": "ema",
            "id_envs": cfg.get("id_envs"),
            "ood_envs": cfg.get("ood_envs"),
        },
    }


def stage(staging: Path, metric: str | None = None) -> list[dict[str, str]]:
    """Copy checkpoints, configs, results, LICENSE into ``staging``."""
    rows = []
    for ckpt_dir, pths in discover().items():
        target = staging / ckpt_dir.relative_to(ROOT)
        target.mkdir(parents=True, exist_ok=True)

        cfg: dict = {}
        for src in sorted(ckpt_dir.iterdir()):
            if src.suffix in {".yaml", ".yml"}:
                cfg = scrub(yaml.safe_load(src.read_text()))
                (target / src.name).write_text(yaml.safe_dump(cfg, sort_keys=True))
            elif src.suffix == ".pth":
                shutil.copy2(src, target / src.name)

        stats = export_weights(pths[-1], target / "model.safetensors")
        (target / "selection.json").write_text(
            json.dumps(selection(cfg, stats, metric), indent=2) + "\n",
        )
        rows.append(describe(ckpt_dir, pths[-1], stats, cfg))

    if RESULTS.is_dir():
        (staging / "results").mkdir()
        for csv in sorted(RESULTS.glob("*.csv")):
            shutil.copy2(csv, staging / "results" / csv.name)

    shutil.copy2(ROOT / "LICENSE", staging / "LICENSE")
    return rows


def model_card(repo_id: str, rows: list[dict[str, str]], metric: str | None) -> str:
    selected_on = (
        f"selected on {metric}" if metric
        else "the metric behind that selection is not recorded in this release"
    )
    header = (
        "| Path | Role | Architecture | Params | Trained to | Full state |\n"
        "|---|---|---|---|---|---|\n"
    )
    table = header + "".join(
        f"| `{r['path']}` | {r['role']} | {r['arch']} | {r['params']} | "
        f"{r['trained']} | {r['size']} |\n"
        for r in sorted(rows, key=lambda r: r["path"])
    )
    return f"""---
license: mit
library_name: pytorch
pipeline_tag: reinforcement-learning
tags:
- reinforcement-learning
- planning
- discrete-diffusion
- remdm
- minihack
- nethack
- pytorch
---

# ReMDM Planner — MiniHack checkpoints

Trained weights accompanying *{PAPER}*: a remasking discrete diffusion model
(ReMDM) used as an action-sequence planner in
[MiniHack](https://github.com/facebookresearch/minihack).

Code, configs and evaluation harness: {CODE_URL}

## Contents

{table}
Each directory holds three things: the original `.pth` training state (weights,
EMA shadow, optimiser, scheduler, and for the DAgger run the curriculum and RNG
state, so training can be resumed exactly), a `model.safetensors` export of the
EMA weights for inference, and the YAML config snapshot the run was trained
under. Paths mirror the source repository, so a snapshot can be dropped
straight into a working copy.

Both files are best-checkpoint selections rather than final-step dumps: each
trainer evaluates every periodic checkpoint on 50 episodes per environment
using EMA weights, and the highest-scoring one is published ({selected_on}).
Each directory's `selection.json` records the selected step, the candidate
cadence and the eval protocol. Directory suffixes
are the sample-equivalents the published model consumed (gradient steps x batch
size, rounded); file names carry each trainer's own counter, DAgger iterations
online and gradient steps offline. The offline run was given the
DAgger-matched budget of 60,000 gradient steps and its best checkpoint fell at
40,000, so the two published models sit at different points on a matched
budget.

`results/` holds the evaluation and ablation tables reported in the paper, as
produced by `experiments/rl_finetuning`. Figures and raw logs stay in the code
repository.

## Download

```python
from huggingface_hub import snapshot_download

# everything
snapshot_download(repo_id="{repo_id}", local_dir=".")

# inference weights only
snapshot_download(
    repo_id="{repo_id}",
    local_dir=".",
    allow_patterns=["**/model.safetensors", "**/config*.yaml"],
)
```

## Use

From a clone of the code repository, after downloading into it:

```bash
DIR=checkpoints/online/Minihack-OnlineDiffusion-DAgger-123M
uv run python main.py --mode inference \\
    --config $DIR/config_iter600.yaml --checkpoint $DIR/iter600.pth
```

Programmatic loading, using the safetensors export:

```python
from safetensors.torch import load_file
from src.config import load_config
from src.models.denoiser import make_model

cfg = load_config(f"{{DIR}}/config_iter600.yaml")
model = make_model(cfg)
model.load_state_dict(load_file(f"{{DIR}}/model.safetensors"))
model.eval()
```

Architecture arguments must come from the checkpoint's own config snapshot
rather than from `configs/defaults.yaml`, which tracks the current code.

## Training

The planners are bidirectional transformers that denoise a masked action plan
conditioned on a cropped MiniHack glyph observation, trained either by offline
behaviour cloning on oracle rollouts or by online DAgger against the oracle
with a dynamic environment curriculum. In-distribution and out-of-distribution
environment sets, remasking strategy, sampling settings and every
hyperparameter are recorded in the per-checkpoint config snapshots, which are
the authoritative record.

## Limitations

These are research artefacts tied to specific MiniHack environment versions and
to the cropped-glyph observation encoding; they are not general-purpose agents
and will not transfer to other environments or to pixel observations.
Evaluation results and their variance are reported in the paper.

## Citation

```bibtex
@inproceedings{{remdm-minihack-planner,
  title  = {{{PAPER}}},
  author = {{Weil, Mathis}},
  year   = {{2026}},
  note   = {{NeurIPS 2026 Workshop: Beyond Next-Token Prediction}}
}}
```

## License

MIT, see `LICENSE`.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--repo-id",
        default="MathisW78/remdm-minihack-checkpoints",
        help="target model repo (default: %(default)s)",
    )
    p.add_argument(
        "--selection-metric",
        help="metric the best checkpoint was chosen on, e.g. 'mean ID+OOD win rate'",
    )
    p.add_argument("--private", action="store_true", help="create the repo private")
    p.add_argument("--dry-run", action="store_true", help="stage and print, do not upload")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    token = os.environ.get("HF_TOKEN")
    if not args.dry_run and not token:
        print("HF_TOKEN is not set.", file=sys.stderr)
        return 1

    if not discover():
        print(f"No .pth checkpoints found under {CKPTS}.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="remdm-minihack-") as tmp:
        staging = Path(tmp)
        rows = stage(staging, args.selection_metric)
        card = model_card(args.repo_id, rows, args.selection_metric)
        (staging / "README.md").write_text(card)

        files = [f for f in staging.rglob("*") if f.is_file()]
        total = sum(f.stat().st_size for f in files) / 1_048_576
        print(f"Staged {len(rows)} checkpoints, {len(files)} files, {total:.0f} MB")
        for r in sorted(rows, key=lambda r: r["path"]):
            print(f"  {r['path']:<60} {r['params']:>7} params  {r['trained']}")
            print(f"  {'':<60} restores {r['restores']}")
        if not args.selection_metric:
            print(
                "Warning: --selection-metric not given; these are best-of-N "
                "checkpoints and the card cannot say what they were chosen on.",
                file=sys.stderr,
            )

        if args.dry_run:
            print(f"\nDry run, nothing uploaded. Card:\n\n{card}")
            return 0

        if not args.yes:
            visibility = "private" if args.private else "public"
            answer = input(f"Upload to {args.repo_id} ({visibility})? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                print("Aborted.")
                return 0

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(staging),
            repo_type="model",
            ignore_patterns=HUB_IGNORE,
            commit_message="Upload MiniHack ReMDM planner checkpoints",
        )
        print(f"Done: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
