"""Upload the trained MiniHack ReMDM checkpoints and results to the Hugging Face Hub.

Discovers every checkpoint under ``checkpoints/``, every ablation run under
``experiments/rl_finetuning/outputs/``, every ``--mode inference`` result
under ``results/inference/`` and the manuscript figure PDFs under
``results/paper_figures/``, stages them with the repo-relative layout
preserved, drops W&B and hub metadata (which carries the author's account,
hostname and local paths), exports safetensors inference weights alongside each
full training state, generates a model card from the checkpoints' own config
snapshots, and uploads.

    HF_TOKEN=hf_xxx uv run python scripts/hf_upload.py \\
        --repo-id mathisweil/remdm-minihack-checkpoints \\
        [--inference-results PATH ...] [--dry-run] [--private]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

# scripts/ is not a package, so the helper is imported by bare name.
# Python already puts this file's directory on sys.path when the script is
# run directly; the explicit insert is for the file-location loaders the
# tests use (tests/test_config.py), which do not.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _git_provenance import copy_tracked_file  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPTS = ROOT / "checkpoints"
RUNS = ROOT / "experiments" / "rl_finetuning" / "outputs"
INFERENCE = ROOT / "results" / "inference"
PAPER_FIGURES = ROOT / "results" / "paper_figures"

PAPER = (
    "Return-Weighted ELBO Fine-Tuning Degrades "
    "Masked Diffusion Planners"
)
CODE_URL = "https://github.com/mathisweil/minihack-ReMDM-planner"
ENV_NAME = "MiniHack"

ROLES = {
    "offline": "Diffusion planner (offline BC)",
    "online": "Diffusion planner (online DAgger)",
    "imported": "Imported baseline",
}

# An ablation run is published as its own summary plus tables and figures; the
# raw per-iteration logs stay in the code repository.
RUN_FILES = ("results.json", "diagnosis.md")
# gdelta/ holds the per-seed gradient measurements behind the paper's
# decomposition appendix; they land inside the run directory, so
# discover_runs() finds them with no further change.
RUN_DIRS = ("tables", "figures", "gdelta")

# Environment provenance, never needed to restore a checkpoint, and dropped
# from every published config in both repos: the nested `_wandb` blob (email,
# host, git remote, absolute paths), every `wandb_*` and `hub_*` key, and
# `use_wandb` itself — which matches neither prefix, so both repos shipped it
# in the released config while claiming to scrub W&B settings. Keys are
# compared lower-cased, because craftax records them UPPERCASE and minihack
# lower-case.
DROP_PREFIXES = ("hub_",)
# `wandb_` is matched anywhere in the key, not only at the start. A prefix
# rule caught `wandb_project` but not `baselines_wandb_project`, which was
# published in both released config.yaml files, nor the sibling's
# `RESUME_WANDB_RUN_ID` -- the same way `use_wandb` slipped past originally.
# The underscore is what keeps it narrow: `wandbish` contains `wandb` but not
# `wandb_`, so the negative assertions still hold in both repos.
DROP_SUBSTRINGS = ("wandb_",)
DROP_KEYS = ("_wandb", "use_wandb")


def is_environment_key(key: str) -> bool:
    """True for a config key that is provenance rather than recipe."""
    lowered = key.lower()
    return (
        lowered in DROP_KEYS
        or lowered.startswith(DROP_PREFIXES)
        or any(mark in lowered for mark in DROP_SUBSTRINGS)
    )


COPY_IGNORE = shutil.ignore_patterns(
    ".DS_Store", "__pycache__", "*.pyc", "wandb-metadata.json",
)
HUB_IGNORE = ["**/.DS_Store", "**/__pycache__/**", "**/wandb-metadata.json"]


# =============================================================================
# Discovery
# =============================================================================

# `checkpoints/hf/` is where a Hub *download* lands. Publishing from it would
# re-upload already-published artefacts into a nested `checkpoints/hf/...` tree
# on the Hub, so it is never a publish source in either repo.
HF_DOWNLOAD_DIR = "hf"


def _is_download_copy(path: Path) -> bool:
    """True for anything under ``checkpoints/hf/``, wherever it sits."""
    return HF_DOWNLOAD_DIR in path.relative_to(CKPTS).parts


def discover_checkpoints() -> dict[Path, list[Path]]:
    """Map each checkpoint directory to its ``.pth`` files, oldest first.

    Discovery is at the **released layout**, ``checkpoints/<role>/<name>/`` —
    the layout the Hub repo mirrors, matching the sibling repo. A recursive
    ``rglob("*.pth")`` found everything instead, which is wrong in both
    directions: measured on this repo's live tree it discovered 10 directories,
    **none of them at the released layout** — seven raw ``dagger_<timestamp>/``
    run directories, the repository root itself with two loose files, and **two
    already-published artefacts under** ``checkpoints/hf/``, which a publish
    would have pushed back up into a nested ``checkpoints/hf/checkpoints/...``
    tree.

    A training run writes to its own directory, so its checkpoints have to be
    copied into ``checkpoints/{offline,online}/<name>/`` before publishing. The
    README documents that; it is the same requirement the sibling repo has
    always had, and it is now stated on both sides rather than one.

    Anything under ``checkpoints/hf/`` is skipped as a download copy, explicitly
    rather than by depth arithmetic.

    Returns:
        ``{checkpoint directory: [.pth paths, oldest first]}``.
    """
    models: dict[Path, list[Path]] = {}
    for pth in sorted(CKPTS.glob("*/*/*.pth")):
        if _is_download_copy(pth):
            continue
        models.setdefault(pth.parent, []).append(pth)
    return models


def discover_runs() -> list[Path]:
    """Every ablation output directory holding a ``results.json``."""
    if not RUNS.is_dir():
        return []
    return sorted(d for d in RUNS.iterdir() if (d / "results.json").is_file())


def discover_paper_figures() -> list[Path]:
    """The manuscript figures, as vector PDFs.

    Built by ``scripts/paper_figures.py``, which lives in the sibling repo
    because each figure reads *both* repositories' ablation ``results.json``
    and puts Craftax Classic and MiniHack side by side in one figure. The
    resulting PDFs therefore describe this repo's results as much as the
    sibling's, and both Hub repos publish the same set from
    ``results/paper_figures/``.

    They were previously published by neither: discovery covered only
    ``checkpoints/``, the ablation run directories and ``results/inference/``,
    so ``results/paper_figures/`` matched nothing and the upload said nothing
    about it.
    """
    if not PAPER_FIGURES.is_dir():
        return []
    return sorted(PAPER_FIGURES.glob("*.pdf"))


def discover_inference(extra: list[str]) -> list[Path]:
    """Inference result JSONs: the default directory plus any given paths."""
    found: list[Path] = []
    for source in [INFERENCE, *(Path(p) for p in extra)]:
        if source.is_dir():
            found.extend(sorted(source.glob("*.json")))
        elif source.is_file():
            found.append(source)
        elif source != INFERENCE:
            print(f"No inference results at {source}.", file=sys.stderr)
    return list(dict.fromkeys(found))


# =============================================================================
# Helpers
# =============================================================================

def dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1_048_576
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576


def human_size(path: Path) -> str:
    mb = dir_size_mb(path)
    return f"{mb:.0f} MB" if mb >= 1 else f"{max(mb * 1024, 1):.0f} KB"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def shorten_paths(value):
    """Shorten absolute cluster paths anywhere in a staged JSON document."""
    if isinstance(value, dict):
        return {k: shorten_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shorten_paths(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        return "/".join(Path(value).parts[-2:])
    return value


def environment_key_paths(value, path: str = "") -> list[str]:
    """Every dotted path at which an environment key appears, at any depth.

    The reporting counterpart to :func:`drop_environment_keys`, so the
    uploader can name what it removed rather than claiming a scrub happened.
    Recursion stops at a key that is itself dropped: its whole subtree goes.
    """
    found: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            name = str(key)
            here = f"{path}.{name}" if path else name
            if is_environment_key(name):
                found.append(here)
            else:
                found.extend(environment_key_paths(sub, here))
    elif isinstance(value, (list, tuple)):
        for i, sub in enumerate(value):
            found.extend(environment_key_paths(sub, f"{path}[{i}]"))
    return found


def drop_environment_keys(value):
    """Strip environment keys at **every** depth, not just the top level.

    Filtering only the top level was a real defect in the sibling repo: its
    path-shortener recursed while its key-filter did not, so W&B project and
    entity survived one level down inside a nested config snapshot and were
    published, while the uploader reported a successful scrub. This repo's
    checkpoints happen to carry no nested config, so the same one-level filter
    leaked nothing here -- clean for a reason that was not the code being
    right, and one schema change away from leaking silently.

    Mapping types are preserved, so an ``OrderedDict`` state dict stays an
    ``OrderedDict`` and the published checkpoint keeps its original structure.
    """
    if isinstance(value, dict):
        kept = {
            key: drop_environment_keys(sub)
            for key, sub in value.items()
            if not is_environment_key(str(key))
        }
        if type(value) is dict:
            return kept
        try:
            return type(value)(kept)
        except TypeError:
            # A mapping needing constructor arguments (e.g. a defaultdict
            # factory). Structure matters less than not shipping the key.
            return kept
    if isinstance(value, list):
        return [drop_environment_keys(sub) for sub in value]
    return value


def scrub(cfg: dict) -> dict:
    """Drop the environment keys and shorten absolute cluster paths.

    The prefixes alone missed `use_wandb`, which starts with neither, so it
    shipped in every released config; the sibling repo dropped a different set
    again, and left `USE_WANDB` with it. Both now drop the same one.

    Both passes recurse. They used to disagree -- ``shorten_paths`` descended
    and the key filter did not -- which is the asymmetry that leaked W&B
    settings from a nested config snapshot in the sibling repo.
    """
    return drop_environment_keys(shorten_paths(cfg))


#: An absolute POSIX path with at least three components, as it appears
#: inside a JSON string, a Markdown table or a LaTeX table cell.
# The lookbehind anchors the match to an absolute path. Without it the regex
# also matched from the middle of a relative one: `experiments/rl_finetuning/
# analysis/tables.py` in a generated LaTeX comment became
# `experimentsanalysis/tables.py`, because matching could start at the slash
# after `experiments`.
#
# `@` and `+` are in the class because the lookbehind makes a narrow class
# leak rather than merely under-match: with them omitted,
# `/home/user@dom/proj/run/file.txt` fails at the start (`user@dom` is not one
# component) and the lookbehind then refuses to restart at `/proj`, so there
# is no fallback match and the whole absolute path ships unshortened. `@` is
# exactly where an email-like component appears in a home-directory path,
# which is the leak class this scrub exists for.
_ABS_PATH = re.compile(r"(?<![\w.@+\-])/(?:[\w.@+\-]+/){2,}[\w.@+\-]+")

#: What a staged tree holds that is worth reading as text.
_TEXT_SUFFIXES = (".md", ".txt", ".csv", ".tex")
_DOC_SUFFIXES = (".json", ".yaml", ".yml")


def shorten_text_paths(text: str) -> str:
    """Apply the ``shorten_paths`` rule to paths embedded in plain text.

    ``shorten_paths`` only reaches strings it can find by walking a parsed
    document. The same cluster paths appear inside ``diagnosis.md`` prose and
    LaTeX table cells, where nothing parses them.
    """
    return _ABS_PATH.sub(lambda m: "/".join(Path(m.group(0)).parts[-2:]), text)


def scrub_staged_tree(staging: Path) -> list[str]:
    """Scrub every document and text file in the staged tree, in place.

    The stagers copy whole directories -- ``tables/``, ``figures/``,
    ``gdelta/`` -- so naming the files to scrub means maintaining a list by
    hand, and a list of known filenames is the same mistake as a known
    nesting. Walking what was actually staged makes the set of files scrubbed
    identical to the set published, by construction.

    This closes a reach bug, not a depth one: ``stage_runs`` copied
    ``results.json`` with ``shutil.copy2`` and scrubbed nothing at all, so the
    released file carried `wandb_project`, `wandb_entity` and the absolute
    path of the machine it ran on. ``scrub`` was correct and was simply never
    called on it.

    A file is rewritten only when scrubbing changes it, so anything already
    clean keeps its bytes.

    Returns:
        Staged-relative paths of the files that changed.
    """
    changed: list[str] = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _DOC_SUFFIXES:
            raw = path.read_text()
            try:
                doc = (json.loads(raw) if suffix == ".json"
                       else yaml.safe_load(raw))
            except (json.JSONDecodeError, yaml.YAMLError):
                continue
            if not isinstance(doc, (dict, list)):
                continue
            clean = drop_environment_keys(shorten_paths(doc))
            if clean == doc:
                continue
            path.write_text(
                json.dumps(clean, indent=2) + "\n" if suffix == ".json"
                else yaml.safe_dump(clean, sort_keys=False)
            )
            changed.append(str(path.relative_to(staging)))
        elif suffix in _TEXT_SUFFIXES:
            raw = path.read_text()
            clean = shorten_text_paths(raw)
            if clean != raw:
                path.write_text(clean)
                changed.append(str(path.relative_to(staging)))
    return changed


def scrub_checkpoint(src: Path, dst: Path) -> None:
    """Copy a ``.pth`` with its provenance keys dropped.

    ``scrub`` only reaches the YAML/JSON sidecars, so `wandb_run_id` shipped
    inside the pickled checkpoints themselves (iter563.pth -> a07wlxl7,
    offline_step50000.pth -> 4nzxat0c) while the uploader reported that W&B
    settings had been scrubbed. Environment keys are dropped with the same
    predicate the configs use; every training-state key is preserved.
    """
    import torch

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        shutil.copy2(src, dst)
        return
    dropped = environment_key_paths(ckpt)
    if not dropped:
        # Nothing to remove, so copy the bytes rather than re-saving them:
        # torch.save is not byte-reproducible, and a clean checkpoint should
        # not change on the Hub just for having been inspected.
        shutil.copy2(src, dst)
        return
    torch.save(drop_environment_keys(ckpt), dst)
    print(f"  scrubbed {', '.join(sorted(dropped))} from {src.name}")


def export_weights(pth: Path, out: Path) -> dict:
    """Write EMA inference weights as safetensors; return checkpoint stats."""
    import torch
    from safetensors.torch import save_file

    ckpt = torch.load(pth, map_location="cpu", weights_only=False)
    model_sd = ckpt["model_state_dict"]
    ema_sd = ckpt.get("ema_state_dict") or {}
    source = "ema_state_dict" if set(ema_sd) >= set(model_sd) else "model_state_dict"
    tensors = {
        k: v.contiguous()
        for k, v in (ema_sd if source == "ema_state_dict" else model_sd).items()
    }
    save_file(tensors, str(out), metadata={"format": "pt", "source": source})

    # The offline trainer stores step x batch under `env_steps`; those are
    # sample-equivalents, not env.step() calls. Only DAgger's are real.
    if "iteration" in ckpt:
        counter, value = "dagger_iteration", ckpt["iteration"]
        step, unit = f"iteration {value:,}", "env steps"
    elif "step" in ckpt:
        counter, value = "gradient_step", ckpt["step"]
        step, unit = f"gradient step {value:,}", "sample-equivalents"
    else:
        counter, value = None, None
        step, unit = "final", "sample-equivalents"

    return {
        "params": sum(v.numel() for v in tensors.values()),
        "source": source,
        "counter": counter,
        "value": value,
        "step": step,
        "detail": f"{ckpt['env_steps']:,} {unit}" if ckpt.get("env_steps") else step,
        "keys": sorted(
            k for k in ckpt
            if k.endswith("state_dict") or k in {"rng_states", "curriculum_state"}
        ),
    }


def required(cfg: dict, key: str) -> object:
    """Read a key the published record depends on, or fail the upload.

    ``.get()`` here published a null instead: every DAgger `selection.json`
    released before this guard recorded `"every": null` because the two keys
    read had been renamed out of the config, while the model card claims each
    file records the budget it came from. A declared-but-null value is still
    accepted -- the four `offline_*` keys are nullable by design.
    """
    if key not in cfg:
        raise KeyError(
            f"{key} is absent from the checkpoint's config snapshot, so "
            f"selection.json cannot record the candidate set it came from"
        )
    return cfg[key]


def selection(cfg: dict, stats: dict, metric: str | None) -> dict:
    """Record the best-of-N selection a published checkpoint came from."""
    if stats["counter"] == "gradient_step":
        candidates = {
            "unit": "gradient_steps",
            "every": required(cfg, "offline_checkpoint_every_grad_steps"),
            "configured_budget": required(cfg, "offline_total_grad_steps"),
        }
    else:
        # DAgger checkpoints on an env-step cadence (`online.py` compares
        # against `env_steps_total`) but names its files by iteration, so the
        # candidate set is denominated in env steps and `selected` is not.
        candidates = {
            "unit": "env_steps",
            "every": required(cfg, "checkpoint_every_timesteps"),
            "configured_max": required(cfg, "total_timesteps"),
        }
    return {
        "policy": "best-of-N over periodic checkpoints",
        "selected": {stats["counter"]: stats["value"]},
        "selection_metric": metric,
        "candidates": candidates,
        "eval_protocol": {
            "episodes_per_env": required(cfg, "checkpoint_eval_episodes"),
            "weights": "ema",
            "id_envs": required(cfg, "id_envs"),
            "ood_envs": required(cfg, "ood_envs"),
        },
    }


# =============================================================================
# Description
# =============================================================================

def describe(
    model_dir: Path, staged: Path, pth: Path, cfg_name: str, stats: dict, cfg: dict,
) -> dict[str, str]:
    """Pull env name and training detail out of a checkpoint's own metadata."""
    return {
        "path": str(model_dir.relative_to(ROOT)),
        "role": ROLES.get(model_dir.parent.name, model_dir.parent.name),
        "env": cfg.get("env_name") or ENV_NAME,
        "arch": (
            f"{cfg['n_layer']}L, d_model {cfg['n_embd']}, {cfg['n_head']} heads, "
            f"horizon {cfg['seq_len']}, {stats['params'] / 1e6:.0f}M params"
        ),
        "step": stats["step"],
        "detail": stats["detail"],
        "size": human_size(staged),
        "restores": ", ".join(f"`{k}`" for k in stats["keys"]),
        "file": pth.name,
        "config": cfg_name,
    }


def describe_run(run: Path, staged: Path) -> dict[str, str]:
    """Summarise what an ablation run contributes to the release."""
    counts = [
        f"{len(list((staged / d).glob('*')))} {d}"
        for d in RUN_DIRS if (staged / d).is_dir()
    ]
    files = [f for f in RUN_FILES if (staged / f).is_file()]
    return {
        "run": run.name,
        "path": str(run.relative_to(ROOT)),
        "contents": ", ".join([*(f"`{f}`" for f in files), *counts]),
        "size": human_size(staged),
    }


def describe_inference(name: str, payload: dict) -> dict[str, str]:
    """Summarise one ``--mode inference`` result JSON."""
    results = payload.get("results", payload)
    per_env = {
        env: stats for env, stats in results.items()
        if isinstance(stats, dict) and "win_rate" in stats
    }
    rates = [stats["win_rate"] for stats in per_env.values()]
    episodes = {stats.get("n_episodes") for stats in per_env.values()}
    return {
        "file": name,
        "env": (
            next(iter(per_env)) if len(per_env) == 1
            else f"{len(per_env)} envs" if per_env else "-"
        ),
        "episodes": (
            f"{max(e for e in episodes if e)} episodes per env"
            if episodes - {None} else "-"
        ),
        "metric": (
            f"mean win rate {sum(rates) / len(rates):.2f}" if rates else "-"
        ),
    }


# =============================================================================
# Staging
# =============================================================================

def stage_checkpoints(
    staging: Path, models: dict[Path, list[Path]], metric: str | None,
) -> list[dict[str, str]]:
    """Copy each checkpoint directory, scrubbing its provenance metadata."""
    rows = []
    for model_dir, pths in models.items():
        target = staging / model_dir.relative_to(ROOT)
        target.mkdir(parents=True, exist_ok=True)

        cfg: dict = {}
        cfg_name = ""
        for src in sorted(model_dir.iterdir()):
            if src.suffix in {".yaml", ".yml"}:
                cfg, cfg_name = scrub(yaml.safe_load(src.read_text())), src.name
                (target / src.name).write_text(yaml.safe_dump(cfg, sort_keys=True))
            elif src.suffix == ".pth":
                scrub_checkpoint(src, target / src.name)

        stats = export_weights(pths[-1], target / "model.safetensors")
        (target / "selection.json").write_text(
            json.dumps(selection(cfg, stats, metric), indent=2) + "\n",
        )
        rows.append(describe(model_dir, target, pths[-1], cfg_name, stats, cfg))
    return rows


def stage_runs(staging: Path, runs: list[Path]) -> list[dict[str, str]]:
    """Copy each ablation run's summary, tables and figures."""
    rows = []
    for run in runs:
        target = staging / run.relative_to(ROOT)
        target.mkdir(parents=True, exist_ok=True)
        for name in RUN_FILES:
            if (run / name).is_file():
                shutil.copy2(run / name, target / name)
        for name in RUN_DIRS:
            if (run / name).is_dir():
                shutil.copytree(run / name, target / name, ignore=COPY_IGNORE)
        rows.append(describe_run(run, target))
    return rows


def stage_inference(staging: Path, files: list[Path]) -> list[dict[str, str]]:
    """Copy each inference result JSON into ``results/inference/``."""
    target_dir = staging / INFERENCE.relative_to(ROOT)
    rows: list[dict[str, str]] = []
    for src in files:
        try:
            payload = json.loads(src.read_text())
        except json.JSONDecodeError:
            print(f"Skipping unreadable inference result {src}.", file=sys.stderr)
            continue
        name = src.name
        if any(r["file"] == name for r in rows):
            name = f"{src.parent.name}-{src.name}"
        target_dir.mkdir(parents=True, exist_ok=True)
        # scrub, not shorten_paths: this shortened cluster paths while doing
        # nothing about environment keys, the same asymmetry as stage_runs one
        # step further along. These payloads carry no config today.
        (target_dir / name).write_text(
            json.dumps(scrub(payload), indent=2) + "\n",
        )
        row = describe_inference(name, payload)
        row["size"] = human_size(target_dir / name)
        rows.append(row)
    return rows


def stage_paper_figures(staging: Path, files: list[Path]) -> list[dict[str, str]]:
    """Copy the manuscript figures into ``results/paper_figures/``."""
    if not files:
        return []
    target_dir = staging / PAPER_FIGURES.relative_to(ROOT)
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in files:
        shutil.copy2(src, target_dir / src.name)
        rows.append({"file": src.name, "size": human_size(target_dir / src.name)})
    return rows


def stage(
    staging: Path,
    models: dict[Path, list[Path]],
    runs: list[Path],
    inference: list[Path],
    figures: list[Path],
    metric: str | None,
) -> tuple[
    list[dict[str, str]], list[dict[str, str]],
    list[dict[str, str]], list[dict[str, str]],
]:
    """Stage checkpoints, results and LICENSE; the card is written by the caller.

    LICENSE comes from git rather than the working tree: a
    ``hf download --local-dir .`` overwrites it with the Hub's own copy, and
    publishing from the tree would push that straight back up as current.
    """
    rows = stage_checkpoints(staging, models, metric)
    run_rows = stage_runs(staging, runs)
    inf_rows = stage_inference(staging, inference)
    fig_rows = stage_paper_figures(staging, figures)
    # ROOT is passed explicitly rather than read inside the helper, so a
    # test can rebind it and have that take effect.
    copy_tracked_file("LICENSE", staging / "LICENSE", ROOT)
    # Last, so it covers everything every stager copied -- including the whole
    # directories stage_runs copies wholesale, which no stager scrubbed.
    for name in scrub_staged_tree(staging):
        print(f"  scrubbed provenance from {name}")
    return rows, run_rows, inf_rows, fig_rows


# =============================================================================
# Model card
# =============================================================================

def table(headers: list[str], lines: list[str]) -> str:
    sep = "|".join(["---"] * len(headers))
    return f"| {' | '.join(headers)} |\n|{sep}|\n" + "".join(f"| {ln} |\n" for ln in lines)


def checkpoint_table(rows: list[dict[str, str]]) -> str:
    return table(
        ["Path", "Role", "Environment", "Architecture", "Selected at", "Training", "Size"],
        [
            f"`{r['path']}` | {r['role']} | `{r['env']}` | {r['arch']} | "
            f"{r['step']} | {r['detail']} | {r['size']}"
            for r in sorted(rows, key=lambda r: r["path"])
        ],
    )


def results_section(
    run_rows: list[dict[str, str]],
    inf_rows: list[dict[str, str]],
    fig_rows: list[dict[str, str]],
) -> str:
    """Ablation and inference tables; empty when the release carries neither."""
    parts = []
    if run_rows:
        parts.append(
            "RL fine-tuning ablation runs, as produced by "
            "`experiments/rl_finetuning/run_ablations.py`. Each run ships its "
            "`results.json` summary, the `diagnosis.md` write-up, and the "
            "tables (`.csv` and `.tex`) and figures generated from it.\n\n"
            + table(
                ["Run", "Contents", "Size"],
                [
                    f"`{r['path']}` | {r['contents']} | {r['size']}"
                    for r in sorted(run_rows, key=lambda r: r["run"])
                ],
            ),
        )
    if inf_rows:
        parts.append(
            "Evaluation results produced by `main.py --mode inference` on the "
            "checkpoints above, under `results/inference/`.\n\n"
            + table(
                ["File", "Environment", "Evaluation", "Headline metric", "Size"],
                [
                    f"`{r['file']}` | `{r['env']}` | {r['episodes']} | "
                    f"{r['metric']} | {r['size']}"
                    for r in sorted(inf_rows, key=lambda r: r["file"])
                ],
            ),
        )
    if fig_rows:
        parts.append(
            "Manuscript figures as vector PDF, under `results/paper_figures/`. "
            "Each puts Craftax Classic and MiniHack side by side, so they are "
            "built from both repositories' ablation `results.json` by the "
            "sibling repo's `scripts/paper_figures.py` and published "
            "identically in both Hub repos.\n\n"
            + table(
                ["Figure", "Size"],
                [
                    f"`{r['file']}` | {r['size']}"
                    for r in sorted(fig_rows, key=lambda r: r["file"])
                ],
            ),
        )
    return "## Results\n\n" + "\n".join(parts) if parts else ""


def featured(rows: list[dict[str, str]]) -> dict[str, str]:
    """The checkpoint the download and usage examples are written against."""
    planners = [r for r in rows if "planner" in r["role"].lower()]
    return sorted(planners or rows, key=lambda r: r["path"])[0]


def model_card(
    repo_id: str,
    rows: list[dict[str, str]],
    run_rows: list[dict[str, str]],
    inf_rows: list[dict[str, str]],
    fig_rows: list[dict[str, str]],
    total_mb: float,
    metric: str | None,
) -> str:
    example = featured(rows)
    selected_on = (
        f"selected on {metric}" if metric
        else "the metric behind that selection is not recorded in this release"
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

# ReMDM Planner: {ENV_NAME} checkpoints

Trained weights accompanying *{PAPER}*: a remasking discrete diffusion model
(ReMDM) used as an action-sequence planner in
[MiniHack](https://github.com/facebookresearch/minihack), together with the BFS
oracle rollouts that supervise it, and the results reported in the paper.

Code, configs and evaluation harness: {CODE_URL}

## Contents

{checkpoint_table(rows)}
Each checkpoint ships the `.pth` training state it was published from (weights,
EMA shadow, optimiser, scheduler, and for the DAgger run the curriculum and RNG
state, so training resumes exactly), a `model.safetensors` export of the EMA
weights for inference, the YAML config snapshot it was trained under, and a
`selection.json` recording how it was chosen.

Weights are PyTorch training states with a `safetensors` export of the EMA
weights alongside, and the paths above mirror the source repository so a
snapshot can be dropped straight into a working copy.

{results_section(run_rows, inf_rows, fig_rows)}
## Download

This repo mirrors the code repository's layout, so a snapshot drops straight
into a working copy -- but it also carries its own `README.md` (this card),
`LICENSE` and `.gitattributes`, and `local_dir="."` would overwrite the code
repository's copies of all three. Exclude them, or download into a directory
of its own.

```python
from huggingface_hub import snapshot_download

# everything (~{total_mb:.0f} MB), into a clone of the code repository
snapshot_download(
    repo_id="{repo_id}",
    local_dir=".",
    ignore_patterns=["README.md", "LICENSE", ".gitattributes"],
)

# or somewhere of its own, leaving any working copy untouched
snapshot_download(repo_id="{repo_id}", local_dir="remdm-minihack")

# a single model
snapshot_download(
    repo_id="{repo_id}",
    local_dir=".",
    allow_patterns="{example['path']}/**",
)
```

## Use

From a clone of the code repository, after downloading into it:

```bash
DIR={example['path']}
uv run python main.py --mode inference \\
    --config $DIR/{example['config']} --checkpoint $DIR/{example['file']} \\
    --output results/inference/eval.json
```

Programmatic loading uses `src.models.denoiser.make_model` with the checkpoint's
own config, then the safetensors export:

```python
from safetensors.torch import load_file
from src.config import load_config
from src.models.denoiser import make_model

cfg = load_config("{example['path']}/{example['config']}")
model = make_model(cfg)
model.load_state_dict(load_file("{example['path']}/model.safetensors"))
model.eval()
```

Architecture arguments should be read from the checkpoint's own config snapshot
rather than from `configs/defaults.yaml`, which tracks the current code.

## Training

The planners are bidirectional transformers that denoise a masked action plan
conditioned on a cropped MiniHack glyph observation, trained either by offline
behaviour cloning on oracle rollouts or by online DAgger against the BFS oracle
under a dynamic environment curriculum. Model size and horizon differ per run
(see the table). Exact hyperparameters for every run, including the
in-distribution and out-of-distribution environment sets, the remasking
strategy, schedule and sampling settings, are in the per-checkpoint config
snapshots listed above, which are the authoritative record.

Both models are best-checkpoint selections rather than final-step dumps: each
trainer evaluates every periodic checkpoint on its configured number of
episodes per environment using EMA weights, and the highest-scoring one is
published ({selected_on}). Directory names encode the sample-equivalents the
published model consumed (gradient steps x batch size, rounded); file names
carry each trainer's own counter, DAgger iterations online and gradient steps
offline. Each checkpoint's `selection.json` records the configured budget it
was drawn from and the step it was selected at.

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


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo-id", required=True, help="e.g. mathisweil/remdm-minihack-checkpoints")
    p.add_argument(
        "--inference-results", nargs="*", default=[], metavar="PATH",
        help=f"extra --mode inference JSONs or directories, on top of "
             f"{INFERENCE.relative_to(ROOT)}/",
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

    models = discover_checkpoints()
    if not models:
        print(
            f"No .pth checkpoints found under {CKPTS}.\n"
            "Discovery expects the released layout, "
            "checkpoints/<role>/<name>/*.pth, and skips checkpoints/hf/ "
            "because that is where Hub downloads land. Copy a run's "
            "checkpoints into checkpoints/{offline,online}/<name> first.",
            file=sys.stderr,
        )
        return 1
    runs = discover_runs()
    inference = discover_inference(args.inference_results)
    figures = discover_paper_figures()

    with tempfile.TemporaryDirectory(prefix="remdm-minihack-") as tmp:
        staging = Path(tmp)
        rows, run_rows, inf_rows, fig_rows = stage(
            staging, models, runs, inference, figures, args.selection_metric,
        )
        total_mb = dir_size_mb(staging)
        card = model_card(
            args.repo_id, rows, run_rows, inf_rows, fig_rows, total_mb,
            args.selection_metric,
        )
        (staging / "README.md").write_text(card)

        files = [f for f in staging.rglob("*") if f.is_file()]
        print(f"Staged {plural(len(rows), 'checkpoint')}, "
              f"{plural(len(run_rows), 'ablation run')}, "
              f"{plural(len(inf_rows), 'inference result')}, "
              f"{plural(len(fig_rows), 'paper figure')}, "
              f"{plural(len(files), 'file')}, {total_mb:.0f} MB")
        for r in sorted(rows, key=lambda r: r["path"]):
            print(f"  {r['path']:<70} {r['size']:>8}  {r['detail']}")
            print(f"  {'':<70} restores {r['restores']}")
        for r in sorted(run_rows, key=lambda r: r["run"]):
            print(f"  {r['path']:<70} {r['size']:>8}  {r['contents']}")
        for r in sorted(inf_rows, key=lambda r: r["file"]):
            print(f"  results/inference/{r['file']:<52} {r['size']:>8}  {r['metric']}")
        for r in sorted(fig_rows, key=lambda r: r["file"]):
            print(f"  results/paper_figures/{r['file']:<48} {r['size']:>8}")
        if not run_rows:
            print(f"Warning: no ablation runs with a results.json under {RUNS}.",
                  file=sys.stderr)
        if not inf_rows:
            print("Warning: no inference results; produce them with "
                  "`main.py --mode inference --output "
                  f"{INFERENCE.relative_to(ROOT)}/<name>.json`.", file=sys.stderr)
        if not fig_rows:
            print(
                "Warning: no manuscript figures; build them with the sibling "
                "repo's `scripts/paper_figures.py` and copy the PDFs into "
                f"{PAPER_FIGURES.relative_to(ROOT)}/.", file=sys.stderr,
            )
        if not args.selection_metric:
            print(
                "Warning: --selection-metric not given; these are best-of-N "
                "checkpoints and the card cannot say what they were chosen on.",
                file=sys.stderr,
            )

        if args.dry_run:
            print(f"Dry run; staged tree left nowhere. Card:\n\n{card}")
            return 0

        if not args.yes:
            visibility = "private" if args.private else "public"
            if input(f"Upload to {args.repo_id} ({visibility})? [y/N] ").strip().lower() not in {"y", "yes"}:
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
            commit_message="Upload MiniHack ReMDM planner checkpoints and results",
        )
        print(f"Done: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
