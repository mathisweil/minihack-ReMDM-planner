"""Entry point for the MiniHack ReMDM RL Fine-Tuning Ablation Suite.

Usage::

    # Run all ablations
    python experiments/rl_finetuning/run_ablations.py \\
        --checkpoint path/to/dagger_checkpoint.pth --all

    # Run specific ablations
    python experiments/rl_finetuning/run_ablations.py \\
        --checkpoint path/to/dagger_checkpoint.pth \\
        --ablations kl_penalty ewc lora

    # Fast smoke test
    python experiments/rl_finetuning/run_ablations.py \\
        --checkpoint path/to/dagger_checkpoint.pth \\
        --ablations baseline_rl kl_penalty --fast

    # Use a W&B artifact as checkpoint
    python experiments/rl_finetuning/run_ablations.py \\
        --checkpoint wandb:entity/project/artifact:version \\
        --ablations baseline_rl kl_penalty --fast

    # List registered ablations
    python experiments/rl_finetuning/run_ablations.py --list

    # Analysis only (load existing results)
    python experiments/rl_finetuning/run_ablations.py \\
        --analyze-only --results-path outputs/run_xyz/results.json

    # Merge results from independent runs (spread across GPUs)
    python experiments/rl_finetuning/run_ablations.py \\
        --merge outputs/gpu0/results.json outputs/gpu1/results.json \\
        --output-dir outputs/merged
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

# The gradient batch is data-limited, so it changes size from
# iteration to iteration, which fragments the caching allocator's fixed
# segments. On a 16 GB card at `batch_size: 4608` the suite reaches ~94%
# of VRAM and the default allocator then OOMs on a backward pass with
# over a gigabyte reserved-but-unallocated. Expandable segments give that
# gigabyte back. This must be set before the first CUDA allocation, and
# an explicit setting from the environment always wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import orjson
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CONFIG_DIR = _PROJECT_ROOT / "experiments" / "rl_finetuning" / "configs"
_DEFAULT_ABLATIONS_CONFIG = _CONFIG_DIR / "ablations_default.yaml"
_FAST_ABLATIONS_CONFIG = _CONFIG_DIR / "ablations_fast.yaml"

from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import (
    AblationHistory,
    run_ablation,
)
from experiments.rl_finetuning.analysis.action_distribution import (
    run_all_action_distribution_analyses,
)
from experiments.rl_finetuning.analysis.plots import generate_all_plots
from experiments.rl_finetuning.analysis.report import generate_diagnosis_report
from experiments.rl_finetuning.analysis.tables import generate_summary_tables
from src.config import validate_keys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_yaml(path: str | None) -> dict:
    """Load a YAML file, returning empty dict if path is None.

    Args:
        path: File path or None.

    Returns:
        Parsed YAML dict.
    """
    if path is None:
        return {}
    with open(Path(path).resolve()) as f:
        return yaml.safe_load(f) or {}


def _load_ablation_config(path: str | None, allowed: set[str] | None = None) -> dict:
    """Load an ablations config on top of ``ablations_default.yaml``.

    Two layers, always: the base carries every ablation hyperparameter and
    the machine config (e.g. ``final_ablations_ucl.yaml``) carries only what
    that machine changes. Configs never inherit from one another.

    Args:
        path: File path or None.
        allowed: Valid config keys. When given, both layers are validated
            under their own names; None skips validation.

    Returns:
        Merged config dict.

    Raises:
        KeyError: If *allowed* is given and a file carries an unknown key.
    """
    if path is None:
        return {}

    resolved = Path(path).expanduser().resolve()
    base = _DEFAULT_ABLATIONS_CONFIG.resolve()
    layers = [base] if resolved != base else []
    layers.append(resolved)

    merged: dict = {}
    for source in layers:  # base first, machine config last
        raw = _load_yaml(str(source))
        if allowed is not None:
            validate_keys(
                raw,
                allowed,
                str(source),
                valid_source=(
                    "configs/defaults.yaml and "
                    "experiments/rl_finetuning/configs/ablations_default.yaml"
                ),
            )
        merged.update(raw)

    if len(layers) > 1:
        logger.info("Ablation config: %s", " -> ".join(p.name for p in layers))
    return merged


def _merge_to_namespace(*dicts: dict) -> SimpleNamespace:
    """Merge config dicts and convert to SimpleNamespace.

    Args:
        *dicts: Config dicts (later override earlier).

    Returns:
        SimpleNamespace with merged keys.
    """
    merged: dict = {}
    for d in dicts:
        merged.update(d)
    return SimpleNamespace(**merged)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    p = argparse.ArgumentParser(
        description="MiniHack ReMDM RL Fine-Tuning Ablation Suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Main config (configs/defaults.yaml).",
    )
    p.add_argument(
        "--ablations-config",
        type=str,
        default=str(_DEFAULT_ABLATIONS_CONFIG),
        help=("Ablations-specific config, layered on top of ablations_default.yaml."),
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Pretrained DAgger checkpoint. Accepts a local .pth path "
            "or a W&B artifact reference: "
            "wandb:entity/project/artifact:version"
        ),
    )

    p.add_argument("--all", action="store_true", help="Run all ablations.")
    p.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Ablation names. Use --list to see options.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List registered ablations and exit.",
    )

    p.add_argument("--fast", action="store_true", help="Fast smoke-test.")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--results-path", type=str, default=None)
    p.add_argument(
        "--merge",
        nargs="+",
        metavar="JSON",
        help=(
            "Merge multiple results.json files from independent runs "
            "and regenerate analysis. E.g.:\n"
            "  --merge outputs/gpu0/results.json outputs/gpu1/results.json"
        ),
    )

    p.add_argument(
        "--action-dist",
        action="store_true",
        help=(
            "After training, roll out the pretrained and fine-tuned models to "
            "compare action distributions (figures/action_dist/). Off by "
            "default: MiniHack rollouts are not vectorised, so this adds "
            "roughly num_envs * episodes * (1 + n_ablations) episodes."
        ),
    )
    p.add_argument(
        "--action-dist-episodes",
        type=int,
        default=10,
        help="Episodes per environment per model for --action-dist.",
    )

    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--num-seeds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)

    p.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable W&B logging (overrides the config's use_wandb).",
    )
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument(
        "--wandb-resume-id",
        type=str,
        default=None,
        help=(
            "W&B run ID to resume (curve continuity). "
            "Find it in the W&B dashboard URL: wandb.ai/.../runs/<id>"
        ),
    )

    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)

    return p


def _history_finals(history: AblationHistory) -> dict:
    """Final logged value per history field for one seed.

    Captures numeric finals, dict finals (e.g. per_env_win_rates) and
    numeric-list finals, so per-seed evaluation endpoints survive the merge
    instead of only the first seed's history (the App C single-run defect).
    """
    finals: dict = {}
    for k, v in history.to_dict().items():
        if isinstance(v, list) and v:
            last = v[-1]
            if isinstance(last, (int, float, dict)) or (
                isinstance(last, list)
                and all(isinstance(x, (int, float)) for x in last)
            ):
                finals[k] = last
    return finals


def _results_to_json(
    results: dict[str, dict],
    pretrained_score: float,
    config: dict,
) -> bytes:
    """Serialise results to orjson bytes.

    Args:
        results: ``{name: {"score": float, "history": AblationHistory}}``.
        pretrained_score: Pretrained eval score.
        config: Merged config dict.

    Returns:
        UTF-8 JSON bytes.
    """
    serialisable = {
        "pretrained_score": pretrained_score,
        "config": {
            k: v
            for k, v in config.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        },
        "ablations": {
            name: {
                "score": res["score"],
                "score_std": res.get("score_std", 0.0),
                "all_scores": res.get("all_scores", [res["score"]]),
                "history": res["history"].to_dict(),
                # seeds, wall clock and per-seed finals when present
                **{
                    k: res[k]
                    for k in ("base_seed", "seeds", "wall_clock_s", "per_seed_finals")
                    if k in res
                },
            }
            for name, res in results.items()
        },
    }
    return orjson.dumps(serialisable, option=orjson.OPT_INDENT_2)


def _results_from_json(
    path: str,
) -> tuple[dict[str, dict], float, dict]:
    """Load results from JSON.

    Args:
        path: Path to results.json.

    Returns:
        Tuple of (results, pretrained_score, config).
    """
    with open(Path(path).resolve(), "rb") as f:
        data = orjson.loads(f.read())

    pretrained_score = data["pretrained_score"]
    config = data.get("config", {})
    results: dict[str, dict] = {}
    for name, res_data in data["ablations"].items():
        score = res_data["score"]
        results[name] = {
            "score": score,
            "score_std": res_data.get("score_std", 0.0),
            "all_scores": res_data.get("all_scores", [score]),
            "history": AblationHistory.from_dict(res_data["history"]),
        }
        for _k in (
            "base_seed",
            "seeds",
            "wall_clock_s",
            "per_seed_finals",
        ):
            if _k in res_data:
                results[name][_k] = res_data[_k]
    return results, pretrained_score, config


def _merge_result_files(
    paths: list[str],
) -> tuple[dict[str, dict], float, dict]:
    """Merge multiple results.json files from independent runs.

    When the same ablation appears in multiple files its ``all_scores``
    lists are concatenated and ``score`` / ``score_std`` are recomputed
    over the union.  The history from the first file encountered is kept.

    Args:
        paths: List of paths to results.json files.

    Returns:
        Tuple of (merged_results, pretrained_score, config).
    """
    merged: dict[str, dict] = {}
    pretrained_scores: list[float] = []
    config: dict = {}

    for p in paths:
        results, pt_score, cfg = _results_from_json(p)
        pretrained_scores.append(pt_score)
        if not config:
            config = cfg

        for name, res in results.items():
            if name not in merged:
                merged[name] = {
                    "score": res["score"],
                    "score_std": res.get("score_std", 0.0),
                    "all_scores": list(res.get("all_scores", [res["score"]])),
                    "history": res["history"],
                    # carry seed, wall-clock and per-seed final records
                    **{
                        k: list(res[k])
                        for k in ("seeds", "wall_clock_s", "per_seed_finals")
                        if k in res
                    },
                    **({"base_seed": res["base_seed"]} if "base_seed" in res else {}),
                }
            else:
                # Concatenate scores from this file
                new_scores = list(res.get("all_scores", [res["score"]]))
                merged[name]["all_scores"].extend(new_scores)
                for _k in ("seeds", "wall_clock_s", "per_seed_finals"):
                    if _k in res:
                        merged[name].setdefault(_k, []).extend(res[_k])
                # Recompute mean/std over all seeds
                all_s = merged[name]["all_scores"]
                merged[name]["score"] = float(np.mean(all_s))
                merged[name]["score_std"] = float(np.std(all_s))

    pretrained_score = float(np.mean(pretrained_scores))
    return merged, pretrained_score, config


def _evaluate_pretrained(
    checkpoint_path: str,
    cfg: SimpleNamespace,
    device: torch.device,
) -> float:
    """Evaluate the pretrained model to get baseline score.

    Args:
        checkpoint_path: DAgger checkpoint path.
        cfg: Config namespace.
        device: Torch device.

    Returns:
        Mean ID win rate.
    """
    from src.models.denoiser import make_model
    from src.planners.inference import Evaluator

    model = make_model(cfg).to(device)
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["ema_state_dict"])
    model.eval()

    evaluator = Evaluator()
    n_eps = getattr(cfg, "eval_episodes", 20)
    results = evaluator.evaluate(cfg.id_envs, model, n_eps, cfg, device)
    return float(np.mean([v["win_rate"] for v in results.values()]))


def _load_pretrained_model(
    checkpoint_path: str,
    cfg: SimpleNamespace,
    device: torch.device | str,
) -> torch.nn.Module:
    """Load the pretrained (BC-only) model in eval mode.

    Args:
        checkpoint_path: DAgger checkpoint path.
        cfg: Config namespace.
        device: Torch device.

    Returns:
        Model with the EMA weights loaded.
    """
    from src.models.denoiser import make_model

    model = make_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema_state_dict"])
    model.eval()
    return model


def _iter_ablation_models(
    results: dict[str, dict],
    output_dir: Path,
    cfg: SimpleNamespace,
    device: torch.device | str,
) -> Iterator[tuple[str, torch.nn.Module]]:
    """Yield ``(name, model)`` for each ablation checkpoint on disk.

    Loads one model at a time so the whole suite never sits in memory.
    Ablations whose checkpoint is missing are skipped with a warning.

    Args:
        results: Completed ablation results dict.
        output_dir: Run output directory holding ``checkpoint_{name}.pth``.
        cfg: Config namespace.
        device: Torch device.

    Yields:
        Tuples of ablation name and the fine-tuned model in eval mode.
    """
    from src.models.denoiser import make_model

    for name in results:
        ckpt_path = output_dir / f"checkpoint_{name}.pth"
        if not ckpt_path.exists():
            logger.warning("No checkpoint for %s, skipping action dist.", name)
            continue
        model = make_model(cfg).to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False),
        )
        model.eval()
        yield name, model
        del model


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ablation suite.

    Args:
        argv: Optional argument list.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("Registered ablations:")
        for name, spec in sorted(REGISTRY.items()):
            print(f"  [{spec.group}] {name:30s} -- {spec.description}")
        return

    # Output directory
    run_id = args.run_id or (f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (_PROJECT_ROOT / "experiments" / "rl_finetuning" / "outputs" / run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Analysis-only mode
    if args.analyze_only:
        # Resolve results path: explicit --results-path, or results.json
        # inside --output-dir
        results_path = (
            str(Path(args.results_path).resolve()) if args.results_path else None
        )
        if not results_path:
            candidate = output_dir / "results.json"
            if candidate.exists():
                results_path = str(candidate)
            else:
                parser.error(
                    "--analyze-only requires --results-path or an "
                    "--output-dir containing results.json"
                )

        logger.info("Loading results from %s", results_path)
        results, pretrained_score, _ = _results_from_json(results_path)
        logger.info("Loaded %d ablation results.", len(results))

        # Filter to requested subset (--ablations); --all keeps everything
        if args.ablations and not args.all:
            missing = [n for n in args.ablations if n not in results]
            for n in missing:
                logger.warning(
                    "Ablation '%s' not found in results — skipping.",
                    n,
                )
            results = {k: v for k, v in results.items() if k in args.ablations}
            logger.info(
                "Filtered to %d ablation(s): %s",
                len(results),
                list(results.keys()),
            )

        generate_summary_tables(results, pretrained_score, output_dir)
        generate_all_plots(results, pretrained_score, output_dir)
        generate_diagnosis_report(results, pretrained_score, output_dir)
        logger.info("Analysis complete. Outputs in %s", output_dir)
        return

    # Merge mode: combine results from independent runs
    if args.merge:
        merge_paths = [str(Path(p).resolve()) for p in args.merge]
        for p in merge_paths:
            if not Path(p).exists():
                parser.error(f"Results file not found: {p}")
        logger.info(
            "Merging %d results files: %s",
            len(merge_paths),
            merge_paths,
        )
        results, pretrained_score, config = _merge_result_files(merge_paths)
        logger.info(
            "Merged %d ablation(s): %s",
            len(results),
            list(results.keys()),
        )
        for name, res in results.items():
            n = len(res["all_scores"])
            logger.info(
                "  %s: %.4f +/- %.4f  (%d seed%s)",
                name,
                res["score"],
                res["score_std"],
                n,
                "s" if n != 1 else "",
            )

        # Save merged results
        merged_path = output_dir / "results.json"
        merged_path.write_bytes(
            _results_to_json(results, pretrained_score, config),
        )
        logger.info("Saved merged results to %s", merged_path)

        # Regenerate analysis
        generate_summary_tables(results, pretrained_score, output_dir)
        generate_all_plots(results, pretrained_score, output_dir)
        generate_diagnosis_report(results, pretrained_score, output_dir)
        logger.info("Analysis complete. Outputs in %s", output_dir)
        return

    # Training mode: load configs
    main_cfg = _load_yaml(
        args.config or str(_PROJECT_ROOT / "configs" / "defaults.yaml")
    )
    # Valid ablation keys: the main config supplies model/env/diffusion
    # settings, ablations_default.yaml the RL fine-tuning knobs. A key in
    # neither is a typo, which would otherwise be silent: every consumer
    # reads through `getattr(cfg, key, fallback)`, so a misspelt key just
    # leaves the real one at its inherited value.
    allowed = set(main_cfg) | set(_load_yaml(str(_DEFAULT_ABLATIONS_CONFIG)))
    abl_cfg = _load_ablation_config(args.ablations_config, allowed=allowed)

    # Fast overrides. An overlay applied on top of whichever ablations config
    # is in use, so it must contribute only its own keys and never drag
    # ablations_default.yaml back over a machine-specific config.
    fast_cfg: dict = {}
    if args.fast:
        fast_cfg = _load_yaml(str(_FAST_ABLATIONS_CONFIG))
        validate_keys(fast_cfg, allowed, str(_FAST_ABLATIONS_CONFIG))

    # CLI overrides
    cli_overrides: dict = {}
    for key in ("max_iter", "batch_size", "eval_every", "lr", "seed"):
        val = getattr(args, key)
        if val is not None:
            cli_overrides[key] = val

    cfg = _merge_to_namespace(main_cfg, abl_cfg, fast_cfg, cli_overrides)

    # Device
    if args.device:
        cfg.device = args.device
    elif not hasattr(cfg, "device"):
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.device)

    # Enable TF32 for faster float32 matmuls on Ampere+ GPUs
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Checkpoint (resolve W&B artifact if needed)
    if not args.checkpoint:
        parser.error("--checkpoint is required for training mode.")
    if args.checkpoint.startswith("wandb:"):
        from src.planners.logging import download_artifact

        artifact_ref = args.checkpoint[len("wandb:") :]
        resolved = download_artifact(artifact_ref)
        if resolved is None:
            parser.error(f"Failed to download W&B artifact: {artifact_ref}")
        checkpoint_path = resolved
    else:
        checkpoint_path = str(Path(args.checkpoint).resolve())

    # Select ablations
    if args.all:
        selected = list(REGISTRY.keys())
    elif args.ablations:
        unknown = [n for n in args.ablations if n not in REGISTRY]
        if unknown:
            parser.error(f"Unknown ablation(s): {unknown}. Use --list.")
        selected = args.ablations
    else:
        parser.error("Specify --all or --ablations NAME [NAME ...]")

    logger.info("Selected ablations (%d): %s", len(selected), selected)

    # Evaluate pretrained baseline
    # Seed the pretrained evaluation with the base seed
    import random as _random

    _eval_seed = (
        args.seed if args.seed is not None else (getattr(cfg, "seed", None) or 0)
    )
    torch.manual_seed(_eval_seed)
    np.random.seed(_eval_seed)
    _random.seed(_eval_seed)
    logger.info("Evaluating pretrained model (seed %d)...", _eval_seed)
    pretrained_score = _evaluate_pretrained(checkpoint_path, cfg, device)
    logger.info("Pretrained baseline ID win rate: %.4f", pretrained_score)

    # W&B (optional dependency). The CLI flag overrides the config key
    # when given; otherwise the config governs, as in the craftax twin.
    use_wandb = args.use_wandb if args.use_wandb is not None else getattr(
        cfg, "use_wandb", False
    )
    wandb_run = None
    if use_wandb:
        try:
            import wandb
        except ImportError:
            wandb = None
            logger.warning("wandb not installed; skipping.")
        else:
            resume_id = args.wandb_resume_id
            wandb_run = wandb.init(
                # Config governs the project name (spec-config §6.5;
                # the old "remdm-*" literal was a dead fallback).
                project=(args.wandb_project or cfg.wandb_project),
                name=run_id,
                config=vars(cfg),
                tags=["ablations"] + selected,
                id=resume_id or None,
                resume="must" if resume_id else "never",
            )
            # Define metric x-axes
            wandb.define_metric("iteration")
            for ns in (
                "train/*",
                "speed/*",
                "online/*",
                "eval/*",
                "model/*",
                "diag/*",
            ):
                wandb.define_metric(ns, step_metric="iteration")

    # Run ablations
    num_seeds = args.num_seeds or getattr(cfg, "num_seeds", 1)
    base_seed = (
        args.seed if args.seed is not None else (getattr(cfg, "seed", None) or 0)
    )
    results: dict[str, dict] = {}
    max_iter = getattr(cfg, "max_iter", 1000)
    wandb_global_step = 0  # monotonically increasing across ablations/seeds

    for abl_name in selected:
        spec = REGISTRY[abl_name]
        seed_scores: list[float] = []
        seed_histories: list[AblationHistory] = []
        seeds_used: list[int] = []
        seed_times: list[float] = []
        trained_model: torch.nn.Module | None = None

        try:
            for seed_idx in range(num_seeds):
                abl_seed = (
                    base_seed + seed_idx
                )  # literal seed set base+idx (default 0, 1, 2)
                seeds_used.append(abl_seed)
                logger.info(
                    "Running %s (seed %d/%d)...",
                    abl_name,
                    seed_idx + 1,
                    num_seeds,
                )

                _t0 = time.monotonic()
                history, final_score, trained_model = run_ablation(
                    spec=spec,
                    cfg=cfg,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    seed=abl_seed,
                    wandb_step_offset=wandb_global_step,
                )
                wandb_global_step += max_iter
                seed_times.append(
                    round(time.monotonic() - _t0, 1)
                )  # per-seed wall clock
                seed_scores.append(final_score)
                seed_histories.append(history)
        except Exception:
            logger.exception(
                "Ablation '%s' FAILED — skipping to next.",
                abl_name,
            )
            continue

        mean_score = float(np.mean(seed_scores))
        std_score = float(np.std(seed_scores))
        logger.info(
            "%s: score = %.4f +/- %.4f (seeds=%d)",
            abl_name,
            mean_score,
            std_score,
            num_seeds,
        )

        results[abl_name] = {
            "history": seed_histories[0],
            "score": mean_score,
            "score_std": std_score,
            "all_scores": seed_scores,
            "base_seed": base_seed,
            "seeds": seeds_used,
            "wall_clock_s": seed_times,  # per-seed wall clock
            "per_seed_finals": [_history_finals(h) for h in seed_histories],
        }

        # W&B summary for this ablation
        if wandb_run is not None:
            try:
                wandb_run.summary[f"{abl_name}/final_score"] = mean_score
                wandb_run.summary[f"{abl_name}/score_std"] = std_score
            except Exception:
                pass

        # Incremental save
        results_path = output_dir / "results.json"
        results_path.write_bytes(
            _results_to_json(results, pretrained_score, vars(cfg)),
        )
        logger.info(
            "Saved partial results (%d/%d) to %s",
            len(results),
            len(selected),
            results_path,
        )

        # Per-ablation model checkpoint (last seed)
        if trained_model is not None:
            ckpt_path = output_dir / f"checkpoint_{abl_name}.pth"
            torch.save(trained_model.state_dict(), ckpt_path)
            logger.info("Saved model checkpoint to %s", ckpt_path)

        # Intermediate analysis (regenerates after each ablation)
        logger.info("Generating plots, tables, and report...")
        generate_summary_tables(results, pretrained_score, output_dir)
        generate_all_plots(results, pretrained_score, output_dir)
        generate_diagnosis_report(results, pretrained_score, output_dir)

    if args.action_dist and results:
        logger.info("Running action distribution analysis...")
        try:
            run_all_action_distribution_analyses(
                _load_pretrained_model(checkpoint_path, cfg, device),
                _iter_ablation_models(results, output_dir, cfg, device),
                cfg,
                device,
                output_dir,
                num_episodes=args.action_dist_episodes,
            )
        except Exception:
            logger.exception("Action distribution analysis failed — continuing.")

    if wandb_run is not None:
        try:
            wandb_run.summary["pretrained_baseline"] = pretrained_score
            wandb_run.summary["n_ablations_completed"] = len(results)
        except Exception:
            pass
        wandb_run.finish()

    logger.info("=" * 60)
    logger.info("Ablation suite complete.")
    logger.info("  Results:  %s", output_dir / "results.json")
    logger.info("  Report:   %s", output_dir / "diagnosis.md")
    logger.info("  Outputs:  %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
