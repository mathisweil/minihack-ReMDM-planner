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
        --checkpoint wandb://entity/project/artifact:version \\
        --ablations baseline_rl kl_penalty --fast

    # List registered ablations
    python experiments/rl_finetuning/run_ablations.py --list

    # Analysis only (load existing results)
    python experiments/rl_finetuning/run_ablations.py \\
        --analyze_only --results_path outputs/run_xyz/results.json
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import orjson
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import (
    AblationHistory,
    run_ablation,
)
from experiments.rl_finetuning.analysis.plots import generate_all_plots
from experiments.rl_finetuning.analysis.report import generate_diagnosis_report
from experiments.rl_finetuning.analysis.tables import generate_summary_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------


def _load_yaml(path: str | None) -> dict:
    """Load a YAML file, returning empty dict if path is None.

    Args:
        path: File path or None.

    Returns:
        Parsed YAML dict.
    """
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


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



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        "--config", type=str, default=None,
        help="Main config (configs/defaults.yaml).",
    )
    p.add_argument(
        "--ablations_config", type=str,
        default=str(
            _PROJECT_ROOT
            / "experiments/rl_finetuning/configs/ablations_default.yaml"
        ),
        help="Ablations-specific config.",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Pretrained DAgger checkpoint. Accepts a local .pth path "
            "or a W&B artifact reference: "
            "wandb://entity/project/artifact:version"
        ),
    )

    p.add_argument("--all", action="store_true", help="Run all ablations.")
    p.add_argument(
        "--ablations", nargs="+", default=None, metavar="NAME",
        help="Ablation names. Use --list to see options.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List registered ablations and exit.",
    )

    p.add_argument("--fast", action="store_true", help="Fast smoke-test.")
    p.add_argument("--analyze_only", action="store_true")
    p.add_argument("--results_path", type=str, default=None)

    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--num_seeds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default=None)

    p.add_argument("--max_iter", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)

    return p


# ---------------------------------------------------------------------------
# Result serialisation (orjson)
# ---------------------------------------------------------------------------


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
            k: v for k, v in config.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        },
        "ablations": {
            name: {
                "score": res["score"],
                "score_std": res.get("score_std", 0.0),
                "history": res["history"].to_dict(),
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
    with open(path, "rb") as f:
        data = orjson.loads(f.read())

    pretrained_score = data["pretrained_score"]
    config = data.get("config", {})
    results: dict[str, dict] = {}
    for name, res_data in data["ablations"].items():
        results[name] = {
            "score": res_data["score"],
            "history": AblationHistory.from_dict(res_data["history"]),
        }
    return results, pretrained_score, config


# ---------------------------------------------------------------------------
# Pretrained evaluation
# ---------------------------------------------------------------------------


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
        checkpoint_path, map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt["ema_state_dict"])
    model.eval()

    evaluator = Evaluator()
    n_eps = getattr(cfg, "eval_episodes", 20)
    results = evaluator.evaluate(cfg.id_envs, model, n_eps, cfg, device)
    return float(np.mean([v["win_rate"] for v in results.values()]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    run_id = args.run_id or (
        f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "experiments" / "rl_finetuning" / "outputs" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Analysis-only mode
    if args.analyze_only:
        # Resolve results path: explicit --results_path, or results.json
        # inside --output_dir
        results_path = args.results_path
        if not results_path:
            candidate = output_dir / "results.json"
            if candidate.exists():
                results_path = str(candidate)
            else:
                parser.error(
                    "--analyze_only requires --results_path or an "
                    "--output_dir containing results.json"
                )

        logger.info("Loading results from %s", results_path)
        results, pretrained_score, _ = _results_from_json(results_path)
        logger.info("Loaded %d ablation results.", len(results))

        # Filter to requested subset (--ablations); --all keeps everything
        if args.ablations and not args.all:
            missing = [n for n in args.ablations if n not in results]
            for n in missing:
                logger.warning(
                    "Ablation '%s' not found in results — skipping.", n,
                )
            results = {
                k: v for k, v in results.items()
                if k in args.ablations
            }
            logger.info(
                "Filtered to %d ablation(s): %s",
                len(results), list(results.keys()),
            )

        generate_summary_tables(results, pretrained_score, output_dir)
        generate_all_plots(results, pretrained_score, output_dir)
        generate_diagnosis_report(results, pretrained_score, output_dir)
        logger.info("Analysis complete. Outputs in %s", output_dir)
        return

    # Training mode: load configs
    main_cfg = _load_yaml(
        args.config
        or str(_PROJECT_ROOT / "configs" / "defaults.yaml")
    )
    abl_cfg = _load_yaml(args.ablations_config)

    # Fast overrides
    fast_cfg: dict = {}
    if args.fast:
        fast_path = (
            _PROJECT_ROOT
            / "experiments/rl_finetuning/configs/ablations_fast.yaml"
        )
        fast_cfg = _load_yaml(str(fast_path))

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
    if args.checkpoint.startswith("wandb://"):
        from src.planners.logging import download_artifact
        artifact_ref = args.checkpoint[len("wandb://"):]
        resolved = download_artifact(artifact_ref)
        if resolved is None:
            parser.error(
                f"Failed to download W&B artifact: {artifact_ref}"
            )
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
    logger.info("Evaluating pretrained model...")
    pretrained_score = _evaluate_pretrained(checkpoint_path, cfg, device)
    logger.info("Pretrained baseline ID win rate: %.4f", pretrained_score)

    # W&B (optional dependency)
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError:
            wandb = None
            logger.warning("wandb not installed; skipping.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project or "remdm-minihack-ablations",
                name=run_id,
                config=vars(cfg),
                tags=["ablations"] + selected,
            )
            # Define metric x-axes
            wandb.define_metric("iteration")
            for ns in (
                "train/*", "speed/*", "online/*", "eval/*",
                "model/*", "diag/*",
            ):
                wandb.define_metric(ns, step_metric="iteration")

    # Run ablations
    num_seeds = args.num_seeds or getattr(cfg, "num_seeds", 1)
    base_seed = args.seed if args.seed is not None else (getattr(cfg, "seed", None) or 0)
    results: dict[str, dict] = {}

    for abl_name in selected:
        spec = REGISTRY[abl_name]
        seed_scores: list[float] = []
        seed_histories: list[AblationHistory] = []
        trained_model: torch.nn.Module | None = None

        try:
            for seed_idx in range(num_seeds):
                abl_seed = base_seed + seed_idx * 1000
                logger.info(
                    "Running %s (seed %d/%d)...",
                    abl_name, seed_idx + 1, num_seeds,
                )

                history, final_score, trained_model = run_ablation(
                    spec=spec,
                    cfg=cfg,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    seed=abl_seed,
                )
                seed_scores.append(final_score)
                seed_histories.append(history)
        except Exception:
            logger.exception(
                "Ablation '%s' FAILED — skipping to next.", abl_name,
            )
            continue

        mean_score = float(np.mean(seed_scores))
        std_score = float(np.std(seed_scores))
        logger.info(
            "%s: score = %.4f +/- %.4f (seeds=%d)",
            abl_name, mean_score, std_score, num_seeds,
        )

        results[abl_name] = {
            "history": seed_histories[0],
            "score": mean_score,
            "score_std": std_score,
            "all_scores": seed_scores,
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
            len(results), len(selected), results_path,
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
