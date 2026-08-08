from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.planners.baselines import ALL_BASELINE_ALGOS, run_baselines
from src.planners.logging import Logger
from src.planners.offline import run_offline
from src.planners.online import run_dagger
from src.planners.inference import run_inference
from src.planners.collect_oracle import run_collect
from src.planners.smoke import run_smoke


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

WANDB_PREFIX = "wandb:"


# =============================================================================
# Utils
# =============================================================================

def _parse_overrides(pairs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(
                f"--override expects KEY=VALUE, got '{item}'"
            )
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def _set_seed(seed: int | None) -> int:
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ReMDM-MiniHack: Masked Diffusion Planner",
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "smoke", "offline", "online", "inference", "collect", "baselines",
        ],
    )
    parser.add_argument(
        "--config", default="configs/defaults.yaml",
        help="Experiment config, deep-merged onto configs/defaults.yaml",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Run seed (overrides the config value)",
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help=(
            "Config override, repeatable. Keys are validated against "
            "configs/defaults.yaml; unknown keys are an error."
        ),
    )
    parser.add_argument(
        "--algo", default=None, choices=list(ALL_BASELINE_ALGOS),
        help="Baseline algorithm (required for --mode baselines)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=(
            "Explicit list of seeds for --mode baselines "
            "(e.g. --seeds 0 1 2)."
        ),
    )
    parser.add_argument(
        "--num-seeds", type=int, default=None,
        help=(
            "Number of seeds starting from 0 (alternative to --seeds; "
            "only used by --mode baselines)."
        ),
    )

    parser.add_argument(
        "--data", default=None,
        help=(
            "Dataset path: read by --mode offline, written by "
            "--mode collect (default: collect_output from config)"
        ),
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help=(
            "Checkpoint .pth path, or a W&B artifact reference "
            "'wandb:entity/project/checkpoint-iter1000:latest'"
        ),
    )
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--no-ema", action="store_true")

    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument(
        "--des", nargs="+", default=None,
        help="Paths to .des scenario files for custom environment evaluation",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help=(
            "Episodes per environment at inference "
            "(default: eval_episodes_per_env from config)"
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--blind-global", action="store_true",
        help="Zero out global map observations (local-only ablation)",
    )

    return parser.parse_args()


# =============================================================================
# Config
# =============================================================================

def build_config(args):
    config_path = args.config
    if args.mode == "smoke" and config_path == "configs/defaults.yaml":
        config_path = "configs/smoke.yaml"

    cfg = load_config(config_path, _parse_overrides(args.override))

    if args.seed is not None:
        cfg.seed = args.seed

    seed = _set_seed(cfg.seed)
    logger.info(f"Seed: {seed}")

    return cfg


# =============================================================================
# Validation
# =============================================================================

def validate(args) -> None:
    if args.mode == "inference" and not args.checkpoint:
        raise ValueError("--checkpoint required for inference mode")
    if args.mode == "baselines" and args.algo is None:
        raise ValueError(
            "--algo is required for --mode baselines "
            f"(choose one of {list(ALL_BASELINE_ALGOS)})"
        )


def _resolve_seeds(args, cfg) -> list[int]:
    """Build the seed list for --mode baselines."""
    if args.seeds is not None:
        return list(args.seeds)
    if args.num_seeds is not None:
        return list(range(int(args.num_seeds)))
    return [cfg.seed if cfg.seed is not None else 0]


# =============================================================================
# Dispatch
# =============================================================================

def _resolve_path(p: str | None) -> str | None:
    """Resolve a user-provided path to absolute, or return None."""
    if p is None:
        return None
    return str(Path(p).resolve())


def _resolve_checkpoint(args) -> str | None:
    """Return a local checkpoint path from --checkpoint (path or wandb: ref)."""
    ref = args.checkpoint
    if not ref:
        return None
    if ref.startswith(WANDB_PREFIX):
        from src.planners.logging import download_artifact
        path = download_artifact(ref[len(WANDB_PREFIX):])
        if path is None:
            raise RuntimeError(f"Failed to download W&B artifact: {ref}")
        return path
    return _resolve_path(ref)


def run_mode(mode: str, cfg, args) -> None:
    data_path = _resolve_path(args.data)
    output_path = _resolve_path(args.output)
    des_files = (
        [str(Path(d).resolve()) for d in args.des]
        if args.des else None
    )

    if mode == "smoke":
        run_smoke(cfg)

    elif mode == "offline":
        ckpt = _resolve_checkpoint(args)
        run_offline(cfg, data_path, checkpoint_path=ckpt)

    elif mode == "online":
        ckpt = _resolve_checkpoint(args)
        run_dagger(cfg, ckpt, args.no_warm_start)

    elif mode == "collect":
        if data_path is not None:
            cfg.collect_output = data_path
        run_collect(cfg)

    elif mode == "baselines":
        run_baselines(
            cfg,
            algo=args.algo,
            seeds=_resolve_seeds(args, cfg),
            output_path=output_path,
        )

    elif mode == "inference":
        ckpt = _resolve_checkpoint(args)
        if ckpt is None:
            raise ValueError("--checkpoint required for inference")
        episodes = (
            args.episodes if args.episodes is not None
            else cfg.eval_episodes_per_env
        )
        log = Logger(cfg)
        run_inference(
            cfg,
            ckpt,
            args.envs,
            episodes,
            output_path,
            not args.no_ema,
            log=log,
            des_files=des_files,
            blind_global=args.blind_global,
        )
        log.finish()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args = parse_args()
    validate(args)
    cfg = build_config(args)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    run_mode(args.mode, cfg, args)


if __name__ == "__main__":
    main()
