from __future__ import annotations

import argparse
import logging
import random
from typing import Any

import numpy as np
import torch

from src.config import load_config
from src.planners.logging import Logger
from src.planners.offline import run_offline
from src.planners.online import run_dagger
from src.planners.inference import run_inference
from src.planners.smoke import run_smoke


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Utils
# =============================================================================

def _parse_overrides(extras: list[str]) -> dict[str, Any]:
    return {
        k.lstrip("-"): v
        for item in extras if "=" in item
        for k, v in [item.split("=", 1)]
    }


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

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="ReMDM-MiniHack: Masked Diffusion Planner",
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["smoke", "offline", "dagger", "inference"],
    )
    parser.add_argument("--config", default="configs/defaults.yaml")

    parser.add_argument("--data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--wandb-artifact", default=None,
        help=(
            "W&B artifact reference to download as checkpoint, e.g. "
            "'entity/project/checkpoint-iter1000:latest'"
        ),
    )
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--no-ema", action="store_true")

    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument(
        "--des", nargs="+", default=None,
        help="Paths to .des scenario files for custom environment evaluation",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--blind-global", action="store_true",
        help="Zero out global map observations (local-only ablation)",
    )

    return parser.parse_known_args()


# =============================================================================
# Config
# =============================================================================

def build_config(args, extras):
    config_path = args.config
    if args.mode == "smoke" and config_path == "configs/defaults.yaml":
        config_path = "configs/smoke.yaml"

    cfg = load_config(config_path, _parse_overrides(extras))

    seed = _set_seed(cfg.seed)
    logger.info(f"Seed: {seed}")

    return cfg


# =============================================================================
# Validation
# =============================================================================

def validate(args) -> None:
    if args.mode == "inference" and not args.checkpoint and not args.wandb_artifact:
        raise ValueError(
            "--checkpoint or --wandb-artifact required for inference mode"
        )


# =============================================================================
# Dispatch (no lambdas, cleaner)
# =============================================================================

def _resolve_checkpoint(args, cfg) -> str | None:
    """Return a local checkpoint path from --checkpoint or --wandb-artifact."""
    if args.checkpoint:
        return args.checkpoint
    artifact_ref = args.wandb_artifact
    if artifact_ref:
        from src.planners.logging import download_artifact
        path = download_artifact(artifact_ref)
        if path is None:
            raise RuntimeError(
                f"Failed to download W&B artifact: {artifact_ref}"
            )
        return path
    return None


def run_mode(mode: str, cfg, args) -> None:
    if mode == "smoke":
        run_smoke(cfg)

    elif mode == "offline":
        run_offline(cfg, args.data)

    elif mode == "dagger":
        ckpt = _resolve_checkpoint(args, cfg)
        run_dagger(cfg, ckpt, args.no_warm_start)

    elif mode == "inference":
        ckpt = _resolve_checkpoint(args, cfg)
        if ckpt is None:
            raise ValueError(
                "--checkpoint or --wandb-artifact required for inference"
            )
        log = Logger(cfg)
        run_inference(
            cfg,
            ckpt,
            args.envs,
            args.episodes,
            args.output,
            not args.no_ema,
            log=log,
            des_files=args.des,
            blind_global=args.blind_global,
        )
        log.finish()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args, extras = parse_args()
    validate(args)
    cfg = build_config(args, extras)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    run_mode(args.mode, cfg, args)


if __name__ == "__main__":
    main()