from __future__ import annotations

import argparse
import logging
import random
from typing import Any

import numpy as np
import torch

from src.config import load_config
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
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--no-ema", action="store_true")

    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", default=None)

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
    if args.mode == "inference" and not args.checkpoint:
        raise ValueError("--checkpoint required for inference mode")


# =============================================================================
# Dispatch (no lambdas, cleaner)
# =============================================================================

def run_mode(mode: str, cfg, args) -> None:
    if mode == "smoke":
        run_smoke(cfg)

    elif mode == "offline":
        run_offline(cfg, args.data)

    elif mode == "dagger":
        run_dagger(cfg, args.checkpoint, args.no_warm_start)

    elif mode == "inference":
        run_inference(
            cfg,
            args.checkpoint,
            args.envs,
            args.episodes,
            args.output,
            not args.no_ema,
        )


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args, extras = parse_args()
    validate(args)
    cfg = build_config(args, extras)
    run_mode(args.mode, cfg, args)


if __name__ == "__main__":
    main()