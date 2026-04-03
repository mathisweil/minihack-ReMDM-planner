"""MiniHack environment discovery and diagnostic utilities.

Provides tools for scanning the gymnasium registry, validating action-space
consistency across environments, and benchmarking inference throughput.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

_NAV_KEYWORDS = ("Room", "Corridor", "Maze", "River")
_EXCLUDED_KEYWORDS = ("KeyRoom",)
_REFERENCE_ENV_ID = "MiniHack-Room-15x15-v0"


def list_working_minihack_tasks() -> list[str]:
    """Scan the gymnasium registry for working MiniHack navigation tasks.

    Filters to environments whose names contain at least one navigation
    keyword and attempts to instantiate each. Returns the IDs of all
    successfully created environments.

    Returns:
        Sorted list of working MiniHack navigation environment IDs.
    """
    import gymnasium as gym
    import minihack  # noqa: F401 — registers envs

    all_ids = list(gym.envs.registry.keys())
    candidates = [
        e for e in all_ids
        if "MiniHack" in e
        and any(k in e for k in _NAV_KEYWORDS)
        and not any(x in e for x in _EXCLUDED_KEYWORDS)
    ]

    working: list[str] = []
    broken: list[str] = []
    for env_id in sorted(candidates):
        try:
            env = gym.make(env_id)
            working.append(env_id)
            env.close()
        except Exception:
            broken.append(env_id)

    logger.info(
        f"MiniHack navigation tasks — working: {len(working)}, "
        f"broken: {len(broken)}"
    )
    return working


def check_action_consistency_with_fixed_ref(
    env_list: list[str],
) -> list[tuple[str, str, int]]:
    """Validate action-space ordering against a fixed reference environment.

    Compares each environment's action list against
    ``MiniHack-Room-15x15-v0`` and classifies the relationship as one of:
    ``REFERENCE``, ``EXACT``, ``SUPERSET (+N)``, ``SUBSET (-N)``,
    ``CONFLICT``, or ``CRASHED``.

    Args:
        env_list: MiniHack environment IDs to check.

    Returns:
        List of ``(env_id, status, action_space_size)`` tuples.
    """
    import gymnasium as gym
    import minihack  # noqa: F401

    ref_env = gym.make(_REFERENCE_ENV_ID)
    reference_actions = ref_env.unwrapped.actions  # type: ignore[attr-defined]
    ref_env.close()

    results: list[tuple[str, str, int]] = []
    for env_id in sorted(env_list):
        if env_id == _REFERENCE_ENV_ID:
            results.append((env_id, "REFERENCE", len(reference_actions)))
            continue
        try:
            env = gym.make(env_id)
            try:
                env_actions = env.unwrapped.actions  # type: ignore[attr-defined]
                limit = min(len(reference_actions), len(env_actions))
                is_match = all(
                    reference_actions[i] == env_actions[i]
                    for i in range(limit)
                )
                diff = len(env_actions) - len(reference_actions)
                if is_match and diff == 0:
                    status = "EXACT"
                elif diff > 0:
                    status = f"SUPERSET (+{diff})"
                elif is_match:
                    status = f"SUBSET ({diff})"
                else:
                    status = "CONFLICT"
                results.append((env_id, status, len(env_actions)))
            finally:
                env.close()
        except Exception:
            results.append((env_id, "CRASHED", 0))

    for name, status, size in results:
        logger.info(f"  {name:<40} | {status:<14} | n_actions={size}")
    return results


def benchmark_inference(
    model: torch.nn.Module,
    cfg: SimpleNamespace,
    device: torch.device | str,
    n_actions: int = 100,
) -> tuple[float, float]:
    """Measure ReMDM inference throughput.

    Runs ``n_actions`` planning calls with dummy observations and
    measures wall-clock time.

    Args:
        model: Denoising model in eval mode.
        cfg: Config namespace (used for ``seq_len``, ``mask_token``, etc.).
        device: Torch device.
        n_actions: Number of planning calls to benchmark.

    Returns:
        ``(diffusion_steps_per_sec, actions_per_sec)`` as floats.
    """
    from src.diffusion.sampling import remdm_sample

    model.eval()
    local_dummy = torch.zeros(
        (1, cfg.crop_size, cfg.crop_size), dtype=torch.long, device=device,
    )
    global_dummy = torch.zeros(
        (1, cfg.map_h, cfg.map_w), dtype=torch.long, device=device,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_actions):
        remdm_sample(model, local_dummy, global_dummy, cfg, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_steps = n_actions * cfg.diffusion_steps_eval
    steps_per_sec = total_steps / elapsed if elapsed > 0 else 0.0
    actions_per_sec = n_actions / elapsed if elapsed > 0 else 0.0

    logger.info(
        f"Benchmark ({n_actions} actions): "
        f"{steps_per_sec:.1f} diffusion-steps/s | "
        f"{actions_per_sec:.1f} actions/s"
    )
    return steps_per_sec, actions_per_sec
