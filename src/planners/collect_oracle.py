"""Standalone BFS oracle data collection for offline training datasets.

Runs the BFS oracle across in-distribution MiniHack environments using
multiprocessing and saves the resulting trajectories in the dict format
expected by ``ReplayBuffer.load_offline_data()``.

Usage::

    python main.py --mode collect
    python main.py --mode collect --override collect_episodes_per_env=2000
    python main.py --mode collect --data data/small.pt
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import torch

from src.envs.minihack_env import collect_oracle_trajectory

logger = logging.getLogger(__name__)


def _collect_single(
    args: tuple[str, int, SimpleNamespace],
) -> dict | None:
    """Process-pool worker: collect one oracle trajectory.

    Module-level function so ``ProcessPoolExecutor`` can pickle it.

    Args:
        args: ``(env_id, seed, cfg)`` tuple.

    Returns:
        Trajectory dict with ``"local"``, ``"global"``,
        ``"actions"``, ``"env_id"`` keys, or ``None`` on failure.
    """
    env_id, seed, cfg = args
    return collect_oracle_trajectory(env_id, seed, cfg)


def _format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string.

    Args:
        seconds: Remaining time in seconds.

    Returns:
        Formatted string like ``"2m 30s"`` or ``"45s"``.
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def run_collect(cfg: SimpleNamespace) -> None:
    """Collect BFS oracle demonstrations and save as a .pt dataset.

    Collects ``collect_episodes_per_env`` episodes per ID environment
    using ``ProcessPoolExecutor`` for parallelism, then saves the
    trajectories in the dict format consumed by
    ``ReplayBuffer.load_offline_data()``.

    The output file can be loaded directly by ``--mode offline``::

        python main.py --mode collect
        python main.py --mode offline --data data/dataset.pt

    Args:
        cfg: Config namespace. Reads ``collect_episodes_per_env``,
            ``collect_num_workers``, ``collect_output``, ``id_envs``,
            ``seed``.
    """
    eps_per_env: int = cfg.collect_episodes_per_env
    max_workers: int = min(
        cfg.collect_num_workers,
        os.cpu_count() or 4,
    )
    output_path: str = cfg.collect_output
    id_envs: list[str] = cfg.id_envs
    base_seed: int = cfg.seed if cfg.seed is not None else 0

    total_episodes = eps_per_env * len(id_envs)
    logger.info(
        "Collecting %d oracle episodes (%d per env, %d envs, %d workers)",
        total_episodes,
        eps_per_env,
        len(id_envs),
        max_workers,
    )

    # Deterministic task list: (env_id, seed, cfg) per episode
    tasks: list[tuple[str, int, SimpleNamespace]] = []
    for env_idx, env_id in enumerate(id_envs):
        for ep in range(eps_per_env):
            seed = base_seed + env_idx * eps_per_env + ep
            tasks.append((env_id, seed, cfg))

    trajectories: list[dict] = []
    per_env_count: dict[str, int] = dict.fromkeys(id_envs, 0)
    per_env_steps: dict[str, int] = dict.fromkeys(id_envs, 0)
    failures = 0
    completed = 0
    t_start = time.perf_counter()
    log_interval = max(1, total_episodes // 50)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_env: dict = {
            executor.submit(_collect_single, task): task[0] for task in tasks
        }

        for future in as_completed(future_to_env):
            env_id = future_to_env[future]
            completed += 1

            try:
                result = future.result()
            except Exception:
                logger.error(
                    "Worker crashed for %s",
                    env_id,
                    exc_info=True,
                )
                result = None

            if result is not None:
                trajectories.append(result)
                per_env_count[env_id] += 1
                per_env_steps[env_id] += len(result["actions"])
            else:
                failures += 1

            if completed % log_interval == 0 or completed == total_episodes:
                elapsed = time.perf_counter() - t_start
                rate = completed / max(elapsed, 1e-6)
                eta = (total_episodes - completed) / max(rate, 1e-6)
                env_summary = "  ".join(
                    f"{eid.split('-')[-2]}:{per_env_count[eid]}" for eid in id_envs
                )
                logger.info(
                    "  %d/%d (%.1f%%)  %.1f eps/s  ETA: %s  |  %s",
                    completed,
                    total_episodes,
                    100 * completed / total_episodes,
                    rate,
                    _format_eta(eta),
                    env_summary,
                )

    elapsed = time.perf_counter() - t_start

    # Summary
    total_steps = sum(per_env_steps.values())
    logger.info("Collection complete in %.1fs", elapsed)
    logger.info(
        "  Trajectories: %d (%d failures)",
        len(trajectories),
        failures,
    )
    logger.info("  Total steps: %d", total_steps)
    for env_id in id_envs:
        n = per_env_count[env_id]
        s = per_env_steps[env_id]
        avg = s / max(n, 1)
        logger.info(
            "  %s: %d eps, %d steps, avg %.1f steps/ep",
            env_id,
            n,
            s,
            avg,
        )

    # Save in the dict format expected by ReplayBuffer.load_offline_data()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    dataset: dict = {"trajectories": trajectories}
    torch.save(dataset, str(out))

    file_mb = out.stat().st_size / (1024 * 1024)
    logger.info(
        "Saved %d trajectories to %s (%.1f MB)",
        len(trajectories),
        out,
        file_mb,
    )
