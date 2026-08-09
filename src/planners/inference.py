"""Stateless evaluation runner.

Runs episodes using the diffusion model and collects per-environment
win rates, average rewards, and step counts.  All episodes for a given
environment are rolled out in lockstep so that replanning calls are
batched into single GPU forward passes (B = n_episodes).
"""

from __future__ import annotations

import json
import logging
import zlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.models.denoiser import ModelEMA, make_model
from src.planners.logging import Logger

logger = logging.getLogger(__name__)


class Evaluator:
    """Stateless evaluation runner.

    Runs the model on a set of environments and returns aggregate
    statistics per environment.  Episodes within each environment are
    executed in lockstep so replanning calls are GPU-batched.
    """

    @torch.no_grad()
    def evaluate(
        self,
        env_ids: list[str],
        model: torch.nn.Module,
        n_episodes: int,
        cfg: SimpleNamespace,
        device: torch.device | str,
        des_files: list[str] | None = None,
        blind_global: bool = False,
    ) -> dict[str, dict]:
        """Evaluate *model* on each environment in *env_ids*.

        All *n_episodes* for a given environment run in lockstep so
        that replanning forward passes are batched (B = active envs
        needing a replan).

        Args:
            env_ids: List of MiniHack environment IDs.
            model: Denoising model (eval mode).
            n_episodes: Episodes per environment.
            cfg: Config namespace.
            device: Torch device.
            des_files: Optional list of ``.des`` file paths for custom
                scenario evaluation. Each file yields one extra env entry
                keyed by its filename stem.
            blind_global: If ``True``, zero out global map observations
                (local-only ablation mode).

        Returns:
            ``{env_id: {"win_rate", "wins", "avg_reward", "avg_steps",
            "n_episodes"}}``
        """
        model.eval()
        results: dict[str, dict] = {}

        # Build list of (env_id, des_content) pairs
        eval_targets: list[tuple[str, str | None]] = [
            (eid, None) for eid in env_ids
        ]
        if des_files:
            for des_path in des_files:
                from pathlib import Path
                stem = Path(des_path).stem
                with open(des_path) as fh:
                    eval_targets.append((stem, fh.read()))

        for env_id, des_content in eval_targets:
            # Python's hash() is salted per process; crc32 keeps the derived
            # environment seeds stable across invocations.
            seeds = [
                42 + zlib.crc32(f"{env_id}:{ep}".encode()) % (2**31)
                for ep in range(n_episodes)
            ]
            ep_results = self._run_episodes_batched(
                model, env_id, n_episodes, cfg, device,
                seeds=seeds,
                des_content=des_content,
                blind_global=blind_global,
            )

            wins = sum(1 for r in ep_results if r["won"])
            total_reward = sum(r["total_reward"] for r in ep_results)
            total_steps = sum(r["steps"] for r in ep_results)
            n = max(len(ep_results), 1)
            results[env_id] = {
                "win_rate": wins / n,
                "wins": wins,
                "avg_reward": total_reward / n,
                "avg_steps": total_steps / n,
                "n_episodes": len(ep_results),
            }

        return results

    @torch.no_grad()
    def _run_episodes_batched(
        self,
        model: torch.nn.Module,
        env_id: str,
        n_episodes: int,
        cfg: SimpleNamespace,
        device: torch.device | str,
        seeds: list[int],
        des_content: str | None = None,
        blind_global: bool = False,
    ) -> list[dict]:
        """Run episodes in lockstep with batched model inference.

        Creates one environment per episode, steps them in lockstep,
        and batches all replanning calls into single forward passes
        (B = number of active envs needing a replan at each step).

        Args:
            model: Denoising model (eval mode).
            env_id: MiniHack environment ID.
            n_episodes: Number of episodes to run.
            cfg: Config namespace.
            device: Torch device.
            seeds: Per-episode RNG seeds (length *n_episodes*).
            des_content: Optional ``.des`` file content for custom
                scenarios.
            blind_global: If ``True``, zero out global map observations.

        Returns:
            List of per-episode dicts with ``"won"``, ``"steps"``,
            ``"total_reward"`` keys.  Failed episodes report
            ``won=False``.
        """
        from src.diffusion.sampling import remdm_sample
        from src.envs.minihack_env import make_env

        n = n_episodes
        max_steps = 500
        cs = cfg.crop_size

        # Create and reset all envs
        envs: list = []
        cur_local = np.zeros((n, cs, cs), dtype=np.int16)
        cur_global = np.zeros(
            (n, cfg.map_h, cfg.map_w), dtype=np.int16,
        )
        failed = np.zeros(n, dtype=bool)

        for i in range(n):
            try:
                env = make_env(env_id, des_content, cfg)
                (local, glb), _ = env.reset(seed=seeds[i])
                envs.append(env)
                cur_local[i] = local
                cur_global[i] = glb
            except Exception:
                logger.warning(
                    "Failed to create env %s (ep %d)",
                    env_id, i, exc_info=True,
                )
                envs.append(None)
                failed[i] = True

        # Per-episode state vectors
        plans = np.zeros((n, cfg.seq_len), dtype=np.int64)
        step_in_plan = np.zeros(n, dtype=np.int32)
        need_replan = np.ones(n, dtype=bool)
        done = failed.copy()
        won = np.zeros(n, dtype=bool)
        total_reward = np.zeros(n, dtype=np.float64)
        n_steps = np.zeros(n, dtype=np.int32)

        try:
            for _ in range(max_steps):
                # Batch replan for active envs that need it
                replan_idx = np.where(need_replan & ~done)[0]
                if len(replan_idx) > 0:
                    local_t = torch.from_numpy(
                        cur_local[replan_idx],
                    ).long().to(device)  # [B_r, cs, cs]
                    glb_t = torch.from_numpy(
                        cur_global[replan_idx],
                    ).long().to(device)  # [B_r, map_h, map_w]
                    batch_plans = remdm_sample(
                        model, local_t, glb_t, cfg, device,
                        physics_aware=getattr(
                            cfg, "physics_aware_sampling", False,
                        ),
                        blind_global=blind_global,
                    ).cpu().numpy()  # [B_r, seq_len]
                    plans[replan_idx] = batch_plans
                    step_in_plan[replan_idx] = 0
                    need_replan[replan_idx] = False

                # Step all active envs
                any_active = False
                for i in range(n):
                    if done[i]:
                        continue
                    any_active = True

                    action = int(plans[i, step_in_plan[i]])
                    action = max(
                        0, min(action, cfg.action_dim - 1),
                    )
                    step_in_plan[i] += 1
                    n_steps[i] += 1

                    if step_in_plan[i] >= cfg.replan_every:
                        need_replan[i] = True

                    try:
                        obs, reward, term, trunc, info = (
                            envs[i].step(action)
                        )
                        local, glb = obs
                        total_reward[i] += reward
                        cur_local[i] = local
                        cur_global[i] = glb

                        if info.get("won", False):
                            won[i] = True
                        if term or trunc:
                            done[i] = True
                    except Exception:
                        logger.warning(
                            "Episode %d step failed for %s",
                            i, env_id, exc_info=True,
                        )
                        done[i] = True

                if not any_active:
                    break
        finally:
            for env in envs:
                if env is not None:
                    env.close()

        return [
            {
                "won": bool(won[i]),
                "steps": int(n_steps[i]),
                "total_reward": float(total_reward[i]),
            }
            for i in range(n)
        ]


def format_eval_results(
    results: dict[str, dict], label: str = "Eval",
) -> str:
    """Format evaluation results as an ASCII table.

    Args:
        results: Output of ``Evaluator.evaluate``.
        label: Table header label.

    Returns:
        Formatted string.
    """
    lines = [f"{'=' * 60}", f"  {label} Results", f"{'=' * 60}"]
    lines.append(
        f"  {'Environment':<35} {'WinRate':>8} {'Steps':>8}"
    )
    lines.append(f"  {'-' * 53}")
    for env_id, stats in results.items():
        wr = f"{stats['win_rate']:.2%}"
        st = f"{stats['avg_steps']:.1f}"
        lines.append(f"  {env_id:<35} {wr:>8} {st:>8}")
    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def save_eval_json(
    results: dict,
    path: str,
    metadata: dict | None = None,
) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: Evaluation results dict.
        path: Output file path.
        metadata: Optional extra metadata (e.g. iteration).
    """
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    if metadata:
        payload["metadata"] = metadata
    resolved = str(Path(path).resolve())
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(resolved, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception:
        logger.error(f"Failed to save eval JSON to {resolved}", exc_info=True)


def run_inference(
    cfg,
    checkpoint_path: str,
    env_ids: list[str] | None,
    episodes: int,
    output_path: str | None,
    use_ema: bool,
    log: Logger | None = None,
    des_files: list[str] | None = None,
    blind_global: bool = False,
) -> None:
    """Evaluate a checkpoint on specified environments."""

    device = cfg.device
    logger.info(f"Inference on {device}")

    model = make_model(cfg).to(device)
    ckpt = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if use_ema and "ema_state_dict" in ckpt:
            ema = ModelEMA(model, decay=cfg.ema_decay)
            ema.load_state_dict(ckpt["ema_state_dict"])
            ema.apply_to(model)
    else:
        model.load_state_dict(ckpt)

    model.eval()

    if env_ids is None:
        env_ids = cfg.id_envs + cfg.ood_envs

    evaluator = Evaluator()
    results = evaluator.evaluate(
        env_ids, model, episodes, cfg, device,
        des_files=des_files, blind_global=blind_global,
    )

    print(format_eval_results(results, label="Inference"))

    if log is not None:
        log.log_eval(results, step=0, prefix="inference")
        log.log_summary(
            {f"inference/{env_id}/win_rate": stats["win_rate"]
             for env_id, stats in results.items()}
        )

    if output_path:
        save_eval_json(results, output_path)
        logger.info(f"Results saved to {output_path}")
