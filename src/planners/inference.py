"""Stateless evaluation runner.

Runs episodes using the diffusion model and collects per-environment
win rates, average rewards, and step counts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import torch

from src.models.denoiser import ModelEMA, make_model

logger = logging.getLogger(__name__)


class Evaluator:
    """Stateless evaluation runner.

    Runs the model on a set of environments and returns aggregate
    statistics per environment.
    """

    @torch.no_grad()
    def evaluate(
        self,
        env_ids: list[str],
        model: torch.nn.Module,
        n_episodes: int,
        cfg: SimpleNamespace,
        device: torch.device | str,
    ) -> dict[str, dict]:
        """Evaluate *model* on each environment in *env_ids*.

        Args:
            env_ids: List of MiniHack environment IDs.
            model: Denoising model (eval mode).
            n_episodes: Episodes per environment.
            cfg: Config namespace.
            device: Torch device.

        Returns:
            ``{env_id: {"win_rate", "avg_reward", "avg_steps",
            "n_episodes"}}``
        """
        from src.planners.collect import run_model_episode

        model.eval()
        results: dict[str, dict] = {}

        for env_id in env_ids:
            wins = 0
            total_reward = 0.0
            total_steps = 0
            completed = 0

            for ep in range(n_episodes):
                seed = 42 + hash((env_id, ep)) % (2**31)
                try:
                    ep_result = run_model_episode(
                        model, env_id, cfg, device, seed,
                    )
                    if ep_result["won"]:
                        wins += 1
                    total_steps += ep_result["steps"]
                    completed += 1
                except Exception:
                    logger.warning(
                        f"Episode {ep} failed for {env_id}",
                        exc_info=True,
                    )
                    completed += 1  # count as loss

            n = max(completed, 1)
            results[env_id] = {
                "win_rate": wins / n,
                "avg_reward": total_reward / n,
                "avg_steps": total_steps / n,
                "n_episodes": completed,
            }

        return results


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
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception:
        logger.error(f"Failed to save eval JSON to {path}", exc_info=True)


def run_inference(
    cfg,
    checkpoint_path: str,
    env_ids: list[str] | None,
    episodes: int,
    output_path: str | None,
    use_ema: bool,
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
    results = evaluator.evaluate(env_ids, model, episodes, cfg, device)

    print(format_eval_results(results, label="Inference"))

    if output_path:
        save_eval_json(results, output_path)
        logger.info(f"Results saved to {output_path}")
