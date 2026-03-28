"""Centralised W&B and stdout logging.

Mirrors the Craftax logging conventions with metric namespaces:
``diffusion/``, ``train/``, ``eval_id/``, ``eval_ood/``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)


class Logger:
    """Centralised logger for W&B and stdout.

    Args:
        cfg: Config namespace with ``use_wandb``, ``wandb_project``,
            ``wandb_entity``, ``seed``.
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        self._use_wandb = cfg.use_wandb
        self._run = None
        if self._use_wandb:
            try:
                import wandb
                self._run = wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity or None,
                    config=vars(cfg),
                )
            except Exception:
                logger.error("W&B init failed", exc_info=True)
                self._use_wandb = False

    def log(self, metrics: dict, step: int) -> None:
        """Log a dict of metrics.

        Args:
            metrics: Flat ``{namespace/key: value}`` dict.
            step: Global step index.
        """
        if self._use_wandb and self._run is not None:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

        # Stdout summary every 10 steps
        if step % 10 == 0:
            parts = [f"step={step}"]
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            logger.info("  ".join(parts))

    def log_eval(
        self, results: dict[str, dict], step: int, prefix: str,
    ) -> None:
        """Flatten evaluation results and log them.

        Args:
            results: ``{env_id: {"win_rate", ...}}``
            step: Global step.
            prefix: Metric namespace prefix (e.g. ``"eval_id"``).
        """
        flat: dict[str, float] = {}
        for env_id, stats in results.items():
            for key, val in stats.items():
                if isinstance(val, (int, float)):
                    flat[f"{prefix}/{env_id}/{key}"] = val
        self.log(flat, step=step)

    def finish(self) -> None:
        """Close the W&B run if active."""
        if self._use_wandb and self._run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
