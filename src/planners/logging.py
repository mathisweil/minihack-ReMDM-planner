"""Centralised W&B and stdout logging.

Mirrors the Craftax logging conventions with metric namespaces:
``diffusion/``, ``train/``, ``eval_id/``, ``eval_ood/``.
"""

from __future__ import annotations

import logging
import torch
from typing import TYPE_CHECKING
from types import SimpleNamespace

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as _WandbRun

logger = logging.getLogger(__name__)


def download_artifact(
    artifact_ref: str, dst_dir: str = "artifacts",
) -> str | None:
    """Download a W&B artifact via the public API (no active run needed).

    Args:
        artifact_ref: Fully qualified artifact reference, e.g.
            ``"entity/project/checkpoint-iter1000:latest"``.
        dst_dir: Local directory to download into.

    Returns:
        Path to the ``.pth`` file inside the downloaded artifact
        directory, or ``None`` on failure.
    """
    try:
        import wandb
        from pathlib import Path

        api = wandb.Api()
        artifact = api.artifact(artifact_ref)
        artifact_dir = artifact.download(root=dst_dir)
        pth_files = list(Path(artifact_dir).glob("*.pth"))
        if not pth_files:
            logger.error(
                f"No .pth file found in artifact {artifact_ref}"
            )
            return None
        path = str(pth_files[0])
        logger.info(f"Downloaded artifact {artifact_ref} -> {path}")
        return path
    except Exception:
        logger.error(
            f"Failed to download artifact {artifact_ref}",
            exc_info=True,
        )
        return None


def _auto_run_name(cfg: SimpleNamespace) -> str:
    """Generate a descriptive W&B run name from key hyperparameters.

    Format: ``seq{seq_len}_d{n_embd}_L{n_layer}_lr{dagger_lr}_bs{batch}_eta{eta}_{remask}``

    Args:
        cfg: Config namespace.

    Returns:
        A concise, human-readable run name.
    """
    parts = [
        f"seq{cfg.seq_len}",
        f"d{cfg.n_embd}",
        f"L{cfg.n_layer}",
        f"lr{cfg.dagger_lr:.0e}",
        f"bs{cfg.dagger_batch_size}",
        f"eta{cfg.eta}",
        f"{cfg.remask_strategy}",
    ]
    if cfg.use_importance_weighting:
        parts.append("subs")
    if getattr(cfg, "physics_aware_sampling", False):
        parts.append("phys")
    if cfg.seed is not None:
        parts.append(f"s{cfg.seed}")
    return "_".join(parts)


class Logger:
    """Centralised logger for W&B and stdout.

    Args:
        cfg: Config namespace with ``use_wandb``, ``wandb_project``,
            ``wandb_entity``, ``seed``.
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        self._use_wandb = cfg.use_wandb
        self._run: _WandbRun | None = None
        if self._use_wandb:
            try:
                import wandb
                run_name = getattr(cfg, "wandb_run_name", None)
                if not run_name:
                    run_name = _auto_run_name(cfg)
                resume_id = getattr(cfg, "wandb_resume_id", None)
                self._run = wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity or None,
                    name=run_name,
                    config=vars(cfg),
                    id=resume_id or None,
                    resume="must" if resume_id else "never",
                )
                # Define custom metric x-axes
                wandb.define_metric("iteration")
                for ns in (
                    "diffusion/*", "train/*", "perf/*", "speed/*",
                    "model/*",
                    "eval_id/*", "eval_ood/*",
                    "curriculum/*",
                    "ckpt_eval_id/*", "ckpt_eval_ood/*", "ckpt_eval/*",
                    "inference/*",
                ):
                    wandb.define_metric(ns, step_metric="iteration")
            except Exception:
                logger.error("W&B init failed", exc_info=True)
                self._use_wandb = False

    def log_summary(self, metrics: dict) -> None:
        """Write key/value pairs to the wandb run summary (final aggregates).

        Args:
            metrics: Flat ``{key: value}`` dict.
        """
        if self._use_wandb and self._run is not None:
            try:
                self._run.summary.update(metrics)
            except Exception:
                pass

    def log(self, metrics: dict, step: int) -> None:
        """Log a dict of metrics.

        Args:
            metrics: Flat ``{namespace/key: value}`` dict.
            step: Global step index.
        """
        if self._use_wandb and self._run is not None:
            try:
                import wandb
                # Include "iteration" so define_metric(step_metric="iteration") works
                wandb.log({**metrics, "iteration": step}, step=step)
            except Exception:
                pass

        # Stdout summary every 10 steps
        if step % 10 == 0:
            parts = [f"step={step}"]
            for k, v in metrics.items():
                if isinstance(v, float):
                    if abs(v) < 1e-3 and v != 0.0:
                        parts.append(f"{k}={v:.2e}")
                    else:
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

    def log_checkpoint_artifact(
        self,
        checkpoint_path: str,
        config_path: str | None,
        iteration: int,
        metadata: dict | None = None,
        artifact_name: str | None = None,
    ) -> None:
        """Upload a checkpoint as a W&B artifact with config attached.

        Args:
            checkpoint_path: Path to the ``.pth`` checkpoint file.
            config_path: Path to the YAML config snapshot to attach.
                If ``None``, only the checkpoint is uploaded.
            iteration: Iteration number (used in the default artifact
                name when ``artifact_name`` is not provided).
            metadata: Optional metadata dict stored on the artifact.
            artifact_name: Optional explicit artifact name. When
                ``None``, defaults to ``f"checkpoint-iter{iteration}"``.
                Offline BC passes a step-based name to avoid the
                misleading "iter" prefix.
        """
        if not self._use_wandb or self._run is None:
            return
        try:
            import wandb

            name = artifact_name or f"checkpoint-iter{iteration}"
            artifact = wandb.Artifact(
                name=name,
                type="model",
                metadata=metadata or {},
            )
            artifact.add_file(checkpoint_path)
            if config_path is not None:
                artifact.add_file(config_path, name="config.yaml")
            logged = self._run.log_artifact(artifact)  # type: ignore[union-attr]
            logged.wait()  # block until upload completes
            logger.info("W&B artifact uploaded: %s", name)
        except Exception:
            logger.error("W&B artifact upload failed", exc_info=True)

    def finish(self) -> None:
        """Close the W&B run if active."""
        if self._use_wandb and self._run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Metric helper functions (used by both src/ and experiments/)
# ---------------------------------------------------------------------------


def gpu_memory_mb() -> float:
    """Return peak GPU memory allocated in MB since last reset.

    Returns:
        Peak memory in MB, or 0.0 if CUDA is unavailable.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_memory_stats() -> None:
    """Reset GPU peak memory stats for the current device."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def compute_param_norm(model: torch.nn.Module) -> float:
    """Compute total L2 norm of all model parameters.

    Args:
        model: The model.

    Returns:
        Total L2 norm as a float.
    """
    total = 0.0
    for p in model.parameters():
        total += p.data.norm(2).item() ** 2
    return total ** 0.5


def compute_param_drift(
    model: torch.nn.Module,
    ref_state: dict[str, torch.Tensor],
) -> float:
    """Compute L2 distance between current model params and a reference state.

    Args:
        model: Current model.
        ref_state: Reference state_dict (e.g. pretrained weights).

    Returns:
        L2 distance as a float.
    """
    total = 0.0
    for name, p in model.named_parameters():
        if name in ref_state:
            total += (p.data - ref_state[name]).norm(2).item() ** 2
    return total ** 0.5
