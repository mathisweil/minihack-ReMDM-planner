"""Offline behavioural cloning trainer.

Mirrors the Craftax ``make_train`` closure pattern. Trains the diffusion
model on pre-collected oracle demonstrations using the MDLM ELBO loss
with optional auxiliary goal loss.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import logging
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn as nn

from src.buffer import ReplayBuffer
from src.config import make_run_dir
from src.diffusion.forward import q_sample
from src.diffusion.loss import auxiliary_goal_loss, mdlm_loss
from src.diffusion.schedules import get_schedule
from src.models.denoiser import ModelEMA, make_model, try_compile
from src.planners.logging import Logger

logger = logging.getLogger(__name__)


def make_offline_trainer(cfg: SimpleNamespace) -> Callable:
    """Build the offline BC training closure.

    Args:
        cfg: Config namespace.

    Returns:
        ``train_offline(model, ema_model, buffer, cfg, device) -> dict``
    """
    schedule_fn = get_schedule(cfg.noise_schedule)

    def train_offline(
        model: nn.Module,
        ema_model: ModelEMA,
        buffer: ReplayBuffer,
        cfg: SimpleNamespace,
        device: torch.device | str,
        log: Logger | None = None,
        raw_model: nn.Module | None = None,
        resume_state: dict | None = None,
    ) -> dict:
        """Run offline BC training.

        Args:
            model: Denoising model (may be torch.compiled).
            ema_model: EMA tracker.
            buffer: Replay buffer with offline data.
            cfg: Config namespace.
            device: Torch device.
            log: Optional Logger for wandb and stdout metrics.
            raw_model: Uncompiled model for EMA updates. If ``None``,
                uses *model* directly.
            resume_state: Checkpoint dict to resume from. If provided,
                restores optimizer, scheduler, epoch, and step state.

        Returns:
            Dict with ``"final_loss"`` and ``"loss_history"``.
        """
        _ema_source = raw_model if raw_model is not None else model
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.offline_lr,
            weight_decay=cfg.weight_decay,
        )

        # Unified budget: `total_timesteps` counts env.step()-equivalent
        # samples consumed during training. Each gradient step consumes
        # `offline_batch_size` samples, so total grad steps derives
        # directly from the budget and is independent of dataset size
        # — this is what gives offline / DAgger / SB3 runs a common
        # denominator when comparing curves.
        total_grad_steps = max(
            1, cfg.total_timesteps // cfg.offline_batch_size,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_grad_steps,
            eta_min=cfg.offline_lr * 0.1,
        )
        # Checkpoint every `checkpoint_every_timesteps` samples processed
        # → cadence in gradient-step units. 0 disables.
        ckpt_every_step = (
            cfg.checkpoint_every_timesteps // cfg.offline_batch_size
            if cfg.checkpoint_every_timesteps > 0 else 0
        )
        # Logging cadence. `offline_log_every` is the configured default, but
        # for very short budgets (smoke tests, ablations) we clamp it so that
        # every run emits at least ~10 log points — otherwise short runs go
        # silent and curves cannot be compared across budgets.
        log_every = min(
            max(1, cfg.offline_log_every),
            max(1, total_grad_steps // 10),
        )

        # Restore optimizer/scheduler state if resuming
        step = 0
        if resume_state is not None:
            if "optimizer_state_dict" in resume_state:
                optimizer.load_state_dict(
                    resume_state["optimizer_state_dict"],
                )
            if "scheduler_state_dict" in resume_state:
                scheduler.load_state_dict(
                    resume_state["scheduler_state_dict"],
                )
            step = resume_state.get("step", 0)
            logger.info(
                f"Resumed offline training from step {step}/"
                f"{total_grad_steps}"
            )

        # AMP: enabled when use_amp=true and on CUDA
        _use_amp = (
            getattr(cfg, "use_amp", False)
            and str(device).startswith("cuda")
        )
        scaler = torch.amp.GradScaler("cuda", enabled=_use_amp)

        loss_history: list[float] = []
        _batch_start = time.perf_counter()
        last_ckpt_step = step

        while step < total_grad_steps:
            batch = buffer.sample(cfg.offline_batch_size)
            if batch is None:
                break
            local_np, global_np, actions_np = batch
            local_t = torch.from_numpy(local_np).long().to(device)
            global_t = torch.from_numpy(global_np).long().to(device)
            actions_t = torch.from_numpy(actions_np).long().to(device)

            B = actions_t.shape[0]
            t = torch.rand(B, device=device)  # [B] in [0, 1)
            t = t.clamp(1e-5, 1.0 - 1e-5)

            zt = q_sample(
                actions_t, t, cfg.mask_token, cfg.pad_token,
                schedule_fn,
            )
            t_discrete = (
                t * cfg.num_diffusion_steps
            ).long().clamp(0, cfg.num_diffusion_steps - 1)  # [B]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=_use_amp):
                out = model(local_t, global_t, zt, t_discrete)

                loss_diff = mdlm_loss(
                    out["actions"], actions_t, zt, t,
                    cfg.mask_token, cfg.pad_token, schedule_fn,
                    weight_clip=cfg.loss_weight_clip,
                    label_smoothing=cfg.label_smoothing,
                    use_importance_weighting=cfg.use_importance_weighting,
                )

                loss_aux = torch.tensor(0.0, device=device)
                if "goal_pred" in out:
                    loss_aux = auxiliary_goal_loss(
                        out["goal_pred"], global_t,
                    )

                loss = loss_diff + cfg.aux_loss_weight * loss_aux

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), cfg.offline_grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ema_model.update(_ema_source)
            loss_history.append(loss.item())
            step += 1

            # env-step equivalent: samples processed so far.
            env_steps = step * cfg.offline_batch_size

            if log is not None and step % log_every == 0:
                step_time = time.perf_counter() - _batch_start
                metrics = {
                    "diffusion/loss": loss.item(),
                    "diffusion/loss_diff": loss_diff.item(),
                    "diffusion/loss_aux": loss_aux.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/env_steps": env_steps,
                    "train/progress": step / total_grad_steps,
                    "train/grad_norm": grad_norm.item(),
                    "perf/grad_steps_per_sec": (
                        log_every / max(step_time, 1e-6)
                    ),
                }
                _ema_source_ref = _ema_source
                if hasattr(_ema_source_ref, "global_gate"):
                    metrics["train/global_gate"] = torch.sigmoid(
                        _ema_source_ref.global_gate,
                    ).item()
                log.log(metrics, step=step)
                _batch_start = time.perf_counter()
                logger.info(
                    f"step {step}/{total_grad_steps} "
                    f"(env_steps={env_steps}) loss={loss.item():.4f}"
                )

            # Periodic step-level checkpoint (cadence derived from
            # checkpoint_every_timesteps)
            if (
                ckpt_every_step > 0
                and step - last_ckpt_step >= ckpt_every_step
            ):
                _save_offline_checkpoint(
                    _ema_source, ema_model, optimizer, scheduler,
                    step, cfg, log,
                )
                last_ckpt_step = step

        if log is not None:
            log.log_summary({
                "offline/final_loss": loss_history[-1] if loss_history else 0.0,
                "offline/total_steps": step,
                "offline/total_timesteps": step * cfg.offline_batch_size,
            })

        return {
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
        }

    return train_offline


def _save_offline_checkpoint(
    model: nn.Module,
    ema_model: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    cfg: SimpleNamespace,
    log: Logger | None,
) -> None:
    """Save an offline training checkpoint with W&B run ID.

    Args:
        model: Raw (uncompiled) model.
        ema_model: EMA tracker.
        optimizer: Optimizer.
        scheduler: LR scheduler.
        step: Global gradient step count.
        cfg: Config namespace.
        log: Logger (used to extract W&B run ID).
    """
    wandb_run_id: str | None = None
    if log is not None and log._use_wandb and log._run is not None:
        wandb_run_id = log._run.id

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"offline_step{step}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "env_steps": step * cfg.offline_batch_size,
            "wandb_run_id": wandb_run_id,
        },
        path,
    )
    logger.info(f"Offline checkpoint saved: {path}")


def load_offline_dataset(
    path: str | None, cfg: SimpleNamespace,
) -> dict | None:
    """Load an offline dataset from disk.

    Args:
        path: Path to a ``.pt`` file, or ``None``.
        cfg: Config namespace (unused, reserved for future).

    Returns:
        Loaded dict or ``None``.
    """
    if path is None:
        return None
    try:
        import torch as _torch
        return _torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        logger.error(f"Failed to load dataset from {path}", exc_info=True)
        return None


def run_offline(
    cfg: SimpleNamespace,
    data_path: str | None,
    checkpoint_path: str | None = None,
) -> None:
    """Offline BC training on pre-collected data.

    Args:
        cfg: Config namespace.
        data_path: Path to ``.pt`` dataset file.
        checkpoint_path: Optional checkpoint to resume from. Restores
            model, EMA, optimizer, scheduler, and W&B run for curve
            continuity.
    """
    make_run_dir(cfg, tag="offline")

    device = cfg.device
    logger.info(f"Offline BC on {device}")

    data = load_offline_dataset(data_path, cfg)
    if data is None:
        logger.error("No dataset provided or failed to load. Exiting.")
        sys.exit(1)

    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seq_len, cfg.pad_token)
    buffer.load_offline_data(data, cfg.id_envs)
    logger.info(f"Loaded {len(buffer)} windows")

    if len(buffer) == 0:
        logger.error(
            "Buffer is empty after loading dataset — no trajectories matched "
            f"id_envs={cfg.id_envs}. Exiting."
        )
        sys.exit(1)

    raw_model = make_model(cfg).to(device)

    # torch.compile: wrap for training only; shares params with raw_model
    model = try_compile(raw_model, cfg)

    ema = ModelEMA(raw_model, decay=cfg.ema_decay)

    # If resuming, extract W&B run ID from checkpoint before Logger init
    resume_state: dict | None = None
    if checkpoint_path:
        resume_state = torch.load(
            checkpoint_path, map_location=device, weights_only=False,
        )
        raw_model.load_state_dict(resume_state["model_state_dict"])
        ema.load_state_dict(resume_state["ema_state_dict"])
        resume_id = getattr(cfg, "wandb_resume_id", None)
        if not resume_id:
            saved_id = resume_state.get("wandb_run_id")
            if saved_id:
                cfg.wandb_resume_id = saved_id
                logger.info(f"W&B run ID from checkpoint: {saved_id}")

    log = Logger(cfg)
    train_fn = make_offline_trainer(cfg)
    result = train_fn(
        model, ema, buffer, cfg, device, log=log,
        raw_model=raw_model, resume_state=resume_state,
    )
    logger.info(
        f"Offline training done. Final loss: {result['final_loss']:.4f}"
    )

    # Save final checkpoint for downstream compatibility (DAgger, inference)
    wandb_run_id: str | None = None
    if log._use_wandb and log._run is not None:
        wandb_run_id = log._run.id

    ckpt_dir = Path(cfg.checkpoint_dir)
    path = ckpt_dir / "offline_final.pth"
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "wandb_run_id": wandb_run_id,
        },
        path,
    )
    logger.info(f"Saved offline checkpoint: {path}")
    log.finish()
