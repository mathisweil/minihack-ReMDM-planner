"""Offline behavioural cloning trainer.

Mirrors the Craftax ``make_train`` closure pattern. Trains the diffusion
model on pre-collected oracle demonstrations using the MDLM ELBO loss
with optional auxiliary goal loss.
"""

from __future__ import annotations

import sys
from pathlib import Path
import logging
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn as nn

from src.buffer import ReplayBuffer
from src.diffusion.forward import q_sample
from src.diffusion.loss import auxiliary_goal_loss, mdlm_loss
from src.diffusion.schedules import get_schedule
from src.models.denoiser import ModelEMA, make_model
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
    ) -> dict:
        """Run offline BC training.

        Args:
            model: Denoising model.
            ema_model: EMA tracker.
            buffer: Replay buffer with offline data.
            cfg: Config namespace.
            device: Torch device.
            log: Optional Logger for wandb and stdout metrics.

        Returns:
            Dict with ``"final_loss"`` and ``"loss_history"``.
        """
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.offline_lr,
            weight_decay=cfg.weight_decay,
        )

        total_steps = cfg.offline_epochs * max(
            1, len(buffer) // cfg.offline_batch_size
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps),
            eta_min=cfg.offline_lr * 0.1,
        )

        # AMP: enabled when use_amp=true and on CUDA
        _use_amp = (
            getattr(cfg, "use_amp", False)
            and str(device).startswith("cuda")
        )
        scaler = torch.amp.GradScaler("cuda", enabled=_use_amp)

        loss_history: list[float] = []
        step = 0

        for epoch in range(cfg.offline_epochs):
            n_batches = max(1, len(buffer) // cfg.offline_batch_size)
            for _ in range(n_batches):
                batch = buffer.sample(cfg.offline_batch_size)
                if batch is None:
                    continue
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
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.offline_grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                ema_model.update(model)
                loss_history.append(loss.item())
                step += 1

                if log is not None and step % cfg.offline_log_every == 0:
                    log.log(
                        {
                            "diffusion/loss": loss.item(),
                            "diffusion/loss_diff": loss_diff.item(),
                            "diffusion/loss_aux": loss_aux.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        },
                        step=step,
                    )

            logger.info(
                f"Epoch {epoch + 1}/{cfg.offline_epochs} "
                f"loss={loss_history[-1]:.4f}"
            )

        if log is not None:
            log.log_summary({
                "offline/final_loss": loss_history[-1] if loss_history else 0.0,
                "offline/total_steps": step,
                "offline/epochs": cfg.offline_epochs,
            })

        return {
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
        }

    return train_offline


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


def run_offline(cfg, data_path: str | None) -> None:
    """Offline BC training on pre-collected data."""

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
    if getattr(cfg, "torch_compile", False) and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile")
        model = torch.compile(raw_model, mode="default")
    else:
        model = raw_model

    ema = ModelEMA(raw_model, decay=cfg.ema_decay)

    log = Logger(cfg)
    train_fn = make_offline_trainer(cfg)
    result = train_fn(model, ema, buffer, cfg, device, log=log)
    logger.info(f"Offline training done. Final loss: {result['final_loss']:.4f}")

    # Save checkpoint
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "offline_final.pth"
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "ema_state_dict": ema.state_dict(),
        },
        path,
    )
    logger.info(f"Saved offline checkpoint: {path}")
    log.finish()
