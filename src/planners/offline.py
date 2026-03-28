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
    ) -> dict:
        """Run offline BC training.

        Args:
            model: Denoising model.
            ema_model: EMA tracker.
            buffer: Replay buffer with offline data.
            cfg: Config namespace.
            device: Torch device.

        Returns:
            Dict with ``"final_loss"`` and ``"loss_history"``.
        """
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.offline_lr,
        )

        total_steps = cfg.offline_epochs * max(
            1, len(buffer) // cfg.offline_batch_size
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps),
            eta_min=cfg.offline_lr * 0.1,
        )

        loss_history: list[float] = []
        step = 0

        for epoch in range(cfg.offline_epochs):
            n_batches = max(1, len(buffer) // cfg.offline_batch_size)
            for _ in range(n_batches):
                local_np, global_np, actions_np = buffer.sample(
                    cfg.offline_batch_size,
                )
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
                    t * (cfg.num_diffusion_steps - 1)
                ).long()  # [B]

                out = model(local_t, global_t, zt, t_discrete)

                loss_diff = mdlm_loss(
                    out["actions"], actions_t, zt, t,
                    cfg.mask_token, cfg.pad_token, schedule_fn,
                    weight_clip=cfg.loss_weight_clip,
                    label_smoothing=cfg.label_smoothing,
                )

                loss_aux = torch.tensor(0.0, device=device)
                if "goal_pred" in out:
                    loss_aux = auxiliary_goal_loss(
                        out["goal_pred"], global_t,
                    )

                loss = loss_diff + cfg.aux_loss_weight * loss_aux

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.offline_grad_clip,
                )
                optimizer.step()
                scheduler.step()

                ema_model.update(model)
                loss_history.append(loss.item())
                step += 1

            logger.info(
                f"Epoch {epoch + 1}/{cfg.offline_epochs} "
                f"loss={loss_history[-1]:.4f}"
            )

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

    model = make_model(cfg).to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)

    train_fn = make_offline_trainer(cfg)
    result = train_fn(model, ema, buffer, cfg, device)
    logger.info(f"Offline training done. Final loss: {result['final_loss']:.4f}")

    # Save checkpoint
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "offline_final.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
        },
        path,
    )
    logger.info(f"Saved offline checkpoint: {path}")
