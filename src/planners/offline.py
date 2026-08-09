"""Offline behavioural cloning trainer.

Mirrors the Craftax ``make_train`` closure pattern. Trains the diffusion
model on pre-collected oracle demonstrations using the MDLM ELBO loss
with optional auxiliary goal loss.
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.buffer import ReplayBuffer
from src.config import make_run_dir
from src.diffusion.forward import q_sample
from src.diffusion.loss import auxiliary_goal_loss, mdlm_loss
from src.diffusion.schedules import get_schedule
from src.models.denoiser import ModelEMA, make_model, try_compile
from src.planners.inference import Evaluator, save_eval_json
from src.planners.logging import (
    Logger,
    compute_param_drift,
    compute_param_norm,
    gpu_memory_mb,
    reset_gpu_memory_stats,
)

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
        evaluator: Evaluator | None = None,
        id_envs: list[str] | None = None,
        ood_envs: list[str] | None = None,
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
            evaluator: Optional ``Evaluator`` instance for periodic ID/OOD
                evaluation. When ``None``, no eval is run during training.
            id_envs: In-distribution environment IDs for periodic eval.
                Required (non-empty) if ``evaluator`` is provided and
                ``cfg.id_eval_every_timesteps > 0``.
            ood_envs: Out-of-distribution environment IDs for periodic
                eval. Required (non-empty) if ``evaluator`` is provided
                and ``cfg.ood_eval_every_timesteps > 0``.

        Returns:
            Dict with ``"final_loss"`` and ``"loss_history"``.
        """
        _ema_source = raw_model if raw_model is not None else model
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.offline_lr,
            weight_decay=cfg.weight_decay,
        )

        # Unified budget: `total_timesteps` counts env.step()-equivalent
        # samples consumed during training. Each gradient step consumes
        # `offline_batch_size` samples, so total grad steps derives
        # directly from the budget and is independent of dataset size
        # — this is what gives offline / DAgger / SB3 runs a common
        # denominator when comparing curves.
        total_grad_steps = max(
            1,
            cfg.total_timesteps // cfg.offline_batch_size,
        )
        # Optional override: pin offline gradient budget independently
        # of `total_timesteps`. Used for paper-fair compute matching
        # against a specific DAgger iteration count, e.g.
        # `offline_total_grad_steps: 60000` to match 600 DAgger iters
        # × `grad_steps_per_iteration: 100` AdamW updates regardless of
        # what env-step budget DAgger consumed in those iters.
        _grad_override = getattr(cfg, "offline_total_grad_steps", None)
        if _grad_override is not None and _grad_override > 0:
            total_grad_steps = int(_grad_override)
            logger.info(
                "Offline grad budget pinned via offline_total_grad_steps="
                f"{total_grad_steps} (overrides total_timesteps)"
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_grad_steps,
            eta_min=cfg.offline_lr * 0.1,
        )
        # Checkpoint cadence — defaults to deriving from
        # `checkpoint_every_timesteps` (env-step units → grad-step units
        # via // batch_size). The optional `offline_checkpoint_every_grad_steps`
        # override is used when an offline run is pinned via
        # `offline_total_grad_steps` and needs an aligned cadence in
        # grad-step units (env-step cadence diverges wildly from grad-step
        # cadence between offline and DAgger because their sample-to-step
        # ratios differ by ~50x).
        _ckpt_grad_override = getattr(
            cfg,
            "offline_checkpoint_every_grad_steps",
            None,
        )
        if _ckpt_grad_override is not None and _ckpt_grad_override > 0:
            ckpt_every_step = int(_ckpt_grad_override)
        else:
            ckpt_every_step = (
                cfg.checkpoint_every_timesteps // cfg.offline_batch_size
                if cfg.checkpoint_every_timesteps > 0
                else 0
            )
        # Eval cadence — same override pattern. Without this, an offline
        # run pinned at e.g. 60k grad steps with the default
        # `id_eval_every_timesteps=250000` would fire ~491 evals
        # (250000 // 2048 = 122 grad steps per eval), which is
        # impractically dense.
        _eval_grad_override = getattr(
            cfg,
            "offline_eval_every_grad_steps",
            None,
        )
        if _eval_grad_override is not None and _eval_grad_override > 0:
            id_eval_every_env_steps = int(_eval_grad_override) * cfg.offline_batch_size
            ood_eval_every_env_steps = id_eval_every_env_steps
        else:
            id_eval_every_env_steps = cfg.id_eval_every_timesteps
            ood_eval_every_env_steps = cfg.ood_eval_every_timesteps
        # Logging cadence. `offline_log_every` is the *minimum* cadence;
        # the actual `log_every` is clamped on both ends so the number of
        # log points stays in [~10, ~1000] regardless of run length:
        #
        #   * Lower bound (`floor`): on very long runs, force `log_every`
        #     up so total log points cap at ~1000. Without this, a 600k
        #     grad-step run with the default `offline_log_every=10` would
        #     emit 60,000 W&B points — silent log spam.
        #
        #   * Upper bound (`ceiling`): on very short runs (smoke, fast
        #     ablations) clamp `log_every` down so every run emits at
        #     least ~10 log points and curves stay comparable across
        #     budgets.
        #
        # When the configured value sits inside the [floor, ceiling]
        # window (the common case), it is used unchanged.
        _floor = max(1, total_grad_steps // 1000)
        _ceiling = max(1, total_grad_steps // 10)
        log_every = min(
            _ceiling,
            max(_floor, cfg.offline_log_every),
        )

        # Restore optimizer/scheduler state if resuming
        step = 0
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            step = resume_state["step"]
            # FIX-B4 (offline arm): a resume without the saved RNG state
            # would silently continue with fresh randomness; see the
            # online.py counterpart.
            rng = resume_state.get("rng_states", {})
            if not rng:
                raise RuntimeError(
                    "Offline checkpoint has no rng_states; refusing to "
                    "resume with fresh RNG. Resume from a checkpoint "
                    "written by the current code."
                )
            try:
                torch.set_rng_state(rng["torch"])
                np.random.set_state(rng["numpy"])
                random.setstate(rng["python"])
            except Exception as err:
                raise RuntimeError(
                    "Offline checkpoint carries rng_states that could not "
                    "be restored; refusing to resume with fresh RNG."
                ) from err
            logger.info(f"Resumed offline training from step {step}/{total_grad_steps}")

        # AMP: enabled when use_amp=true and on CUDA
        _use_amp = getattr(cfg, "use_amp", False) and str(device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=_use_amp)

        loss_history: list[float] = []
        _batch_start = time.perf_counter()
        last_ckpt_step = step
        # Periodic eval anchors (env-step units, mirroring online.py).
        # Snapping to current env_steps avoids accumulated drift across
        # resumes; the next eval fires once another full interval has
        # been processed since the resume point.
        last_id_eval_env_steps = step * cfg.offline_batch_size
        last_ood_eval_env_steps = step * cfg.offline_batch_size

        # Snapshot of initial weights for `model/param_drift_from_init`.
        # Mirrors online.py:Trainer.__init__.
        _init_state = {
            k: v.detach().clone()
            for k, v in _ema_source.state_dict().items()
            if v.is_floating_point()
        }
        # Counts logging emissions (not raw grad steps), used to gate
        # the once-per-10-windows model health metrics analogously to
        # online.py's `iteration % 10 == 0` cadence.
        log_windows = 0
        reset_gpu_memory_stats()

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
                actions_t,
                t,
                cfg.mask_token,
                cfg.pad_token,
                schedule_fn,
            )
            t_discrete = (
                (t * cfg.num_diffusion_steps)
                .long()
                .clamp(0, cfg.num_diffusion_steps - 1)
            )  # [B]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=_use_amp):
                out = model(local_t, global_t, zt, t_discrete)

                loss_diff = mdlm_loss(
                    out["actions"],
                    actions_t,
                    zt,
                    t,
                    cfg.mask_token,
                    cfg.pad_token,
                    schedule_fn,
                    weight_clip=cfg.loss_weight_clip,
                    label_smoothing=cfg.label_smoothing,
                )

                loss_aux = torch.tensor(0.0, device=device)
                if "goal_pred" in out:
                    loss_aux = auxiliary_goal_loss(
                        out["goal_pred"],
                        global_t,
                    )

                loss = loss_diff + cfg.aux_loss_weight * loss_aux

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.offline_grad_clip,
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
                log_windows += 1

                # Buffer state — for offline mode `offline_size` always
                # equals `len(buffer)` (no online appends), so the
                # online fraction is always 0.0. Logged anyway for
                # symmetry with the DAgger curves.
                buf_total = len(buffer)
                buf_online_frac = (
                    (buf_total - buffer.offline_size) / max(buf_total, 1)
                    if hasattr(buffer, "offline_size")
                    else 0.0
                )

                # Throughput: samples processed in this logging window.
                samples_window = log_every * cfg.offline_batch_size
                samples_per_sec = samples_window / max(step_time, 1e-6)

                _ema_source_ref = _ema_source
                metrics = {
                    "diffusion/loss": loss.item(),
                    "diffusion/loss_diff": loss_diff.item(),
                    "diffusion/loss_aux": loss_aux.item(),
                    "train/buffer_size": buf_total,
                    "train/buffer_online_frac": buf_online_frac,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/env_steps": env_steps,
                    "train/progress": step / total_grad_steps,
                    "train/grad_norm": grad_norm.item(),
                    "speed/train_step_time_sec": step_time,
                    "speed/samples_per_sec": samples_per_sec,
                    "speed/gpu_memory_mb": gpu_memory_mb(),
                }
                if hasattr(_ema_source_ref, "global_gate"):
                    gate_val = torch.sigmoid(
                        _ema_source_ref.global_gate,
                    ).item()
                    metrics["train/global_gate"] = gate_val
                    metrics["model/ema_gate_value"] = gate_val

                # Model health (every 10 logging windows to keep overhead
                # low — matches online.py's `iteration % 10 == 0`).
                if log_windows % 10 == 1:
                    metrics["model/param_norm"] = compute_param_norm(
                        _ema_source_ref,
                    )
                    metrics["model/param_drift_from_init"] = compute_param_drift(
                        _ema_source_ref,
                        _init_state,
                    )

                log.log(metrics, step=step)
                _batch_start = time.perf_counter()
                reset_gpu_memory_stats()
                logger.info(
                    f"step {step}/{total_grad_steps} "
                    f"(env_steps={env_steps}) loss={loss.item():.4f}"
                )

            # Periodic ID eval — env-step delta-check (mirrors
            # online.py:277-305). Eval is opt-in: skipped entirely when
            # no Evaluator was threaded through. The cadence variable
            # already accounts for the optional
            # `offline_eval_every_grad_steps` override.
            if (
                evaluator is not None
                and id_envs
                and id_eval_every_env_steps > 0
                and env_steps - last_id_eval_env_steps >= id_eval_every_env_steps
            ):
                eval_model = ema_model.make_eval_model(_ema_source)
                results = evaluator.evaluate(
                    id_envs,
                    eval_model,
                    cfg.eval_episodes_per_env,
                    cfg,
                    device,
                )
                if log is not None:
                    log.log_eval(results, step=step, prefix="eval_id")
                    mean_id_wr = (
                        (sum(s["win_rate"] for s in results.values()) / len(results))
                        if results
                        else 0.0
                    )
                    log.log(
                        {"eval_id/mean_win_rate": mean_id_wr},
                        step=step,
                    )
                last_id_eval_env_steps = env_steps

            # Periodic OOD eval — same delta-check pattern.
            if (
                evaluator is not None
                and ood_envs
                and ood_eval_every_env_steps > 0
                and env_steps - last_ood_eval_env_steps >= ood_eval_every_env_steps
            ):
                eval_model = ema_model.make_eval_model(_ema_source)
                results = evaluator.evaluate(
                    ood_envs,
                    eval_model,
                    cfg.eval_episodes_per_env,
                    cfg,
                    device,
                )
                if log is not None:
                    log.log_eval(results, step=step, prefix="eval_ood")
                    mean_ood_wr = (
                        (sum(s["win_rate"] for s in results.values()) / len(results))
                        if results
                        else 0.0
                    )
                    log.log(
                        {"eval_ood/mean_win_rate": mean_ood_wr},
                        step=step,
                    )
                last_ood_eval_env_steps = env_steps

            # Periodic step-level checkpoint (cadence derived from
            # checkpoint_every_timesteps)
            if ckpt_every_step > 0 and step - last_ckpt_step >= ckpt_every_step:
                _save_offline_checkpoint(
                    _ema_source,
                    ema_model,
                    optimizer,
                    scheduler,
                    step,
                    cfg,
                    log,
                    evaluator=evaluator,
                    id_envs=id_envs,
                    ood_envs=ood_envs,
                    device=device,
                )
                last_ckpt_step = step

        if log is not None:
            log.log_summary(
                {
                    "offline/final_loss": loss_history[-1] if loss_history else 0.0,
                    "offline/total_steps": step,
                    "offline/total_timesteps": step * cfg.offline_batch_size,
                }
            )

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
    evaluator: Evaluator | None = None,
    id_envs: list[str] | None = None,
    ood_envs: list[str] | None = None,
    device: torch.device | str | None = None,
) -> None:
    """Save an offline training checkpoint, eval, and W&B artifact.

    Mirrors the DAgger ``Trainer.save_checkpoint`` flow:
        1. Persist model + EMA + optimizer + scheduler state to disk.
        2. Save a YAML config snapshot alongside the checkpoint.
        3. Run an EMA-weight ID + OOD eval and emit ``ckpt_eval_*``
           metrics + an eval JSON sidecar.
        4. Upload the checkpoint + config snapshot as a W&B artifact.

    Steps 3 and 4 are skipped gracefully when ``evaluator`` / envs /
    ``device`` are not provided, so callers that just want the bare
    state dump still work.

    Args:
        model: Raw (uncompiled) model — used both for ``state_dict``
            persistence and as the source argument to
            ``ema_model.make_eval_model``.
        ema_model: EMA tracker.
        optimizer: Optimizer.
        scheduler: LR scheduler.
        step: Global gradient step count (used in filenames + metadata).
        cfg: Config namespace.
        log: Logger (used to extract W&B run ID, log eval metrics,
            and upload artifact).
        evaluator: Optional evaluator. When ``None``, the checkpoint
            eval is skipped.
        id_envs: ID env IDs for the checkpoint eval.
        ood_envs: OOD env IDs for the checkpoint eval.
        device: Torch device for the checkpoint eval.
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
            "rng_states": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        path,
    )
    logger.info(f"Offline checkpoint saved: {path}")

    # Save config snapshot alongside checkpoint (mirrors DAgger).
    config_path: Path | None = ckpt_dir / f"config_offline_step{step}.yaml"
    # FIX-B3: snapshot failures raise (see online.py counterpart).
    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    with open(config_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

    # Checkpoint-time eval — mirrors Trainer.save_checkpoint in online.py.
    # Skipped when the caller did not thread an evaluator through.
    if evaluator is not None and id_envs and ood_envs and device is not None:
        try:
            eval_model = ema_model.make_eval_model(model)
            id_results = evaluator.evaluate(
                id_envs,
                eval_model,
                cfg.checkpoint_eval_episodes,
                cfg,
                device,
            )
            ood_results = evaluator.evaluate(
                ood_envs,
                eval_model,
                cfg.checkpoint_eval_episodes,
                cfg,
                device,
            )

            id_winrate = (
                (sum(s["win_rate"] for s in id_results.values()) / len(id_results))
                if id_results
                else 0.0
            )
            ood_winrate = (
                (sum(s["win_rate"] for s in ood_results.values()) / len(ood_results))
                if ood_results
                else 0.0
            )

            current_lr = scheduler.get_last_lr()[0]
            training_meta = {
                "step": step,
                "env_steps": step * cfg.offline_batch_size,
                "total_timesteps": cfg.total_timesteps,
                "lr": current_lr,
                "offline_batch_size": cfg.offline_batch_size,
                "aux_loss_weight": cfg.aux_loss_weight,
                "ema_decay": cfg.ema_decay,
                "id_winrate": id_winrate,
                "ood_winrate": ood_winrate,
                "per_env_id": {
                    env_id: {
                        "win_rate": s["win_rate"],
                        "wins": s.get("wins", 0),
                        "avg_reward": s["avg_reward"],
                        "avg_steps": s["avg_steps"],
                        "n_episodes": s["n_episodes"],
                    }
                    for env_id, s in id_results.items()
                },
                "per_env_ood": {
                    env_id: {
                        "win_rate": s["win_rate"],
                        "wins": s.get("wins", 0),
                        "avg_reward": s["avg_reward"],
                        "avg_steps": s["avg_steps"],
                        "n_episodes": s["n_episodes"],
                    }
                    for env_id, s in ood_results.items()
                },
            }

            json_path = ckpt_dir / f"eval_offline_step{step}.json"
            save_eval_json(
                {"id": id_results, "ood": ood_results},
                str(json_path),
                metadata=training_meta,
            )

            if log is not None:
                log.log_eval(
                    id_results,
                    step=step,
                    prefix="ckpt_eval_id",
                )
                log.log_eval(
                    ood_results,
                    step=step,
                    prefix="ckpt_eval_ood",
                )
                log.log(
                    {
                        "ckpt_eval/id_winrate": id_winrate,
                        "ckpt_eval/ood_winrate": ood_winrate,
                    },
                    step=step,
                )
                log.log_summary(
                    {
                        f"ckpt_offline_step{step}/id_winrate": id_winrate,
                        f"ckpt_offline_step{step}/ood_winrate": ood_winrate,
                    }
                )
        except Exception:
            # FIX-B3: selection integrity (see online.py counterpart).
            logger.error("Offline checkpoint eval failed", exc_info=True)
            raise

    # W&B artifact upload (no-op when wandb is not initialised).
    if log is not None:
        log.log_checkpoint_artifact(
            checkpoint_path=str(path),
            config_path=str(config_path) if config_path else None,
            iteration=step,
            metadata={"step": step, "mode": "offline"},
            artifact_name=f"checkpoint-offline-step{step}",
        )


def load_offline_dataset(
    path: str | None,
    cfg: SimpleNamespace,
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

    # Offline buffer must hold the full pre-collected dataset. DAgger's
    # `buffer_capacity` (typically 10k) would silently FIFO-evict 99% of
    # the dataset, so honour the optional `offline_buffer_capacity`
    # override when present.
    _offline_buf_cap = (
        getattr(cfg, "offline_buffer_capacity", None) or cfg.buffer_capacity
    )
    buffer = ReplayBuffer(_offline_buf_cap, cfg.seq_len, cfg.pad_token)
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
            checkpoint_path,
            map_location=device,
            weights_only=False,
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
    evaluator = Evaluator()
    train_fn = make_offline_trainer(cfg)
    result = train_fn(
        model,
        ema,
        buffer,
        cfg,
        device,
        log=log,
        raw_model=raw_model,
        resume_state=resume_state,
        evaluator=evaluator,
        id_envs=cfg.id_envs,
        ood_envs=cfg.ood_envs,
    )
    logger.info(f"Offline training done. Final loss: {result['final_loss']:.4f}")

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
