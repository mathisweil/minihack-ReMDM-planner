"""DAgger online training loop.

Orchestrates the full DAgger pipeline: collect data via model + oracle,
train on buffer, evaluate periodically, and checkpoint.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.buffer import ReplayBuffer
from src.config import make_run_dir
from src.curriculum import DynamicCurriculum
from src.diffusion.forward import q_sample
from src.diffusion.loss import auxiliary_goal_loss, mdlm_loss
from src.diffusion.schedules import get_schedule
from src.envs.minihack_env import collect_oracle_trajectory
from src.models.denoiser import ModelEMA, make_model, try_compile
from src.planners.collect import DataCollector
from src.planners.inference import Evaluator, save_eval_json
from src.planners.logging import (
    Logger,
    compute_param_drift,
    compute_param_norm,
    gpu_memory_mb,
    reset_gpu_memory_stats,
)

logger = logging.getLogger(__name__)

# Order of the per-step metrics accumulated on device in `train`.
_STEP_METRIC_KEYS = ("loss", "loss_diff", "loss_aux", "grad_norm")


class Trainer:
    """Full DAgger training loop.

    Args:
        model: Denoising model.
        ema_model: EMA tracker.
        optimizer: Torch optimizer.
        scheduler: Optional LR scheduler.
        buffer: Replay buffer.
        collector: DAgger data collector.
        evaluator: Evaluation runner.
        log: Centralised logger.
        cfg: Config namespace.
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        ema_model: ModelEMA,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        buffer: ReplayBuffer,
        collector: DataCollector,
        evaluator: Evaluator,
        log: Logger,
        cfg: SimpleNamespace,
        device: torch.device | str,
        raw_model: nn.Module | None = None,
    ) -> None:
        self.model = model
        # raw_model is the uncompiled model used for eval deep-copies.
        # When torch.compile is off, raw_model is the same as model.
        self._raw_model = raw_model if raw_model is not None else model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.buffer = buffer
        self.collector = collector
        self.evaluator = evaluator
        self.log = log
        self.cfg = cfg
        self.device = device
        self._schedule_fn = get_schedule(cfg.noise_schedule)
        # Snapshot of initial weights for param drift tracking
        self._init_state = {
            k: v.clone()
            for k, v in self._raw_model.state_dict().items()
            if v.is_floating_point()
        }
        # AMP scaler: enabled only when use_amp=true and on CUDA
        self._use_amp = getattr(cfg, "use_amp", False) and str(device).startswith(
            "cuda"
        )
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

    def train(
        self,
        start_iter: int = 0,
        start_env_steps: int = 0,
    ) -> None:
        """Run the DAgger training loop.

        The budget is ``cfg.total_timesteps`` — total env.step() calls
        across model + oracle rollouts. Iteration count is derived; it
        depends on how many env steps each iteration consumes (which in
        turn depends on episode length and efficiency filter outcomes).

        Args:
            start_iter: Iteration index to resume from (for logging).
            start_env_steps: Cumulative env steps already consumed.
        """
        cfg = self.cfg
        env_steps_total = start_env_steps
        iteration = start_iter
        last_id_eval_step = start_env_steps
        last_ood_eval_step = start_env_steps
        last_ckpt_step = start_env_steps

        while env_steps_total < cfg.total_timesteps:
            reset_gpu_memory_stats()
            iter_start = time.perf_counter()

            # 1. Collect N episodes per iteration
            n_eps = getattr(cfg, "episodes_per_iteration", 1)
            num_workers = getattr(cfg, "num_collection_workers", 0)  # fallback 0 != defaults.yaml (8); partial-config callers rely on it
            model_wins = 0
            added_total = 0
            # Accumulators across all n_eps episodes — must be summed,
            # NOT taken from a single (last) episode, otherwise the
            # unified env-step budget undercounts by ~n_eps×.
            model_steps_iter = 0
            oracle_steps_iter = 0
            last_env_id: str = ""

            collect_start = time.perf_counter()
            use_gpu_batch = str(self.device).startswith("cuda") and n_eps > 1
            if use_gpu_batch:
                # GPU-batched collection (all envs in lockstep)
                batch_stats = self.collector.collect_batch_gpu(n_eps)
                for s in batch_stats:
                    model_wins += int(s["model_won"])
                    added_total += int(s["added_to_buffer"])
                    model_steps_iter += int(s["model_steps"])
                    oracle_steps_iter += int(s["oracle_steps"])
                    last_env_id = s.get("env_id", last_env_id)
            elif num_workers > 0 and n_eps > 1:
                # Threaded CPU collection (fallback)
                batch_stats = self.collector.collect_batch_parallel(
                    n_eps,
                )
                for s in batch_stats:
                    model_wins += int(s["model_won"])
                    added_total += int(s["added_to_buffer"])
                    model_steps_iter += int(s["model_steps"])
                    oracle_steps_iter += int(s["oracle_steps"])
                    last_env_id = s.get("env_id", last_env_id)
            else:
                # Sequential collection (reference behaviour)
                for _ in range(n_eps):
                    s = self.collector.collect_one_iteration()
                    model_wins += int(s["model_won"])
                    added_total += int(s["added_to_buffer"])
                    model_steps_iter += int(s["model_steps"])
                    oracle_steps_iter += int(s["oracle_steps"])
                    last_env_id = s.get("env_id", last_env_id)
            collect_time = time.perf_counter() - collect_start

            collect_stats = {
                "env_id": last_env_id,
                "model_won": model_wins,
                "added_to_buffer": added_total,
                "model_steps": model_steps_iter,
                "oracle_steps": oracle_steps_iter,
            }

            # This budget charges oracle env.step() calls; the craftax repo
            # counts learner frames only.
            # Advance the unified env-step budget. Both model and oracle
            # rollouts consume real env.step() calls (the oracle rollout
            # runs in its own env instance in collect_oracle_trajectory),
            # so both contribute to the budget.
            iter_env_steps = model_steps_iter + oracle_steps_iter
            env_steps_total += iter_env_steps

            # 2. Gradient steps (EMA updated after each step)
            # The per-step metrics stay on the device and are
            # summed there. Calling `.item()` per step forced four device
            # syncs per gradient step (400 per iteration), each stalling
            # the CPU until the GPU drained. The single `.tolist()` below
            # is the only sync, and it happens where the metrics are
            # actually logged. The averages are unchanged.
            self.model.train()
            train_start = time.perf_counter()
            metric_sums = torch.zeros(len(_STEP_METRIC_KEYS), device=self.device)
            n_steps = 0
            for _ in range(cfg.grad_steps_per_iteration):
                m = self._train_step_device()
                metric_sums += torch.stack([m[k] for k in _STEP_METRIC_KEYS])
                n_steps += 1
                self.ema_model.update(self._raw_model)
            train_time = time.perf_counter() - train_start

            iter_time = time.perf_counter() - iter_start

            # 4. Log
            n_steps = n_steps or 1
            avg_loss, avg_loss_diff, avg_loss_aux, avg_grad_norm = (
                metric_sums / n_steps
            ).tolist()
            current_lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler is not None
                else self.cfg.dagger_lr
            )

            # Global gate value (how open is the global stream)
            gate_val = None
            if hasattr(self._raw_model, "global_gate"):
                gate_val = torch.sigmoid(self._raw_model.global_gate).item()

            # Buffer online fraction
            buf_total = len(self.buffer)
            buf_online_frac = (
                (buf_total - self.buffer.offline_size) / max(buf_total, 1)
                if hasattr(self.buffer, "offline_size")
                else 0.0
            )

            # Samples per second
            total_samples = n_steps * cfg.dagger_batch_size
            samples_per_sec = total_samples / max(train_time, 1e-6)

            # Env steps per second (uses the iter-summed total, not a
            # single episode — same bug class as the env-step budget).
            env_steps_per_sec = iter_env_steps / max(collect_time, 1e-6)

            metrics = {
                "diffusion/loss": avg_loss,
                "diffusion/loss_diff": avg_loss_diff,
                "diffusion/loss_aux": avg_loss_aux,
                "train/buffer_size": buf_total,
                "train/buffer_online_frac": buf_online_frac,
                "train/model_won": int(collect_stats["model_won"]),
                "train/added_to_buffer": int(collect_stats["added_to_buffer"]),
                "train/episodes_collected": n_eps,
                "train/model_steps": collect_stats["model_steps"],
                "train/oracle_steps": collect_stats["oracle_steps"],
                "train/efficiency_ratio": (
                    collect_stats["model_steps"] / max(collect_stats["oracle_steps"], 1)
                ),
                "train/lr": current_lr,
                "train/grad_norm": avg_grad_norm,
                "train/env_steps": env_steps_total,
                "train/progress": env_steps_total / cfg.total_timesteps,
                "speed/iter_time_sec": iter_time,
                "speed/collect_time_sec": collect_time,
                "speed/train_step_time_sec": train_time,
                "speed/samples_per_sec": samples_per_sec,
                "speed/env_steps_per_sec": env_steps_per_sec,
                "speed/gpu_memory_mb": gpu_memory_mb(),
            }
            if gate_val is not None:
                metrics["train/global_gate"] = gate_val
                metrics["model/ema_gate_value"] = gate_val

            # Model health (every 10 iters to avoid overhead)
            if iteration % 10 == 0:
                metrics["model/param_norm"] = compute_param_norm(self._raw_model)
                metrics["model/param_drift_from_init"] = compute_param_drift(
                    self._raw_model, self._init_state
                )

            # Profile breakdown from GPU-batched collection
            _profile = getattr(self.collector, "_last_profile", {})
            for _pk, _pv in _profile.items():
                metrics[f"profile/{_pk}"] = _pv

            self.log.log(metrics, step=iteration)

            # 5. ID eval — triggered when env-step delta crosses threshold
            if (
                cfg.id_eval_every_timesteps > 0
                and env_steps_total - last_id_eval_step >= cfg.id_eval_every_timesteps
            ):
                eval_model = self.ema_model.make_eval_model(self._raw_model)
                results = self.evaluator.evaluate(
                    cfg.id_envs,
                    eval_model,
                    cfg.eval_episodes_per_env,
                    cfg,
                    self.device,
                )
                self.log.log_eval(results, step=iteration, prefix="eval_id")
                mean_id_wr = (
                    float(np.mean([s["win_rate"] for s in results.values()]))
                    if results
                    else 0.0
                )
                self.log.log(
                    {
                        "eval_id/mean_win_rate": mean_id_wr,
                        **{
                            f"curriculum/{env_id}/win_rate": self.collector.curriculum.win_rate(
                                env_id
                            )
                            for env_id in self.cfg.id_envs
                        },
                    },
                    step=iteration,
                )
                last_id_eval_step = env_steps_total

            # 6. OOD eval — env-step-triggered
            if (
                cfg.ood_eval_every_timesteps > 0
                and env_steps_total - last_ood_eval_step >= cfg.ood_eval_every_timesteps
            ):
                eval_model = self.ema_model.make_eval_model(self._raw_model)
                results = self.evaluator.evaluate(
                    cfg.ood_envs,
                    eval_model,
                    cfg.eval_episodes_per_env,
                    cfg,
                    self.device,
                )
                self.log.log_eval(results, step=iteration, prefix="eval_ood")
                mean_ood_wr = (
                    float(np.mean([s["win_rate"] for s in results.values()]))
                    if results
                    else 0.0
                )
                self.log.log(
                    {"eval_ood/mean_win_rate": mean_ood_wr},
                    step=iteration,
                )
                last_ood_eval_step = env_steps_total

            # 7. Checkpoint — env-step-triggered
            if (
                cfg.checkpoint_every_timesteps > 0
                and env_steps_total - last_ckpt_step >= cfg.checkpoint_every_timesteps
            ):
                self.save_checkpoint(iteration, env_steps_total)
                last_ckpt_step = env_steps_total

            iteration += 1

        # Final checkpoint
        if cfg.save_policy:
            self.save_checkpoint(iteration, env_steps_total)

    def _train_step(self) -> dict[str, float]:
        """One gradient step on a buffer sample, as Python floats.

        Convenience wrapper around :meth:`_train_step_device`. Reading the
        scalars costs one device sync, so the training loop uses the
        device-side variant directly and syncs once per iteration instead.

        Returns:
            Dict with ``"loss"``, ``"loss_diff"``, ``"loss_aux"``,
            and ``"grad_norm"`` float scalars.
        """
        return {k: float(v) for k, v in self._train_step_device().items()}

    def _train_step_device(self) -> dict[str, torch.Tensor]:
        """One gradient step on a buffer sample, metrics left on device.

        Uses AMP (mixed precision) when ``cfg.use_amp`` is ``True``
        and training on CUDA.

        Returns:
            Dict with ``"loss"``, ``"loss_diff"``, ``"loss_aux"``,
            and ``"grad_norm"`` as detached 0-dim tensors.
        """
        cfg = self.cfg
        batch = self.buffer.sample(cfg.dagger_batch_size)
        if batch is None:
            zero = torch.zeros((), device=self.device)
            return dict.fromkeys(_STEP_METRIC_KEYS, zero)
        local_np, global_np, actions_np = batch
        # Widen on the GPU, not the CPU. The buffer holds glyph
        # maps as int16 (minihack_env.py:720-721), so casting before the
        # transfer sends 4x the bytes over PCIe: `global` alone is 27.2 MB
        # per step as int64 against 6.8 MB as int16. The widening is exact
        # for glyph IDs, so values are unchanged.
        local_t = torch.from_numpy(local_np).to(self.device).long()
        global_t = torch.from_numpy(global_np).to(self.device).long()
        actions_t = torch.from_numpy(actions_np).to(self.device).long()

        B = actions_t.shape[0]
        t = torch.rand(B, device=self.device).clamp(1e-5, 1.0 - 1e-5)

        zt = q_sample(
            actions_t,
            t,
            cfg.mask_token,
            cfg.pad_token,
            self._schedule_fn,
        )
        t_discrete = (
            (t * cfg.num_diffusion_steps)
            .long()
            .clamp(
                0,
                cfg.num_diffusion_steps - 1,
            )
        )

        self.optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            out = self.model(local_t, global_t, zt, t_discrete)

            loss_diff = mdlm_loss(
                out["actions"],
                actions_t,
                zt,
                t,
                cfg.mask_token,
                cfg.pad_token,
                self._schedule_fn,
                weight_clip=cfg.loss_weight_clip,
                label_smoothing=cfg.label_smoothing,
            )

            loss_aux = torch.tensor(0.0, device=self.device)
            if "goal_pred" in out:
                loss_aux = auxiliary_goal_loss(out["goal_pred"], global_t)

            loss = loss_diff + cfg.aux_loss_weight * loss_aux

        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            cfg.dagger_grad_clip,
        )
        self._scaler.step(self.optimizer)
        self._scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": loss.detach(),
            "loss_diff": loss_diff.detach(),
            "loss_aux": loss_aux.detach(),
            "grad_norm": grad_norm.detach(),
        }

    def save_checkpoint(
        self,
        iteration: int,
        env_steps: int,
    ) -> None:
        """Save a training checkpoint.

        Args:
            iteration: Current iteration number (for filename + metadata).
            env_steps: Cumulative env.step() count consumed so far.
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"iter{iteration}.pth"

        # Capture W&B run ID for seamless resumption
        wandb_run_id: str | None = None
        if self.log._use_wandb and self.log._run is not None:
            wandb_run_id = self.log._run.id

        state = {
            "model_state_dict": self._raw_model.state_dict(),
            "ema_state_dict": self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "curriculum_state": self.collector.curriculum.state_dict(),
            "iteration": iteration,
            "env_steps": env_steps,
            "wandb_run_id": wandb_run_id,
            "rng_states": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

        # A failed save must halt the run, not let hours of
        # training finish with no usable checkpoint.
        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")

        # Save config snapshot alongside checkpoint
        config_path = ckpt_dir / f"config_iter{iteration}.yaml"
        # The snapshot backs the evaluate-with-snapshot workflow;
        # a checkpoint without one is a defect, so failures raise.
        cfg_dict = {
            k: v for k, v in vars(self.cfg).items() if not k.startswith("_")
        }
        with open(config_path, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)

        # Run eval at checkpoint and save JSON
        try:
            eval_model = self.ema_model.make_eval_model(self._raw_model)
            id_results = self.evaluator.evaluate(
                self.cfg.id_envs,
                eval_model,
                self.cfg.checkpoint_eval_episodes,
                self.cfg,
                self.device,
            )
            ood_results = self.evaluator.evaluate(
                self.cfg.ood_envs,
                eval_model,
                self.cfg.checkpoint_eval_episodes,
                self.cfg,
                self.device,
            )

            id_winrate = (
                float(np.mean([s["win_rate"] for s in id_results.values()]))
                if id_results
                else 0.0
            )
            ood_winrate = (
                float(np.mean([s["win_rate"] for s in ood_results.values()]))
                if ood_results
                else 0.0
            )
            current_lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler is not None
                else self.cfg.dagger_lr
            )
            training_meta = {
                "iteration": iteration,
                "env_steps": env_steps,
                "total_timesteps": self.cfg.total_timesteps,
                "lr": current_lr,
                "dagger_batch_size": self.cfg.dagger_batch_size,
                "aux_loss_weight": self.cfg.aux_loss_weight,
                "buffer_size": len(self.buffer),
                "buffer_capacity": self.cfg.buffer_capacity,
                "ema_decay": self.cfg.ema_decay,
                "grad_steps_per_iteration": self.cfg.grad_steps_per_iteration,
                "episodes_per_iteration": getattr(
                    self.cfg, "episodes_per_iteration", 1
                ),
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

            json_path = ckpt_dir / f"eval_iter{iteration}.json"
            save_eval_json(
                {"id": id_results, "ood": ood_results},
                str(json_path),
                metadata=training_meta,
            )

            # W&B checkpoint log — per-env step metrics + aggregates
            self.log.log_eval(
                id_results,
                step=iteration,
                prefix="ckpt_eval_id",
            )
            self.log.log_eval(
                ood_results,
                step=iteration,
                prefix="ckpt_eval_ood",
            )
            self.log.log(
                {
                    "ckpt_eval/id_winrate": id_winrate,
                    "ckpt_eval/ood_winrate": ood_winrate,
                },
                step=iteration,
            )
            self.log.log_summary(
                {
                    f"ckpt_{iteration}/id_winrate": id_winrate,
                    f"ckpt_{iteration}/ood_winrate": ood_winrate,
                }
            )
        except Exception:
            # Best-checkpoint selection reads these evals; a
            # silently skipped eval biases selection toward the
            # checkpoints whose eval happened to succeed.
            logger.error("Checkpoint eval failed", exc_info=True)
            raise

        # HuggingFace Hub upload (no-op if HF_TOKEN or hub_run_id not set)
        try:
            from scripts.hf_upload import maybe_upload_checkpoint

            maybe_upload_checkpoint(
                str(ckpt_dir),
                getattr(self.cfg, "hub_run_id", None),
                getattr(self.cfg, "hub_repo_id", None),
            )
        except Exception:
            logger.error("HF Hub upload failed", exc_info=True)

        # W&B artifact upload
        self.log.log_checkpoint_artifact(
            checkpoint_path=str(path),
            config_path=str(config_path) if config_path else None,
            iteration=iteration,
            metadata={
                "iteration": iteration,
                "buffer_size": len(self.buffer),
            },
        )

    def load_checkpoint(self, path: str) -> tuple[int, int]:
        """Load a training checkpoint.

        Args:
            path: Path to ``.pth`` checkpoint file.

        Returns:
            ``(start_iter, start_env_steps)`` — the iteration and
            cumulative env-step count to resume from.
        """
        ckpt = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        self._raw_model.load_state_dict(ckpt["model_state_dict"])
        self.ema_model.load_state_dict(ckpt["ema_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.scheduler is not None and ckpt["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.collector.curriculum.load_state_dict(ckpt["curriculum_state"])

        # A resume that cannot restore the saved RNG state would
        # silently continue with fresh randomness while claiming to be a
        # resume, corrupting the seed provenance the interval estimates
        # depend on. All checkpoints from the current code carry
        # rng_states, so absent or unusable state means a malformed
        # checkpoint and raises.
        rng = ckpt.get("rng_states", {})
        if not rng:
            raise RuntimeError(
                f"Checkpoint {path} has no rng_states; refusing to resume "
                "with fresh RNG. Resume from a checkpoint written by the "
                "current code."
            )
        try:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        except Exception as err:
            raise RuntimeError(
                f"Checkpoint {path} carries rng_states that could not "
                "be restored; refusing to resume with fresh RNG. "
                "Retrain or resume from an intact checkpoint."
            ) from err

        iteration = ckpt["iteration"]
        env_steps = ckpt["env_steps"]
        resume_from = iteration + 1
        logger.info(
            f"Resumed from checkpoint: {path} (iter {iteration}, "
            f"env_steps={env_steps}), starting at iter {resume_from}"
        )
        return resume_from, env_steps


def run_dagger(
    cfg: SimpleNamespace,
    checkpoint_path: str | None,
    no_warm_start: bool,
) -> None:
    """DAgger online training loop."""
    make_run_dir(cfg, tag="dagger")

    device = cfg.device
    logger.info(f"DAgger training on {device}")

    raw_model = make_model(cfg).to(device)

    # EMA and eval always use the raw (uncompiled) model — deep-copying
    # a compiled model breaks FX tracing.
    ema = ModelEMA(raw_model, decay=cfg.ema_decay)

    # torch.compile: wrap for training only; shares parameters with raw_model
    model = try_compile(raw_model, cfg)

    # `fused=True` folds the whole AdamW update into one kernel
    # instead of a per-tensor loop over the 72 parameter tensors. CUDA only;
    # it is safe alongside AMP's GradScaler, which passes its inf check
    # through to the fused kernel.
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg.dagger_lr,
        weight_decay=cfg.weight_decay,
        fused=str(device).startswith("cuda"),
    )

    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seq_len, cfg.pad_token)
    curriculum = DynamicCurriculum(
        cfg.id_envs,
        cfg.curriculum_queue_size,
        cfg.curriculum_preseed,
    )

    # DAgger warm start: seed the buffer with 3 oracle trajectories per
    # ID environment (README §DAgger); --no-warm-start disables exactly
    # this seeding (was defect §8.9: the flag only gated checkpoint and
    # W&B resumption while the seeding ran unconditionally).
    if no_warm_start:
        logger.info("Warm-start seeding disabled (--no-warm-start)")
    else:
        for i, env_id in enumerate(cfg.id_envs):
            for s in range(3):
                traj = collect_oracle_trajectory(env_id, seed=i * 100 + s, cfg=cfg)
                if traj is not None:
                    buffer.add(traj)
        logger.info(f"Buffer seeded with {len(buffer)} windows")

    # If resuming, extract W&B run ID from checkpoint before Logger init
    # so the same W&B run is continued (curve continuity).
    if checkpoint_path and not no_warm_start:
        resume_id = getattr(cfg, "wandb_resume_id", None)
        if not resume_id:
            ckpt_peek = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            saved_id = ckpt_peek.get("wandb_run_id")
            if saved_id:
                cfg.wandb_resume_id = saved_id
                logger.info(f"W&B run ID from checkpoint: {saved_id}")
            del ckpt_peek

    # DataCollector uses raw_model for eval copies (not compiled)
    collector = DataCollector(ema, raw_model, buffer, curriculum, cfg, device)
    evaluator = Evaluator()
    log = Logger(cfg)

    trainer = Trainer(
        model,
        ema,
        optimizer,
        None,
        buffer,
        collector,
        evaluator,
        log,
        cfg,
        device,
        raw_model=raw_model,
    )

    start_iter = 0
    start_env_steps = 0
    if checkpoint_path and not no_warm_start:
        start_iter, start_env_steps = trainer.load_checkpoint(
            checkpoint_path,
        )

    trainer.train(
        start_iter=start_iter,
        start_env_steps=start_env_steps,
    )
    log.finish()
