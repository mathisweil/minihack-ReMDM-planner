"""DAgger online training loop.

Orchestrates the full DAgger pipeline: collect data via model + oracle,
train on buffer, evaluate periodically, and checkpoint.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from src.buffer import ReplayBuffer
from src.diffusion.forward import q_sample
from src.diffusion.loss import auxiliary_goal_loss, mdlm_loss
from src.diffusion.schedules import get_schedule
from src.models.denoiser import ModelEMA, make_model
from src.planners.collect import DataCollector
from src.planners.inference import Evaluator, save_eval_json
from src.planners.logging import Logger
from src.curriculum import DynamicCurriculum
from src.envs.minihack_env import collect_oracle_trajectory

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.model = model
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

    # ── Main loop ────────────────────────────────────────────────

    def train(self, start_iter: int = 0) -> None:
        """Run the DAgger training loop.

        Args:
            start_iter: Iteration to resume from.
        """
        cfg = self.cfg
        for iteration in range(start_iter, cfg.max_iterations):
            # 1. Collect N episodes per iteration
            n_eps = getattr(cfg, "episodes_per_iteration", 1)
            model_wins = 0
            added_total = 0
            last_stats: dict = {}
            for _ in range(n_eps):
                last_stats = self.collector.collect_one_iteration()
                model_wins += int(last_stats["model_won"])
                added_total += int(last_stats["added_to_buffer"])
            collect_stats = {
                **last_stats,
                "model_won": model_wins,
                "added_to_buffer": added_total,
            }

            # 2. Gradient steps
            self.model.train()
            losses: list[float] = []
            for _ in range(cfg.grad_steps_per_iteration):
                loss_val = self._train_step()
                losses.append(loss_val)

            # 3. EMA update
            self.ema_model.update(self.model)

            # 4. Log
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            current_lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler is not None
                else self.cfg.dagger_lr
            )
            self.log.log(
                {
                    "diffusion/loss": avg_loss,
                    "train/buffer_size": len(self.buffer),
                    "train/model_won": int(collect_stats["model_won"]),
                    "train/added_to_buffer": int(
                        collect_stats["added_to_buffer"]
                    ),
                    "train/episodes_collected": n_eps,
                    "train/model_steps": collect_stats["model_steps"],
                    "train/oracle_steps": collect_stats["oracle_steps"],
                    "train/efficiency_ratio": (
                        collect_stats["model_steps"]
                        / max(collect_stats["oracle_steps"], 1)
                    ),
                    "train/lr": current_lr,
                },
                step=iteration,
            )

            # 5. ID eval
            if (
                cfg.id_eval_every > 0
                and iteration > 0
                and iteration % cfg.id_eval_every == 0
            ):
                eval_model = self.ema_model.make_eval_model(self.model)
                results = self.evaluator.evaluate(
                    cfg.id_envs,
                    eval_model,
                    cfg.eval_episodes_per_env,
                    cfg,
                    self.device,
                )
                self.log.log_eval(results, step=iteration, prefix="eval_id")
                mean_id_wr = float(np.mean(
                    [s["win_rate"] for s in results.values()]
                )) if results else 0.0
                self.log.log(
                    {
                        "eval_id/mean_win_rate": mean_id_wr,
                        **{
                            f"curriculum/{env_id}/win_rate":
                                self.collector.curriculum.win_rate(env_id)
                            for env_id in self.cfg.id_envs
                        },
                    },
                    step=iteration,
                )

            # 6. OOD eval
            if (
                cfg.ood_eval_every > 0
                and iteration > 0
                and iteration % cfg.ood_eval_every == 0
            ):
                eval_model = self.ema_model.make_eval_model(self.model)
                results = self.evaluator.evaluate(
                    cfg.ood_envs,
                    eval_model,
                    cfg.eval_episodes_per_env,
                    cfg,
                    self.device,
                )
                self.log.log_eval(results, step=iteration, prefix="eval_ood")
                mean_ood_wr = float(np.mean(
                    [s["win_rate"] for s in results.values()]
                )) if results else 0.0
                self.log.log(
                    {"eval_ood/mean_win_rate": mean_ood_wr}, step=iteration,
                )

            # 7. Checkpoint
            if (
                cfg.checkpoint_every > 0
                and iteration > 0
                and iteration % cfg.checkpoint_every == 0
            ):
                self.save_checkpoint(iteration)

        # Final checkpoint
        if cfg.save_policy:
            self.save_checkpoint(cfg.max_iterations)

    # ── Single gradient step ─────────────────────────────────────

    def _train_step(self) -> float:
        """One gradient step on a buffer sample.

        Returns:
            Scalar loss value.
        """
        cfg = self.cfg
        local_np, global_np, actions_np = self.buffer.sample(
            cfg.dagger_batch_size,
        )
        local_t = torch.from_numpy(local_np).long().to(self.device)
        global_t = torch.from_numpy(global_np).long().to(self.device)
        actions_t = torch.from_numpy(actions_np).long().to(self.device)

        B = actions_t.shape[0]
        t = torch.rand(B, device=self.device).clamp(1e-5, 1.0 - 1e-5)

        zt = q_sample(
            actions_t, t, cfg.mask_token, cfg.pad_token,
            self._schedule_fn,
        )
        t_discrete = (t * (cfg.num_diffusion_steps - 1)).long()

        out = self.model(local_t, global_t, zt, t_discrete)

        loss_diff = mdlm_loss(
            out["actions"], actions_t, zt, t,
            cfg.mask_token, cfg.pad_token, self._schedule_fn,
            weight_clip=cfg.loss_weight_clip,
            label_smoothing=cfg.label_smoothing,
        )

        loss_aux = torch.tensor(0.0, device=self.device)
        if "goal_pred" in out:
            loss_aux = auxiliary_goal_loss(out["goal_pred"], global_t)

        loss = loss_diff + cfg.aux_loss_weight * loss_aux

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(), cfg.dagger_grad_clip,
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    # ── Checkpointing ────────────────────────────────────────────

    def save_checkpoint(self, iteration: int) -> None:
        """Save a training checkpoint.

        Args:
            iteration: Current iteration number.
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"iter{iteration}.pth"

        state = {
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "curriculum_state": self.collector.curriculum.state_dict(),
            "iteration": iteration,
            "rng_states": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

        try:
            torch.save(state, path)
            logger.info(f"Checkpoint saved: {path}")
        except Exception:
            logger.error(
                f"Failed to save checkpoint to {path}", exc_info=True,
            )

        # Run eval at checkpoint and save JSON
        try:
            eval_model = self.ema_model.make_eval_model(self.model)
            id_results = self.evaluator.evaluate(
                self.cfg.id_envs, eval_model,
                self.cfg.checkpoint_eval_episodes,
                self.cfg, self.device,
            )
            ood_results = self.evaluator.evaluate(
                self.cfg.ood_envs, eval_model,
                self.cfg.checkpoint_eval_episodes,
                self.cfg, self.device,
            )

            id_winrate = float(np.mean(
                [s["win_rate"] for s in id_results.values()]
            )) if id_results else 0.0
            ood_winrate = float(np.mean(
                [s["win_rate"] for s in ood_results.values()]
            )) if ood_results else 0.0
            current_lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler is not None
                else self.cfg.dagger_lr
            )
            training_meta = {
                "iteration": iteration,
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
                id_results, step=iteration, prefix="ckpt_eval_id",
            )
            self.log.log_eval(
                ood_results, step=iteration, prefix="ckpt_eval_ood",
            )
            self.log.log(
                {
                    "ckpt_eval/id_winrate": id_winrate,
                    "ckpt_eval/ood_winrate": ood_winrate,
                },
                step=iteration,
            )
            self.log.log_summary({
                f"ckpt_{iteration}/id_winrate": id_winrate,
                f"ckpt_{iteration}/ood_winrate": ood_winrate,
            })
        except Exception:
            logger.error("Checkpoint eval failed", exc_info=True)

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

    def load_checkpoint(self, path: str) -> int:
        """Load a training checkpoint.

        Args:
            path: Path to ``.pth`` checkpoint file.

        Returns:
            Iteration to resume from.
        """
        ckpt = torch.load(
            path, map_location=self.device, weights_only=False,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.ema_model.load_state_dict(ckpt["ema_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if (
            self.scheduler is not None
            and ckpt.get("scheduler_state_dict") is not None
        ):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if "curriculum_state" in ckpt:
            self.collector.curriculum.load_state_dict(
                ckpt["curriculum_state"],
            )

        # Restore RNG states (best-effort)
        rng = ckpt.get("rng_states", {})
        try:
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "python" in rng:
                random.setstate(rng["python"])
        except Exception:
            logger.warning(
                "RNG state restore failed; continuing with fresh state",
            )

        iteration = ckpt.get("iteration", 0)
        logger.info(f"Resumed from checkpoint: {path} (iter {iteration})")
        return iteration


def run_dagger(
    cfg, checkpoint_path: str | None, no_warm_start: bool,
) -> None:
    """DAgger online training loop."""

    device = cfg.device
    logger.info(f"DAgger training on {device}")

    model = make_model(cfg).to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.dagger_lr)

    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seq_len, cfg.pad_token)
    curriculum = DynamicCurriculum(cfg.id_envs, cfg.curriculum_queue_size)

    # Seed buffer with some oracle data
    for i, env_id in enumerate(cfg.id_envs):
        for s in range(3):
            traj = collect_oracle_trajectory(env_id, seed=i * 100 + s, cfg=cfg)
            if traj is not None:
                buffer.add(traj)
    logger.info(f"Buffer seeded with {len(buffer)} windows")

    eval_model = ema.make_eval_model(model)
    collector = DataCollector(eval_model, buffer, curriculum, cfg, device)
    evaluator = Evaluator()
    log = Logger(cfg)

    trainer = Trainer(
        model, ema, optimizer, None, buffer, collector,
        evaluator, log, cfg, device,
    )

    start_iter = 0
    if checkpoint_path and not no_warm_start:
        start_iter = trainer.load_checkpoint(checkpoint_path)

    trainer.train(start_iter=start_iter)
    log.finish()
