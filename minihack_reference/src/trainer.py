"""
Trainer: DAgger training loop, gradient steps, EMA, evaluation triggers.
"""

import os
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import date, datetime

from config import (
    ID_ENVS,
    OOD_ENVS,
    BUFFER_CAPACITY,
    EPISODES_PER_ITERATION,
    GRAD_STEPS_PER_ITERATION,
    ID_EVAL_EVERY_ITERATIONS,
    OOD_EVAL_EVERY_ITERATIONS,
    CHECKPOINT_EVERY_ITERATIONS,
    CHECKPOINT_EVAL_EPISODES,
    SMOKE_TEST_MAX_ITERATIONS,
    MAIN_RUN_MAX_ITERATIONS,
    HYPERPARAMS,
    ACTION_DIM,
    MASK_TOKEN,
    PAD_TOKEN,
)
from collections import deque

from buffer import ReplayBuffer
from curriculum import DynamicCurriculum
from collector import DataCollector
from evaluator import Evaluator
from utils import find_staircase_from_glyphs
from utils import evaluate_envs
from utils import HF_GENERIZATION_REPO


class Trainer:
    """DAgger training: buffer, curriculum, collector, gradient steps, EMA, eval triggers."""

    def __init__(
        self,
        model,
        data_path,
        metadata=None,
        buffer_capacity=None,
        lr=3e-5,
        aux_loss_weight=0.5,
        save_dir=None,
        hub_run_id=None,
        hub_token=None,
        resume_from=None,
    ):
        self.model = model
        self.device = HYPERPARAMS["device"]
        self.buffer_capacity = buffer_capacity or BUFFER_CAPACITY
        self.aux_loss_weight = aux_loss_weight
        self.lr = lr
        self.save_dir = save_dir or "."
        self.hub_run_id = hub_run_id
        self.hub_token = hub_token
        self.resume_from = resume_from

        self.buffer = ReplayBuffer(capacity=self.buffer_capacity)
        n_loaded = self.buffer.load_offline_data(data_path, ID_ENVS, metadata)
        print(f"Buffer loaded: {n_loaded} offline samples (filtered to ID_ENVS)")

        self.curriculum = DynamicCurriculum(env_ids=ID_ENVS)
        self.ema_model = copy.deepcopy(model)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Restore from checkpoint if resuming
        self.start_iteration = 1
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.ema_model.load_state_dict(ckpt.get("ema_state_dict", ckpt["model_state_dict"]), strict=False)
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.scheduler and ckpt.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if self.scaler and ckpt.get("scaler_state_dict") is not None:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            if "rng_states" in ckpt:
                rs = ckpt["rng_states"]
                if "torch" in rs:
                    torch.set_rng_state(rs["torch"].to("cpu"))
                if "numpy" in rs:
                    np.random.set_state(rs["numpy"])
                if "python" in rs:
                    random.setstate(rs["python"])
            if "curriculum_state" in ckpt:
                cs = ckpt["curriculum_state"]
                self.curriculum.env_ids = cs.get("env_ids", self.curriculum.env_ids)
                self.curriculum.queue_size = cs.get("queue_size", self.curriculum.queue_size)
                if "_queues" in cs:
                    self.curriculum._queues = {
                        e: deque(lst, maxlen=self.curriculum.queue_size)
                        for e, lst in cs["_queues"].items()
                    }
            self.start_iteration = ckpt.get("iteration", 0) + 1
            print(f"Resumed from {resume_from} (next iteration: {self.start_iteration})")

        self.collector = DataCollector(
            self.ema_model,
            device=self.device,
            diffusion_steps=5,
            replan_every=16,
            max_steps=200,
        )
        self.evaluator = Evaluator(
            self.ema_model,
            device=self.device,
            diffusion_steps=5,
            max_steps=200,
        )

        self.ema_decay = HYPERPARAMS.get("ema_decay", 0.999)
        self.batch_size = HYPERPARAMS["batch_size"]
        self.loss_history = []  # List of {"total", "diffusion", "aux"} for checkpoint JSON
        self.scheduler = None  # Optional LR scheduler (CosineAnnealingLR, etc.)
        self.scaler = None  # Optional GradScaler if AMP used
        self.training_start_time = None  # Set at run() start for timing
        os.makedirs(self.save_dir, exist_ok=True)

    def _train_step(self):
        """One gradient step on a random batch from buffer. Returns (loss_total, loss_diffusion, loss_aux)."""
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return 0.0, 0.0, 0.0
        local, global_obs, actions = batch
        b_local = torch.tensor(local, dtype=torch.long).to(self.device)
        b_global = torch.tensor(global_obs, dtype=torch.long).to(self.device)
        b_actions = torch.tensor(actions, dtype=torch.long).to(self.device)

        mask_ratios = torch.rand(len(batch[0]), device=self.device)
        noisy_input = b_actions.clone()
        mask_mask = torch.rand(b_actions.shape, device=self.device) < mask_ratios.unsqueeze(1)
        noisy_input[mask_mask] = MASK_TOKEN
        t_discrete = (mask_ratios * 100).long()

        self.model.train()
        model_out = self.model(b_local, b_global, noisy_input, t_discrete)
        pred_logits = model_out["actions"] if isinstance(model_out, dict) else model_out[0]
        goal_pred = model_out.get("goal_pred") if isinstance(model_out, dict) else None

        loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=PAD_TOKEN)
        loss_diffusion = (
            loss_fn(pred_logits.reshape(-1, ACTION_DIM), b_actions.reshape(-1))
            .view(len(b_local), -1) * mask_mask.float()
        ).sum() / (mask_mask.sum() + 1e-6)

        loss_aux = torch.tensor(0.0, device=self.device)
        if goal_pred is not None and self.aux_loss_weight > 0:
            b_stair = torch.tensor(
                np.array([find_staircase_from_glyphs(x) for x in global_obs]),
                dtype=torch.float32,
            ).to(self.device)
            stair_visible = b_stair[:, 0] >= 0
            if stair_visible.any():
                loss_aux = nn.functional.mse_loss(
                    goal_pred[stair_visible], b_stair[stair_visible]
                )
        loss = loss_diffusion + self.aux_loss_weight * loss_aux

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
                p_ema.mul_(self.ema_decay).add_(p, alpha=1 - self.ema_decay)

        return loss.item(), loss_diffusion.item(), loss_aux.item()

    def run_iteration(self):
        """One iteration: 10 episodes (collect + filter) then 50 grad steps."""
        self.ema_model.eval()
        iter_added = 0

        for ep in range(EPISODES_PER_ITERATION):
            env_id = self.curriculum.sample_next_env()
            seed = np.random.randint(0, 2**31)

            trajectory, should_add, model_won = self.collector.collect_episode(env_id, seed)
            self.curriculum.update(env_id, model_won)

            if trajectory is None:
                continue

            if should_add:
                self.buffer.add(trajectory)
                iter_added += 1

        loss_records = []
        for _ in range(GRAD_STEPS_PER_ITERATION):
            total, diff, aux = self._train_step()
            loss_records.append({"total": total, "diffusion": diff, "aux": aux})

        avg_total = np.mean([r["total"] for r in loss_records]) if loss_records else 0.0
        return len(loss_records), avg_total, iter_added, loss_records

    def run(self, mode="main"):
        """Run training. mode: 'smoke' or 'main'. Resumes from start_iteration if set by resume_from."""
        if mode == "smoke":
            max_iter = SMOKE_TEST_MAX_ITERATIONS
        else:
            max_iter = MAIN_RUN_MAX_ITERATIONS

        start = self.start_iteration
        self.training_start_time = datetime.now()
        print(f"Training: mode={mode}, max_iter={max_iter}, start_iter={start}")
        for iteration in range(start, max_iter + 1):
            n_steps, avg_loss, n_added, loss_records = self.run_iteration()
            iter_agg = {
                "total": np.mean([r["total"] for r in loss_records]) if loss_records else 0.0,
                "diffusion": np.mean([r["diffusion"] for r in loss_records]) if loss_records else 0.0,
                "aux": np.mean([r["aux"] for r in loss_records]) if loss_records else 0.0,
            }
            self.loss_history.append(iter_agg)

            if iteration % 10 == 0:
                print(f"  iter {iteration}/{max_iter} | loss={avg_loss:.4f} | "
                      f"buffer={len(self.buffer)} | added={n_added}")

            if mode == "main":
                if iteration % ID_EVAL_EVERY_ITERATIONS == 0:
                    id_rate = self.evaluator.evaluate_id()
                    print(f"  Eval/ID_WinRate: {id_rate:.2%}")
                if iteration % OOD_EVAL_EVERY_ITERATIONS == 0:
                    ood_rate = self.evaluator.evaluate_ood()
                    print(f"  Eval/OOD_WinRate: {ood_rate:.2%}")

                # Every 1000 iters: save checkpoint + full eval (ID+OOD, 20 eps, ema_model)
                if iteration % CHECKPOINT_EVERY_ITERATIONS == 0:
                    curriculum_state = {
                        "env_ids": self.curriculum.env_ids,
                        "queue_size": self.curriculum.queue_size,
                        "_queues": {e: list(q) for e, q in self.curriculum._queues.items()},
                    }
                    rng_states = {
                        "torch": torch.get_rng_state(),
                        "numpy": np.random.get_state(),
                        "python": random.getstate(),
                    }
                    ckpt_dict = {
                        "model_state_dict": self.model.state_dict(),
                        "ema_state_dict": self.ema_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "curriculum_state": curriculum_state,
                        "iteration": iteration,
                        "rng_states": rng_states,
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                        "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                    }
                    ckpt_path = os.path.join(
                        self.save_dir, f"generalization_iter{iteration}.pth"
                    )
                    torch.save(ckpt_dict, ckpt_path)
                    print(f"  Checkpoint saved: {ckpt_path}")
                    # Upload to HF generalization repo if hub params set
                    if self.hub_run_id and self.hub_token:
                        try:
                            from huggingface_hub import HfApi
                            api = HfApi(token=self.hub_token)
                            repo_path = f"{self.hub_run_id}/generalization_iter{iteration}.pth"
                            api.upload_file(
                                path_or_fileobj=ckpt_path,
                                path_in_repo=repo_path,
                                repo_id=HF_GENERIZATION_REPO,
                                token=self.hub_token,
                            )
                            print(f"  Uploaded to HF: {repo_path}")
                        except Exception as e:
                            print(f"  HF upload failed: {e}")

                    self.ema_model.eval()
                    all_envs = ID_ENVS + OOD_ENVS
                    results = evaluate_envs(
                        self.ema_model,
                        env_ids=all_envs,
                        episodes=CHECKPOINT_EVAL_EPISODES,
                        diffusion_steps=5,
                    )

                    # Save checkpoint eval JSON (training params, loss breakdown, per-env results)
                    recent = self.loss_history[-CHECKPOINT_EVERY_ITERATIONS:]
                    eval_json = {
                        "iteration": iteration,
                        "date": str(date.today()),
                        "timestamp": datetime.now().isoformat(),
                        "training": {
                            "lr": self.lr,
                            "batch_size": self.batch_size,
                            "aux_loss_weight": self.aux_loss_weight,
                            "buffer_size": len(self.buffer),
                            "buffer_capacity": self.buffer_capacity,
                            "ema_decay": self.ema_decay,
                            "recent_loss_total": float(np.mean([r["total"] for r in recent])) if recent else 0.0,
                            "recent_loss_diffusion": float(np.mean([r["diffusion"] for r in recent])) if recent else 0.0,
                            "recent_loss_aux": float(np.mean([r["aux"] for r in recent])) if recent else 0.0,
                            "grad_steps_per_iteration": GRAD_STEPS_PER_ITERATION,
                            "episodes_per_iteration": EPISODES_PER_ITERATION,
                        },
                        "eval_episodes_per_env": CHECKPOINT_EVAL_EPISODES,
                        "id_winrate": float(np.mean([r["win_rate"] for r in results if r["env_id"] in ID_ENVS])),
                        "ood_winrate": float(np.mean([r["win_rate"] for r in results if r["env_id"] in OOD_ENVS])),
                        "per_env": {
                            r["env_id"]: {
                                "win_rate": r["win_rate"],
                                "avg_reward": r["avg_reward"],
                                "avg_steps": r["avg_steps"],
                                "wins": r["wins"],
                                "episodes": r["episodes"],
                            }
                            for r in results
                        },
                    }
                    eval_path = os.path.join(self.save_dir, f"checkpoint_eval_iter{iteration}.json")
                    with open(eval_path, "w") as f:
                        json.dump(eval_json, f, indent=2)
                    print(f"  Checkpoint eval saved: {eval_path}")

                    # Upload eval JSON to HF
                    if self.hub_run_id and self.hub_token:
                        try:
                            from huggingface_hub import HfApi
                            api = HfApi(token=self.hub_token)
                            repo_path = f"{self.hub_run_id}/checkpoint_eval_iter{iteration}.json"
                            api.upload_file(
                                path_or_fileobj=eval_path,
                                path_in_repo=repo_path,
                                repo_id=HF_GENERIZATION_REPO,
                                token=self.hub_token,
                            )
                            print(f"  Uploaded to HF: {repo_path}")
                        except Exception as e:
                            print(f"  HF eval JSON upload failed: {e}")

                    # Log to WandB if available
                    try:
                        import wandb
                        if wandb.run is not None:
                            log_dict = {
                                f"ckpt_{iteration}/ID_WinRate": eval_json["id_winrate"],
                                f"ckpt_{iteration}/OOD_WinRate": eval_json["ood_winrate"],
                            }
                            for r in results:
                                k = r["env_id"].replace("-", "_").replace(".", "_")
                                log_dict[f"ckpt_{iteration}/{k}/win_rate"] = r["win_rate"]
                                log_dict[f"ckpt_{iteration}/{k}/avg_reward"] = r["avg_reward"]
                                log_dict[f"ckpt_{iteration}/{k}/avg_steps"] = r["avg_steps"]
                            wandb.log(log_dict, step=iteration)
                    except ImportError:
                        pass

        return self.ema_model
