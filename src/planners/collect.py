"""Data collection with DAgger and oracle replay.

Implements model episode rollout with replanning and DAgger-style
data collection using the BFS oracle and efficiency filter.
Supports parallel episode collection via ``ThreadPoolExecutor``.
"""

from __future__ import annotations

import copy
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.buffer import ReplayBuffer
from src.curriculum import DynamicCurriculum, efficiency_filter
from src.diffusion.sampling import greedy_sample, remdm_sample
from src.envs.minihack_env import collect_oracle_trajectory, make_env

if TYPE_CHECKING:
    from src.models.denoiser import ModelEMA

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_model_episode(
    model: torch.nn.Module,
    env_id: str,
    cfg: SimpleNamespace,
    device: torch.device | str,
    seed: int | None = None,
    max_steps: int = 500,
    des_file: str | None = None,
    blind_global: bool = False,
    stochastic: bool = False,
) -> dict:
    """Roll out the diffusion model on a single episode.

    Maintains a ``seq_len``-length plan and replans every
    ``cfg.replan_every`` steps.

    Args:
        model: Denoising model (eval mode).
        env_id: MiniHack registry ID.
        cfg: Config namespace.
        device: Torch device.
        seed: Optional RNG seed.
        max_steps: Maximum episode length.
        des_file: Optional ``.des`` file content for custom scenarios.
        blind_global: If ``True``, zero out global map (local-only ablation).
        stochastic: If ``True``, use stochastic ReMDM sampling (evaluation).
            If ``False`` (default), use greedy argmax (DAgger collection).

    Returns:
        Dict with ``"local"`` ``[T,9,9]``, ``"global"`` ``[T,21,79]``,
        ``"actions"`` ``[T]``, ``"won"`` bool, ``"steps"`` int,
        ``"total_reward"`` float, ``"seed"`` int.
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    env = make_env(env_id, des_file, cfg)
    try:
        (local, glb), _info = env.reset(seed=seed)

        locals_list = [local]
        globals_list = [glb]
        actions_list: list[int] = []
        won = False
        total_reward = 0.0
        plan: torch.Tensor | None = None
        step_in_plan = 0

        model.eval()
        for step_idx in range(max_steps):
            if plan is None or step_in_plan >= cfg.replan_every:
                local_t = (
                    torch.from_numpy(local[np.newaxis]).long().to(device)
                )  # [1, 9, 9]
                glb_t = (
                    torch.from_numpy(glb[np.newaxis]).long().to(device)
                )  # [1, 21, 79]
                if stochastic:
                    plan = remdm_sample(
                        model,
                        local_t,
                        glb_t,
                        cfg,
                        device,
                        physics_aware=getattr(
                            cfg,
                            "physics_aware_sampling",
                            False,
                        ),
                        blind_global=blind_global,
                    )
                else:
                    plan = greedy_sample(
                        model,
                        local_t,
                        glb_t,
                        cfg,
                        device,
                        blind_global=blind_global,
                        num_steps=getattr(
                            cfg,
                            "diffusion_steps_collect",
                            cfg.diffusion_steps_eval,
                        ),
                    )  # [1, seq_len]
                step_in_plan = 0

            action = plan[0, step_in_plan].item()
            action = max(0, min(action, cfg.action_dim - 1))
            actions_list.append(action)
            step_in_plan += 1

            (local, glb), reward, terminated, truncated, info = env.step(
                action,
            )
            total_reward += reward
            locals_list.append(local)
            globals_list.append(glb)

            if info.get("won", False):
                won = True
            if terminated or truncated:
                break
    finally:
        env.close()

    # Trim trailing obs
    locals_arr = np.stack(locals_list[:-1], axis=0).astype(np.int16)
    globals_arr = np.stack(globals_list[:-1], axis=0).astype(np.int16)
    actions_arr = np.array(actions_list, dtype=np.int64)

    return {
        "local": locals_arr,
        "global": globals_arr,
        "actions": actions_arr,
        "won": won,
        "steps": len(actions_list),
        "total_reward": total_reward,
        "seed": seed,
    }


def _collect_episode_thread(
    model: torch.nn.Module,
    env_id: str,
    seed: int,
    cfg: SimpleNamespace,
) -> dict | None:
    """Thread worker: run one paired (model + oracle) episode.

    Both NLE (C code) and PyTorch CPU inference release the GIL,
    so true parallelism is achieved with threads. Each call uses
    its own model copy and env instance.

    Args:
        model: CPU-resident eval-mode model (thread's own copy).
        env_id: MiniHack environment ID.
        seed: RNG seed for the episode.
        cfg: Config namespace.

    Returns:
        Stats dict or ``None`` on failure.
    """
    try:
        model_result = run_model_episode(
            model,
            env_id,
            cfg,
            "cpu",
            seed,
        )
        oracle_result = collect_oracle_trajectory(env_id, seed, cfg)
        oracle_steps = len(oracle_result["actions"]) if oracle_result else 999
        return {
            "env_id": env_id,
            "seed": seed,
            "model_won": model_result["won"],
            "model_steps": model_result["steps"],
            "oracle_steps": oracle_steps,
            "oracle_result": oracle_result,
        }
    except Exception:
        logger.error(
            f"Thread worker failed for {env_id} seed={seed}",
            exc_info=True,
        )
        return None


class DataCollector:
    """DAgger-style data collector.

    Each iteration: sample an environment from the curriculum, run the
    model, run the oracle on the same seed, apply efficiency filter, and
    optionally add the oracle trajectory to the buffer.

    Supports parallel episode collection via ``cfg.num_collection_workers``.

    Uses a live reference to the ``ModelEMA`` object so the collector
    always uses the latest EMA weights (synced before each rollout).

    Args:
        ema: EMA tracker holding shadow weights.
        model: Training model (architecture template for EMA snapshot).
        buffer: Replay buffer to populate.
        curriculum: Dynamic environment curriculum.
        cfg: Config namespace.
        device: Torch device.
    """

    def __init__(
        self,
        ema: "ModelEMA",
        model: torch.nn.Module,
        buffer: ReplayBuffer,
        curriculum: DynamicCurriculum,
        cfg: SimpleNamespace,
        device: torch.device | str,
    ) -> None:
        self._ema = ema
        # Materialise an eval-mode copy; refreshed before each rollout
        self.ema_model = ema.make_eval_model(model)
        self.buffer = buffer
        self.curriculum = curriculum
        self.cfg = cfg
        self.device = device
        self._num_workers = getattr(cfg, "num_collection_workers", 0)  # fallback 0 != defaults.yaml (8); partial-config callers rely on it
        self._last_profile: dict[str, float] = {}
        self._thread_pool: ThreadPoolExecutor | None = None
        self._thread_models: list[torch.nn.Module] = []
        if self._num_workers > 0:
            n = min(self._num_workers, os.cpu_count() or 4)
            self._thread_pool = ThreadPoolExecutor(max_workers=n)
            # Create one CPU model copy per thread
            for _ in range(n):
                m = copy.deepcopy(model).cpu()
                m.eval()
                self._thread_models.append(m)

    def _sync_ema(self) -> None:
        """Copy latest EMA shadow weights into the eval model."""
        self._ema.apply_to(self.ema_model)
        self.ema_model.eval()

    def collect_one_iteration(self) -> dict:
        """Run one DAgger collection iteration (single episode).

        Returns:
            Stats dict with ``"env_id"``, ``"model_won"``,
            ``"model_steps"``, ``"oracle_steps"``,
            ``"added_to_buffer"`` keys.
        """
        self._sync_ema()
        env_id = self.curriculum.sample_env()
        seed = random.randint(0, 2**31 - 1)

        # Model rollout
        model_result = run_model_episode(
            self.ema_model,
            env_id,
            self.cfg,
            self.device,
            seed,
        )

        # Oracle rollout (same seed)
        oracle_result = collect_oracle_trajectory(
            env_id,
            seed,
            self.cfg,
        )
        oracle_steps = len(oracle_result["actions"]) if oracle_result else 999

        # Efficiency filter
        add = efficiency_filter(
            model_result["won"],
            model_result["steps"],
            oracle_steps,
            self.cfg.efficiency_multiplier,
        )

        if add and oracle_result is not None:
            self.buffer.add(oracle_result)

        self.curriculum.update(env_id, model_result["won"])

        return {
            "env_id": env_id,
            "model_won": model_result["won"],
            "model_steps": model_result["steps"],
            "oracle_steps": oracle_steps,
            "added_to_buffer": add and oracle_result is not None,
        }

    def collect_batch_parallel(
        self,
        n_episodes: int,
    ) -> list[dict]:
        """Collect multiple episodes in parallel using threads.

        Both NLE env calls and PyTorch CPU inference release the GIL,
        enabling true parallelism. Each thread uses a pre-allocated
        CPU model copy. Weights are synced from EMA once per call.

        Args:
            n_episodes: Number of episodes to collect.

        Returns:
            List of per-episode stats dicts.
        """
        assert self._thread_pool is not None, (
            "collect_batch_parallel requires num_collection_workers > 0"
        )
        self._sync_ema()

        # Sync EMA weights to all thread-local CPU models
        ema_sd = self.ema_model.state_dict()
        cpu_sd = {k: v.cpu() for k, v in ema_sd.items()}
        for tm in self._thread_models:
            tm.load_state_dict(cpu_sd)
            tm.eval()

        # Build task list
        tasks = []
        for _ in range(n_episodes):
            env_id = self.curriculum.sample_env()
            seed = random.randint(0, 2**31 - 1)
            tasks.append((env_id, seed))

        # Round-robin assign models to tasks
        n_models = len(self._thread_models)
        futures = []
        for i, (env_id, seed) in enumerate(tasks):
            model = self._thread_models[i % n_models]
            f = self._thread_pool.submit(
                _collect_episode_thread,
                model,
                env_id,
                seed,
                self.cfg,
            )
            futures.append(f)

        results = [f.result() for f in futures]

        # Process results: efficiency filter + buffer add
        stats_list = []
        for res in results:
            if res is None:
                continue

            add = efficiency_filter(
                res["model_won"],
                res["model_steps"],
                res["oracle_steps"],
                self.cfg.efficiency_multiplier,
            )

            oracle_result = res["oracle_result"]
            if add and oracle_result is not None:
                self.buffer.add(oracle_result)

            self.curriculum.update(res["env_id"], res["model_won"])

            stats_list.append(
                {
                    "env_id": res["env_id"],
                    "model_won": res["model_won"],
                    "model_steps": res["model_steps"],
                    "oracle_steps": res["oracle_steps"],
                    "added_to_buffer": add and oracle_result is not None,
                }
            )

        return stats_list

    def collect_batch_gpu(self, n_episodes: int) -> list[dict]:
        """Collect episodes with GPU-batched model inference.

        Runs all model episodes with batched GPU forward passes
        (B=n_episodes instead of B=1), then runs oracle rollouts
        in parallel threads for efficiency filtering.

        Args:
            n_episodes: Number of episodes to collect.

        Returns:
            List of per-episode stats dicts.
        """
        self._sync_ema()
        cfg = self.cfg
        self._last_profile = {}

        tasks = [
            (self.curriculum.sample_env(), random.randint(0, 2**31 - 1))
            for _ in range(n_episodes)
        ]

        # Phase 1: GPU-batched model rollouts
        t0 = time.perf_counter()
        model_results = self._run_model_episodes_batched(tasks)
        model_time = time.perf_counter() - t0

        # Phase 2: Oracle rollouts (threaded, CPU-only BFS)
        t0 = time.perf_counter()
        n_workers = min(n_episodes, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            oracle_futures = [
                pool.submit(
                    collect_oracle_trajectory,
                    env_id,
                    seed,
                    cfg,
                )
                for env_id, seed in tasks
            ]
            oracle_results = [f.result() for f in oracle_futures]
        oracle_time = time.perf_counter() - t0

        # Phase 3: Efficiency filter + buffer add
        stats_list: list[dict] = []
        for (env_id, _seed), m_res, o_res in zip(
            tasks,
            model_results,
            oracle_results,
        ):
            oracle_steps = len(o_res["actions"]) if o_res else 999
            add = efficiency_filter(
                m_res["won"],
                m_res["steps"],
                oracle_steps,
                cfg.efficiency_multiplier,
            )
            if add and o_res is not None:
                self.buffer.add(o_res)
            self.curriculum.update(env_id, m_res["won"])
            stats_list.append(
                {
                    "env_id": env_id,
                    "model_won": m_res["won"],
                    "model_steps": m_res["steps"],
                    "oracle_steps": oracle_steps,
                    "added_to_buffer": add and o_res is not None,
                }
            )

        self._last_profile["model_rollout_sec"] = model_time
        self._last_profile["oracle_rollout_sec"] = oracle_time
        return stats_list

    @torch.no_grad()
    def _run_model_episodes_batched(
        self,
        tasks: list[tuple[str, int]],
    ) -> list[dict]:
        """Run model episodes with batched GPU forward passes.

        Creates one env per episode, steps them in lockstep, and
        batches all replanning into single GPU forward passes
        (B = number of active envs needing a replan).

        Args:
            tasks: List of ``(env_id, seed)`` pairs.

        Returns:
            List of trajectory dicts matching
            ``run_model_episode`` output format.
        """
        cfg = self.cfg
        device = self.device
        model = self.ema_model
        model.eval()
        n = len(tasks)
        max_steps = 500
        K = getattr(
            cfg,
            "diffusion_steps_collect",
            cfg.diffusion_steps_eval,
        )
        cs = cfg.crop_size

        # Create and reset all envs
        envs: list = []
        cur_local = np.zeros((n, cs, cs), dtype=np.int16)
        cur_global = np.zeros(
            (n, cfg.map_h, cfg.map_w),
            dtype=np.int16,
        )

        t_reset = time.perf_counter()
        for i, (env_id, seed) in enumerate(tasks):
            env = make_env(env_id, None, cfg)
            (local, glb), _ = env.reset(seed=seed)
            envs.append(env)
            cur_local[i] = local
            cur_global[i] = glb
        reset_time = time.perf_counter() - t_reset

        # Pre-allocate history buffers
        obs_local = np.zeros(
            (n, max_steps + 1, cs, cs),
            dtype=np.int16,
        )
        obs_global = np.zeros(
            (n, max_steps + 1, cfg.map_h, cfg.map_w),
            dtype=np.int16,
        )
        act_buf = np.zeros((n, max_steps), dtype=np.int64)
        obs_local[:, 0] = cur_local
        obs_global[:, 0] = cur_global

        # Per-episode state vectors
        plans = np.zeros((n, cfg.seq_len), dtype=np.int64)
        step_in_plan = np.zeros(n, dtype=np.int32)
        need_replan = np.ones(n, dtype=bool)
        done = np.zeros(n, dtype=bool)
        won = np.zeros(n, dtype=bool)
        total_reward = np.zeros(n, dtype=np.float64)
        n_steps = np.zeros(n, dtype=np.int32)

        inference_time = 0.0
        env_step_time = 0.0

        try:
            for _ in range(max_steps):
                # Batch replan on GPU
                replan_idx = np.where(
                    need_replan & ~done,
                )[0]
                if len(replan_idx) > 0:
                    t0 = time.perf_counter()
                    local_t = (
                        torch.from_numpy(
                            cur_local[replan_idx],
                        )
                        .long()
                        .to(device)
                    )
                    glb_t = (
                        torch.from_numpy(
                            cur_global[replan_idx],
                        )
                        .long()
                        .to(device)
                    )
                    batch_plans = (
                        greedy_sample(
                            model,
                            local_t,
                            glb_t,
                            cfg,
                            device,
                            num_steps=K,
                        )
                        .cpu()
                        .numpy()
                    )
                    plans[replan_idx] = batch_plans
                    step_in_plan[replan_idx] = 0
                    need_replan[replan_idx] = False
                    inference_time += time.perf_counter() - t0

                # Step all active envs
                t0 = time.perf_counter()
                any_active = False
                for i in range(n):
                    if done[i]:
                        continue
                    any_active = True

                    action = int(plans[i, step_in_plan[i]])
                    action = max(
                        0,
                        min(action, cfg.action_dim - 1),
                    )
                    act_buf[i, n_steps[i]] = action
                    step_in_plan[i] += 1
                    n_steps[i] += 1

                    if step_in_plan[i] >= cfg.replan_every:
                        need_replan[i] = True

                    obs, reward, term, trunc, info = envs[i].step(action)
                    local, glb = obs
                    total_reward[i] += reward
                    cur_local[i] = local
                    cur_global[i] = glb
                    obs_local[i, n_steps[i]] = local
                    obs_global[i, n_steps[i]] = glb

                    if info.get("won", False):
                        won[i] = True
                    if term or trunc:
                        done[i] = True
                env_step_time += time.perf_counter() - t0

                if not any_active:
                    break
        finally:
            for env in envs:
                env.close()

        # Build result dicts
        results: list[dict] = []
        for i in range(n):
            T = int(n_steps[i])
            results.append(
                {
                    "local": obs_local[i, :T].copy(),
                    "global": obs_global[i, :T].copy(),
                    "actions": act_buf[i, :T].copy(),
                    "won": bool(won[i]),
                    "steps": T,
                    "total_reward": float(total_reward[i]),
                    "seed": tasks[i][1],
                }
            )

        self._last_profile.update(
            {
                "env_reset_sec": reset_time,
                "gpu_inference_sec": inference_time,
                "env_step_sec": env_step_time,
            }
        )
        return results
