"""Data collection with DAgger and oracle replay.

Implements model episode rollout with replanning and DAgger-style
data collection using the BFS oracle and efficiency filter.
"""

from __future__ import annotations

import logging
import random
from types import SimpleNamespace

import numpy as np
import torch

from src.buffer import ReplayBuffer
from src.curriculum import DynamicCurriculum, efficiency_filter
from src.diffusion.sampling import remdm_sample
from src.envs.minihack_env import collect_oracle_trajectory, make_env

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_model_episode(
    model: torch.nn.Module,
    env_id: str,
    cfg: SimpleNamespace,
    device: torch.device | str,
    seed: int | None = None,
    max_steps: int = 500,
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

    Returns:
        Dict with ``"local"`` ``[T,9,9]``, ``"global"`` ``[T,21,79]``,
        ``"actions"`` ``[T]``, ``"won"`` bool, ``"steps"`` int,
        ``"seed"`` int.
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    env = make_env(env_id, None, cfg)
    (local, glb), _info = env.reset(seed=seed)

    locals_list = [local]
    globals_list = [glb]
    actions_list: list[int] = []
    won = False
    plan: torch.Tensor | None = None
    step_in_plan = 0

    model.eval()
    for step_idx in range(max_steps):
        # Replan when needed
        if plan is None or step_in_plan >= cfg.replan_every:
            local_t = torch.from_numpy(
                local[np.newaxis]
            ).long().to(device)  # [1, 9, 9]
            glb_t = torch.from_numpy(
                glb[np.newaxis]
            ).long().to(device)  # [1, 21, 79]
            plan = remdm_sample(
                model, local_t, glb_t, cfg, device,
            )  # [1, seq_len]
            step_in_plan = 0

        action = plan[0, step_in_plan].item()
        action = max(0, min(action, cfg.action_dim - 1))
        actions_list.append(action)
        step_in_plan += 1

        (local, glb), _reward, terminated, truncated, info = env.step(action)
        locals_list.append(local)
        globals_list.append(glb)

        if info.get("won", False):
            won = True
        if terminated or truncated:
            break

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
        "seed": seed,
    }


class DataCollector:
    """DAgger-style data collector.

    Each iteration: sample an environment from the curriculum, run the
    model, run the oracle on the same seed, apply efficiency filter, and
    optionally add the oracle trajectory to the buffer.

    Args:
        ema_model: EMA model for inference.
        buffer: Replay buffer to populate.
        curriculum: Dynamic environment curriculum.
        cfg: Config namespace.
        device: Torch device.
    """

    def __init__(
        self,
        ema_model: torch.nn.Module,
        buffer: ReplayBuffer,
        curriculum: DynamicCurriculum,
        cfg: SimpleNamespace,
        device: torch.device | str,
    ) -> None:
        self.ema_model = ema_model
        self.buffer = buffer
        self.curriculum = curriculum
        self.cfg = cfg
        self.device = device

    def collect_one_iteration(self) -> dict:
        """Run one DAgger collection iteration.

        Returns:
            Stats dict with ``"env_id"``, ``"model_won"``,
            ``"model_steps"``, ``"oracle_steps"``,
            ``"added_to_buffer"`` keys.
        """
        env_id = self.curriculum.sample_env()
        seed = random.randint(0, 2**31 - 1)

        # Model rollout
        model_result = run_model_episode(
            self.ema_model, env_id, self.cfg, self.device, seed,
        )

        # Oracle rollout (same seed)
        oracle_result = collect_oracle_trajectory(
            env_id, seed, self.cfg,
        )
        oracle_steps = (
            len(oracle_result["actions"]) if oracle_result else 999
        )

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
