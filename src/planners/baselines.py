"""SB3 + Decision Transformer baselines for the ReMDM diffusion planner.

This module wraps standard discrete-action RL baselines (PPO, A2C, DQN,
recurrent PPO) plus two imitation baselines (Behavioural Cloning and
Decision Transformer) into the project's unified config + dispatch
surface so they can be compared head-to-head against the DAgger /
offline-BC diffusion planner on the same MiniHack environments.

Entry point: :func:`run_baselines`.

Hyperparameters live in ``configs/defaults.yaml`` under the
``baselines_*`` namespace; the unified env-step training budget
(``cfg.total_timesteps``) is shared with DAgger and offline BC.

W&B logging routes through the project's :class:`Logger` (with the W&B
project temporarily swapped to ``cfg.baselines_wandb_project``); SB3's
standard ``WandbCallback`` piggybacks on the active run and syncs its
tensorboard scalars automatically. No file in this module calls
``wandb.log(...)`` directly.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import random
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import orjson
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.data import DataLoader, Dataset
from wandb.integration.sb3 import WandbCallback

from src.envs.minihack_env import (
    AdvancedObservationEnv,
    collect_oracle_trajectory,
)
from src.planners.logging import Logger

logger = logging.getLogger(__name__)


SB3_RL_ALGOS: tuple[str, ...] = ("ppo", "a2c", "dqn", "ppo-rnn")
IMITATION_ALGOS: tuple[str, ...] = ("bc", "dt")
ALL_BASELINE_ALGOS: tuple[str, ...] = SB3_RL_ALGOS + IMITATION_ALGOS


class _SB3MiniHackWrapper(gym.Wrapper):
    """Reshape ``AdvancedObservationEnv`` tuple obs into an SB3 dict obs.

    The underlying env returns ``(local_crop, global_map)`` with shapes
    ``(crop, crop)`` and ``(map_h, map_w)``; SB3's ``MultiInputPolicy``
    needs a ``Dict`` space with explicit channel dims. Also remaps
    ``info["won"]`` -> ``info["is_success"]`` so SB3's success tracking
    reports our win rate.
    """

    def __init__(self, env: AdvancedObservationEnv) -> None:
        super().__init__(env)
        local_h, local_w = env.observation_space.shape
        cfg = env._cfg  # AdvancedObservationEnv stores cfg here
        self.observation_space = gym.spaces.Dict(
            {
                "local": gym.spaces.Box(
                    low=0,
                    high=6000,
                    shape=(1, local_h, local_w),
                    dtype=np.int16,
                ),
                "global": gym.spaces.Box(
                    low=0,
                    high=6000,
                    shape=(1, cfg.map_h, cfg.map_w),
                    dtype=np.int16,
                ),
            }
        )

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict]:
        (local, glob), info = self.env.reset(**kwargs)
        return self._pack(local, glob), info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        (local, glob), reward, terminated, truncated, info = self.env.step(action)
        if "won" in info:
            info["is_success"] = info["won"]
        return self._pack(local, glob), reward, terminated, truncated, info

    @staticmethod
    def _pack(
        local: np.ndarray,
        glob: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "local": np.expand_dims(local, axis=0),  # [1, crop, crop]
            "global": np.expand_dims(glob, axis=0),  # [1, H, W]
        }


class _MiniHackCNN(BaseFeaturesExtractor):
    """Dual-stream CNN for the SB3 dict observation.

    Local stream: ``Conv(1->16, 3) -> Conv(16->32, 3)``.
    Global stream: ``Conv(1->16, 5, stride 2) -> Conv(16->32, 3, stride 2)``.
    Both streams are flattened and concatenated, then projected to
    ``features_dim`` via a single linear + ReLU.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.local_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.global_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_loc = torch.zeros(1, *observation_space["local"].shape)
            dummy_glob = torch.zeros(1, *observation_space["global"].shape)
            n_flatten = (
                self.local_cnn(dummy_loc).shape[1]
                + self.global_cnn(dummy_glob).shape[1]
            )
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(
        self,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loc = self.local_cnn(observations["local"].float())  # [B, F_l]
        glob = self.global_cnn(observations["global"].float())  # [B, F_g]
        return self.linear(torch.cat([loc, glob], dim=1))


class _MiniHackStateEncoder(nn.Module):
    """CNN encoder mapping a (local, global) obs pair to a state embedding."""

    def __init__(
        self,
        embed_dim: int = 128,
        crop_h: int = 9,
        crop_w: int = 9,
        map_h: int = 21,
        map_w: int = 79,
    ) -> None:
        super().__init__()
        self.local_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.global_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_loc = torch.zeros(1, 1, crop_h, crop_w)
            dummy_glob = torch.zeros(1, 1, map_h, map_w)
            local_flat = self.local_cnn(dummy_loc).shape[1]
            global_flat = self.global_cnn(dummy_glob).shape[1]
        self.proj = nn.Linear(local_flat + global_flat, embed_dim)

    def forward(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
    ) -> torch.Tensor:
        # Accepts (B, T, 1, H, W) or (B, 1, H, W).
        if local_obs.dim() == 5:
            B, T = local_obs.shape[:2]
            local_obs = local_obs.view(B * T, *local_obs.shape[2:])
            global_obs = global_obs.view(B * T, *global_obs.shape[2:])
            reshape = True
        else:
            B, T = local_obs.shape[0], 1
            reshape = False

        loc_feat = self.local_cnn(local_obs.float())  # [B*T, F_l]
        glob_feat = self.global_cnn(global_obs.float())  # [B*T, F_g]
        out = self.proj(torch.cat([loc_feat, glob_feat], dim=-1))  # [B*T, D]
        if reshape:
            out = out.view(B, T, -1)
        return out


class _DecisionTransformer(nn.Module):
    """Causal Decision Transformer over interleaved (R, s, a) tokens."""

    def __init__(
        self,
        n_actions: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        context_len: int = 30,
        max_ep_len: int = 500,
        dropout: float = 0.1,
        crop_h: int = 9,
        crop_w: int = 9,
        map_h: int = 21,
        map_w: int = 79,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.n_actions = n_actions
        self.max_ep_len = max_ep_len

        self.state_encoder = _MiniHackStateEncoder(
            embed_dim,
            crop_h,
            crop_w,
            map_h,
            map_w,
        )
        self.action_embed = nn.Embedding(n_actions + 1, embed_dim)  # +1 for pad
        self.return_embed = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Embedding(max_ep_len, embed_dim)
        self.token_type_embed = nn.Embedding(3, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(embed_dim, n_actions)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        returns_to_go: torch.Tensor,  # [B, T, 1]
        local_obs: torch.Tensor,  # [B, T, 1, H_l, W_l]
        global_obs: torch.Tensor,  # [B, T, 1, H_g, W_g]
        actions: torch.Tensor,  # [B, T]
        timesteps: torch.Tensor,  # [B, T]
        attention_mask: torch.Tensor | None = None,  # [B, T]
    ) -> torch.Tensor:
        B, T = returns_to_go.shape[:2]
        device = returns_to_go.device

        rtg_embed = self.return_embed(returns_to_go)  # [B, T, D]
        state_embed = self.state_encoder(local_obs, global_obs)  # [B, T, D]
        action_embed = self.action_embed(actions)  # [B, T, D]

        pos_embed = self.pos_embed(timesteps)  # [B, T, D]
        rtg_embed = rtg_embed + pos_embed + self.token_type_embed.weight[0]
        state_embed = state_embed + pos_embed + self.token_type_embed.weight[1]
        action_embed = action_embed + pos_embed + self.token_type_embed.weight[2]

        # Interleave (R_0, s_0, a_0, R_1, s_1, a_1, ...) -> [B, 3T, D]
        stacked = torch.stack([rtg_embed, state_embed, action_embed], dim=2)
        stacked = stacked.view(B, 3 * T, self.embed_dim)
        stacked = self.dropout(self.embed_ln(stacked))

        seq_len = 3 * T
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1,
        ).bool()

        key_padding_mask = None
        if attention_mask is not None:
            expanded = attention_mask.unsqueeze(-1).repeat(1, 1, 3).view(B, 3 * T)
            key_padding_mask = expanded == 0

        hidden = self.transformer(
            stacked,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        # State token positions are 1, 4, 7, ... -> stride 3.
        state_hidden = hidden[:, 1::3, :]  # [B, T, D]
        return self.action_head(state_hidden)  # [B, T, A]

    @torch.no_grad()
    def get_action(
        self,
        returns_to_go: torch.Tensor,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        logits = self.forward(
            returns_to_go,
            local_obs,
            global_obs,
            actions,
            timesteps,
        )
        return logits[:, -1, :].argmax(dim=-1)


class _DTDataset(Dataset):
    """Sliding-window dataset over Decision Transformer trajectories."""

    def __init__(
        self,
        trajectories: list[dict[str, np.ndarray]],
        context_len: int,
        max_ep_len: int,
        n_actions: int,
    ) -> None:
        self.trajectories = trajectories
        self.context_len = context_len
        self.max_ep_len = max_ep_len
        self.n_actions = n_actions
        self.indices: list[tuple[int, int]] = [
            (traj_idx, start)
            for traj_idx, traj in enumerate(trajectories)
            for start in range(len(traj["actions"]))
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_idx, start = self.indices[idx]
        traj = self.trajectories[traj_idx]
        traj_len = len(traj["actions"])
        end = min(start + self.context_len, traj_len)
        actual_len = end - start

        local = traj["local"][start:end].copy()
        glob = traj["global"][start:end].copy()
        actions = traj["actions"][start:end].copy()
        rtg = traj["returns_to_go"][start:end].copy()
        timesteps = np.arange(start, end)

        # Clamp to valid embedding ranges.
        timesteps = np.clip(timesteps, 0, self.max_ep_len - 1)
        actions = np.clip(actions, 0, self.n_actions - 1)

        pad_len = self.context_len - actual_len
        if pad_len > 0:
            local = np.pad(
                local,
                ((0, pad_len), (0, 0), (0, 0), (0, 0)),
                mode="constant",
            )
            glob = np.pad(
                glob,
                ((0, pad_len), (0, 0), (0, 0), (0, 0)),
                mode="constant",
            )
            actions = np.pad(actions, (0, pad_len), mode="constant")
            rtg = np.pad(rtg, (0, pad_len), mode="constant")
            timesteps = np.pad(timesteps, (0, pad_len), mode="constant")

        attention_mask = np.zeros(self.context_len, dtype=np.float32)
        attention_mask[:actual_len] = 1.0

        return {
            "local": torch.tensor(local, dtype=torch.float32),
            "global": torch.tensor(glob, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "returns_to_go": torch.tensor(rtg, dtype=torch.float32).unsqueeze(-1),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }


class _PrefixedEvalCallback(EvalCallback):
    """``EvalCallback`` that records mean_reward / avg_steps / win_rate
    under a unique per-environment prefix.

    SB3 truncates metric names at 36 chars, which collides on long
    MiniHack env IDs; the prefix lets us strip ``MiniHack-`` / ``-v0``
    cleanly.
    """

    def __init__(
        self,
        eval_env: SubprocVecEnv,
        prefix: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(eval_env, **kwargs)
        self.prefix = prefix

    def _on_step(self) -> bool:
        cont = super()._on_step()
        if self.evaluations_results:
            self.logger.record(
                f"{self.prefix}/mean_reward",
                float(np.mean(self.evaluations_results[-1])),
            )
            self.logger.record(
                f"{self.prefix}/avg_steps",
                float(np.mean(self.evaluations_length[-1])),
            )
        if self.evaluations_successes:
            self.logger.record(
                f"{self.prefix}/win_rate",
                float(np.mean(self.evaluations_successes[-1])),
            )
        return cont


def quiet_multiprocessing_tempdir_teardown() -> None:
    """Stop multiprocessing printing a traceback it cannot act on at exit.

    ``SubprocVecEnv`` makes multiprocessing create a ``pymp-*`` directory
    under ``TMPDIR`` and register a finalizer that ``shutil.rmtree``s it at
    interpreter exit. That finalizer tolerates ``FileNotFoundError`` and
    re-raises everything else, and on a shared filesystem an
    unlinked-but-still-open file survives as ``.nfsXXXX`` until its last
    handle closes, so the rmtree raises ``OSError: Directory not empty``.
    ``multiprocessing.util._run_finalizers`` catches that and prints the
    traceback, which is why ``--mode baselines`` ends with a traceback on
    stderr, exit code 0 and every artefact written.

    This widens the finalizer's tolerance to any removal error. Nothing is
    left behind that was not already: the directory is inside the process's
    own temp root, and the leftover ``.nfsXXXX`` entries are reaped by the
    filesystem once the last handle closes.

    Must run before the first ``SubprocVecEnv``: multiprocessing captures the
    callback when it creates the directory, so a later patch is ignored.
    Idempotent, and a no-op if a future Python drops the private hook.
    """
    import multiprocessing.util as mp_util

    original = getattr(mp_util, "_remove_temp_dir", None)
    if original is None or getattr(original, "_remdm_tolerant", False):
        return

    def tolerant_remove_temp_dir(rmtree, tempdir):
        # `**kwargs` swallows the `onerror`/`onexc` handler the caller
        # supplies, whose whole job is to re-raise; `ignore_errors` replaces
        # it and takes precedence in `shutil.rmtree` regardless.
        def quiet_rmtree(path, **kwargs):
            rmtree(path, ignore_errors=True)

        try:
            original(quiet_rmtree, tempdir)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            logger.debug("Left multiprocessing temp dir %s: %s", tempdir, exc)

    tolerant_remove_temp_dir._remdm_tolerant = True
    mp_util._remove_temp_dir = tolerant_remove_temp_dir


def _make_sb3_env_fn(env_id: str, cfg: SimpleNamespace, log_dir: str):
    """Return a picklable thunk that builds one wrapped+monitored env."""

    def _init() -> Monitor:
        os.makedirs(log_dir, exist_ok=True)
        env = AdvancedObservationEnv(env_id, des_file=None, cfg=cfg)
        env = _SB3MiniHackWrapper(env)
        return Monitor(env, log_dir)

    return _init


def _short(env_id: str) -> str:
    return env_id.replace("MiniHack-", "").replace("-v0", "")


def _eval_episodes_per_env(cfg: SimpleNamespace) -> int:
    override = getattr(cfg, "baselines_eval_episodes_per_env", None)
    if override is not None:
        return int(override)
    return int(cfg.eval_episodes_per_env)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _resolve_output_dir(cfg: SimpleNamespace, override: str | None) -> Path:
    out = Path(override) if override else Path(cfg.baselines_output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _init_baseline_logger(
    cfg: SimpleNamespace,
    run_name: str,
) -> Logger:
    """Init the project Logger with W&B project swapped to baselines.

    Mutates ``cfg.wandb_project`` / ``cfg.wandb_run_name`` /
    ``cfg.wandb_resume_id`` for the duration of the call so the existing
    Logger constructor picks them up. We deliberately do not restore the
    originals — each baseline seed reuses this helper, and main.py exits
    after ``run_baselines`` returns.
    """

    project_override = getattr(cfg, "baselines_wandb_project", None)
    if project_override:
        cfg.wandb_project = project_override
    cfg.wandb_run_name = run_name
    cfg.wandb_resume_id = None
    return Logger(cfg)


def _collect_bc_dataset(
    cfg: SimpleNamespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out the BFS oracle on each ID env and stack flat (s, a) pairs."""

    n_per_env = int(cfg.baselines_bc_oracle_episodes_per_env)
    locals_, globals_, actions_ = [], [], []
    for env_id in cfg.id_envs:
        for traj_seed in range(n_per_env):
            traj = collect_oracle_trajectory(env_id, traj_seed, cfg)
            if traj is None:
                continue
            # (T, H, W) -> (T, 1, H, W)
            locals_.append(np.expand_dims(traj["local"], axis=1))
            globals_.append(np.expand_dims(traj["global"], axis=1))
            actions_.append(traj["actions"])
    if not actions_:
        raise RuntimeError("BC oracle collection produced zero trajectories")
    return (
        np.concatenate(locals_, axis=0),
        np.concatenate(globals_, axis=0),
        np.concatenate(actions_, axis=0),
    )


class _BCDataset(Dataset):
    def __init__(
        self,
        loc: np.ndarray,
        glob: np.ndarray,
        acts: np.ndarray,
    ) -> None:
        self.loc = torch.tensor(loc, dtype=torch.float32)
        self.glob = torch.tensor(glob, dtype=torch.float32)
        self.acts = torch.tensor(acts, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.acts)

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        return {
            "obs": {"local": self.loc[idx], "global": self.glob[idx]},
            "acts": self.acts[idx],
        }


def evaluation_seeds(env_id: str, n_episodes: int) -> list[int]:
    """The per-episode evaluation seeds, matching the planner's exactly.

    The planner evaluates on `42 + crc32(f"{env_id}:{ep}") % 2**31`
    (`src/planners/inference.py`, pinned by
    `test_evaluator_seeds_are_fixed_and_run_seed_independent`). The baselines
    have to evaluate on the same episodes or the headline planner-vs-baseline
    comparison is between two different sets of levels.

    They did not. `_make_sb3_env_fn` built the env with no seed at all and
    `_eval_sb3_policy_manually` ran it inside a `SubprocVecEnv`, a child
    process that never inherited the parent's `_seed_everything`; `_eval_dt`
    had the same shape. The only seeding was the Python/NumPy/torch globals
    plus `seed=` into the SB3 constructors, which seeds action sampling and
    **not MiniHack level generation** — gymnasium's `reset(seed=...)` does not
    reach the NetHack core RNG, which is why `AdvancedObservationEnv.reset`
    seeds it explicitly. Measured before the fix: `_seed_everything(0)` twice
    gave first-observation hashes `62a012aa6f073246` and `2ee59c01b9bfd35d`.

    Python's `hash()` is salted per process, so crc32 is what keeps these
    stable across invocations. The formula is duplicated from the planner
    rather than imported, and
    `test_baseline_eval_seeds_match_the_planners` fails if the two ever drift.

    Args:
        env_id: The MiniHack environment id being evaluated.
        n_episodes: How many episodes the evaluation runs.

    Returns:
        One seed per episode, in episode order.
    """
    return [
        42 + zlib.crc32(f"{env_id}:{ep}".encode()) % (2**31)
        for ep in range(n_episodes)
    ]


def _eval_sb3_policy_manually(
    policy: ActorCriticPolicy,
    env_id: str,
    cfg: SimpleNamespace,
    log_dir: str,
    n_episodes: int,
) -> tuple[float, float]:
    """Run ``policy.predict`` on a Monitor-wrapped env and return
    (win_rate, avg_steps).

    Each episode is reset on its own seed from :func:`evaluation_seeds`, so the
    baseline is scored on the same levels the planner is. That is also why the
    env is built here rather than inside a ``SubprocVecEnv``: the vec env auto-
    resets between episodes with no seed to give it, and it ran in a child
    process that never inherited the parent's ``_seed_everything`` — one env in
    one subprocess bought nothing and cost the seeding.
    """
    os.makedirs(log_dir, exist_ok=True)
    eval_env = Monitor(
        _SB3MiniHackWrapper(AdvancedObservationEnv(env_id, des_file=None, cfg=cfg)),
        log_dir,
    )
    seeds = evaluation_seeds(env_id, n_episodes)
    try:
        wins = 0
        total_steps = 0
        for episode in range(n_episodes):
            obs, _info = eval_env.reset(seed=seeds[episode])
            terminated = truncated = False
            info: dict = {}
            while not (terminated or truncated):
                action, _ = policy.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = eval_env.step(int(action))
            if info.get("won", False):
                wins += 1
            total_steps += info["episode"]["l"]
    finally:
        eval_env.close()
    return wins / n_episodes, total_steps / n_episodes


def _train_bc(
    cfg: SimpleNamespace,
    train_env: SubprocVecEnv,
    log: Logger,
    log_dir: str,
    seed: int,
) -> tuple[ActorCriticPolicy, dict[str, float]]:
    """Train a Behavioural Cloning baseline. Returns (policy, seed_metrics)."""

    device = torch.device(cfg.device)
    n_eval = _eval_episodes_per_env(cfg)

    logger.info("Collecting oracle demonstrations for BC...")
    loc_arr, glob_arr, acts_arr = _collect_bc_dataset(cfg)
    logger.info("BC dataset: %d transitions", len(acts_arr))

    bc_loader = DataLoader(
        _BCDataset(loc_arr, glob_arr, acts_arr),
        batch_size=int(cfg.baselines_bc_batch_size),
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    lr = float(cfg.baselines_bc_lr)
    policy = ActorCriticPolicy(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        lr_schedule=lambda _progress: lr,
        features_extractor_class=_MiniHackCNN,
        features_extractor_kwargs={"features_dim": 256},
    ).to(device)

    n_epochs = int(cfg.baselines_bc_epochs)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=lr,
        weight_decay=float(cfg.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
    )
    policy.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in bc_loader:
            obs = {k: v.to(policy.device) for k, v in batch["obs"].items()}
            acts = batch["acts"].to(policy.device)
            _values, log_prob, _entropy = policy.evaluate_actions(obs, acts)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / max(1, len(bc_loader))
        current_lr = scheduler.get_last_lr()[0]
        log.log(
            {
                "train/bc_loss": avg_loss,
                "train/lr": current_lr,
                "train/epoch": epoch + 1,
            },
            step=epoch + 1,
        )
        logger.info(
            "BC epoch %02d/%02d | loss=%.4f | lr=%.2e",
            epoch + 1,
            n_epochs,
            avg_loss,
            current_lr,
        )

    seed_metrics: dict[str, float] = {}
    for split, env_list in (("ID", cfg.id_envs), ("OOD", cfg.ood_envs)):
        logger.info("--- BC %s evaluation (seed=%d) ---", split, seed)
        for env_id in env_list:
            short = _short(env_id)
            win_rate, avg_steps = _eval_sb3_policy_manually(
                policy,
                env_id,
                cfg,
                f"{log_dir}/eval_{split.lower()}/{env_id}",
                n_eval,
            )
            seed_metrics[f"{split}/{short}/win_rate"] = win_rate * 100
            seed_metrics[f"{split}/{short}/avg_steps"] = avg_steps
            logger.info(
                "%-30s | win_rate=%5.1f%% | avg_steps=%5.1f",
                short,
                win_rate * 100,
                avg_steps,
            )
    log.log(seed_metrics, step=n_epochs + 1)
    return policy, seed_metrics


def _collect_dt_trajectories(
    cfg: SimpleNamespace,
) -> list[dict[str, np.ndarray]]:
    """Collect oracle trajectories with sparse reward + return-to-go labels."""

    n_per_env = int(cfg.baselines_dt_oracle_episodes_per_env)
    trajectories: list[dict[str, np.ndarray]] = []
    for env_id in cfg.id_envs:
        for traj_seed in range(n_per_env):
            traj = collect_oracle_trajectory(env_id, traj_seed, cfg)
            if traj is None:
                continue
            T = len(traj["actions"])
            rewards = np.zeros(T, dtype=np.float32)
            rewards[-1] = 1.0  # sparse goal reward
            rtg = np.zeros(T, dtype=np.float32)
            rtg[-1] = rewards[-1]
            for t in range(T - 2, -1, -1):
                rtg[t] = rewards[t] + rtg[t + 1]
            trajectories.append(
                {
                    "local": np.expand_dims(traj["local"], axis=1),
                    "global": np.expand_dims(traj["global"], axis=1),
                    "actions": traj["actions"],
                    "rewards": rewards,
                    "returns_to_go": rtg,
                }
            )
    return trajectories


def _eval_dt(
    model: _DecisionTransformer,
    env_id: str,
    cfg: SimpleNamespace,
    target_return: float,
    n_episodes: int,
    max_ep_len: int,
    eval_max_steps: int,
    context_len: int,
) -> tuple[float, float]:
    """Roll out a trained Decision Transformer with target-return conditioning."""

    device = torch.device(cfg.device)
    env = AdvancedObservationEnv(env_id, des_file=None, cfg=cfg)
    env = _SB3MiniHackWrapper(env)
    # The same episodes the planner and the SB3 baselines are scored on.
    seeds = evaluation_seeds(env_id, n_episodes)
    model.eval()
    wins = 0
    total_steps = 0
    try:
        for _ep in range(n_episodes):
            obs, _ = env.reset(seed=seeds[_ep])
            done = False

            local_hist: list[np.ndarray] = []
            global_hist: list[np.ndarray] = []
            action_hist: list[int] = []
            rtg_hist: list[float] = []
            ts_hist: list[int] = []

            current_rtg = float(target_return)
            t = 0
            info: dict = {}
            while not done and t < eval_max_steps:
                local_hist.append(obs["local"])
                global_hist.append(obs["global"])
                rtg_hist.append(current_rtg)
                ts_hist.append(min(t, max_ep_len - 1))

                ctx = min(len(local_hist), context_len)
                local_in = np.stack(local_hist[-ctx:], axis=0)
                global_in = np.stack(global_hist[-ctx:], axis=0)
                rtg_in = np.array(rtg_hist[-ctx:], dtype=np.float32)
                ts_in = np.array(ts_hist[-ctx:], dtype=np.int64)
                if len(action_hist) < ctx:
                    act_in = np.zeros(ctx, dtype=np.int64)
                    if action_hist:
                        act_in[-len(action_hist) :] = action_hist[-ctx:]
                else:
                    act_in = np.array(action_hist[-ctx:], dtype=np.int64)

                local_t = (
                    torch.tensor(local_in, dtype=torch.float32).unsqueeze(0).to(device)
                )
                global_t = (
                    torch.tensor(global_in, dtype=torch.float32).unsqueeze(0).to(device)
                )
                rtg_t = (
                    torch.tensor(rtg_in, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .to(device)
                )
                act_t = torch.tensor(act_in, dtype=torch.long).unsqueeze(0).to(device)
                ts_t = torch.tensor(ts_in, dtype=torch.long).unsqueeze(0).to(device)

                with torch.no_grad():
                    action = int(
                        model.get_action(rtg_t, local_t, global_t, act_t, ts_t).item()
                    )
                action = max(0, min(action, int(cfg.action_dim) - 1))
                action_hist.append(action)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                current_rtg -= float(reward)
                t += 1

            if info.get("won", False):
                wins += 1
            total_steps += t
    finally:
        env.close()

    return wins / n_episodes, total_steps / n_episodes


def _train_dt(
    cfg: SimpleNamespace,
    log: Logger,
    log_dir: str,
    seed: int,
) -> tuple[_DecisionTransformer, dict[str, float]]:
    """Train a Decision Transformer baseline. Returns (model, seed_metrics)."""

    device = torch.device(cfg.device)
    context_len = int(cfg.baselines_dt_context_len)
    max_ep_len = int(cfg.baselines_dt_max_ep_len)
    eval_max_steps = int(cfg.baselines_dt_eval_max_steps)
    n_eval = _eval_episodes_per_env(cfg)
    n_epochs = int(cfg.baselines_dt_epochs)

    logger.info("Collecting oracle demonstrations for DT...")
    trajectories = _collect_dt_trajectories(cfg)
    if not trajectories:
        raise RuntimeError("DT oracle collection produced zero trajectories")

    traj_lengths = [len(t["actions"]) for t in trajectories]
    logger.info(
        "DT dataset: %d trajectories, %d transitions (len: min=%d max=%d mean=%.1f)",
        len(trajectories),
        sum(traj_lengths),
        min(traj_lengths),
        max(traj_lengths),
        float(np.mean(traj_lengths)),
    )
    if max(traj_lengths) > max_ep_len:
        logger.warning(
            "Longest oracle trajectory (%d) exceeds baselines_dt_max_ep_len (%d); "
            "positions will be clamped.",
            max(traj_lengths),
            max_ep_len,
        )

    target_return = float(max(t["returns_to_go"][0] for t in trajectories))

    dataset = _DTDataset(
        trajectories,
        context_len=context_len,
        max_ep_len=max_ep_len,
        n_actions=int(cfg.action_dim),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.baselines_dt_batch_size),
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = _DecisionTransformer(
        n_actions=int(cfg.action_dim),
        embed_dim=int(cfg.baselines_dt_embed_dim),
        n_heads=int(cfg.baselines_dt_n_heads),
        n_layers=int(cfg.baselines_dt_n_layers),
        context_len=context_len,
        max_ep_len=max_ep_len,
        crop_h=int(cfg.crop_size),
        crop_w=int(cfg.crop_size),
        map_h=int(cfg.map_h),
        map_w=int(cfg.map_w),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("DT parameters: %d", n_params)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.baselines_dt_lr),
        weight_decay=float(cfg.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
    )

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            local = batch["local"].to(device)
            glob = batch["global"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(rtg, local, glob, actions, timesteps, attention_mask)
            logits_flat = logits.reshape(-1, int(cfg.action_dim))
            targets_flat = actions.reshape(-1)
            mask_flat = attention_mask.reshape(-1)
            ce = nn.functional.cross_entropy(
                logits_flat,
                targets_flat,
                reduction="none",
            )
            loss = (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)
        log.log(
            {
                "train/dt_loss": avg_loss,
                "train/lr": float(scheduler.get_last_lr()[0]),
                "train/epoch": epoch + 1,
            },
            step=epoch + 1,
        )
        logger.info(
            "DT epoch %02d/%02d | loss=%.4f | lr=%.2e",
            epoch + 1,
            n_epochs,
            avg_loss,
            float(scheduler.get_last_lr()[0]),
        )

    seed_metrics: dict[str, float] = {}
    logger.info("DT eval target return = %.2f", target_return)
    for split, env_list in (("ID", cfg.id_envs), ("OOD", cfg.ood_envs)):
        logger.info("--- DT %s evaluation (seed=%d) ---", split, seed)
        for env_id in env_list:
            short = _short(env_id)
            win_rate, avg_steps = _eval_dt(
                model,
                env_id,
                cfg,
                target_return=target_return,
                n_episodes=n_eval,
                max_ep_len=max_ep_len,
                eval_max_steps=eval_max_steps,
                context_len=context_len,
            )
            seed_metrics[f"{split}/{short}/win_rate"] = win_rate * 100
            seed_metrics[f"{split}/{short}/avg_steps"] = avg_steps
            logger.info(
                "%-30s | win_rate=%5.1f%% | avg_steps=%5.1f",
                short,
                win_rate * 100,
                avg_steps,
            )
    log.log(seed_metrics, step=n_epochs + 1)
    return model, seed_metrics


def _build_sb3_model(
    algo: str,
    train_env: SubprocVecEnv,
    cfg: SimpleNamespace,
    seed: int,
    tb_log_dir: str,
):
    """Construct one of {ppo, a2c, dqn, ppo-rnn} with the MiniHack CNN."""

    policy_kwargs = {
        "features_extractor_class": _MiniHackCNN,
        "features_extractor_kwargs": {"features_dim": 256},
    }
    # SB3 raises at learn() time if tensorboard_log is set without tensorboard
    # installed, and tensorboard is not a dependency of this project.
    if not importlib.util.find_spec("tensorboard"):
        logger.info("tensorboard not installed; SB3 tensorboard logging off")
        tb_log_dir = None

    if algo == "ppo":
        return PPO(
            "MultiInputPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tb_log_dir,
            seed=seed,
        )
    if algo == "ppo-rnn":
        return RecurrentPPO(
            "MultiInputLstmPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tb_log_dir,
            seed=seed,
        )
    if algo == "a2c":
        return A2C(
            "MultiInputPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tb_log_dir,
            seed=seed,
        )
    if algo == "dqn":
        return DQN(
            "MultiInputPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tb_log_dir,
            seed=seed,
            buffer_size=int(cfg.baselines_dqn_buffer_size),
        )
    raise ValueError(f"Unknown SB3 algo: {algo!r}")


def _build_sb3_callbacks(
    cfg: SimpleNamespace,
    train_env: SubprocVecEnv,
    log_dir: str,
    model_dir: str,
) -> CallbackList:
    # WandbCallback requires an active run; without one it raises on
    # construction, which would make every SB3 baseline unrunnable with
    # use_wandb=false.
    import wandb

    callbacks: list = []
    if wandb.run is not None:
        callbacks.append(WandbCallback(model_save_path=model_dir))

    n_eval = _eval_episodes_per_env(cfg)
    eval_freq = max(
        1,
        int(cfg.baselines_eval_freq_env_steps) // train_env.num_envs,
    )
    for env_id in cfg.id_envs:
        short = _short(env_id)
        eval_env = SubprocVecEnv(
            [_make_sb3_env_fn(env_id, cfg, f"{log_dir}/eval_id/{env_id}")]
        )
        callbacks.append(
            _PrefixedEvalCallback(
                eval_env,
                prefix=f"ID/{short}",
                best_model_save_path=f"{model_dir}/best_{env_id}/",
                log_path=f"{log_dir}/eval_id/{env_id}/",
                eval_freq=eval_freq,
                n_eval_episodes=n_eval,
                deterministic=True,
            )
        )
    for env_id in cfg.ood_envs:
        short = _short(env_id)
        eval_env = SubprocVecEnv(
            [_make_sb3_env_fn(env_id, cfg, f"{log_dir}/eval_ood/{env_id}")]
        )
        callbacks.append(
            _PrefixedEvalCallback(
                eval_env,
                prefix=f"OOD/{short}",
                best_model_save_path=None,
                log_path=f"{log_dir}/eval_ood/{env_id}/",
                eval_freq=eval_freq,
                n_eval_episodes=n_eval,
                deterministic=True,
            )
        )
    return CallbackList(callbacks)


def _aggregate(
    all_seed_results: list[dict[str, Any]],
) -> dict[str, dict[str, float | list[float]]]:
    """Compute mean/std across seeds for every shared metric key."""

    if not all_seed_results:
        return {}
    metric_keys = [k for k in all_seed_results[0] if k != "seed"]
    agg: dict[str, dict[str, float | list[float]]] = {}
    for key in metric_keys:
        values = [r[key] for r in all_seed_results if key in r]
        if values:
            agg[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values],
            }
    return agg


def _print_aggregated(seeds: list[int], agg: dict[str, dict[str, Any]]) -> None:
    if not agg:
        logger.info(
            "No per-environment metrics to aggregate (RL eval is callback-driven)"
        )
        return
    logger.info("Aggregated results across %d seeds: %s", len(seeds), seeds)
    for split in ("ID", "OOD"):
        env_metrics: dict[str, dict[str, dict[str, Any]]] = {}
        for key, stats in agg.items():
            if not key.startswith(f"{split}/"):
                continue
            _split, env_name, metric_name = key.split("/", 2)
            env_metrics.setdefault(env_name, {})[metric_name] = stats
        if not env_metrics:
            continue
        logger.info("--- %s environments ---", split)
        for env_name, metrics in sorted(env_metrics.items()):
            wr = metrics.get("win_rate", {})
            steps = metrics.get("avg_steps", {})
            logger.info(
                "%-30s | win_rate=%5.1f%% +/- %4.1f | avg_steps=%5.1f +/- %4.1f",
                env_name,
                wr.get("mean", 0.0),
                wr.get("std", 0.0),
                steps.get("mean", 0.0),
                steps.get("std", 0.0),
            )


def _save_aggregated(
    out_path: Path,
    algo: str,
    seeds: list[int],
    all_seed_results: list[dict[str, Any]],
    agg: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "algorithm": algo,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "per_seed_results": all_seed_results,
        "aggregated": {k: {"mean": v["mean"], "std": v["std"]} for k, v in agg.items()},
    }
    out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    logger.info("Aggregated results written to %s", out_path)


def run_baselines(
    cfg: SimpleNamespace,
    algo: str,
    seeds: list[int] | None = None,
    output_path: str | None = None,
) -> None:
    """Train and evaluate one baseline algorithm across one or more seeds.

    Args:
        cfg: Project config namespace (must contain ``baselines_*`` keys).
        algo: One of ``ppo``, ``a2c``, ``dqn``, ``ppo-rnn``, ``bc``, ``dt``.
        seeds: Optional list of seeds. ``None`` -> ``[cfg.seed]`` (or
            a single seed of ``0`` if ``cfg.seed`` is ``None``).
        output_path: Optional override for the aggregated-results JSON
            destination. When ``None``, results land under
            ``cfg.baselines_output_dir``.
    """

    if algo not in ALL_BASELINE_ALGOS:
        raise ValueError(f"Unknown algo {algo!r}. Choose one of {ALL_BASELINE_ALGOS}.")

    # Before the first SubprocVecEnv, or the finalizer is already registered.
    quiet_multiprocessing_tempdir_teardown()

    if seeds is None:
        seeds = [cfg.seed if cfg.seed is not None else 0]
    if not seeds:
        raise ValueError("seeds must be non-empty")

    out_dir = _resolve_output_dir(cfg, None)
    if output_path is not None:
        agg_json_path = Path(output_path)
        agg_json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        agg_json_path = out_dir / f"results_{algo}_{len(seeds)}seeds.json"

    logger.info(
        "Running baseline %s on %d seed(s): %s (output -> %s)",
        algo,
        len(seeds),
        seeds,
        agg_json_path,
    )

    all_seed_results: list[dict[str, Any]] = []
    n_envs_per_id = int(cfg.baselines_n_envs_per_id)

    for seed_idx, seed in enumerate(seeds):
        logger.info(
            "============================================================\n"
            " %s seed %d (%d/%d)\n"
            "============================================================",
            algo.upper(),
            seed,
            seed_idx + 1,
            len(seeds),
        )
        _seed_everything(seed)

        run_name = f"{algo}-multitask-seed{seed}"
        log = _init_baseline_logger(cfg, run_name)
        run_id = (
            log._run.id  # type: ignore[union-attr]
            if log._use_wandb and log._run is not None
            else f"local-{algo}-seed{seed}"
        )
        log_dir = str(out_dir / "logs" / run_id)
        model_dir = str(out_dir / "models" / run_id)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        seed_results: dict[str, Any] = {"seed": seed}
        try:
            if algo == "dt":
                model, dt_metrics = _train_dt(cfg, log, log_dir, seed)
                seed_results.update(dt_metrics)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "n_actions": int(cfg.action_dim),
                            "embed_dim": int(cfg.baselines_dt_embed_dim),
                            "n_heads": int(cfg.baselines_dt_n_heads),
                            "n_layers": int(cfg.baselines_dt_n_layers),
                            "context_len": int(cfg.baselines_dt_context_len),
                            "max_ep_len": int(cfg.baselines_dt_max_ep_len),
                        },
                    },
                    f"{model_dir}/dt_final_seed{seed}.pt",
                )
            else:
                # SB3 RL families and BC both need the parallel train env.
                train_env_fns = [
                    _make_sb3_env_fn(env_id, cfg, log_dir)
                    for env_id in list(cfg.id_envs) * n_envs_per_id
                ]
                train_env = SubprocVecEnv(train_env_fns)
                try:
                    if algo == "bc":
                        policy, bc_metrics = _train_bc(
                            cfg,
                            train_env,
                            log,
                            log_dir,
                            seed,
                        )
                        seed_results.update(bc_metrics)
                        policy.save(f"{model_dir}/bc_final_seed{seed}")
                    else:
                        sb3_model = _build_sb3_model(
                            algo,
                            train_env,
                            cfg,
                            seed,
                            tb_log_dir=str(out_dir / "tb" / run_id),
                        )
                        callbacks = _build_sb3_callbacks(
                            cfg,
                            train_env,
                            log_dir,
                            model_dir,
                        )
                        logger.info(
                            "Training %s for %d env-steps across %d ID maps "
                            "(%d parallel envs)...",
                            algo.upper(),
                            int(cfg.total_timesteps),
                            len(cfg.id_envs),
                            train_env.num_envs,
                        )
                        sb3_model.learn(
                            total_timesteps=int(cfg.total_timesteps),
                            callback=callbacks,
                        )
                        sb3_model.save(f"{model_dir}/{algo}_final_seed{seed}")
                finally:
                    train_env.close()

            all_seed_results.append(seed_results)
        finally:
            log.finish()
        logger.info("%s seed %d complete.", algo.upper(), seed)

    agg = _aggregate(all_seed_results)
    _print_aggregated(seeds, agg)
    if agg:
        _save_aggregated(agg_json_path, algo, seeds, all_seed_results, agg)
        # Final summary write to the project Logger so the aggregated
        # numbers land on a dedicated W&B run.
        summary_run_name = f"{algo}-multitask-summary"
        summary_log = _init_baseline_logger(cfg, summary_run_name)
        try:
            summary_payload: dict[str, float] = {}
            for key, stats in agg.items():
                summary_payload[f"summary/{key}/mean"] = stats["mean"]
                summary_payload[f"summary/{key}/std"] = stats["std"]
            summary_log.log_summary(summary_payload)
        finally:
            summary_log.finish()
    logger.info("All %d seed(s) complete.", len(seeds))
