"""MiniHack environment wrapper with BFS oracle and shaped rewards.

Provides dual-stream
observations (9x9 local crop + 21x79 global map), a multi-tier BFS
oracle, and reward shaping (win bonus, BFS progress, exploration, step
penalty).
"""

from __future__ import annotations

import collections
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import minihack  # noqa: F401 — registers MiniHack envs
import numpy as np

logger = logging.getLogger(__name__)

# Suppress noisy NLE INFO spam ("Not saving any NLE data." on every env create)
logging.getLogger("nle.env.base").setLevel(logging.WARNING)




_nhdat_patch_checked = False


def _patch_nhdat_py(self, des_file: str) -> None:
    """Python port of MiniHack's ``mh_patch_nhdat.sh``.

    Behaviour matches the shell script step for step, but paths are passed
    as argv entries instead of being interpolated into a command line.
    """
    import minihack.base as mh_base

    if not des_file.endswith(".des"):
        fpath = os.path.join(self.nethack._vardir, "mylevel.des")
        with open(fpath, "w") as fh:
            fh.writelines(des_file)
        des_file = fpath

    des_path = os.path.abspath(des_file)
    if not os.path.exists(des_path):
        des_path = os.path.abspath(
            os.path.join(mh_base.PATH_DAT_DIR, des_file)
        )
    if not os.path.exists(des_path):
        raise FileNotFoundError(f"des file not found: {des_path}")

    vardir = Path(self.nethack._vardir)
    libdir = vardir / "lib"
    if not libdir.is_dir():
        shutil.copytree(mh_base.LIB_DIR, libdir)

    shutil.copy(des_path, libdir / "mylevel.des")
    _run_nethack_tool(
        [os.path.join(mh_base.HACKDIR, "lev_comp"), "mylevel.des"], libdir,
    )
    (libdir / "mylevel.des").unlink()

    # Bash `*` skips dotfiles and sorts; match that so the archive contents
    # are identical to the shell implementation's.
    contents = sorted(
        p.name for p in libdir.iterdir() if not p.name.startswith(".")
    )
    _run_nethack_tool(
        [os.path.join(mh_base.HACKDIR, "dlb"), "cf", "nhdat", *contents],
        libdir,
    )
    shutil.move(str(libdir / "nhdat"), str(vardir / "nhdat"))


def _run_nethack_tool(cmd: list[str], cwd: Path) -> None:
    """Run a NetHack build tool, surfacing its output when it fails."""
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{Path(cmd[0]).name} failed ({result.returncode}) in {cwd}\n"
            f"{result.stdout}\n{result.stderr}"
        )


def _ensure_nhdat_patch_works() -> None:
    """Replace MiniHack's nhdat patch step when its shell script would fail.

    ``mh_patch_nhdat.sh`` interpolates its path arguments unquoted, so any
    whitespace in the install path splits them into separate words. The
    script then fails without a non-zero exit status that MiniHack checks:
    no nhdat file is produced, every environment silently falls back to the
    same default level with no goal staircase, and the BFS oracle has
    nothing to navigate towards. Swap in the Python port in that case and
    leave the upstream path alone everywhere else.
    """
    global _nhdat_patch_checked
    if _nhdat_patch_checked:
        return
    _nhdat_patch_checked = True

    import minihack.base as mh_base

    paths = (mh_base.LIB_DIR, mh_base.HACKDIR, mh_base.PATH_DAT_DIR)
    if not any(re.search(r"\s", p) for p in paths):
        return

    mh_base.MiniHack._patch_nhdat = _patch_nhdat_py
    logger.warning(
        "MiniHack is installed under a path containing whitespace; using the "
        "Python nhdat patcher because mh_patch_nhdat.sh would fail silently."
    )




def find_staircase_from_glyphs(global_obs: np.ndarray) -> np.ndarray:
    """Locate the staircase '>' in the global glyph map.

    Args:
        global_obs: Glyph map, shape ``[B, H, W]`` or ``[H, W]``.

    Returns:
        Normalised ``(row/H, col/W)`` coords, shape ``[B, 2]``
        (float32). ``(-1, -1)`` when not visible.
    """
    squeeze = global_obs.ndim == 2
    if squeeze:
        global_obs = global_obs[np.newaxis]
    B, H, W = global_obs.shape
    coords = np.full((B, 2), -1.0, dtype=np.float32)
    for b in range(B):
        is_stair = (
            (global_obs[b] == 62)
            | (global_obs[b] == 2310)
            | (global_obs[b] == 2368)
            | (global_obs[b] == 2383)
        )
        positions = np.argwhere(is_stair)
        if positions.shape[0] > 0:
            coords[b, 0] = positions[0, 0] / max(1, H - 1)
            coords[b, 1] = positions[0, 1] / max(1, W - 1)
    return coords




class AdvancedObservationEnv(gym.Env):
    """MiniHack wrapper with dual-stream obs, BFS oracle, shaped rewards.

    Observations are ``(local_crop, global_map)`` where
    ``local_crop`` is a ``[crop_size, crop_size]`` glyph window centred
    on the agent and ``global_map`` is the full ``[21, 79]`` glyph grid.

    Args:
        env_id: MiniHack registry ID.
        des_file: Optional ``.des`` file content (for custom levels).
        cfg: Configuration namespace with ``crop_size``, ``action_dim``,
            ``pad_token``, ``map_h``, ``map_w``.
    """

    _UNWALKABLE = frozenset({32, 45, 124, 125})  # space, -, |, }
    _CLOSED_DOOR = 43  # '+'
    _DIR_MAP = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    _CARDINAL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(
        self,
        env_id: str,
        des_file: str | None,
        cfg: SimpleNamespace,
    ) -> None:
        super().__init__()
        _ensure_nhdat_patch_works()
        self.env_id = env_id
        self._cfg = cfg
        self._crop_half = cfg.crop_size // 2

        obs_keys = ("glyphs", "chars", "pixel")
        if des_file is not None:
            self._inner = gym.make(
                "MiniHack-Navigation-Custom-v0",
                des_file=des_file,
                observation_keys=obs_keys,
            )
        else:
            self._inner = gym.make(
                env_id, observation_keys=obs_keys,
            )

        self.observation_space = gym.spaces.Box(
            low=0, high=6000,
            shape=(cfg.crop_size, cfg.crop_size),
            dtype=np.int16,
        )
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(cfg.action_dim)

        self._visited: set[tuple[int, int]] = set()
        self._prev_bfs_dist: int | None = None
        self.last_raw_obs: dict | None = None


    def reset(
        self, seed: int | None = None, options: dict | None = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
        """Reset environment and tracking state.

        Args:
            seed: Optional RNG seed.
            options: Passed through to the inner env.

        Returns:
            ``((local_crop, global_map), info)``
        """
        # gymnasium's reset(seed=...) does not reach the NetHack
        # core RNG, so dungeons were entropy-random regardless of the seed
        # (verified: identical seed+actions diverge). Seed the NLE core
        # explicitly; reseed=False keeps it fixed for this env instance.
        if seed is not None:
            _u = getattr(self._inner, "unwrapped", self._inner)
            if hasattr(_u, "seed"):
                _u.seed(core=seed, disp=seed, reseed=False)
        obs, info = self._inner.reset(seed=seed, options=options)
        self.last_raw_obs = obs
        self._prev_bfs_dist = self._get_bfs_distance(obs)
        self._visited = set()
        agent_pos = self._get_agent_pos(obs)
        if agent_pos is not None:
            self._visited.add(agent_pos)
        return self._get_obs(obs), info

    def step(
        self, action: int,
    ) -> tuple[tuple[np.ndarray, np.ndarray], float, bool, bool, dict]:
        """Execute one environment step with shaped reward.

        Reward shaping:
        - Win bonus: ``+20.0``
        - BFS progress toward staircase: ``+0.5 * (prev - curr)``
        - New-tile exploration: ``+0.05``
        - Step penalty: ``-0.01``

        Args:
            action: Integer action in ``[0, action_dim)``.

        Returns:
            ``(obs, shaped_reward, terminated, truncated, info)``
        """
        inner_n = self._inner.action_space.n
        if action >= inner_n:
            action = action % inner_n

        obs, raw_reward, terminated, truncated, info = self._inner.step(action)
        self.last_raw_obs = obs
        reward = float(raw_reward)

        # Win bonus
        if terminated and reward > 0:
            info["won"] = True
            reward += 20.0
        else:
            info["won"] = False

        # BFS shaping
        curr_dist = self._get_bfs_distance(obs)
        if curr_dist is not None and self._prev_bfs_dist is not None:
            reward += (self._prev_bfs_dist - curr_dist) * 0.5
            self._prev_bfs_dist = curr_dist

        # Exploration bonus
        agent_pos = self._get_agent_pos(obs)
        if agent_pos is not None and agent_pos not in self._visited:
            reward += 0.05
            self._visited.add(agent_pos)

        # Step penalty
        reward -= 0.01

        return self._get_obs(obs), reward, terminated, truncated, info

    @property
    def unwrapped(self):
        """Access the inner MiniHack env."""
        return self._inner.unwrapped

    def close(self) -> None:
        """Close the inner environment."""
        self._inner.close()


    def _get_obs(
        self, obs: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract dual-stream observation.

        Args:
            obs: Raw NLE observation dict.

        Returns:
            ``(local_crop [crop,crop], global_map [H,W])`` as int16.
        """
        return self._get_crop(obs), obs["glyphs"].copy().astype(np.int16)

    def _get_crop(self, obs: dict) -> np.ndarray:
        """Crop local glyph window centred on agent.

        Args:
            obs: Raw NLE observation dict.

        Returns:
            ``[crop_size, crop_size]`` int16 array.
        """
        glyphs = obs["glyphs"]
        chars = obs["chars"]
        agent_pos = np.argwhere(chars == ord("@"))
        cs = self._cfg.crop_size
        if len(agent_pos) == 0:
            return np.full((cs, cs), self._cfg.pad_token, dtype=np.int16)
        y, x = agent_pos[0]
        h = self._crop_half
        padded = np.pad(
            glyphs, h, mode="constant",
            constant_values=self._cfg.pad_token,
        )
        return padded[y:y + cs, x:x + cs].astype(np.int16)

    def _get_agent_pos(self, obs: dict) -> tuple[int, int] | None:
        """Find agent '@' position in the chars grid.

        Args:
            obs: Raw NLE observation dict.

        Returns:
            ``(row, col)`` or ``None``.
        """
        chars = obs["chars"]
        pos = np.argwhere(chars == ord("@"))
        return tuple(pos[0]) if len(pos) > 0 else None

    def _get_bfs_distance(self, obs: dict) -> int | None:
        """BFS shortest-path distance from agent to staircase.

        Args:
            obs: Raw NLE observation dict.

        Returns:
            Integer distance or ``None`` if unreachable / not visible.
        """
        chars = obs["chars"]
        start = np.argwhere(chars == ord("@"))
        target = np.argwhere(chars == ord(">"))
        if len(start) == 0 or len(target) == 0:
            return None
        start = tuple(start[0])
        target = tuple(target[0])
        if start == target:
            return 0
        queue: collections.deque = collections.deque([(start, 0)])
        visited = {start}
        while queue:
            (r, c), dist = queue.popleft()
            if (r, c) == target:
                return dist
            for dr, dc in self._CARDINAL:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self._cfg.map_h
                    and 0 <= nc < self._cfg.map_w
                    and (nr, nc) not in visited
                    and chars[nr, nc] not in self._UNWALKABLE
                ):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
        return None


    def get_oracle_action(self, obs: dict) -> int:
        """5-tier BFS oracle action.

        Priority:
        1. Kick adjacent closed door.
        2. BFS to staircase '>'.
        3. BFS to frontier (adjacent to unexplored space).
        4. BFS to farthest reachable tile.
        5. Random cardinal direction.

        Args:
            obs: Raw NLE observation dict (needs ``'chars'`` key).

        Returns:
            Action index in ``[0, action_dim)``.
        """
        if obs is None:
            return 0
        chars = obs["chars"]
        start = np.argwhere(chars == ord("@"))
        if len(start) == 0:
            return np.random.randint(0, 4)
        start = tuple(start[0])
        target_list = np.argwhere(chars == ord(">"))

        # 1. Adjacent closed door → kick
        for dr, dc in self._CARDINAL:
            nr, nc = start[0] + dr, start[1] + dc
            if (
                0 <= nr < self._cfg.map_h
                and 0 <= nc < self._cfg.map_w
                and chars[nr, nc] == self._CLOSED_DOOR
            ):
                return 11  # KICK

        # BFS to gather reachable tiles + check staircase
        queue: collections.deque = collections.deque([(start, [])])
        visited = {start}
        reachable: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        target_path: list[tuple[int, int]] | None = None

        while queue:
            (r, c), path = queue.popleft()
            reachable.append(((r, c), path))
            for t_r, t_c in target_list:
                if r == t_r and c == t_c:
                    target_path = path
                    break
            if target_path is not None:
                break
            for dr, dc in self._CARDINAL:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self._cfg.map_h
                    and 0 <= nc < self._cfg.map_w
                    and (nr, nc) not in visited
                ):
                    ch = chars[nr, nc]
                    if ch not in self._UNWALKABLE and ch != self._CLOSED_DOOR:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(dr, dc)]))

        # 2. Path to staircase
        if target_path:
            return self._DIR_MAP.get(target_path[0], 0)

        # 3. Frontier exploration — tiles adjacent to unexplored space
        frontier: list[list[tuple[int, int]]] = []
        for (r, c), path in reachable:
            if not path:
                continue
            for dr, dc in self._CARDINAL:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self._cfg.map_h
                    and 0 <= nc < self._cfg.map_w
                    and chars[nr, nc] == 32
                ):
                    frontier.append(path)
                    break
        if frontier:
            frontier.sort(key=len)
            return self._DIR_MAP.get(frontier[0][0], 0)

        # 4. Farthest reachable tile
        if reachable:
            reachable.sort(key=lambda x: len(x[1]), reverse=True)
            farthest = reachable[0][1]
            if farthest:
                return self._DIR_MAP.get(farthest[0], 0)

        # 5. Random cardinal
        return np.random.randint(0, 4)




def make_env(
    env_id: str,
    des_file: str | None,
    cfg: SimpleNamespace,
) -> AdvancedObservationEnv:
    """Create a wrapped MiniHack environment.

    Args:
        env_id: MiniHack registry ID.
        des_file: Optional ``.des`` file content.
        cfg: Configuration namespace.

    Returns:
        Wrapped environment.
    """
    return AdvancedObservationEnv(env_id, des_file, cfg)


def collect_oracle_trajectory(
    env_id: str,
    seed: int,
    cfg: SimpleNamespace,
    max_steps: int = 500,
) -> dict | None:
    """Roll out the BFS oracle on a single episode.

    Args:
        env_id: MiniHack registry ID.
        seed: RNG seed for the episode.
        cfg: Configuration namespace.
        max_steps: Maximum episode length.

    Returns:
        ``{"local": [T,9,9], "global": [T,21,79],
          "actions": [T], "env_id": str}`` on success,
        or ``None`` on failure.
    """
    env = make_env(env_id, None, cfg)
    try:
        (local, glb), _info = env.reset(seed=seed)
        locals_list = [local]
        globals_list = [glb]
        actions_list: list[int] = []

        for _ in range(max_steps):
            action = env.get_oracle_action(env.last_raw_obs)
            actions_list.append(action)
            (local, glb), _reward, terminated, truncated, _info = env.step(
                action
            )
            locals_list.append(local)
            globals_list.append(glb)
            if terminated or truncated:
                break

        # Trim trailing obs (one more obs than actions)
        locals_arr = np.stack(locals_list[:-1], axis=0).astype(np.int16)
        globals_arr = np.stack(globals_list[:-1], axis=0).astype(np.int16)
        actions_arr = np.array(actions_list, dtype=np.int64)

        return {
            "local": locals_arr,
            "global": globals_arr,
            "actions": actions_arr,
            "env_id": env_id,
        }
    except Exception:
        logger.error(
            f"Oracle trajectory failed for {env_id} seed={seed}",
            exc_info=True,
        )
        return None
    finally:
        env.close()
