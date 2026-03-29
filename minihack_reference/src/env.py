"""
Environment wrapper and discovery utilities for MiniHack.
"""

import numpy as np
import collections
import gymnasium as gym
import minihack

from config import HYPERPARAMS, ACTION_DIM, PAD_TOKEN


# ---------------------------------------------------------------------------
# Module 2: Environment Discovery
# ---------------------------------------------------------------------------

def list_working_minihack_tasks():
    """Scan registry for working MiniHack navigation tasks."""
    all_envs = list(gym.envs.registry.keys())
    minihack_envs = [e for e in all_envs if "MiniHack" in e]
    nav_keywords = ["Room", "Corridor", "Maze", "River"]
    working_nav_tasks, broken_nav_tasks = [], []

    for env_id in sorted(minihack_envs):
        if not any(k in env_id for k in nav_keywords) or "KeyRoom" in env_id:
            continue
        try:
            env = gym.make(env_id)
            working_nav_tasks.append((env_id, env.action_space.n))
            env.close()
        except (AssertionError, TypeError, Exception):
            broken_nav_tasks.append(env_id)

    print(f"Working: {len(working_nav_tasks)} | Broken: {len(broken_nav_tasks)}")
    return np.array(working_nav_tasks)[:, 0] if working_nav_tasks else np.array([])


def check_action_consistency_with_fixed_ref(env_list):
    """Check action-space alignment across envs against a reference."""
    ref_id = "MiniHack-Room-15x15-v0"
    ref_env = gym.make(ref_id)
    raw_ref = ref_env.unwrapped
    reference_actions = raw_ref.actions
    ref_env.close()

    results = []
    for env_id in sorted(env_list):
        if env_id == ref_id:
            results.append((env_id, "REFERENCE", len(reference_actions)))
            continue
        try:
            env = gym.make(env_id)
            raw_env = env.unwrapped
            limit = min(len(reference_actions), len(raw_env.actions))
            is_match = all(reference_actions[i] == raw_env.actions[i] for i in range(limit))
            diff = len(raw_env.actions) - len(reference_actions)
            if is_match and diff == 0:
                status = "EXACT"
            elif diff > 0:
                status = f"SUPERSET (+{diff})"
            elif is_match:
                status = f"SUBSET ({diff})"
            else:
                status = "CONFLICT"
            results.append((env_id, status, len(raw_env.actions)))
            env.close()
        except Exception:
            results.append((env_id, "CRASHED", 0))

    for name, status, size in results:
        print(f"  {name:<40} | {status:<12} | {size}")
    return results


# ---------------------------------------------------------------------------
# Module 3: Advanced Observation Environment
# ---------------------------------------------------------------------------

class AdvancedObservationEnv(gym.Env):
    """
    Wraps MiniHack with dual-stream observations (9x9 crop + full map),
    BFS-based oracle, and shaped rewards.

    Supports two kinds of environments mixed together:
      - Registry IDs  (e.g. "MiniHack-Room-Random-5x5-v0")
      - Custom .des    via des_files=[(name, des_content), ...]

    Optional ``weights`` array covers all entries (registry IDs first,
    then .des files).  If omitted, uniform sampling is used.
    """

    def __init__(self, env_list, des_files=None, weights=None):
        self.env_list = list(env_list)
        self.des_files = des_files or []  # [(name, des_content), ...]
        self._all_names = self.env_list + [name for name, _ in self.des_files]
        self._n_registry = len(self.env_list)

        if weights is not None:
            w = np.array(weights, dtype=np.float64)
            self._weights = w / w.sum()
        else:
            self._weights = None

        self.current_env = None
        self.current_env_id = "Init"
        self.crop_half = HYPERPARAMS["crop_size"] // 2
        self.last_raw_obs = None

        first_id = self.env_list[0] if self.env_list else None
        if first_id:
            temp_env = gym.make(first_id, observation_keys=("glyphs", "chars", "pixel"))
        else:
            _, des_content = self.des_files[0]
            temp_env = gym.make("MiniHack-Navigation-Custom-v0",
                                des_file=des_content,
                                observation_keys=("glyphs", "chars", "pixel"))
        self.observation_space = gym.spaces.Box(
            low=0, high=6000,
            shape=(HYPERPARAMS["crop_size"], HYPERPARAMS["crop_size"]),
            dtype=np.int16
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        temp_env.close()

    def reset(self, seed=None, options=None):
        idx = np.random.choice(len(self._all_names), p=self._weights)
        self.current_env_id = self._all_names[idx]
        if self.current_env:
            self.current_env.close()

        if idx < self._n_registry:
            self.current_env = gym.make(
                self.current_env_id,
                observation_keys=("glyphs", "chars", "pixel"))
        else:
            _, des_content = self.des_files[idx - self._n_registry]
            self.current_env = gym.make(
                "MiniHack-Navigation-Custom-v0",
                des_file=des_content,
                observation_keys=("glyphs", "chars", "pixel"))

        obs, info = self.current_env.reset(seed=seed, options=options)
        self.last_raw_obs = obs
        self.prev_dist = self._get_bfs_distance(obs)
        self.visited_tiles = set()
        agent_pos = self._get_agent_pos(obs)
        if agent_pos:
            self.visited_tiles.add(agent_pos)
        return (self._get_crop(obs), obs['glyphs'].copy()), info

    def step(self, action):
        if action >= self.current_env.action_space.n:
            action = 0
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        self.last_raw_obs = obs
        if terminated and reward > 0:
            info['won'] = True
            reward += 20.0
        else:
            info['won'] = False
        curr_dist = self._get_bfs_distance(obs)
        if curr_dist is not None and self.prev_dist is not None:
            reward += (self.prev_dist - curr_dist) * 0.5
            self.prev_dist = curr_dist
        agent_pos = self._get_agent_pos(obs)
        if agent_pos and agent_pos not in self.visited_tiles:
            reward += 0.05
            self.visited_tiles.add(agent_pos)
        reward -= 0.01
        return self._get_obs(obs), reward, terminated, truncated, info

    def _get_agent_pos(self, obs):
        chars = obs['chars']
        pos = np.argwhere(chars == ord('@'))
        return tuple(pos[0]) if len(pos) > 0 else None

    def get_oracle_action(self, obs):
        """BFS to stairs; frontier exploration fallback; door kick detection."""
        if obs is None:
            return 0
        chars = obs['chars']
        start = np.argwhere(chars == ord('@'))
        if len(start) == 0:
            return np.random.randint(0, 4)
        start = tuple(start[0])
        target_list = np.argwhere(chars == ord('>'))
        UNWALKABLE = {32, 45, 124, 125}  # space, -, |, }
        CLOSED_DOOR = 43  # '+'
        DIR_MAP = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}

        # Check if adjacent to a closed door — kick it open
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < 21 and 0 <= nc < 79 and chars[nr, nc] == CLOSED_DOOR:
                return 11  # KICK

        queue = collections.deque([(start, [])])
        visited = set([start])
        reachable_tiles, target_path = [], None

        while queue:
            (r, c), path = queue.popleft()
            reachable_tiles.append(((r, c), path))
            for t_r, t_c in target_list:
                if r == t_r and c == t_c:
                    target_path = path
                    break
            if target_path:
                break
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 21 and 0 <= nc < 79 and (nr, nc) not in visited:
                    ch = chars[nr, nc]
                    if ch not in UNWALKABLE and ch != CLOSED_DOOR:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(dr, dc)]))

        if target_path:
            return DIR_MAP.get(target_path[0], 0)

        # Frontier exploration: prefer tiles adjacent to unexplored space
        frontier = []
        for (r, c), path in reachable_tiles:
            if not path:
                continue
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 21 and 0 <= nc < 79 and chars[nr, nc] == 32:
                    frontier.append(path)
                    break
        if frontier:
            frontier.sort(key=len)
            return DIR_MAP.get(frontier[0][0], 0)

        # Last resort: go to farthest reachable tile
        if reachable_tiles:
            reachable_tiles.sort(key=lambda x: len(x[1]), reverse=True)
            farthest_path = reachable_tiles[0][1]
            if farthest_path:
                return DIR_MAP.get(farthest_path[0], 0)
        return np.random.randint(0, 4)

    @property
    def valid_action_count(self):
        return self.current_env.action_space.n if self.current_env else 8

    def _get_crop(self, obs):
        glyphs, chars = obs['glyphs'], obs['chars']
        agent_pos = np.argwhere(chars == ord('@'))
        if len(agent_pos) == 0:
            return np.full((HYPERPARAMS["crop_size"], HYPERPARAMS["crop_size"]), PAD_TOKEN)
        y, x = agent_pos[0]
        y_start, y_end = y - self.crop_half, y + self.crop_half + 1
        x_start, x_end = x - self.crop_half, x + self.crop_half + 1
        padded = np.pad(glyphs, self.crop_half, mode='constant', constant_values=PAD_TOKEN)
        return padded[y_start + self.crop_half:y_end + self.crop_half,
                      x_start + self.crop_half:x_end + self.crop_half]

    def _get_obs(self, obs):
        return self._get_crop(obs), obs['glyphs'].copy()

    def _get_bfs_distance(self, obs):
        chars = obs['chars']
        start = np.argwhere(chars == ord('@'))
        target = np.argwhere(chars == ord('>'))
        if len(start) == 0 or len(target) == 0:
            return None
        start, target = tuple(start[0]), tuple(target[0])
        if start == target:
            return 0
        queue = collections.deque([(start, 0)])
        visited = set([start])
        UNWALKABLE = [32, 45, 124, 125]
        while queue:
            (r, c), dist = queue.popleft()
            if (r, c) == target:
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 21 and 0 <= nc < 79 and (nr, nc) not in visited and chars[nr, nc] not in UNWALKABLE:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
        return None
