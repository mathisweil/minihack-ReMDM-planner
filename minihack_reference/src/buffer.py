"""
FIFO Replay Buffer for DAgger training.
"""

import json
import os
import numpy as np
import torch

from config import BUFFER_CAPACITY, PAD_TOKEN, HYPERPARAMS


class ReplayBuffer:
    """FIFO replay buffer with capacity limit. Oldest samples evicted first."""

    def __init__(self, capacity=None):
        self.capacity = capacity or BUFFER_CAPACITY
        self._buffer = []
        self._n_offline = 0  # Offline samples are never evicted (kept at front)

    def load_offline_data(self, data_path, allowed_envs, metadata=None):
        """Load offline dataset and filter by allowed_envs.

        data_path: path to .pt file, or pre-loaded list of (state, action_seq) tuples
        allowed_envs: set or list of env IDs to keep
        metadata: optional dict with 'samples_per_env' or 'env_stats' for env filtering.
                 If None and data_path is str, tries data_path with _meta.json suffix.
        """
        if isinstance(data_path, list):
            data = data_path
        else:
            data = torch.load(data_path, map_location="cpu", weights_only=False)
        if metadata is None and isinstance(data_path, str):
            meta_path = data_path.replace(".pt", "_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    metadata = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Offline data must be a list of (state, action_seq) tuples")

        allowed = set(allowed_envs)
        filtered = []

        if metadata and "samples_per_env" in metadata:
            # Build sample index: dataset ordered by env (env1 samples, env2 samples, ...)
            sample_to_env = []
            for env_id in sorted(metadata["samples_per_env"].keys()):
                count = metadata["samples_per_env"][env_id]
                sample_to_env.extend([env_id] * count)
            sample_to_env = sample_to_env[:len(data)]

            for i, sample in enumerate(data):
                env_id = sample_to_env[i] if i < len(sample_to_env) else None
                if env_id is None or env_id in allowed:
                    filtered.append(sample)
        elif metadata and "env_stats" in metadata:
            # No samples_per_env: use env_stats order (proportional)
            env_order = sorted(metadata["env_stats"].keys())
            total = sum(metadata["env_stats"].get(e, {}).get("demos", 0) for e in env_order)
            if total > 0:
                idx = 0
                for env_id in env_order:
                    demos = metadata["env_stats"].get(env_id, {}).get("demos", 0)
                    n = max(1, int(len(data) * demos / total))
                    for j in range(min(n, len(data) - idx)):
                        if idx + j < len(data) and env_id in allowed:
                            filtered.append(data[idx + j])
                    idx += n
            else:
                filtered = [s for s in data]
        else:
            # No metadata: keep all (assume caller provides pre-filtered data)
            filtered = list(data)

        # Truncate to capacity if over (e.g. smoke test with capacity 50)
        if len(filtered) > self.capacity:
            filtered = filtered[: self.capacity]
        self._buffer = filtered
        self._n_offline = len(self._buffer)
        return len(self._buffer)

    def add(self, trajectory):
        """Add trajectory samples to buffer.

        trajectory: (states, actions) tuple from Oracle rollout.
          states: list of (local_crop, global_map) obs
          actions: list of int action indices
        """
        states, actions = trajectory
        seq_len = HYPERPARAMS["seq_len"]
        for i in range(len(actions)):
            seq = actions[i:i + seq_len]
            if len(seq) < seq_len:
                seq = list(seq) + [PAD_TOKEN] * (seq_len - len(seq))
            self._buffer.append((states[i], tuple(seq)))

        # FIFO eviction: remove oldest ONLINE samples (never evict offline)
        while len(self._buffer) > self.capacity:
            self._buffer.pop(self._n_offline)

    def sample(self, batch_size):
        """Sample batch_size random samples. Returns (local, global, actions) tensors."""
        if len(self._buffer) == 0:
            return None
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=True)
        batch = [self._buffer[i] for i in indices]
        local = np.array([x[0][0] for x in batch])
        global_obs = np.array([x[0][1] for x in batch])
        actions = np.array([x[1] for x in batch])
        return local, global_obs, actions

    def __len__(self):
        return len(self._buffer)

    @property
    def n_offline(self):
        return self._n_offline
