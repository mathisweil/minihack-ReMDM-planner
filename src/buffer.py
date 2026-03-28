"""Replay buffer with offline-protected FIFO eviction.

Ported from minihack_reference/src/buffer.py. Stores observation-action
windows of fixed length ``seq_len``. Offline data is pinned at the front
and never evicted; online samples use FIFO.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Fixed-capacity buffer with offline-protected FIFO eviction.

    Offline samples (loaded once via ``load_offline_data``) are pinned
    and never evicted. Online samples added via ``add`` are FIFO-evicted
    when the total count exceeds ``capacity``.

    Args:
        capacity: Maximum total number of windows.
        seq_len: Action-sequence window length.
        pad_token: Token used to pad short sequences.
    """

    def __init__(
        self, capacity: int, seq_len: int, pad_token: int,
    ) -> None:
        self._capacity = capacity
        self._seq_len = seq_len
        self._pad_token = pad_token

        # Each element: (local [9,9], global [21,79], actions [seq_len])
        self._offline: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._online: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # ── Offline data ─────────────────────────────────────────────

    def load_offline_data(
        self,
        data: dict,
        allowed_envs: list[str],
    ) -> None:
        """Load pre-collected trajectories and slice into windows.

        Args:
            data: Dict with ``"trajectories"`` list of trajectory dicts,
                each having ``"local"``, ``"global"``, ``"actions"``,
                ``"env_id"`` keys. Alternatively, a flat dict with
                those keys for a single trajectory.
            allowed_envs: Only trajectories from these env IDs are kept.
        """
        trajectories = data.get("trajectories", [data])
        for traj in trajectories:
            if traj.get("env_id", "") not in allowed_envs:
                continue
            windows = self._slice_trajectory(traj)
            self._offline.extend(windows)
        # Truncate to capacity
        if len(self._offline) > self._capacity:
            self._offline = self._offline[: self._capacity]

    # ── Online data ──────────���───────────────────────────────────

    def add(self, trajectory: dict) -> None:
        """Add a trajectory, sliced into overlapping windows.

        FIFO-evicts oldest online samples when over capacity.

        Args:
            trajectory: Dict with ``"local"`` ``[T,9,9]``,
                ``"global"`` ``[T,21,79]``, ``"actions"`` ``[T]``.
        """
        windows = self._slice_trajectory(trajectory)
        self._online.extend(windows)
        max_online = self._capacity - len(self._offline)
        if len(self._online) > max_online:
            excess = len(self._online) - max_online
            self._online = self._online[excess:]

    # ── Sampling ─────────────────────────────────────────────────

    def sample(
        self, batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random sample from offline + online combined.

        Args:
            batch_size: Number of windows to sample.

        Returns:
            ``(local [B,9,9], global [B,21,79], actions [B,seq_len])``
            as numpy arrays.
        """
        combined = self._offline + self._online
        n = len(combined)
        indices = np.random.randint(0, n, size=batch_size)
        locals_list, globals_list, actions_list = [], [], []
        for i in indices:
            l, g, a = combined[i]
            locals_list.append(l)
            globals_list.append(g)
            actions_list.append(a)
        return (
            np.stack(locals_list),
            np.stack(globals_list),
            np.stack(actions_list),
        )

    # ── Properties ─────────��─────────────────────────────────────

    def __len__(self) -> int:
        """Total number of windows (offline + online)."""
        return len(self._offline) + len(self._online)

    @property
    def n_offline(self) -> int:
        """Number of pinned offline windows."""
        return len(self._offline)

    # ── Internals ───────────────────────────────────────────���────

    def _slice_trajectory(
        self, traj: dict,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Slice a trajectory into overlapping seq_len windows.

        Args:
            traj: Trajectory dict with ``"local"``, ``"global"``,
                ``"actions"`` arrays.

        Returns:
            List of ``(local, global, actions)`` tuples.
        """
        local_arr = np.asarray(traj["local"])
        global_arr = np.asarray(traj["global"])
        actions_arr = np.asarray(traj["actions"])
        T = len(actions_arr)
        windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for start in range(T):
            end = start + self._seq_len
            if end <= T:
                a = actions_arr[start:end]
            else:
                a = np.full(self._seq_len, self._pad_token, dtype=np.int64)
                a[: T - start] = actions_arr[start:]

            # Use the observation at the window start
            l = local_arr[min(start, len(local_arr) - 1)]
            g = global_arr[min(start, len(global_arr) - 1)]
            windows.append((l.copy(), g.copy(), a))

        return windows
