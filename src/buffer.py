"""Replay buffer with offline-protected FIFO eviction.

Stores observation-action
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
        self,
        capacity: int,
        seq_len: int,
        pad_token: int,
    ) -> None:
        self._capacity = capacity
        self._seq_len = seq_len
        self._pad_token = pad_token

        # Each element: (local [9,9], global [21,79], actions [seq_len])
        self._offline: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._online: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        # Stacked array cache for fast sampling
        self._cache_valid = False
        self._cached_local: np.ndarray | None = None
        self._cached_global: np.ndarray | None = None
        self._cached_actions: np.ndarray | None = None

    def load_offline_data(
        self,
        data: dict,
        allowed_envs: list[str],
    ) -> None:
        """Load pre-collected trajectories and slice into windows.

        Expects the dict format ``{"trajectories": [...]}`` where each
        entry is a dict with ``"local"``, ``"global"``, ``"actions"``,
        ``"env_id"``.

        Args:
            data: Dataset dict.
            allowed_envs: Only samples from these env IDs are kept.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"Offline dataset must be a dict, got {type(data).__name__}; "
                "the legacy list format is no longer supported."
            )

        trajectories = data.get("trajectories", [data])
        for traj in trajectories:
            if traj.get("env_id", "") not in allowed_envs:
                continue
            windows = self._slice_trajectory(traj)
            self._offline.extend(windows)
        # Truncate to capacity
        if len(self._offline) > self._capacity:
            self._offline = self._offline[: self._capacity]
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Mark the stacked array cache as stale."""
        self._cache_valid = False

    def _ensure_cache(self) -> None:
        """Rebuild stacked arrays from offline + online windows."""
        if self._cache_valid:
            return
        combined = self._offline + self._online
        if not combined:
            return
        n = len(combined)
        l0, g0, a0 = combined[0]
        self._cached_local = np.empty(
            (n, *l0.shape),
            dtype=l0.dtype,
        )
        self._cached_global = np.empty(
            (n, *g0.shape),
            dtype=g0.dtype,
        )
        self._cached_actions = np.empty(
            (n, *a0.shape),
            dtype=a0.dtype,
        )
        for i, (loc, glob, act) in enumerate(combined):
            self._cached_local[i] = loc
            self._cached_global[i] = glob
            self._cached_actions[i] = act
        self._cache_valid = True

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
        self._invalidate_cache()

    def sample(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Random sample from offline + online combined.

        Args:
            batch_size: Number of windows to sample.

        Returns:
            ``(local [B,9,9], global [B,21,79], actions [B,seq_len])``
            as numpy arrays, or ``None`` if the buffer is empty.
        """
        if len(self) == 0:
            return None
        self._ensure_cache()
        if self._cached_local is None:
            return None
        indices = np.random.randint(0, len(self), size=batch_size)
        return (
            self._cached_local[indices],
            self._cached_global[indices],
            self._cached_actions[indices],
        )

    def __len__(self) -> int:
        """Total number of windows (offline + online)."""
        return len(self._offline) + len(self._online)

    @property
    def n_offline(self) -> int:
        """Number of pinned offline windows."""
        return len(self._offline)

    @property
    def offline_size(self) -> int:
        """Number of pinned offline windows (alias)."""
        return len(self._offline)

    def _slice_trajectory(
        self,
        traj: dict,
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
            loc = local_arr[min(start, len(local_arr) - 1)]
            glob = global_arr[min(start, len(global_arr) - 1)]
            windows.append((loc.copy(), glob.copy(), a))

        return windows
