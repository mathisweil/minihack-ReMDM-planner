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
        data: dict | list,
        allowed_envs: list[str],
        metadata: dict | None = None,
    ) -> None:
        """Load pre-collected trajectories and slice into windows.

        Supports two dataset formats:

        **New format** (dict): ``{"trajectories": [...]}`` where each entry
        is a dict with ``"local"``, ``"global"``, ``"actions"``, ``"env_id"``.

        **Legacy format** (list): Flat list of ``((local, global), action_seq)``
        tuples produced by the reference pipeline (pre-windowed, already
        ``seq_len``-length). Env filtering uses an optional *metadata* dict
        with a ``"samples_per_env"`` key mapping env IDs to sample counts.

        Args:
            data: Dataset in new dict format or legacy list format.
            allowed_envs: Only samples from these env IDs are kept.
            metadata: Optional sidecar metadata for legacy format env
                filtering. Ignored for the new format.
        """
        if isinstance(data, list):
            self._load_legacy_offline_data(data, allowed_envs, metadata)
            return

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

    def _load_legacy_offline_data(
        self,
        data: list,
        allowed_envs: list[str],
        metadata: dict | None = None,
    ) -> None:
        """Load reference-format datasets (pre-windowed tuples).

        Args:
            data: List of ``((local_crop, global_map), action_seq)`` tuples.
                ``local_crop`` is ``[9, 9]``, ``global_map`` is ``[21, 79]``,
                ``action_seq`` is a sequence of length ``seq_len``.
            allowed_envs: Env IDs to retain.
            metadata: Optional dict with ``"samples_per_env"`` key mapping
                env IDs to per-env sample counts for precise filtering.
        """
        allowed = set(allowed_envs)

        if metadata and "samples_per_env" in metadata:
            # Build a per-sample env_id index from the metadata ordering
            sample_to_env: list[str] = []
            for env_id in sorted(metadata["samples_per_env"].keys()):
                count = metadata["samples_per_env"][env_id]
                sample_to_env.extend([env_id] * count)

            for i, sample in enumerate(data):
                env_id = sample_to_env[i] if i < len(sample_to_env) else None
                if env_id is None or env_id in allowed:
                    self._offline.append(self._unpack_legacy_sample(sample))
        else:
            # No metadata — keep all samples (caller is responsible for
            # pre-filtering)
            for sample in data:
                self._offline.append(self._unpack_legacy_sample(sample))

        if len(self._offline) > self._capacity:
            self._offline = self._offline[: self._capacity]
        self._invalidate_cache()

    @staticmethod
    def _unpack_legacy_sample(
        sample: tuple,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert a legacy ``((local, global), action_seq)`` sample.

        Args:
            sample: Tuple of ``(state, action_seq)`` where state is
                ``(local_crop, global_map)``.

        Returns:
            ``(local [9,9], global [21,79], actions [seq_len])`` as
            numpy int16/int64 arrays.
        """
        (local, glb), action_seq = sample
        return (
            np.asarray(local, dtype=np.int16),
            np.asarray(glb, dtype=np.int16),
            np.asarray(action_seq, dtype=np.int64),
        )

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
