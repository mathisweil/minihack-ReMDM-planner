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
                env_id = (
                    sample_to_env[i] if i < len(sample_to_env) else None
                )
                if env_id is None or env_id in allowed:
                    self._offline.append(self._unpack_legacy_sample(sample))
        else:
            # No metadata — keep all samples (caller is responsible for
            # pre-filtering)
            for sample in data:
                self._offline.append(self._unpack_legacy_sample(sample))

        if len(self._offline) > self._capacity:
            self._offline = self._offline[: self._capacity]

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
