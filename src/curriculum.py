"""Dynamic environment curriculum and efficiency filter.

Tracks per-environment
win rates in a rolling window and uses bucket-based sampling weights to
focus training on environments where the model is struggling.
"""

from __future__ import annotations

import random
from collections import deque


class DynamicCurriculum:
    """Rolling-window curriculum with bucket-based sampling weights.

    Each environment maintains a deque of recent win/loss outcomes.
    Sampling probability is inversely proportional to performance:
    environments with low win rates are sampled more often.

    Args:
        env_ids: List of environment IDs to track.
        queue_size: Rolling window size per environment.
    """

    # Bucket thresholds and weights
    _LOW_THRESHOLD = 0.15
    _HIGH_THRESHOLD = 0.85
    _WEIGHT_LOW = 0.2
    _WEIGHT_MID = 1.0
    _WEIGHT_HIGH = 0.1

    def __init__(
        self,
        env_ids: list[str],
        queue_size: int = 100,
        preseed: bool = True,
    ) -> None:
        self._env_ids = list(env_ids)
        self._queue_size = queue_size
        self._queues: dict[str, deque[bool]] = {}
        for eid in self._env_ids:
            q: deque[bool] = deque(maxlen=queue_size)
            if preseed:
                # 50/50 prior for uniform early sampling
                for _ in range(50):
                    q.append(True)
                for _ in range(50):
                    q.append(False)
            self._queues[eid] = q

    def update(self, env_id: str, won: bool) -> None:
        """Record an episode outcome.

        Args:
            env_id: Environment ID.
            won: Whether the episode was won.
        """
        if env_id not in self._queues:
            self._queues[env_id] = deque(maxlen=self._queue_size)
        self._queues[env_id].append(won)

    def win_rate(self, env_id: str) -> float:
        """Rolling win rate for an environment.

        Args:
            env_id: Environment ID.

        Returns:
            Win rate in ``[0, 1]``. Default 0.5 if empty.
        """
        q = self._queues.get(env_id)
        if q is None or len(q) == 0:
            return 0.5
        return sum(q) / len(q)

    def sample_env(self) -> str:
        """Sample an environment ID using bucket-weighted probabilities.

        Returns:
            Sampled environment ID.
        """
        weights: list[float] = []
        for eid in self._env_ids:
            w = self.win_rate(eid)
            if w < self._LOW_THRESHOLD:
                weights.append(self._WEIGHT_LOW)
            elif w > self._HIGH_THRESHOLD:
                weights.append(self._WEIGHT_HIGH)
            else:
                weights.append(self._WEIGHT_MID)
        return random.choices(self._env_ids, weights=weights, k=1)[0]

    def state_dict(self) -> dict:
        """Serialise curriculum state.

        Returns:
            Dict with ``env_ids``, ``queue_size``, and per-env queues.
        """
        return {
            "env_ids": self._env_ids,
            "queue_size": self._queue_size,
            "queues": {eid: list(q) for eid, q in self._queues.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore curriculum state.

        Args:
            sd: State dict from ``state_dict()``.
        """
        self._queue_size = sd.get("queue_size", self._queue_size)
        for eid, items in sd.get("queues", {}).items():
            q: deque[bool] = deque(maxlen=self._queue_size)
            q.extend(items)
            self._queues[eid] = q


def efficiency_filter(
    model_won: bool,
    model_steps: int,
    oracle_steps: int,
    multiplier: float = 1.5,
) -> bool:
    """Decide whether to add oracle trajectory to the buffer.

    Returns ``True`` (add oracle data) when the model either failed
    or was substantially less efficient than the oracle.

    Args:
        model_won: Whether the model solved the episode.
        model_steps: Steps the model took.
        oracle_steps: Steps the oracle took.
        multiplier: Efficiency threshold multiplier.

    Returns:
        ``True`` if oracle data should be added to the buffer.
    """
    if not model_won:
        return True
    return model_steps > multiplier * oracle_steps
