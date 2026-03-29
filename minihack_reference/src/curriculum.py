"""
Dynamic Curriculum for environment sampling based on rolling win rates.
"""

import numpy as np
from collections import deque

from config import ID_ENVS, CURRICULUM_QUEUE_SIZE, EFFICIENCY_MULTIPLIER


class DynamicCurriculum:
    """100-episode rolling win-rate queue per env. Bucket-based sampling weights."""

    def __init__(self, env_ids=None, queue_size=None):
        self.env_ids = list(env_ids or ID_ENVS)
        self.queue_size = queue_size or CURRICULUM_QUEUE_SIZE
        self._queues = {e: deque(maxlen=self.queue_size) for e in self.env_ids}

    def update(self, env_id, won):
        """Record outcome for env_id. won: bool."""
        if env_id in self._queues:
            self._queues[env_id].append(1 if won else 0)

    def get_win_rate(self, env_id):
        """Rolling win rate for env_id. Returns float 0.0 to 1.0."""
        q = self._queues.get(env_id, deque())
        if len(q) == 0:
            return 0.5  # Neutral prior
        return sum(q) / len(q)

    def _raw_weight(self, w):
        """Bucket method: W < 0.15 -> 0.2, 0.15<=W<=0.85 -> 1.0, W > 0.85 -> 0.1."""
        if w < 0.15:
            return 0.2
        if w <= 0.85:
            return 1.0
        return 0.1

    def sample_next_env(self):
        """Sample next environment using curriculum weights.

        Returns: env_id (str)
        """
        weights = []
        for env_id in self.env_ids:
            w = self.get_win_rate(env_id)
            raw = self._raw_weight(w)
            weights.append(raw)
        total = sum(weights)
        if total <= 0:
            probs = np.ones(len(self.env_ids)) / len(self.env_ids)
        else:
            probs = np.array(weights, dtype=np.float64) / total
        idx = np.random.choice(len(self.env_ids), p=probs)
        return self.env_ids[idx]


def efficiency_filter(model_won, model_steps, oracle_steps):
    """Return True if Oracle trajectory should be added to buffer.

    Condition A (Model Failed): model_won == False -> add
    Condition B (Model Inefficient): model_won and model_steps > oracle_steps * 1.5 -> add
    Condition C (Model Efficient): model_won and model_steps <= oracle_steps * 1.5 -> do not add
    """
    if not model_won:
        return True
    if model_steps > (oracle_steps * EFFICIENCY_MULTIPLIER):
        return True
    return False
