"""
Evaluator: torch.no_grad() testing loops for ID_ENVS and OOD_ENVS.
"""

import numpy as np

from config import ID_ENVS, OOD_ENVS, EVAL_EPISODES_PER_ENV, HYPERPARAMS


class Evaluator:
    """Run evaluation on ID and OOD envs. Does not touch the buffer."""

    def __init__(self, model, device=None, diffusion_steps=5, max_steps=200):
        self.model = model
        self.device = device or HYPERPARAMS["device"]
        self.diffusion_steps = diffusion_steps
        self.max_steps = max_steps

    def evaluate_envs(self, env_ids, episodes_per_env=None):
        """Run episodes_per_env without Oracle. Returns mean win rate."""
        from utils import run_episode

        episodes_per_env = episodes_per_env or EVAL_EPISODES_PER_ENV
        all_wins = []
        for env_id in env_ids:
            for ep in range(episodes_per_env):
                seed = 42 + hash((env_id, ep)) % (2**31)
                r = run_episode(
                    self.model,
                    env_id=env_id,
                    diffusion_steps=self.diffusion_steps,
                    max_steps=self.max_steps,
                    seed=seed,
                )
                all_wins.append(1 if r["won"] else 0)
        return np.mean(all_wins) if all_wins else 0.0

    def evaluate_id(self):
        """Run ID evaluation only. Returns Eval/ID_WinRate."""
        import torch
        self.model.eval()
        with torch.no_grad():
            return self.evaluate_envs(ID_ENVS)

    def evaluate_ood(self):
        """Run OOD evaluation only. Returns Eval/OOD_WinRate."""
        import torch
        self.model.eval()
        with torch.no_grad():
            return self.evaluate_envs(OOD_ENVS)

    def evaluate(self):
        """Run ID and OOD evaluation. Returns dict for logging."""
        return {
            "Eval/ID_WinRate": self.evaluate_id(),
            "Eval/OOD_WinRate": self.evaluate_ood(),
        }
