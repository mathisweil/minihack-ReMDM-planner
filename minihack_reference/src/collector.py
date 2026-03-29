"""
DataCollector: Model rollout, Oracle rollout, and DAgger & Efficiency Filter.
"""

from config import HYPERPARAMS
from curriculum import efficiency_filter
from env import AdvancedObservationEnv


def _collect_oracle_trajectory(env_id, seed, max_steps=200):
    """Oracle plays until win. Returns (states, actions) or None."""
    env = AdvancedObservationEnv([env_id])
    obs, _ = env.reset(seed=seed)
    states, actions = [], []
    for _ in range(max_steps):
        act = env.get_oracle_action(env.last_raw_obs)
        states.append(obs)
        obs, r, term, trunc, info = env.step(act)
        actions.append(act)
        if term or trunc:
            env.current_env.close()
            return (states, actions) if info.get("won", False) else None
    env.current_env.close()
    return None


class DataCollector:
    """Collects episodes: model rollout, Oracle rollout, DAgger & Efficiency Filter."""

    def __init__(self, model, device=None, diffusion_steps=5, replan_every=16, max_steps=200):
        self.model = model
        self.device = device or HYPERPARAMS["device"]
        self.diffusion_steps = diffusion_steps
        self.replan_every = replan_every
        self.max_steps = max_steps

    def collect_episode(self, env_id, seed):
        """Run one episode: model rollout then Oracle on same map.

        Returns: (trajectory, should_add, model_won) or (None, False, model_won) if Oracle failed.
        """
        from utils import run_model_episode

        env = AdvancedObservationEnv([env_id])
        model_result = run_model_episode(
            self.model,
            env,
            diffusion_steps=self.diffusion_steps,
            replan_every=self.replan_every,
            max_steps=self.max_steps,
            device=self.device,
            seed=seed,
        )
        model_won = model_result["won"]
        model_steps = model_result["steps"]
        env.current_env.close()

        oracle_traj = _collect_oracle_trajectory(env_id, seed, self.max_steps)
        if oracle_traj is None:
            return None, False, model_won

        oracle_states, oracle_actions = oracle_traj
        oracle_steps = len(oracle_actions)

        should_add = efficiency_filter(model_won, model_steps, oracle_steps)
        return oracle_traj, should_add, model_won
