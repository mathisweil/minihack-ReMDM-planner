"""In-repo baseline spec tests (step 8).

Sources: research/spec-training.md §6.1 (SB3 PPO/A2C/DQN/PPO-RNN per
Schulman 2017 / Mnih 2016 / Mnih 2015; Decision Transformer per Chen
2021: return-conditioned causal (R, s, a) sequence modelling). The
craftax twin file covers the PPO-expert supervision side (PARITY
"Supervision"/"In-repo baselines": the SB3/DT baselines are
minihack-only).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.planners.baselines import (
    _build_sb3_model,
    _DecisionTransformer,
    _make_sb3_env_fn,
)
from tests.conftest import TINY_ENV, requires_minihack


def _dt_batch(b=2, t=4, n_actions=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "returns_to_go": torch.rand(b, t, 1, generator=g),
        "local_obs": torch.randint(0, 100, (b, t, 1, 9, 9), generator=g),
        "global_obs": torch.randint(0, 100, (b, t, 1, 21, 79), generator=g),
        "actions": torch.randint(0, n_actions, (b, t), generator=g),
        "timesteps": torch.arange(t).repeat(b, 1),
    }


def test_decision_transformer_is_causal_over_interleaved_tokens():
    """DT logits at step t may depend only on (R, s) up to t and actions
    before t (Chen 2021 §3: causal masking over the interleaved
    (R_0, s_0, a_0, ...) sequence; the state token at step t sits at
    position 3t+1, so a_t and everything later is masked out).

    Method: perturb actions[:, 2:] and returns_to_go[:, 3:]; logits for
    steps 0..2 must be bit-identical, and the perturbed suffix must
    change some later logit (otherwise the test proves nothing).
    """
    dt = _DecisionTransformer(
        n_actions=8, embed_dim=32, n_heads=2, n_layers=1, context_len=4
    ).eval()
    batch = _dt_batch()
    with torch.no_grad():
        base = dt(**batch)
        perturbed = {**batch, "actions": batch["actions"].clone()}
        perturbed["actions"][:, 2:] = (perturbed["actions"][:, 2:] + 3) % 8
        perturbed["returns_to_go"] = batch["returns_to_go"].clone()
        perturbed["returns_to_go"][:, 3:] += 5.0
        alt = dt(**perturbed)
    assert torch.equal(base[:, :3], alt[:, :3]), "future tokens leaked into the past"
    assert not torch.equal(base[:, 3], alt[:, 3]), "perturbation had no effect at all"


def test_decision_transformer_one_step_training_reduces_the_ce_loss():
    """One-step training sanity per Chen 2021's objective: cross-entropy
    of action logits at state positions against the taken actions;
    30 Adam steps on a fixed tiny batch must reduce the loss."""
    torch.manual_seed(0)
    dt = _DecisionTransformer(
        n_actions=8, embed_dim=32, n_heads=2, n_layers=1, context_len=4
    ).train()
    batch = _dt_batch(seed=1)
    opt = torch.optim.Adam(dt.parameters(), lr=1e-3)

    def _loss():
        logits = dt(**batch)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, 8), batch["actions"].reshape(-1)
        )

    loss0 = float(_loss())
    for _ in range(30):
        opt.zero_grad()
        loss = _loss()
        loss.backward()
        opt.step()
    assert float(_loss()) < loss0


@requires_minihack
@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ppo", "a2c", "dqn", "ppo-rnn"])
def test_sb3_baselines_construct_and_predict(algo, tiny_cfg, tmp_path):
    """Each documented baseline algorithm constructs the SB3 class the
    spec names (PPO / A2C / DQN / RecurrentPPO, spec-training §6.1)
    over the MiniHack dict observation space, and its untrained policy
    emits a legal discrete action."""
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    tiny_cfg.baselines_dqn_buffer_size = 100
    venv = DummyVecEnv([_make_sb3_env_fn(TINY_ENV, tiny_cfg, str(tmp_path))])
    try:
        model = _build_sb3_model(algo, venv, tiny_cfg, seed=0, tb_log_dir=str(tmp_path))
        expected_cls = {
            "ppo": PPO, "a2c": A2C, "dqn": DQN, "ppo-rnn": RecurrentPPO
        }[algo]
        assert isinstance(model, expected_cls)
        obs = venv.reset()
        action, _ = model.predict(obs, deterministic=True)
        n_actions = venv.action_space.n
        assert 0 <= int(np.asarray(action).ravel()[0]) < n_actions
    finally:
        venv.close()
