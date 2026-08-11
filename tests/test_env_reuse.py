"""Env-pool and observation-key regressions.

Collection speed rests on two claims: a recycled environment behaves
exactly like a freshly constructed one, and no code path reads the
``pixel`` observation. Both are load-bearing for the training recipe, so
they are pinned here rather than left to inspection.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.envs.minihack_env import (
    AdvancedObservationEnv,
    acquire_env,
    close_env_pool,
    collect_oracle_trajectory,
    make_env,
    release_env,
)
from tests.conftest import TINY_ENV, requires_minihack

pytestmark = requires_minihack

SEEDS = (7, 42, 1234)


def _oracle_rollout(env, seed: int, max_steps: int = 200) -> tuple:
    """Deterministic signature of one oracle episode."""
    (local, glb), _ = env.reset(seed=seed)
    digest = hashlib.sha256()
    digest.update(local.tobytes())
    digest.update(glb.tobytes())
    actions: list[int] = []
    rewards: list[float] = []
    for _ in range(max_steps):
        action = env.get_oracle_action(env.last_raw_obs)
        actions.append(int(action))
        (local, glb), reward, terminated, truncated, _ = env.step(action)
        rewards.append(round(float(reward), 6))
        digest.update(local.tobytes())
        digest.update(glb.tobytes())
        if terminated or truncated:
            break
    return actions, digest.hexdigest(), rewards


@pytest.fixture(autouse=True)
def _clean_pool():
    """Never let one test's pooled envs reach another."""
    close_env_pool()
    yield
    close_env_pool()


def test_pixel_observation_is_not_requested():
    """'pixel' costs ~16x per step and nothing reads it."""
    env = make_env(TINY_ENV, None, _cfg())
    try:
        keys = env._inner.unwrapped._observation_keys
        assert "pixel" not in keys, (
            f"pixel re-added to obs_keys: {keys}. It renders the full RGB "
            "screen every step and no code path reads it."
        )
    finally:
        env.close()


def test_recycled_env_matches_fresh_env():
    """A pooled env re-seeded per episode is trajectory identical."""
    cfg = _cfg()

    fresh = []
    for seed in SEEDS:
        np.random.seed(0)
        env = make_env(TINY_ENV, None, cfg)
        try:
            fresh.append(_oracle_rollout(env, seed))
        finally:
            env.close()

    recycled = []
    for seed in SEEDS:
        np.random.seed(0)
        env = acquire_env(TINY_ENV, None, cfg)
        recycled.append(_oracle_rollout(env, seed))
        release_env(env)

    for seed, want, got in zip(SEEDS, fresh, recycled, strict=True):
        assert want[0] == got[0], f"actions diverged at seed {seed}"
        assert want[1] == got[1], f"observations diverged at seed {seed}"
        assert want[2] == got[2], f"rewards diverged at seed {seed}"


def test_pool_recycles_instead_of_reconstructing():
    """The second acquire of an env ID returns the released instance."""
    cfg = _cfg()
    first = acquire_env(TINY_ENV, None, cfg)
    first.reset(seed=0)
    release_env(first)
    second = acquire_env(TINY_ENV, None, cfg)
    try:
        assert second is first, "pool reconstructed instead of recycling"
    finally:
        release_env(second)


def test_pool_is_bounded():
    """Idle envs are capped so eval churn cannot exhaust the box."""
    from src.envs.minihack_env import _EnvPool

    cfg = _cfg()
    pool = _EnvPool(max_idle=2)
    envs = [AdvancedObservationEnv(TINY_ENV, None, cfg) for _ in range(4)]
    for env in envs:
        pool.release(env)
    assert pool.n_idle == 2
    pool.close_all()
    assert pool.n_idle == 0


def test_unshaped_reward_preserves_transitions():
    """shaped_reward=False changes only the reward scalar."""
    cfg = _cfg()

    np.random.seed(0)
    shaped_env = acquire_env(TINY_ENV, None, cfg, shaped_reward=True)
    shaped = _oracle_rollout(shaped_env, 7)
    release_env(shaped_env)

    np.random.seed(0)
    plain_env = acquire_env(TINY_ENV, None, cfg, shaped_reward=False)
    plain = _oracle_rollout(plain_env, 7)
    release_env(plain_env)

    assert shaped[0] == plain[0], "actions changed with shaping disabled"
    assert shaped[1] == plain[1], "observations changed with shaping disabled"


def test_oracle_trajectory_returns_env_to_pool():
    """The hot collection path must not leak or reconstruct envs."""
    from src.envs.minihack_env import _POOL

    cfg = _cfg()
    assert collect_oracle_trajectory(TINY_ENV, seed=3, cfg=cfg) is not None
    assert _POOL.n_idle == 1
    assert collect_oracle_trajectory(TINY_ENV, seed=4, cfg=cfg) is not None
    assert _POOL.n_idle == 1, "each call should recycle the same instance"


def _cfg():
    from src.config import load_config

    cfg = load_config("configs/defaults.yaml")
    cfg.device = "cpu"
    return cfg
