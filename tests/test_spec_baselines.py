"""In-repo baseline spec tests (step 8).

Sources: research/spec-training.md §6.1 (SB3 PPO/A2C/DQN/PPO-RNN per
Schulman 2017 / Mnih 2016 / Mnih 2015; Decision Transformer per Chen
2021: return-conditioned causal (R, s, a) sequence modelling). The
craftax twin file covers the PPO-expert supervision side (PARITY
"Supervision"/"In-repo baselines": the SB3/DT baselines are
minihack-only).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.planners.baselines import (
    _build_sb3_model,
    _DecisionTransformer,
    _make_sb3_env_fn,
    quiet_multiprocessing_tempdir_teardown,
    run_baselines,
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


# ---------------------------------------------------------------------------
# `--mode baselines` teardown noise (step-11 finding U4)
# ---------------------------------------------------------------------------

# Creates multiprocessing's own temp dir, registers its exit-time rmtree, then
# makes that rmtree fail the way a shared filesystem makes it fail. The chmod
# stands in for NFS leaving `.nfsXXXX` behind: both make the removal raise an
# OSError that is not FileNotFoundError, which is exactly what the finalizer
# re-raises and `_run_finalizers` prints.
_TEARDOWN_CHILD = """
import os, pathlib, sys, multiprocessing.util as mpu
if os.environ["QUIET"] == "1":
    sys.path.insert(0, {root!r})
    from src.planners.baselines import quiet_multiprocessing_tempdir_teardown
    quiet_multiprocessing_tempdir_teardown()
tempdir = pathlib.Path(mpu.get_temp_dir())
(tempdir / "held").write_text("x")
os.chmod(tempdir, 0o500)
print(tempdir)
"""


@pytest.mark.slow
@pytest.mark.parametrize("quiet", ["0", "1"])
def test_the_baselines_teardown_prints_no_traceback(quiet, tmp_path):
    """A temp dir multiprocessing cannot remove is silent, not a traceback.

    `--mode baselines` runs SubprocVecEnv, so multiprocessing creates a
    `pymp-*` directory and registers an exit-time rmtree of it. That rmtree
    tolerates FileNotFoundError and re-raises everything else, so on the
    shared filesystem the run ended with a traceback on stderr -- exit code
    0, every artefact written, nothing the reader can act on (U4).

    Both directions are asserted: without the call the traceback is there,
    with it the traceback is gone. Neither changes the exit code, and neither
    removes the directory, because the point is the reporting rather than the
    removal.
    """
    root = str(Path(__file__).resolve().parents[1])
    child = tmp_path / "child.py"
    child.write_text(_TEARDOWN_CHILD.format(root=root))

    result = subprocess.run(
        [sys.executable, str(child)],
        capture_output=True,
        text=True,
        env={**os.environ, "QUIET": quiet, "TMPDIR": str(tmp_path)},
        timeout=120,
        check=False,
    )
    leftover = Path(result.stdout.strip())
    leftover.chmod(0o700)  # so tmp_path teardown can remove it

    assert result.returncode == 0
    assert ("Traceback" in result.stderr) == (quiet == "0"), result.stderr


def test_the_teardown_guard_is_idempotent_and_tolerates_any_removal_error():
    """Installing twice wraps once, and the wrapper swallows the error.

    Called from `run_baselines`, which may run several algorithms and seeds
    in one process; wrapping a wrapper each time would nest indefinitely.
    """
    import multiprocessing.util as mp_util

    original = mp_util._remove_temp_dir
    try:
        quiet_multiprocessing_tempdir_teardown()
        installed = mp_util._remove_temp_dir
        quiet_multiprocessing_tempdir_teardown()
        assert mp_util._remove_temp_dir is installed
        assert installed is not original

        def _always_fails(path, **kwargs):
            raise OSError(39, "Directory not empty", path)

        installed(_always_fails, "/nonexistent/pymp-test")
    finally:
        mp_util._remove_temp_dir = original


def test_run_baselines_installs_the_teardown_guard_before_any_subprocess():
    """Source-anchored: multiprocessing captures the callback when it first
    creates its temp dir, so a guard installed after the first SubprocVecEnv
    is ignored. `run_baselines` must call it before doing anything else that
    can spawn."""
    import inspect

    src = inspect.getsource(run_baselines)
    assert "quiet_multiprocessing_tempdir_teardown()" in src
    assert src.index("quiet_multiprocessing_tempdir_teardown()") < src.index(
        "_resolve_output_dir"
    )
