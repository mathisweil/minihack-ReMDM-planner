"""Spec tests for the DAgger training pipeline (step 8).

Sources: research/spec-training.md §1.6-1.9 (demonstration-variant
DAgger, curriculum, warm start, efficiency filter) -- §1.8 records the
--no-warm-start conflict this file pins, closed at 287a22f. Expected
values come from those loci or in-docstring derivations, never from
current repo output.

The DAgger variant is repo-specific (PARITY "Method pipeline"), so this
file has no craftax twin; craftax's DAgger internals are inline in
jitted closures and are recorded as step-9 seams in the step-8 report.
"""

from __future__ import annotations

import random
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.buffer import ReplayBuffer
from src.curriculum import DynamicCurriculum, efficiency_filter
from src.planners.inference import Evaluator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Curriculum bucket weights (spec-training §1.7)
# ---------------------------------------------------------------------------


def _curriculum_with_rates(rates: dict[str, tuple[int, int]]) -> DynamicCurriculum:
    """Curriculum with exact win/loss counts per env (no preseed)."""
    cur = DynamicCurriculum(list(rates), queue_size=100, preseed=False)
    for eid, (wins, losses) in rates.items():
        for _ in range(wins):
            cur.update(eid, True)
        for _ in range(losses):
            cur.update(eid, False)
    return cur


def test_curriculum_bucket_weights_interior():
    """Sampling weights are 0.2 / 1.0 / 0.1 for win rates in [0,0.15),
    [0.15,0.85) and above (spec-training §1.7; README:343-344,502).

    Derivation: win rates 0.10 / 0.50 / 0.90 fall in the three buckets;
    normalised sampling probabilities are [0.2,1.0,0.1]/1.3 =
    [0.15385, 0.76923, 0.07692]. Statistical: 20000 draws; max sigma =
    sqrt(0.769*0.231/20000) = 0.00298; bound 0.015 = 5.0 sigma.
    """
    cur = _curriculum_with_rates({"low": (1, 9), "mid": (5, 5), "high": (9, 1)})
    random.seed(0)
    counts = {"low": 0, "mid": 0, "high": 0}
    n = 20000
    for _ in range(n):
        counts[cur.sample_env()] += 1
    expected = {"low": 0.2 / 1.3, "mid": 1.0 / 1.3, "high": 0.1 / 1.3}
    for eid, p in expected.items():
        assert abs(counts[eid] / n - p) < 0.015, (eid, counts[eid] / n, p)


def test_curriculum_bucket_high_boundary_is_inclusive():
    """A win rate of exactly 0.85 belongs to the [0.85, 1] bucket
    (weight 0.1), per the documented intervals (was step-8 finding
    S8-3: strict '>' put the boundary in the mid bucket).

    Derivation: envs at win rates 0.85 (17/20) and 0.50: documented
    weights [0.1, 1.0] give P(boundary) = 0.1/1.1 = 0.0909; the MID
    misclassification gives 0.5. Statistical: 5000 draws; sigma =
    sqrt(0.0909*0.909/5000) = 0.00407; bound 0.02 = 4.9 sigma, far from
    the 0.41 separation between hypotheses.
    """
    cur = _curriculum_with_rates({"boundary": (17, 3), "mid": (5, 5)})
    assert cur.win_rate("boundary") == pytest.approx(0.85)
    random.seed(0)
    n = 5000
    hits = sum(cur.sample_env() == "boundary" for _ in range(n))
    assert abs(hits / n - 0.1 / 1.1) < 0.02


def test_curriculum_low_boundary_is_exclusive():
    """A win rate of exactly 0.15 belongs to the MID bucket
    ([0,0.15) is right-open, spec-training §1.7).

    Derivation: envs at 0.15 (3/20) and 0.50 both get weight 1.0 ->
    P = 0.5 each. Statistical: 5000 draws; sigma = 0.00707; bound
    0.03 = 4.2 sigma (vs 0.333 if the boundary env were LOW-weighted:
    0.2/1.2 = 0.1667).
    """
    cur = _curriculum_with_rates({"boundary": (3, 17), "mid": (5, 5)})
    assert cur.win_rate("boundary") == pytest.approx(0.15)
    random.seed(0)
    n = 5000
    hits = sum(cur.sample_env() == "boundary" for _ in range(n))
    assert abs(hits / n - 0.5) < 0.03


# ---------------------------------------------------------------------------
# Efficiency filter (spec-training §1.6)
# ---------------------------------------------------------------------------


def test_efficiency_filter_thresholds():
    """Oracle data is added iff the model failed or took more than
    1.5x the oracle's steps (spec-training §1.6; README:364-372).

    The boundary is strict: exactly 1.5x is efficient enough (the
    documented rule is 'took >1.5x oracle steps').
    """
    assert efficiency_filter(model_won=False, model_steps=1, oracle_steps=100)
    assert not efficiency_filter(model_won=True, model_steps=10, oracle_steps=10)
    assert not efficiency_filter(model_won=True, model_steps=15, oracle_steps=10)
    assert efficiency_filter(model_won=True, model_steps=16, oracle_steps=10)
    assert efficiency_filter(model_won=True, model_steps=31, oracle_steps=20, multiplier=1.5)


# ---------------------------------------------------------------------------
# Buffer window semantics (spec-training §1.8; Amendment 6: PAD-padded tails)
# ---------------------------------------------------------------------------


def _traj(actions: list[int]) -> dict:
    t = len(actions)
    return {
        "local": np.zeros((t, 9, 9), dtype=np.int16),
        "global": np.zeros((t, 21, 79), dtype=np.int16),
        "actions": np.asarray(actions, dtype=np.int64),
        "env_id": "MiniHack-Room-Random-5x5-v0",
    }


def test_buffer_pads_episode_tails():
    """A T-step trajectory yields T windows; tail windows are padded
    with PAD to seq_len (spec-training Amendment 6: minihack trains on
    PAD-padded episode tails, unlike craftax's dropped tails).

    Derivation: T=5, seq_len=8, PAD=13: window at start s holds
    actions[s:5] followed by (8-(5-s)) PAD tokens.
    """
    buf = ReplayBuffer(capacity=100, seq_len=8, pad_token=13)
    buf.add(_traj([0, 1, 2, 3, 4]))
    assert len(buf) == 5
    windows = buf._online
    for s, (_, _, acts) in enumerate(windows):
        expected = [0, 1, 2, 3, 4][s:] + [13] * (8 - (5 - s))
        assert acts.tolist() == expected, (s, acts.tolist())


def test_buffer_offline_data_is_pinned_and_online_fifo_evicted():
    """Offline windows are pinned at the front and never evicted;
    online windows FIFO-evict when the total exceeds capacity
    (spec-training §1.8; README:503).

    Derivation: capacity 6, seq_len 2. Offline trajectory of 3 steps ->
    3 windows (actions starting 7). Online adds of 2 x 3-window
    trajectories (starting 0 and 3) exceed capacity by 3: the 3 oldest
    online windows evict, offline stays.
    """
    buf = ReplayBuffer(capacity=6, seq_len=2, pad_token=13)
    buf.load_offline_data(
        {"trajectories": [_traj([7, 8, 9])]},
        allowed_envs=["MiniHack-Room-Random-5x5-v0"],
    )
    assert len(buf._offline) == 3
    buf.add(_traj([0, 1, 2]))
    buf.add(_traj([3, 4, 5]))
    assert len(buf._offline) == 3, "offline windows must never be evicted"
    assert len(buf._online) == 3, "online windows must FIFO-evict to capacity"
    first_actions = [int(a[0]) for _, _, a in buf._online]
    assert first_actions == [3, 4, 5], "eviction must drop the oldest online rows"


# ---------------------------------------------------------------------------
# Evaluator seeding (spec-training Amendment 6; traceability §9)
# ---------------------------------------------------------------------------


def test_evaluator_seeds_are_fixed_and_run_seed_independent(monkeypatch):
    """Evaluation episode seeds are 42 + crc32("{env_id}:{ep}") mod 2^31,
    independent of any run seed (spec-training Amendment 6).

    Captures the seeds handed to the episode runner instead of running
    episodes; re-seeding the global RNGs must not change them.
    """
    captured: list[list[int]] = []

    def _fake_run(self, model, env_id, n_episodes, cfg, device, seeds, **kw):
        captured.append(list(seeds))
        return [
            {"won": False, "total_reward": 0.0, "steps": 1}
            for _ in range(n_episodes)
        ]

    monkeypatch.setattr(Evaluator, "_run_episodes_batched", _fake_run)
    ev = Evaluator()
    cfg = SimpleNamespace()

    class _Dummy:
        def eval(self):
            return self

    for run_seed in (0, 12345):
        random.seed(run_seed)
        np.random.seed(run_seed)
        ev.evaluate(["MiniHack-Room-Random-5x5-v0"], _Dummy(), 3, cfg, "cpu")

    expected = [
        42 + zlib.crc32(f"MiniHack-Room-Random-5x5-v0:{ep}".encode()) % (2**31)
        for ep in range(3)
    ]
    assert captured[0] == expected
    assert captured[1] == expected, "seeds must not depend on the run seed"


# ---------------------------------------------------------------------------
# --no-warm-start (defect §8.9) and smoke side effects (finding N6)
# ---------------------------------------------------------------------------


def test_no_warm_start_disables_oracle_buffer_seeding(monkeypatch):
    """run_dagger(no_warm_start=True) must not seed the buffer with
    oracle trajectories (README:505 defines DAgger warm-start as exactly
    this seeding; was defect §8.9: the seeding ran unconditionally).

    The oracle collector is stubbed to count invocations (returning
    None, so no environments are needed) and Trainer.train is stubbed
    out, leaving only run_dagger's setup path under test.
    """
    from src.config import load_config
    from src.planners import online

    calls = {"n": 0}

    def _fake_collect(env_id, seed, cfg):
        calls["n"] += 1
        return None

    monkeypatch.setattr(online, "collect_oracle_trajectory", _fake_collect)
    monkeypatch.setattr(online.Trainer, "train", lambda self, **kw: None)

    cfg = load_config(
        None,
        {"use_wandb": False, "torch_compile": False, "use_amp": False},
    )
    import tempfile

    # make_run_dir mutates checkpoint_dir; keep the run dir out of the repo
    cfg.checkpoint_dir = tempfile.mkdtemp(prefix="remdm-nws-test-")
    online.run_dagger(cfg, None, True)
    assert calls["n"] == 0, (
        f"warm-start seeding ran {calls['n']} oracle collections despite "
        "no_warm_start=True"
    )

    # Both directions, or the assertion above is satisfied by deleting the
    # seeding entirely (sweep S11-5: with only the negative direction
    # asserted, `if no_warm_start:` -> `if True:` left the suite green).
    # README §DAgger fixes the count: 3 oracle trajectories per ID env.
    calls["n"] = 0
    cfg.checkpoint_dir = tempfile.mkdtemp(prefix="remdm-ws-test-")
    online.run_dagger(cfg, None, False)
    assert calls["n"] == 3 * len(cfg.id_envs), (
        f"warm-start seeding ran {calls['n']} oracle collections, expected "
        f"3 per ID environment = {3 * len(cfg.id_envs)}"
    )


@pytest.mark.slow
def test_smoke_mode_leaves_no_artifacts_in_the_repo():
    """The documented smoke invocation must not persist artefacts into
    the repository tree (parity with craftax smoke; was step-7 finding
    N6: iterN.pth checkpoints landed in the repo checkpoints/ root).

    Runs `main.py --mode smoke` exactly as documented (no overrides)
    from the repo root and compares the checkpoints/ listing before and
    after.
    """
    from tests.conftest import run_cli

    ckpt_root = PROJECT_ROOT / "checkpoints"
    before = set(ckpt_root.iterdir()) if ckpt_root.exists() else set()
    result = run_cli("main.py", "--mode", "smoke", timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    after = set(ckpt_root.iterdir()) if ckpt_root.exists() else set()
    new_entries = after - before
    for entry in new_entries:  # clean up what the defect wrote
        import shutil

        shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink()
    assert not new_entries, f"smoke wrote into the repo: {sorted(map(str, new_entries))}"


# ---------------------------------------------------------------------------
# Budget accounting (spec-training Amendment 6; step-9 seam)
# ---------------------------------------------------------------------------


def test_budget_charges_model_and_oracle_env_steps():
    """total_timesteps is charged with BOTH the model rollout's and the
    seed-matched oracle rollout's env.step() calls (spec-training
    Amendment 6; PARITY 'Data/budget semantics': craftax charges
    learner frames only - a recorded divergence, not a shared rule)."""
    from src.planners.online import budget_increment

    assert budget_increment(30, 70) == 100
    assert budget_increment(0, 55) == 55
    assert budget_increment(42, 0) == 42
