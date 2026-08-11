"""Regressions for the RL fine-tuning speed pass (PERF-X1 onwards).

Each change here is only admissible if it leaves the ablation's output
untouched, so the equivalence is pinned rather than asserted in a commit
message.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from experiments.rl_finetuning.ablations.training import (
    _collect_training_data_gpu,
)
from src.envs import minihack_env
from src.envs.minihack_env import close_env_pool
from src.models.denoiser import make_model
from tests.conftest import TINY_ENV, TINY_OVERRIDES, requires_minihack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_CONFIGS = PROJECT_ROOT / "experiments" / "rl_finetuning" / "configs"


# ── PERF-X1: collection borrows from the pool ────────────────────────


@pytest.fixture
def collect_cfg() -> SimpleNamespace:
    """Ablation config shrunk enough to collect on the CPU."""
    from src.diffusion.schedules import get_schedule

    merged: dict = {}
    for path in (
        PROJECT_ROOT / "configs" / "defaults.yaml",
        ABLATION_CONFIGS / "ablations_default.yaml",
        ABLATION_CONFIGS / "ablations_fast.yaml",
    ):
        merged.update(yaml.safe_load(path.read_text()) or {})
    merged.update(TINY_OVERRIDES)
    merged.update({"id_envs": [TINY_ENV], "episodes_per_iter": 2})
    cfg = SimpleNamespace(**merged)
    cfg._schedule_fn = get_schedule(cfg.noise_schedule)
    cfg._current_iter = 1
    return cfg


@pytest.fixture
def _clean_pool():
    """Never let one test's pooled envs reach another."""
    close_env_pool()
    yield
    close_env_pool()


@requires_minihack
def test_collection_recycles_envs_between_iterations(
    collect_cfg, _clean_pool, monkeypatch
):
    """The second collection constructs nothing.

    Env construction was measured at 27.7 ms against 0.13 ms for an env
    step, and the suite collects 30 episodes per iteration for 500
    iterations, so this is the whole speed pass in one assertion.
    """
    built: list[int] = []
    real_init = minihack_env.AdvancedObservationEnv.__init__

    def counting_init(self, *a, **kw):
        built.append(1)
        real_init(self, *a, **kw)

    monkeypatch.setattr(
        minihack_env.AdvancedObservationEnv, "__init__", counting_init
    )

    torch.manual_seed(0)
    model = make_model(collect_cfg).eval()
    device = torch.device("cpu")

    _collect_training_data_gpu(model, collect_cfg, device, n_episodes=2)
    first_round = len(built)
    _collect_training_data_gpu(model, collect_cfg, device, n_episodes=2)

    assert first_round == 2, "cold pool builds one env per episode"
    assert len(built) == first_round, "warm pool must build none"


@requires_minihack
def test_collection_discards_envs_on_failure(collect_cfg, _clean_pool, monkeypatch):
    """A broken env is closed, never recycled into the next iteration."""
    closed: list[int] = []
    released: list[int] = []
    monkeypatch.setattr(
        "experiments.rl_finetuning.ablations.training.discard_env",
        lambda env: closed.append(1),
    )
    monkeypatch.setattr(
        "experiments.rl_finetuning.ablations.training.release_env",
        lambda env: released.append(1),
    )

    def boom(*a, **kw):
        raise RuntimeError("sampling blew up")

    monkeypatch.setattr(
        "experiments.rl_finetuning.ablations.training.remdm_sample", boom
    )

    torch.manual_seed(0)
    model = make_model(collect_cfg).eval()

    with pytest.raises(RuntimeError, match="sampling blew up"):
        _collect_training_data_gpu(
            model, collect_cfg, torch.device("cpu"), n_episodes=2
        )

    assert closed == [1, 1], "both envs discarded"
    assert released == [], "nothing recycled after a failure"
