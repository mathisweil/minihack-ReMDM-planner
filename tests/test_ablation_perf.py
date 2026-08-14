"""Regressions for the RL fine-tuning speed pass.

Each change here is only admissible if it leaves the ablation's output
untouched, so the equivalence is pinned rather than asserted in a commit
message: collection recycles environments instead of rebuilding them,
the int16 transfer widens losslessly, the reused eval model carries
exactly the weights the per-iteration deepcopy produced, and the fused
health metrics equal the Python loop they replace.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from experiments.rl_finetuning.ablations.training import (
    _collect_training_data_gpu,
    _extract_windows,
)
from src.envs import minihack_env
from src.envs.minihack_env import close_env_pool
from src.models.denoiser import ModelEMA, make_model
from tests.conftest import TINY_ENV, TINY_OVERRIDES, requires_minihack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_CONFIGS = PROJECT_ROOT / "experiments" / "rl_finetuning" / "configs"

# NLE glyph IDs index an Embedding(6000, ...), and the two sentinels sit
# just past the action space. Everything the buffers can hold is inside
# int16, which is what makes the device-side widening lossless.
GLYPH_MAX = 5999


def _episode(n_steps: int, seed: int = 0) -> dict:
    """A collection-shaped episode dict with int16 glyph buffers."""
    rng = np.random.default_rng(seed)
    return {
        "local": rng.integers(0, GLYPH_MAX, (n_steps, 9, 9), dtype=np.int16),
        "global": rng.integers(0, GLYPH_MAX, (n_steps, 21, 79), dtype=np.int16),
        "actions": rng.integers(0, 12, (n_steps,), dtype=np.int64),
        "total_reward": 1.25,
    }


# ── Int16 across the transfer, widened on the device ────────


def test_extract_windows_keeps_the_buffer_dtype():
    """Windows leave ``_extract_windows`` as int16, not int64."""
    lo, go, x0, ret = _extract_windows(_episode(40), seq_len=8, pad_token=13)

    assert lo.dtype == torch.int16
    assert go.dtype == torch.int16
    # Actions index the model's action embedding directly and stay int64.
    assert x0.dtype == torch.int64
    assert ret == 1.25


def test_device_side_widening_equals_the_host_side_cast():
    """The reordered widening returns the values the old order did."""
    ep = _episode(40, seed=1)
    lo, go, x0, _ = _extract_windows(ep, seq_len=8, pad_token=13)

    # The unoptimised form: cast on the host, then move.
    host_local = torch.from_numpy(ep["local"]).long()
    host_global = torch.from_numpy(ep["global"]).long()
    host_lo, host_go, host_x0, _ = _extract_windows(
        {**ep, "local": host_local.numpy(), "global": host_global.numpy()},
        seq_len=8,
        pad_token=13,
    )

    assert torch.equal(lo.long(), host_lo)
    assert torch.equal(go.long(), host_go)
    assert torch.equal(x0, host_x0)


def test_short_episode_padding_survives_the_dtype_change():
    """Episodes shorter than the horizon still pad to one window."""
    lo, go, x0, _ = _extract_windows(_episode(3), seq_len=8, pad_token=13)

    assert lo.shape == (1, 9, 9)
    assert go.shape == (1, 21, 79)
    assert x0.shape == (1, 8)
    assert (x0[0, 3:] == 13).all(), "tail should be PAD"
    assert lo.dtype == torch.int16


def test_empty_episode_returns_empty_int16_windows():
    """The T == 0 guard keeps the dtype contract."""
    lo, go, x0, _ = _extract_windows(_episode(0), seq_len=8, pad_token=13)

    assert lo.shape[0] == 0
    assert lo.dtype == torch.int16
    assert go.dtype == torch.int16
    assert x0.dtype == torch.int64


def test_int16_to_int64_is_exact_over_the_whole_glyph_range():
    """No glyph ID the buffers can hold is altered by the widening."""
    values = np.arange(0, GLYPH_MAX + 1, dtype=np.int16)

    widened = torch.from_numpy(values).long()

    assert torch.equal(widened, torch.arange(0, GLYPH_MAX + 1, dtype=torch.int64))


# ── One eval model, refreshed in place ──────────────────────


@pytest.fixture
def tiny_model_cfg() -> SimpleNamespace:
    """Toy denoiser config, CPU-sized."""
    merged: dict = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "defaults.yaml").read_text()
    )
    merged.update(TINY_OVERRIDES)
    return SimpleNamespace(**merged)


def test_refreshed_eval_model_matches_a_fresh_deepcopy(tiny_model_cfg):
    """``apply_to`` on a kept model reproduces ``make_eval_model``."""
    torch.manual_seed(0)
    raw = make_model(tiny_model_cfg)
    ema = ModelEMA(raw, decay=0.9)

    persistent = ema.make_eval_model(raw)

    for _ in range(5):
        with torch.no_grad():
            for p in raw.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        ema.update(raw)

        ema.apply_to(persistent)
        persistent.eval()
        fresh = ema.make_eval_model(raw)

        fresh_params = dict(fresh.named_parameters())
        for name, param in persistent.named_parameters():
            assert torch.equal(param, fresh_params[name]), name
        assert not persistent.training


def test_refreshed_eval_model_does_not_track_the_training_weights(tiny_model_cfg):
    """The kept model holds EMA weights, not the live ones."""
    torch.manual_seed(0)
    raw = make_model(tiny_model_cfg)
    ema = ModelEMA(raw, decay=0.9)
    persistent = ema.make_eval_model(raw)

    with torch.no_grad():
        for p in raw.parameters():
            p.add_(1.0)
    ema.update(raw)
    ema.apply_to(persistent)

    raw_params = dict(raw.named_parameters())
    differs = [
        n for n, p in persistent.named_parameters() if not torch.equal(p, raw_params[n])
    ]
    assert differs, "EMA weights should lag the live weights"


def test_the_denoiser_has_no_buffers_to_go_stale(tiny_model_cfg):
    """Reuse is only safe because there is no non-parameter state.

    ``apply_to`` refreshes named parameters and nothing else, so a kept
    eval model would drift from a fresh deepcopy the moment the denoiser
    gained a running statistic.
    """
    raw = make_model(tiny_model_cfg)

    assert list(raw.named_buffers()) == []


# ── Fused health metrics ────────────────────────────────────


def test_fused_param_norm_matches_the_per_tensor_loop(tiny_model_cfg):
    """``_foreach_norm`` reproduces the sum-of-squares the loop computed."""
    torch.manual_seed(0)
    raw = make_model(tiny_model_cfg)
    init = {k: v.clone() for k, v in raw.state_dict().items() if v.is_floating_point()}
    with torch.no_grad():
        for p in raw.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    loop_norm = sum(p.data.norm(2).item() ** 2 for p in raw.parameters()) ** 0.5
    loop_drift = (
        sum(
            (p.data - init[n]).norm(2).item() ** 2
            for n, p in raw.named_parameters()
            if n in init
        )
        ** 0.5
    )

    params = [p.data for p in raw.parameters()]
    pairs = [(p.data, init[n]) for n, p in raw.named_parameters() if n in init]
    fused_norm = torch.linalg.vector_norm(
        torch.stack(torch._foreach_norm(params))
    ).item()
    fused_drift = torch.linalg.vector_norm(
        torch.stack(
            torch._foreach_norm(
                torch._foreach_sub([a for a, _ in pairs], [b for _, b in pairs])
            )
        )
    ).item()

    assert fused_norm == pytest.approx(loop_norm, rel=1e-6)
    assert fused_drift == pytest.approx(loop_drift, rel=1e-6)


# ── Collection borrows from the pool ────────────────────────


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
