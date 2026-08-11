"""End-to-end smoke tests for the ``experiments/rl_finetuning/`` pipeline.

The RL fine-tuning suite reuses the ``src/`` denoiser but layers its own
config merge, 25-entry ablation registry, reward model and optimizer
factories on top. Those are what this file exercises.
"""

from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
import yaml

from tests.conftest import (
    PROJECT_ROOT,
    TINY_OVERRIDES,
    assert_cli_ok,
    discover_modules,
    run_cli,
)

EXPERIMENT_MODULES = discover_modules("experiments")
RUN_ABLATIONS = "experiments/rl_finetuning/run_ablations.py"
ABLATION_CONFIGS = PROJECT_ROOT / "experiments" / "rl_finetuning" / "configs"


# ── Config ───────────────────────────────────────────────────────────


@pytest.fixture
def abl_cfg() -> SimpleNamespace:
    """Mirror run_ablations' merge order, then shrink to toy dimensions."""
    from src.diffusion.schedules import get_schedule

    merged: dict = {}
    for path in (
        PROJECT_ROOT / "configs" / "defaults.yaml",
        ABLATION_CONFIGS / "ablations_default.yaml",
        ABLATION_CONFIGS / "ablations_fast.yaml",
    ):
        merged.update(yaml.safe_load(path.read_text()) or {})

    merged.update(TINY_OVERRIDES)
    merged.update(
        {
            "batch_size": 4,
            "max_iter": 1,
            "grad_steps_per_iter": 1,
            "episodes_per_iter": 1,
            "eval_episodes": 1,
            "lr": 3e-4,
        }
    )

    cfg = SimpleNamespace(**merged)
    cfg._schedule_fn = get_schedule(cfg.noise_schedule)
    return cfg


@pytest.fixture
def abl_batch(abl_cfg):
    """Synthetic RL fine-tuning batch: obs, clean actions and advantages."""
    batch = abl_cfg.batch_size
    return {
        "local_obs": torch.randint(
            0, 1000, (batch, abl_cfg.crop_size, abl_cfg.crop_size)
        ).long(),
        "global_obs": torch.randint(
            0, 1000, (batch, abl_cfg.map_h, abl_cfg.map_w)
        ).long(),
        "x0": torch.randint(
            0, abl_cfg.action_dim, (batch, abl_cfg.seq_len)
        ).long(),
        "advantages": torch.rand(batch),
    }


# ── 1. Imports ───────────────────────────────────────────────────────


def test_experiment_module_list_is_not_empty():
    assert len(EXPERIMENT_MODULES) > 10, EXPERIMENT_MODULES


@pytest.mark.parametrize("module_name", EXPERIMENT_MODULES)
def test_experiment_module_imports_cleanly(module_name):
    importlib.import_module(module_name)


# ── 2. Registry and model instantiation ──────────────────────────────


def test_registry_is_well_formed():
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    assert len(REGISTRY) == 25

    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert callable(spec.loss_factory), name
        assert callable(spec.optimizer_factory), name
        assert spec.group in {"Baseline", "A", "B", "C", "D"}, name


def test_reward_model_forward(abl_cfg):
    from experiments.rl_finetuning.ablations.training import RewardModel

    model = RewardModel(obs_dim=16, width=8, depth=2).eval()
    with torch.no_grad():
        out = model(torch.randn(5, 16))

    assert out.shape == (5,)
    assert out.dtype is torch.float32
    assert torch.isfinite(out).all()


def test_reward_model_training_step():
    from experiments.rl_finetuning.ablations.training import RewardModel

    model = RewardModel(obs_dim=16, width=8, depth=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss = torch.nn.functional.mse_loss(model(torch.randn(5, 16)), torch.randn(5))
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_mixed_replay_buffer_roundtrip(abl_cfg):
    from experiments.rl_finetuning.ablations.training import MixedReplayBuffer

    buffer = MixedReplayBuffer(
        capacity=8, seq_len=abl_cfg.seq_len, device=torch.device("cpu")
    )
    buffer.push(
        torch.zeros(4, 9, 9, dtype=torch.long),
        torch.zeros(4, 21, 79, dtype=torch.long),
        torch.zeros(4, abl_cfg.seq_len, dtype=torch.long),
        torch.ones(4),
    )

    assert buffer.size == 4

    local, glob, x0, returns = buffer.sample(3)
    assert local.shape == (3, 9, 9)
    assert glob.shape == (3, 21, 79)
    assert x0.shape == (3, abl_cfg.seq_len)
    assert torch.isfinite(returns).all()


def _push_marked(buffer, seq_len, values):
    """Push one window per entry of *values*, tagged by its return."""
    n = len(values)
    marks = torch.tensor(values, dtype=torch.float32)
    buffer.push(
        torch.zeros(n, 9, 9, dtype=torch.long),
        torch.zeros(n, 21, 79, dtype=torch.long),
        torch.zeros(n, seq_len, dtype=torch.long),
        marks,
    )


def test_mixed_replay_buffer_wraps_without_losing_rows(abl_cfg):
    """A push that straddles the ring boundary keeps the newest windows."""
    from experiments.rl_finetuning.ablations.training import MixedReplayBuffer

    buffer = MixedReplayBuffer(
        capacity=8, seq_len=abl_cfg.seq_len, device=torch.device("cpu")
    )
    _push_marked(buffer, abl_cfg.seq_len, list(range(6)))
    _push_marked(buffer, abl_cfg.seq_len, list(range(6, 12)))

    assert buffer.size == 8
    held = sorted(buffer._returns.tolist())
    assert held == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]


def test_mixed_replay_buffer_survives_a_push_larger_than_itself(abl_cfg):
    """One iteration can collect more windows than the buffer holds.

    Under ``--fast`` the buffer is 500 windows and a single iteration
    collected 1,061, which raised
    ``RuntimeError: The expanded size of the tensor (500) must match the
    existing size (561)`` and the suite silently skipped the ablation.
    """
    from experiments.rl_finetuning.ablations.training import MixedReplayBuffer

    buffer = MixedReplayBuffer(
        capacity=8, seq_len=abl_cfg.seq_len, device=torch.device("cpu")
    )

    _push_marked(buffer, abl_cfg.seq_len, list(range(20)))

    assert buffer.size == 8
    held = sorted(buffer._returns.tolist())
    assert held == [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], (
        "an oversized push must leave the most recent `capacity` windows"
    )

    local, glob, x0, returns = buffer.sample(4)
    assert local.shape == (4, 9, 9)
    assert torch.isfinite(returns).all()


def test_mixed_replay_buffer_handles_an_oversized_push_after_a_partial_fill(abl_cfg):
    """The overflow path is also correct when the write index is not 0."""
    from experiments.rl_finetuning.ablations.training import MixedReplayBuffer

    buffer = MixedReplayBuffer(
        capacity=8, seq_len=abl_cfg.seq_len, device=torch.device("cpu")
    )
    _push_marked(buffer, abl_cfg.seq_len, [100.0, 101.0, 102.0])

    _push_marked(buffer, abl_cfg.seq_len, list(range(20)))

    assert buffer.size == 8
    held = sorted(buffer._returns.tolist())
    assert held == [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]


def test_compute_advantages_is_finite():
    from experiments.rl_finetuning.ablations.training import compute_advantages

    returns = torch.tensor([0.0, 1.0, 5.0, -2.0])
    adv, mean, std = compute_advantages(
        returns,
        floor=0.1,
        cap=5.0,
        wins_only=False,
        win_thresh=0.5,
        use_running_stats=False,
        ema_decay=0.99,
        running_mean=0.0,
        running_std=1.0,
    )

    assert adv.shape == returns.shape
    assert torch.isfinite(adv).all()
    assert all(map(torch.isfinite, (torch.tensor(mean), torch.tensor(std))))


# ── 3 & 4. Forward pass and one training step, per ablation ──────────


def _build_ablation(spec, cfg):
    """Reproduce run_ablation's model/loss/optimizer wiring, minus rollouts."""
    from experiments.rl_finetuning.ablations.losses import LossContext
    from experiments.rl_finetuning.ablations.optimizers import (
        apply_lora_to_model,
        make_optimizer_lora,
    )
    from src.models.denoiser import make_model

    model = make_model(cfg)
    ref_model = copy.deepcopy(model).eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    extra = {}
    if spec.name == "ewc":
        extra["fisher"] = {
            name: torch.ones_like(param)
            for name, param in model.named_parameters()
        }

    if spec.use_lora:
        lora_params = apply_lora_to_model(
            model,
            getattr(cfg, "lora_rank", 8),
            getattr(cfg, "lora_alpha", 16.0),
        )
        optimizer = make_optimizer_lora(cfg, lora_params)
    else:
        optimizer = spec.optimizer_factory(cfg, model)

    ctx = LossContext(
        ref_model=ref_model, schedule_fn=cfg._schedule_fn, cfg=cfg
    )
    return model, spec.loss_factory(ctx, **extra), optimizer


def _registry_names():
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    return sorted(REGISTRY)


@pytest.mark.parametrize("ablation_name", _registry_names())
def test_ablation_training_step_is_finite(ablation_name, abl_cfg, abl_batch):
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    spec = REGISTRY[ablation_name]
    model, loss_fn, optimizer = _build_ablation(spec, abl_cfg)

    model.train()
    optimizer.zero_grad()
    loss = loss_fn(
        model,
        abl_batch["local_obs"],
        abl_batch["global_obs"],
        abl_batch["x0"],
        abl_batch["advantages"],
        abl_cfg,
        torch.device("cpu"),
    )
    loss.backward()
    optimizer.step()

    assert loss.ndim == 0
    assert loss.dtype is torch.float32
    assert torch.isfinite(loss), f"{ablation_name} produced {loss.item()}"

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"{ablation_name}/{name}"


def test_ablation_forward_pass_shapes(abl_cfg, abl_batch):
    from experiments.rl_finetuning.ablations.losses import _forward_and_loss
    from src.models.denoiser import make_model

    model = make_model(abl_cfg).eval()
    per_sample, aux, logits, zt, t_discrete = _forward_and_loss(
        model,
        abl_batch["local_obs"],
        abl_batch["global_obs"],
        abl_batch["x0"],
        abl_cfg,
        torch.device("cpu"),
    )

    batch = abl_cfg.batch_size
    assert per_sample.shape == (batch,)
    assert logits.shape == (batch, abl_cfg.seq_len, abl_cfg.action_dim)
    assert logits.dtype is torch.float32
    assert zt.shape == (batch, abl_cfg.seq_len)
    assert t_discrete.shape == (batch,)
    assert torch.isfinite(per_sample).all()
    assert torch.isfinite(aux)
    assert torch.isfinite(logits).all()


# ── 5. Save and reload ───────────────────────────────────────────────


def test_finetuned_checkpoint_roundtrip_preserves_output(
    abl_cfg, abl_batch, tmp_path
):
    """Round-trip through the format run_ablation loads (``ema_state_dict``)."""
    from src.models.denoiser import ModelEMA, make_model

    model = make_model(abl_cfg).eval()
    ema = ModelEMA(model, decay=getattr(abl_cfg, "ema_decay", 0.999))
    t = torch.zeros(abl_cfg.batch_size, dtype=torch.long)

    eval_model = ema.make_eval_model(model)
    with torch.no_grad():
        before = eval_model(
            abl_batch["local_obs"], abl_batch["global_obs"], abl_batch["x0"], t
        )["actions"]

    path = tmp_path / "finetuned.pth"
    torch.save({"ema_state_dict": ema.state_dict()}, path)

    reloaded = make_model(abl_cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(ckpt["ema_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(
            abl_batch["local_obs"], abl_batch["global_obs"], abl_batch["x0"], t
        )["actions"]

    assert torch.equal(before, after)


def test_lora_weights_roundtrip(abl_cfg, abl_batch, tmp_path):
    from experiments.rl_finetuning.ablations.optimizers import apply_lora_to_model
    from src.models.denoiser import make_model

    model = make_model(abl_cfg)
    apply_lora_to_model(
        model, getattr(abl_cfg, "lora_rank", 8), getattr(abl_cfg, "lora_alpha", 16.0)
    )
    model.eval()
    t = torch.zeros(abl_cfg.batch_size, dtype=torch.long)

    with torch.no_grad():
        before = model(
            abl_batch["local_obs"], abl_batch["global_obs"], abl_batch["x0"], t
        )["actions"]

    path = tmp_path / "lora.pth"
    torch.save(model.state_dict(), path)

    reloaded = make_model(abl_cfg)
    apply_lora_to_model(
        reloaded,
        getattr(abl_cfg, "lora_rank", 8),
        getattr(abl_cfg, "lora_alpha", 16.0),
    )
    reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(
            abl_batch["local_obs"], abl_batch["global_obs"], abl_batch["x0"], t
        )["actions"]

    assert torch.equal(before, after)


# ── 6. Entry points ──────────────────────────────────────────────────


def test_run_ablations_help():
    result = run_cli(RUN_ABLATIONS, "--help")

    assert_cli_ok(result)
    assert "--ablations" in result.stdout


def test_run_ablations_list():
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    result = run_cli(RUN_ABLATIONS, "--list")

    assert_cli_ok(result)
    for name in REGISTRY:
        assert name in result.stdout


def test_run_ablations_requires_a_checkpoint(tmp_path):
    result = run_cli(
        RUN_ABLATIONS,
        "--ablations",
        "baseline_rl",
        "--fast",
        "--output-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode != 0
    assert "checkpoint" in (result.stdout + result.stderr).lower()


def test_run_ablations_rejects_unknown_ablation(tmp_path, tiny_checkpoint_file):
    result = run_cli(
        RUN_ABLATIONS,
        "--ablations",
        "not-an-ablation",
        "--checkpoint",
        str(tiny_checkpoint_file),
        "--output-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode != 0
    assert "unknown ablation" in (result.stdout + result.stderr).lower()
