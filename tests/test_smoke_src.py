"""End-to-end smoke tests for the ``src/`` diffusion-planner pipeline.

Proves the model builds, runs, trains, round-trips through disk, and that
every ``main.py`` mode starts and finishes. Nothing here asserts anything
about result quality.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from tests.conftest import (
    TINY_ENV,
    assert_cli_ok,
    discover_modules,
    requires_cuda,
    requires_minihack,
    run_cli,
)

SRC_MODULES = discover_modules("src")


# ── 1. Imports ───────────────────────────────────────────────────────


def test_src_module_list_is_not_empty():
    assert len(SRC_MODULES) > 10, SRC_MODULES


@pytest.mark.parametrize("module_name", SRC_MODULES)
def test_src_module_imports_cleanly(module_name):
    importlib.import_module(module_name)


# ── 2. Instantiation from the real config ────────────────────────────


def test_model_instantiates_from_real_config(real_cfg):
    from src.models.denoiser import LocalDiffusionPlannerWithGlobal, make_model

    model = make_model(real_cfg)

    assert isinstance(model, LocalDiffusionPlannerWithGlobal)
    assert sum(p.numel() for p in model.parameters()) > 0
    assert model.head.out_features == real_cfg.action_dim


def test_local_only_ablation_instantiates_from_real_config(real_cfg):
    import copy

    from src.models.denoiser import LocalDiffusionPlanner, make_model

    cfg = copy.copy(real_cfg)
    cfg.use_global_stream = False

    assert isinstance(make_model(cfg), LocalDiffusionPlanner)


def test_ema_wraps_real_config_model(real_cfg):
    from src.models.denoiser import ModelEMA, make_model

    model = make_model(real_cfg)
    ema = ModelEMA(model, decay=real_cfg.ema_decay)
    ema.update(model)

    assert set(ema.state_dict()) == {n for n, _ in model.named_parameters()}


# ── 3. Forward pass ──────────────────────────────────────────────────


def test_forward_pass_shape_dtype_and_finiteness(tiny_cfg, tiny_batch):
    from src.models.denoiser import make_model

    local, glob, actions = tiny_batch
    model = make_model(tiny_cfg).eval()

    with torch.no_grad():
        out = model(local, glob, actions, torch.zeros(local.shape[0], dtype=torch.long))

    assert set(out) == {"actions", "goal_pred"}
    assert out["actions"].shape == (
        local.shape[0],
        tiny_cfg.seq_len,
        tiny_cfg.action_dim,
    )
    assert out["goal_pred"].shape == (local.shape[0], 2)
    assert out["actions"].dtype is torch.float32
    assert out["goal_pred"].dtype is torch.float32
    assert torch.isfinite(out["actions"]).all()
    assert torch.isfinite(out["goal_pred"]).all()


def test_forward_pass_accepts_scalar_timestep(tiny_cfg, tiny_batch):
    from src.models.denoiser import make_model

    local, glob, actions = tiny_batch
    model = make_model(tiny_cfg).eval()

    with torch.no_grad():
        out = model(local, glob, actions, 0)

    assert torch.isfinite(out["actions"]).all()


def test_local_only_forward_pass(tiny_cfg, tiny_batch):
    import copy

    from src.models.denoiser import make_model

    cfg = copy.copy(tiny_cfg)
    cfg.use_global_stream = False
    local, glob, actions = tiny_batch
    model = make_model(cfg).eval()

    with torch.no_grad():
        out = model(local, glob, actions, torch.zeros(local.shape[0], dtype=torch.long))

    assert out["actions"].shape == (local.shape[0], cfg.seq_len, cfg.action_dim)
    assert torch.isfinite(out["actions"]).all()


def test_forward_masking_and_loss_are_finite(tiny_cfg, tiny_batch):
    from src.diffusion.forward import q_sample
    from src.diffusion.loss import mdlm_loss
    from src.diffusion.schedules import get_schedule
    from src.models.denoiser import make_model

    local, glob, actions = tiny_batch
    schedule_fn = get_schedule(tiny_cfg.noise_schedule)
    t = torch.rand(actions.shape[0]).clamp(1e-5, 1 - 1e-5)

    zt = q_sample(actions, t, tiny_cfg.mask_token, tiny_cfg.pad_token, schedule_fn)
    assert zt.shape == actions.shape
    assert zt.dtype is torch.int64

    model = make_model(tiny_cfg).eval()
    t_discrete = (t * tiny_cfg.num_diffusion_steps).long().clamp(
        0, tiny_cfg.num_diffusion_steps - 1
    )
    with torch.no_grad():
        out = model(local, glob, zt, t_discrete)

    loss = mdlm_loss(
        out["actions"],
        actions,
        zt,
        t,
        tiny_cfg.mask_token,
        tiny_cfg.pad_token,
        schedule_fn,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("strategy", ["rescale", "cap", "conf"])
def test_remdm_sampler_runs(tiny_cfg, tiny_batch, strategy):
    import copy

    from src.diffusion.sampling import remdm_sample
    from src.models.denoiser import make_model

    cfg = copy.copy(tiny_cfg)
    cfg.remask_strategy = strategy
    local, glob, _ = tiny_batch
    model = make_model(cfg).eval()

    seq = remdm_sample(model, local, glob, cfg, "cpu", physics_aware=False)

    assert seq.shape == (local.shape[0], cfg.seq_len)
    assert seq.dtype is torch.int64
    assert (seq != cfg.mask_token).all()
    assert (seq >= 0).all() and (seq < cfg.action_dim).all()


def test_greedy_sampler_runs(tiny_cfg, tiny_batch):
    from src.diffusion.sampling import greedy_sample
    from src.models.denoiser import make_model

    local, glob, _ = tiny_batch
    model = make_model(tiny_cfg).eval()

    seq = greedy_sample(model, local, glob, tiny_cfg, "cpu")

    assert seq.shape == (local.shape[0], tiny_cfg.seq_len)
    assert (seq >= 0).all() and (seq < tiny_cfg.action_dim).all()


# ── Environment sanity ───────────────────────────────────────────────


@requires_minihack
def test_environment_exposes_a_goal_staircase(real_cfg):
    """Regression: MiniHack's nhdat patch step can fail silently.

    When it does, every env falls back to the same default level with no
    staircase, so the BFS oracle has nothing to target and no episode can
    ever be won.
    """
    from src.envs.minihack_env import make_env

    env = make_env(TINY_ENV, None, real_cfg)
    try:
        env.reset(seed=0)
        raw = env.last_raw_obs
        assert (raw["chars"] == ord(">")).sum() >= 1, "no staircase on the map"
        assert env._get_bfs_distance(raw) is not None
    finally:
        env.close()


@requires_minihack
def test_distinct_envs_produce_distinct_levels(real_cfg):
    """A silent nhdat failure collapses every env onto one default level."""
    from src.envs.minihack_env import make_env

    sizes = []
    for env_id in ("MiniHack-Room-Random-5x5-v0", "MiniHack-Room-Random-15x15-v0"):
        env = make_env(env_id, None, real_cfg)
        try:
            env.reset(seed=0)
            sizes.append(int((env.last_raw_obs["chars"] != ord(" ")).sum()))
        finally:
            env.close()

    assert sizes[0] != sizes[1], f"both envs rendered {sizes[0]} cells"


# ── 4. One training step ─────────────────────────────────────────────


def _make_trainer(cfg, trajectory):
    """Build a real Trainer with the collector/evaluator/logger left out."""
    from src.buffer import ReplayBuffer
    from src.models.denoiser import ModelEMA, make_model
    from src.planners.online import Trainer

    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seq_len, cfg.pad_token)
    buffer.add(trajectory)

    model = make_model(cfg)
    ema = ModelEMA(model, decay=cfg.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.dagger_lr)

    trainer = Trainer(
        model,
        ema,
        optimizer,
        None,
        buffer,
        collector=None,
        evaluator=None,
        log=None,
        cfg=cfg,
        device="cpu",
        raw_model=model,
    )
    return trainer, model, ema


def test_single_training_step_produces_finite_loss(tiny_cfg, tiny_trajectory):
    trainer, model, ema = _make_trainer(tiny_cfg, tiny_trajectory)

    model.train()
    metrics = trainer._train_step()
    ema.update(model)

    for key in ("loss", "loss_diff", "loss_aux", "grad_norm"):
        assert key in metrics
        assert torch.isfinite(torch.tensor(metrics[key])), (key, metrics[key])


def test_training_step_updates_parameters(tiny_cfg, tiny_trajectory):
    trainer, model, _ = _make_trainer(tiny_cfg, tiny_trajectory)
    before = model.head.weight.detach().clone()

    model.train()
    trainer._train_step()

    assert not torch.equal(before, model.head.weight.detach())


def test_training_step_on_empty_buffer_is_a_no_op(tiny_cfg):
    from src.buffer import ReplayBuffer
    from src.models.denoiser import ModelEMA, make_model
    from src.planners.online import Trainer

    model = make_model(tiny_cfg)
    trainer = Trainer(
        model,
        ModelEMA(model, decay=tiny_cfg.ema_decay),
        torch.optim.AdamW(model.parameters(), lr=tiny_cfg.dagger_lr),
        None,
        ReplayBuffer(tiny_cfg.buffer_capacity, tiny_cfg.seq_len, tiny_cfg.pad_token),
        collector=None,
        evaluator=None,
        log=None,
        cfg=tiny_cfg,
        device="cpu",
        raw_model=model,
    )

    assert trainer._train_step() == {
        "loss": 0.0,
        "loss_diff": 0.0,
        "loss_aux": 0.0,
        "grad_norm": 0.0,
    }


@requires_cuda
def test_amp_training_step_on_cuda(tiny_cfg, tiny_trajectory):
    import copy

    cfg = copy.copy(tiny_cfg)
    cfg.device = "cuda"
    cfg.use_amp = True

    from src.buffer import ReplayBuffer
    from src.models.denoiser import ModelEMA, make_model
    from src.planners.online import Trainer

    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seq_len, cfg.pad_token)
    buffer.add(tiny_trajectory)
    model = make_model(cfg).to("cuda")
    trainer = Trainer(
        model,
        ModelEMA(model, decay=cfg.ema_decay),
        torch.optim.AdamW(model.parameters(), lr=cfg.dagger_lr),
        None,
        buffer,
        collector=None,
        evaluator=None,
        log=None,
        cfg=cfg,
        device="cuda",
        raw_model=model,
    )

    model.train()
    assert torch.isfinite(torch.tensor(trainer._train_step()["loss"]))


# ── 5. Save and reload ───────────────────────────────────────────────


def test_checkpoint_roundtrip_preserves_output(tiny_cfg, tiny_batch, tmp_path):
    from src.models.denoiser import ModelEMA, make_model

    local, glob, actions = tiny_batch
    t = torch.zeros(local.shape[0], dtype=torch.long)

    model = make_model(tiny_cfg).eval()
    ema = ModelEMA(model, decay=tiny_cfg.ema_decay)
    with torch.no_grad():
        before = model(local, glob, actions, t)

    path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
        },
        path,
    )

    reloaded = make_model(tiny_cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(ckpt["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(local, glob, actions, t)

    assert torch.equal(before["actions"], after["actions"])
    assert torch.equal(before["goal_pred"], after["goal_pred"])


def test_ema_weights_roundtrip(tiny_cfg, tiny_batch, tmp_path):
    from src.models.denoiser import ModelEMA, make_model

    local, glob, actions = tiny_batch
    t = torch.zeros(local.shape[0], dtype=torch.long)

    model = make_model(tiny_cfg)
    ema = ModelEMA(model, decay=0.5)
    ema.update(model)

    path = tmp_path / "ema.pth"
    torch.save({"ema_state_dict": ema.state_dict()}, path)

    eval_model = ema.make_eval_model(model)
    with torch.no_grad():
        before = eval_model(local, glob, actions, t)["actions"]

    reloaded = make_model(tiny_cfg)
    reloaded_ema = ModelEMA(reloaded, decay=0.5)
    reloaded_ema.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)["ema_state_dict"]
    )
    after_model = reloaded_ema.make_eval_model(reloaded)
    with torch.no_grad():
        after = after_model(local, glob, actions, t)["actions"]

    assert torch.equal(before, after)


# ── 6. Entry points ──────────────────────────────────────────────────


def test_main_help():
    result = run_cli("main.py", "--help")

    assert_cli_ok(result)
    assert "--mode" in result.stdout


def test_main_rejects_unknown_mode():
    assert run_cli("main.py", "--mode", "not-a-mode").returncode != 0


def test_main_inference_requires_a_checkpoint(tiny_config_file):
    result = run_cli(
        "main.py", "--mode", "inference", "--config", str(tiny_config_file)
    )

    assert result.returncode != 0
    assert "checkpoint" in (result.stdout + result.stderr).lower()


@requires_minihack
def test_main_smoke_mode_runs(tiny_config_file):
    result = run_cli("main.py", "--mode", "smoke", "--config", str(tiny_config_file))

    assert_cli_ok(result)
    assert "Smoke Results" in result.stdout


@requires_minihack
def test_main_collect_mode_runs(tiny_config_file, tmp_path):
    result = run_cli("main.py", "--mode", "collect", "--config", str(tiny_config_file))

    assert_cli_ok(result)
    assert (tmp_path / "dataset.pt").exists()


def test_main_offline_mode_runs(tiny_config_file, tiny_dataset_file, tmp_path):
    result = run_cli(
        "main.py",
        "--mode",
        "offline",
        "--config",
        str(tiny_config_file),
        "--data",
        str(tiny_dataset_file),
        "--override",
        "total_timesteps=8",
    )

    assert_cli_ok(result)
    assert list((tmp_path / "checkpoints").glob("offline_*/offline_final.pth"))


@requires_minihack
def test_main_inference_mode_runs(tiny_config_file, tiny_checkpoint_file, tmp_path):
    output = tmp_path / "eval.json"
    result = run_cli(
        "main.py",
        "--mode",
        "inference",
        "--config",
        str(tiny_config_file),
        "--checkpoint",
        str(tiny_checkpoint_file),
        "--envs",
        TINY_ENV,
        "--episodes",
        "1",
        "--output",
        str(output),
    )

    assert_cli_ok(result)
    assert output.exists()


@requires_minihack
def test_main_online_mode_runs(tiny_config_file):
    result = run_cli(
        "main.py",
        "--mode",
        "online",
        "--config",
        str(tiny_config_file),
        "--no-warm-start",
    )

    assert_cli_ok(result)


def test_main_baselines_mode_validates_algo(tiny_config_file):
    result = run_cli(
        "main.py",
        "--mode",
        "baselines",
        "--algo",
        "not-an-algo",
        "--config",
        str(tiny_config_file),
    )

    assert result.returncode != 0


@requires_minihack
@pytest.mark.slow
def test_main_baselines_bc_runs(tiny_config_file):
    result = run_cli(
        "main.py",
        "--mode",
        "baselines",
        "--algo",
        "bc",
        "--seeds",
        "0",
        "--config",
        str(tiny_config_file),
        "--override", "baselines_bc_oracle_episodes_per_env=1",
        "--override", "baselines_bc_epochs=1",
        "--override", "baselines_bc_batch_size=8",
        "--override", "baselines_n_envs_per_id=1",
        "--override", "baselines_eval_episodes_per_env=1",
        "--override", "baselines_eval_freq_env_steps=1000000",
    )

    assert_cli_ok(result)


@requires_minihack
@pytest.mark.slow
def test_main_baselines_ppo_runs_without_wandb(tiny_config_file):
    """Regression: SB3 baselines used to die before their first env step.

    ``WandbCallback`` was built unconditionally, and behind it SB3 refused to
    start because ``tensorboard_log`` was set without tensorboard installed.
    """
    result = run_cli(
        "main.py",
        "--mode",
        "baselines",
        "--algo",
        "ppo",
        "--seeds",
        "0",
        "--config",
        str(tiny_config_file),
        "--override", "total_timesteps=64",
        "--override", "baselines_n_envs_per_id=1",
        "--override", "baselines_eval_episodes_per_env=1",
        "--override", "baselines_eval_freq_env_steps=1000000",
    )

    assert_cli_ok(result)


def test_offline_builds_one_eval_model_per_eval_point(tiny_cfg, tiny_trajectory):
    """PERF-O2: the ID and OOD eval blocks share one EMA copy.

    Both cadences derive from ``offline_eval_every_grad_steps``
    (``offline.py:157-161``), so they fire on the same step against the
    same ``_ema_source``. Each block used to build its own
    ``copy.deepcopy(model)`` plus EMA apply.
    """
    import copy as _copy

    import torch

    from src.buffer import ReplayBuffer
    from src.models.denoiser import ModelEMA, make_model
    from src.planners.offline import make_offline_trainer

    cfg = _copy.deepcopy(tiny_cfg)
    cfg.offline_total_grad_steps = 3
    cfg.offline_eval_every_grad_steps = 1
    cfg.offline_checkpoint_every_grad_steps = 10**9
    cfg.checkpoint_every_timesteps = 10**9
    cfg.offline_log_every = 10**9
    cfg.use_amp = False
    cfg.torch_compile = False

    buffer = ReplayBuffer(1000, cfg.seq_len, cfg.pad_token)
    buffer.load_offline_data(
        {"trajectories": [tiny_trajectory]}, [tiny_trajectory["env_id"]]
    )

    torch.manual_seed(0)
    model = make_model(cfg)
    ema = ModelEMA(model, decay=0.9)

    built: list[int] = []
    real_make = ModelEMA.make_eval_model

    def counting_make(self, src):
        built.append(1)
        return real_make(self, src)

    seen: list[tuple[str, int]] = []

    class _StubEvaluator:
        def evaluate(self, env_ids, eval_model, n_episodes, cfg_, device):
            seen.append((env_ids[0], id(eval_model)))
            return {e: {"win_rate": 0.0, "avg_reward": 0.0} for e in env_ids}

    ModelEMA.make_eval_model = counting_make
    try:
        make_offline_trainer(cfg)(
            model=model,
            ema_model=ema,
            buffer=buffer,
            cfg=cfg,
            device=torch.device("cpu"),
            evaluator=_StubEvaluator(),
            id_envs=["ID_ENV"],
            ood_envs=["OOD_ENV"],
        )
    finally:
        ModelEMA.make_eval_model = real_make

    id_calls = [s for s in seen if s[0] == "ID_ENV"]
    ood_calls = [s for s in seen if s[0] == "OOD_ENV"]
    assert id_calls, "the ID eval never fired, so the test proves nothing"
    assert len(id_calls) == len(ood_calls), "both blocks should fire together"
    # One EMA copy per eval point, not two.
    assert len(built) == len(id_calls), (
        f"{len(built)} eval models built for {len(id_calls)} eval points"
    )
    # And both blocks were handed the very same model object.
    for (_, id_obj), (_, ood_obj) in zip(id_calls, ood_calls, strict=True):
        assert id_obj == ood_obj


def test_building_either_model_warns_about_nothing(tiny_cfg):
    """No warning may originate in this repo's own source (sweep S11-7).

    `LocalDiffusionPlanner` left `nn.TransformerEncoder` at its default
    `enable_nested_tensor=True` while its encoder layer sets
    `norm_first=True`, which makes the nested-tensor path unavailable, so
    PyTorch warned on every construction. It was the only repo-origin warning
    in either suite. Filtering it would have hidden the next one; the keyword
    is passed instead, as the dual-stream sibling always has.
    """
    import warnings
    from pathlib import Path

    from src.models.denoiser import (
        LocalDiffusionPlanner,
        LocalDiffusionPlannerWithGlobal,
    )

    root = str(Path(__file__).resolve().parents[1] / "src")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LocalDiffusionPlannerWithGlobal(tiny_cfg)
        LocalDiffusionPlanner(tiny_cfg)

    ours = [w for w in caught if str(w.filename).startswith(root)]
    assert not ours, [f"{w.filename}:{w.lineno} {w.message}" for w in ours]
