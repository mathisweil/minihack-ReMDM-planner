"""Failure-behaviour tests: hard failures raise instead of degrading silently.

Each fix converts a silently-swallowed failure that could produce
plausible-looking wrong results into a raised error. These tests prove
the new failure behaviour; the parity harness proves the success paths
are bit-identical.
"""

from __future__ import annotations

import pytest


def test_eval_raises_on_broken_env(tiny_cfg):
    """An evaluation episode that cannot be created raises instead
    of counting as a silent loss in the win-rate denominator."""
    from src.models.denoiser import make_model
    from src.planners.inference import Evaluator

    model = make_model(tiny_cfg).eval()
    with pytest.raises(Exception):
        Evaluator().evaluate(
            ["Not-A-Real-Env-v0"], model, 1, tiny_cfg, "cpu"
        )


def test_collector_halts_on_persistent_oracle_failure(tiny_cfg):
    """Ten consecutive missing oracle trajectories raise instead of
    silently removing DAgger supervision (oracle_steps=999 masquerade)."""
    from src.buffer import ReplayBuffer
    from src.curriculum import DynamicCurriculum
    from src.models.denoiser import ModelEMA, make_model
    from src.planners.collect import (
        _MAX_CONSECUTIVE_ORACLE_FAILURES,
        DataCollector,
    )

    model = make_model(tiny_cfg)
    collector = DataCollector(
        ModelEMA(model, decay=0.5),
        model,
        ReplayBuffer(8, tiny_cfg.seq_len, tiny_cfg.pad_token),
        DynamicCurriculum(tiny_cfg.id_envs, 10, True),
        tiny_cfg,
        "cpu",
    )
    with pytest.raises(RuntimeError, match="oracle failed"):
        for _ in range(_MAX_CONSECUTIVE_ORACLE_FAILURES):
            collector._note_oracle_result(None, "SomeEnv-v0")

    collector._oracle_failure_streak = 0
    for _ in range(_MAX_CONSECUTIVE_ORACLE_FAILURES - 1):
        collector._note_oracle_result(None, "SomeEnv-v0")
    collector._note_oracle_result({"actions": [0]}, "SomeEnv-v0")
    assert collector._oracle_failure_streak == 0, "success must reset the streak"


def test_offline_snapshot_save_failure_raises(tiny_cfg, tmp_path, monkeypatch):
    """A checkpoint whose config snapshot cannot be written raises
    instead of shipping a checkpoint the snapshot-evaluation workflow
    cannot use."""
    import yaml as yaml_mod

    from src.models.denoiser import ModelEMA, make_model
    from src.planners import offline as offline_mod

    model = make_model(tiny_cfg)
    ema = ModelEMA(model, decay=0.5)
    tiny_cfg.checkpoint_dir = str(tmp_path)

    def boom(*a, **k):
        raise OSError("disk full")

    import torch

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1)
    monkeypatch.setattr(yaml_mod, "dump", boom)
    with pytest.raises(OSError):
        offline_mod._save_offline_checkpoint(
            model, ema, opt, sched, step=1, cfg=tiny_cfg, log=None,
        )


def test_offline_checkpoint_saves_and_requires_rng_states(
    tiny_cfg, tmp_path, monkeypatch
):
    """Offline arm: offline checkpoints carry rng_states, and an
    offline resume without them raises instead of continuing with fresh
    randomness under a resume's identity."""
    import torch

    from src.buffer import ReplayBuffer
    from src.models.denoiser import ModelEMA, make_model
    from src.planners import offline as offline_mod

    model = make_model(tiny_cfg)
    ema = ModelEMA(model, decay=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1)
    tiny_cfg.checkpoint_dir = str(tmp_path)
    offline_mod._save_offline_checkpoint(
        model, ema, opt, sched, step=1, cfg=tiny_cfg, log=None,
    )
    ckpt = torch.load(
        tmp_path / "offline_step1.pth", map_location="cpu", weights_only=False,
    )
    assert set(ckpt["rng_states"]) == {"torch", "numpy", "python"}

    stripped = {k: v for k, v in ckpt.items() if k != "rng_states"}
    tiny_cfg.use_wandb = False
    buffer = ReplayBuffer(8, tiny_cfg.seq_len, tiny_cfg.pad_token)
    train_fn = offline_mod.make_offline_trainer(tiny_cfg)
    with pytest.raises(RuntimeError, match="rng_states"):
        train_fn(
            model, ema, buffer, tiny_cfg, "cpu",
            raw_model=model, resume_state=stripped,
        )


def test_buffer_rejects_legacy_list_dataset(tiny_cfg):
    """Legacy drop: the list dataset format raises instead of loading."""
    from src.buffer import ReplayBuffer

    buffer = ReplayBuffer(8, tiny_cfg.seq_len, tiny_cfg.pad_token)
    with pytest.raises(TypeError, match="legacy list format"):
        buffer.load_offline_data([], tiny_cfg.id_envs)


def test_inference_rejects_bare_state_dict(tiny_cfg, tmp_path):
    """Checkpoint-read strictness: a bare state-dict file (no
    model_state_dict wrapper) raises instead of loading silently."""
    import torch

    from src.models.denoiser import make_model
    from src.planners.inference import run_inference

    model = make_model(tiny_cfg)
    bare = tmp_path / "bare.pth"
    torch.save(model.state_dict(), bare)
    with pytest.raises(ValueError, match="model_state_dict"):
        run_inference(
            tiny_cfg, str(bare), ["x"], 1, None, use_ema=False,
        )


def test_resume_refuses_corrupt_rng_state(tiny_cfg, tmp_path):
    """rng_states present but unrestorable raises instead of
    continuing with fresh randomness under a resume's identity; a
    checkpoint without rng_states raises (all current checkpoints
    carry it, so absence means malformed)."""
    import torch

    from src.buffer import ReplayBuffer
    from src.curriculum import DynamicCurriculum
    from src.models.denoiser import ModelEMA, make_model
    from src.planners.collect import DataCollector
    from src.planners.inference import Evaluator
    from src.planners.logging import Logger
    from src.planners.online import Trainer

    tiny_cfg.use_wandb = False
    model = make_model(tiny_cfg)
    ema = ModelEMA(model, decay=0.5)
    buffer = ReplayBuffer(8, tiny_cfg.seq_len, tiny_cfg.pad_token)
    collector = DataCollector(
        ema, model, buffer,
        DynamicCurriculum(tiny_cfg.id_envs, 10, True),
        tiny_cfg, "cpu",
    )
    trainer = Trainer(
        model,
        ema,
        torch.optim.AdamW(model.parameters(), lr=1e-4),
        None,
        buffer,
        collector,
        Evaluator(),
        Logger(tiny_cfg),
        tiny_cfg,
        "cpu",
        raw_model=model,
    )

    base = {
        "model_state_dict": model.state_dict(),
        "ema_state_dict": trainer.ema_model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": None,
        "curriculum_state": {},
        "iteration": 3,
        "env_steps": 100,
    }

    corrupt = tmp_path / "corrupt.pth"
    torch.save({**base, "rng_states": {"torch": "not-a-state"}}, corrupt)
    with pytest.raises(RuntimeError, match="rng_states"):
        trainer.load_checkpoint(str(corrupt))

    missing = tmp_path / "missing_rng.pth"
    torch.save(base, missing)
    with pytest.raises(RuntimeError, match="rng_states"):
        trainer.load_checkpoint(str(missing))

    intact = tmp_path / "intact.pth"
    import random

    import numpy as np

    torch.save(
        {
            **base,
            "rng_states": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        intact,
    )
    resume_from, env_steps = trainer.load_checkpoint(str(intact))
    assert resume_from == 4 and env_steps == 100
