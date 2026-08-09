"""Failure-behaviour tests for FIX-B1..B4 (ADJUDICATION handler ruling).

Each fix converts a silently-swallowed failure that could produce
plausible-looking wrong results into a raised error. These tests prove
the new failure behaviour; the parity harness proves the success paths
are bit-identical.
"""

from __future__ import annotations

import pytest


def test_eval_raises_on_broken_env(tiny_cfg):
    """FIX-B1: an evaluation episode that cannot be created raises instead
    of counting as a silent loss in the win-rate denominator."""
    from src.models.denoiser import make_model
    from src.planners.inference import Evaluator

    model = make_model(tiny_cfg).eval()
    with pytest.raises(Exception):
        Evaluator().evaluate(
            ["Not-A-Real-Env-v0"], model, 1, tiny_cfg, "cpu"
        )


def test_collector_halts_on_persistent_oracle_failure(tiny_cfg):
    """FIX-B2: ten consecutive missing oracle trajectories raise instead of
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
    """FIX-B3: a checkpoint whose config snapshot cannot be written raises
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
