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
