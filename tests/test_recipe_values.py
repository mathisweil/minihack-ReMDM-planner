"""Value-level pins for the canonical scientific recipe (step 8).

Sources: research/spec-method.md §7 (method parameters, minihack
column), research/spec-training.md §1.6-1.9/§4/§5 (DAgger constants,
EMA, optimisation), research/spec-config.md §3/§4 (offline silent pins,
documented budgets). configs/defaults.yaml IS the final paper recipe
(README.md:172); these tests pin its scientific values against the spec
loci so a silent recipe drift fails loudly. The craftax twin file pins
the same surfaces for its recipe.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_DEFAULTS = Path(__file__).resolve().parents[1] / "configs" / "defaults.yaml"

# spec-method §7 (minihack column) + spec-training/config anchors.
_RECIPE_PINS = {
    # tokens and architecture (spec-method §1.3; spec anchors defaults 26-45)
    "action_dim": 12,
    "mask_token": 12,
    "pad_token": 13,
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 4,
    "seq_len": 64,
    "dropout": 0.0,
    "use_global_stream": True,
    # method parameters (spec-method §7)
    "noise_schedule": "linear",
    "num_diffusion_steps": 100,
    "diffusion_steps_eval": 10,
    "diffusion_steps_collect": 5,
    "remask_strategy": "conf",
    "eta": 0.15,
    "temperature": 0.5,
    "top_p": 0.9,
    "replan_every": 16,
    "loss_weight_clip": 1000.0,
    "label_smoothing": 0.0,
    "physics_aware_sampling": False,
    # EMA (spec-training §4.1)
    "ema_decay": 0.999,
    # budgets and loop constants (spec-training §1.6/1.9; spec-config §4)
    "total_timesteps": 5_650_000,
    "episodes_per_iteration": 30,
    "grad_steps_per_iteration": 100,
    "dagger_batch_size": 2048,
    "buffer_capacity": 10_000,
    "curriculum_queue_size": 100,
    "curriculum_preseed": True,
    # optimisation (spec-training §5)
    "dagger_lr": 3e-5,
    "offline_lr": 3e-4,
    "offline_batch_size": 2048,
    "weight_decay": 1e-4,
    "use_amp": True,
    "torch_compile": True,
    # offline silent pins (spec-config §3.1: 60000/5000/10000/1.5M)
    "offline_total_grad_steps": 60_000,
    "offline_eval_every_grad_steps": 5_000,
    "offline_checkpoint_every_grad_steps": 10_000,
    "offline_buffer_capacity": 1_500_000,
    # BC auxiliary head (spec-training §2.3)
    "aux_loss_weight": 0.5,
}


def test_recipe_values_match_the_spec():
    """Every pinned key in defaults.yaml equals its spec-recorded value."""
    raw = yaml.safe_load(_DEFAULTS.read_text())
    for key, expected in _RECIPE_PINS.items():
        got = raw[key]
        if isinstance(expected, float):
            got = float(got)
            assert got == pytest.approx(expected), f"{key}: {got} != {expected}"
        else:
            assert got == expected, f"{key}: {got!r} != {expected!r}"


def test_offline_grad_step_pin_equals_the_dagger_compute_match():
    """offline_total_grad_steps = 600 DAgger iterations x 100 grad steps.

    Source: spec-training §2.4 / spec-config §3.2 (compute-fairness
    invariant: the pinned 60000 equals DAgger's gradient work at
    iter600). Derivation: 600 * grad_steps_per_iteration(100) = 60000.
    """
    raw = yaml.safe_load(_DEFAULTS.read_text())
    assert raw["offline_total_grad_steps"] == 600 * raw["grad_steps_per_iteration"]


def test_offline_default_grad_steps_formula():
    """Unpinned offline budget resolves as total_timesteps //
    offline_batch_size (spec-config §4 shared formula).

    Derivation: 5,650,000 // 2048 = 2758 (the recipe pins 60000 instead;
    this pins the documented fallback formula's inputs).
    """
    raw = yaml.safe_load(_DEFAULTS.read_text())
    assert raw["total_timesteps"] // raw["offline_batch_size"] == 2758
