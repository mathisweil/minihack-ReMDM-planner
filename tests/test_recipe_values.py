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
    "weight_decay": 0.0,  # author decision 2026-08-16 (was 1e-4)
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


# ---------------------------------------------------------------------------
# Ablation-suite recipe values (spec-ablations §1.6)
#
# The suite's own recipe had no value-level pin in either repo (sweep S11-1):
# every test that read `ablations_default.yaml` treated it as a key set or
# compared it to itself, so eight of nine shipped-value mutations survived a
# full run — `lr` at 1000x included. The values below are transcribed from
# §1.6, not read back from the config.
# ---------------------------------------------------------------------------

#: Values §1.6 records as shared with the sibling repo. Byte-identical there.
_ABLATION_SHARED_PINS = {
    "adv_clip_eps": 0.2,
    "diffusion_steps_collect": 5,
    "entropy_coef": 0.01,
    "eval_every": 25,
    "ewc_fisher_batches": 20,
    "ewc_lambda": 100.0,
    "kl_coef": 0.1,
    "llrd_decay": 0.9,
    "lora_alpha": 16.0,
    "lora_rank": 8,
    "lr": 3.0e-4,
    "max_grad_norm": 1.0,
    "max_iter": 500,
    "mixed_replay_buffer_size": 10000,
    "mixed_replay_ratio": 0.25,
    "num_seeds": 3,
    "return_weight_cap": 5.0,
    "return_weight_floor": 0.1,
    "reward_filter_percentile": 75,
    "reward_model_depth": 2,
    "reward_model_lr": 1.0e-3,
    "reward_model_train_steps": 50,
    "reward_model_width": 64,
    "running_stats_ema_decay": 0.99,
    "t_curriculum_end": 0.2,
    "t_curriculum_start": 0.8,
    "t_curriculum_steps": 200,
    "t_max_low": 0.2,
    "trust_region_kl": 0.05,
    "use_wandb": False,
    "weight_decay": 1.0e-4,
    "win_threshold": 0.5,
}

#: The values §1.6 records as split, this repo's column. Episodes per
#: iteration are `episodes_per_iter` here and `num_envs * num_steps` there.
#: `use_amp` is not in §1.6 but `run_ablations._RESULT_AFFECTING` declares it
#: result-affecting, so it is pinned rather than excused.
_ABLATION_SPLIT_PINS = {
    "batch_size": 3072,
    "episodes_per_iter": 30,
    "eval_episodes": 20,
    "grad_steps_per_iter": 1,
    "use_amp": True,
}

#: Keys §1.6 does not record as recipe values, each pinned or governed
#: elsewhere. Anything else new in the config must join a pin table above.
_ABLATION_UNPINNED = {
    # Backend switch, not a recipe value.
    "torch_compile",
    # Diagnostic cadences: wall-clock only, and spec-ablations §1.4 records
    # them as not affecting poolability.
    "cka_batch_size",
    "cka_every",
    "grad_align_every",
    "per_layer_every",
    "repr_drift_every",
    "t_analysis_every",
    "t_analysis_n_bins",
    # W&B naming, pinned by tests/test_config.py.
    "wandb_entity",
    "wandb_project",
}

_ABLATIONS_DEFAULT = (
    _DEFAULTS.parents[1]
    / "experiments"
    / "rl_finetuning"
    / "configs"
    / "ablations_default.yaml"
)


def _ablations_default() -> dict:
    return yaml.safe_load(_ABLATIONS_DEFAULT.read_text())


@pytest.mark.parametrize(("key", "expected"), sorted(_ABLATION_SHARED_PINS.items()))
def test_ablation_recipe_shared_value(key, expected):
    """spec-ablations §1.6, shared column."""
    got = _ablations_default()[key]
    assert got == expected, f"{key}: {got} != {expected} (spec-ablations §1.6)"


@pytest.mark.parametrize(("key", "expected"), sorted(_ABLATION_SPLIT_PINS.items()))
def test_ablation_recipe_split_value(key, expected):
    """spec-ablations §1.6, this repo's column of the recorded split."""
    got = _ablations_default()[key]
    assert got == expected, f"{key}: {got} != {expected} (spec-ablations §1.6)"


def test_every_ablation_config_key_is_pinned_or_explicitly_not():
    """A new key in the suite's config must join a pin table or the excused
    set, so a scientific value cannot be added unguarded — which is how the
    whole recipe came to be unpinned."""
    keys = set(_ablations_default())
    accounted = (
        set(_ABLATION_SHARED_PINS) | set(_ABLATION_SPLIT_PINS) | _ABLATION_UNPINNED
    )
    assert keys - accounted == set(), (
        f"unaccounted ablation config key(s): {sorted(keys - accounted)}. "
        "Pin the value against spec-ablations §1.6 or add it to "
        "_ABLATION_UNPINNED with the reason."
    )
    assert accounted - keys == set(), (
        f"pin table names key(s) the config no longer has: "
        f"{sorted(accounted - keys)}"
    )


# ---------------------------------------------------------------------------
# Numerical floors (sweep S11-2, S11-3)
#
# Both are shared constants whose values the sibling repo must match exactly.
# Neither was guarded: the loss floor moved 100x and the sampler floor 10^6x
# with both suites green. Pinned by value and by the behaviour the value
# produces, so a change has to be deliberate in two places.
# ---------------------------------------------------------------------------


def test_the_nelbo_weight_denominator_floor():
    """`w_t = -alpha_dot / clamp(1 - alpha_t, min=_WEIGHT_DENOM_EPS)`.

    The floor caps the weight as t -> 0, where `1 - alpha_t` goes to zero.
    At `1 - alpha_t = 1e-8`, far below the floor, the denominator is the
    floor itself, so the weight is `-alpha_dot / 1e-5` = 1e5 * -alpha_dot,
    not 1e8 * -alpha_dot. Moving the floor to 1e-3 would make it 1e3.

    The craftax twin's `_EPS` must hold the same value; there it is also the
    default `t_min` of `compute_loss`, which has no counterpart here.
    """
    from src.diffusion import loss as loss_mod

    assert loss_mod._WEIGHT_DENOM_EPS == 1e-5
    assert loss_mod._MAX_WEIGHT == 1000.0

    alpha_t = 1.0 - 1e-8
    weight = 1.0 / max(1.0 - alpha_t, loss_mod._WEIGHT_DENOM_EPS)
    assert weight == pytest.approx(1e5, rel=1e-9)


def test_the_sampler_stability_floor():
    """`sigma_max = min(1, (1 - alpha_s) / max(alpha_t, _SIGMA_DENOM_EPS))`,
    and the same constant floors `1 - alpha_t` in the unmask posterior.

    At `alpha_t = 1e-12`, below the floor, the divisor is 1e-8, so a
    `1 - alpha_s` of 1e-9 gives 0.1. Moving the floor to 1e-2 would give
    1e-7 — six orders of magnitude of remasking probability at the high-t
    end of every sampled plan.
    """
    from src.diffusion import sampling as samp

    assert samp._SIGMA_DENOM_EPS == 1e-8

    got = min(1.0, 1e-9 / max(1e-12, samp._SIGMA_DENOM_EPS))
    assert got == pytest.approx(0.1, rel=1e-9)
