"""Tests for the g_delta measurement behind the paper's gradient decomposition.

Checkpoint-free: the algebra is exercised on a tiny model and a synthetic
batch. The expensive part of ``analysis/gdelta.py`` is the checkpoint restore
and the on-policy rollout, and neither is what could be wrong.

The craftax twin file (`tests/test_gdelta.py` there) carries the same
assertions on the same quantities; the differences are torch for JAX.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from experiments.rl_finetuning.ablations import losses
from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import (
    AblationHistory,
    compute_advantages,
)
from experiments.rl_finetuning.analysis import gdelta as gd
from experiments.rl_finetuning.analysis.tables import write_tex_macros

WEIGHT_BATCH = 32
SEED = 0


@pytest.fixture(scope="module")
def gd_config() -> dict:
    """The four weighting knobs, read from the real ablations config."""
    import yaml

    from tests.conftest import PROJECT_ROOT

    raw = yaml.safe_load(
        (
            PROJECT_ROOT
            / "experiments"
            / "rl_finetuning"
            / "configs"
            / "ablations_default.yaml"
        ).read_text()
    )
    return {
        key: raw[key]
        for key in (
            "adv_clip_eps",
            "win_threshold",
            "return_weight_floor",
            "return_weight_cap",
        )
    }


@pytest.fixture(scope="module")
def returns() -> torch.Tensor:
    """Sparse-reward returns: most windows earn nothing, a few earn a lot."""
    gen = torch.Generator().manual_seed(SEED)
    win = torch.rand(WEIGHT_BATCH, generator=gen) < 0.25
    magnitude = 1.0 + 5.0 * torch.rand(WEIGHT_BATCH, generator=gen)
    return torch.where(win, magnitude, torch.zeros(WEIGHT_BATCH))


@pytest.fixture(scope="module")
def advantages(returns, gd_config) -> torch.Tensor:
    adv, _, _ = compute_advantages(
        returns,
        gd_config["return_weight_floor"],
        gd_config["return_weight_cap"],
        wins_only=False,
        win_thresh=gd_config["win_threshold"],
        use_running_stats=False,
        ema_decay=0.99,
        running_mean=0.0,
        running_std=1.0,
    )
    return adv


@pytest.fixture(scope="module")
def variants(advantages, returns, gd_config) -> dict:
    return gd.build_variants(advantages, returns, gd_config, WEIGHT_BATCH)


# ---------------------------------------------------------------------------
# The weight vectors are the ones the trainer applies
# ---------------------------------------------------------------------------


def test_registry_still_matches_the_assumed_variants(variants):
    """The four measured variants are hardcoded, so a registry edit could
    silently leave the measurement describing a weighting the trainer no
    longer applies. `verify_registry` is what stops that.
    """
    gd.verify_registry()
    assert set(variants) == set(gd.REGISTRY_RULES)
    assert set(gd.REGISTRY_RULES) == {
        "baseline_clipped_ratio",
        "advantage_clip",
        "normalized_adv",
        "bc_wins",
    }


def test_build_variants_on_a_hand_computed_batch():
    """Two of four windows win, so the mask rescales by B / n_wins = 2."""
    returns = torch.tensor([0.0, 0.0, 1.0, 2.0])
    adv = torch.tensor([0.1, 0.1, 1.0, 2.0])
    cfg = {"adv_clip_eps": 0.2, "win_threshold": 0.5}
    built = gd.build_variants(adv, returns, cfg, batch=4)

    assert gd.centred_delta(built["baseline_clipped_ratio"])[2]
    assert gd.centred_delta(built["advantage_clip"])[2]
    assert not gd.centred_delta(built["normalized_adv"])[2]
    assert torch.allclose(built["bc_wins"], torch.tensor([0.0, 0.0, 2.0, 2.0]))


@pytest.mark.parametrize("variant", ["advantage_clip", "normalized_adv", "bc_wins"])
def test_variant_matches_the_loss_the_trainer_runs(
    variant, variants, returns, gd_config, tiny_cfg, monkeypatch
):
    """Each reconstructed weight vector equals the one its loss factory applies.

    ``_core_loss`` is the single point where every factory hands its transformed
    weights to the forward pass, so capturing its argument reads the trainer's
    own vector rather than a re-implementation of it.
    """
    captured = {}

    def capture(model, local_obs, global_obs, x0, advantages, cfg, device, *a, **k):
        captured["w"] = advantages
        return torch.zeros(())

    monkeypatch.setattr(losses, "_core_loss", capture)

    cfg = SimpleNamespace(**{**vars(tiny_cfg), **gd_config})
    ctx = losses.LossContext(ref_model=None, schedule_fn=None, cfg=cfg)
    name, _, wins_only = gd.REGISTRY_RULES[variant]
    loss_fn = REGISTRY[name].loss_factory(ctx)

    # The trainer hands the loss whatever compute_advantages returned under
    # this ablation's wins_only flag, not the baseline vector.
    trainer_adv, _, _ = compute_advantages(
        returns,
        gd_config["return_weight_floor"],
        gd_config["return_weight_cap"],
        wins_only=wins_only,
        win_thresh=gd_config["win_threshold"],
        use_running_stats=False,
        ema_decay=0.99,
        running_mean=0.0,
        running_std=1.0,
    )
    dummy = torch.zeros((WEIGHT_BATCH, cfg.seq_len), dtype=torch.long)
    loss_fn(None, dummy, dummy, dummy, trainer_adv, cfg, torch.device("cpu"))

    assert torch.allclose(captured["w"], variants[variant], atol=1e-5)


# ---------------------------------------------------------------------------
# delta and (A1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant", ["baseline_clipped_ratio", "advantage_clip", "bc_wins"]
)
def test_delta_is_zero_mean_where_a1_holds(variant, variants):
    delta, abar, a1_holds = gd.centred_delta(variants[variant])
    assert a1_holds
    assert abar > 0.0
    assert abs(float(delta.mean())) < 1e-5


def test_normalized_adv_is_flagged_as_a1_violating(variants):
    delta, abar, a1_holds = gd.centred_delta(variants["normalized_adv"])
    assert not a1_holds
    assert abs(abar) < 1e-5
    # The fallback centres rather than dividing by a vanishing mean.
    assert bool(torch.isfinite(delta).all())


def test_cv_a_and_ess_agree(variants):
    """ESS/B == 1 / (1 + CV_A^2) exactly, which is Eq. 5 of the paper."""
    for name in ["baseline_clipped_ratio", "advantage_clip", "bc_wins"]:
        delta, _, _ = gd.centred_delta(variants[name])
        cv_a = float(torch.sqrt((delta**2).mean()))
        assert gd.effective_sample_size(variants[name]) == pytest.approx(
            1.0 / (1.0 + cv_a**2), rel=1e-4
        )


# ---------------------------------------------------------------------------
# The decomposition itself
# ---------------------------------------------------------------------------


def test_decomposition_identity(variants, tiny_cfg):
    """grad L_RW == Abar * (grad L_BC + g_delta) under a shared (z_t, t) draw."""
    from src.diffusion.schedules import get_schedule
    from src.models.denoiser import make_model

    cfg = tiny_cfg
    cfg._schedule_fn = get_schedule(cfg.noise_schedule)
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    model = make_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    gen = torch.Generator().manual_seed(SEED + 1)
    local = torch.randint(
        0, 1000, (WEIGHT_BATCH, cfg.crop_size, cfg.crop_size), generator=gen
    )
    glob = torch.randint(0, 1000, (WEIGHT_BATCH, cfg.map_h, cfg.map_w), generator=gen)
    x0 = torch.randint(0, cfg.action_dim, (WEIGHT_BATCH, cfg.seq_len), generator=gen)

    def gradient(weights):
        torch.manual_seed(SEED + 2)
        model.zero_grad(set_to_none=True)
        per_sample, _, _, _, _ = losses._forward_and_loss(
            model, local, glob, x0, cfg, device
        )
        loss = per_sample.mean() if weights is None else (per_sample * weights).mean()
        loss.backward()
        return torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in params
        ]).detach()

    weights = variants["baseline_clipped_ratio"]
    delta, abar, a1_holds = gd.centred_delta(weights)
    assert a1_holds

    g_bc = gradient(None)
    g_delta = gradient(delta)
    g_rw = gradient(weights)

    residual = float((g_rw - abar * (g_bc + g_delta)).norm() / g_rw.norm())
    assert residual < 1e-4


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


def _fake_seed_blob(seed: int, ratio: float, cos: float, abar: float) -> dict:
    return {
        "aggregate": False, "seed": seed, "n_draws": 8, "n_params": 100,
        "random_cos_sd": 0.1, "batch": 32, "bc_self_cos_mean": 0.9,
        "bc_self_cos_std": 0.01, "eq4_residual_max": 1e-5 * (seed + 1),
        "variants": {
            "baseline_clipped_ratio": {
                "cv_a": 1.0, "abar": abar, "abar_ratio_to_baseline": 1.0,
                "ess_fraction": 0.5, "a1_violated": False,
                "ratio_mean": ratio, "ratio_std_draws": 0.5,
                "cos_mean": cos, "cos_std_draws": 0.5,
                "ratio_shuffled_mean": 0.4, "ratio_shuffled_std": 0.01,
                "cos_shuffled_mean": 0.0, "cos_shuffled_std": 0.01,
            }
        },
    }


@pytest.fixture(scope="module")
def fake_aggregate() -> dict:
    return gd.aggregate([
        _fake_seed_blob(seed, ratio, cos, abar)
        for seed, (ratio, cos, abar) in enumerate(
            [(0.40, 0.00, 0.30), (0.50, 0.02, 0.32), (0.60, 0.04, 0.34)]
        )
    ])


def test_aggregate_reports_dispersion_across_seeds(fake_aggregate):
    rec = fake_aggregate["variants"]["baseline_clipped_ratio"]

    assert fake_aggregate["n_seeds"] == 3
    assert fake_aggregate["seeds"] == [0, 1, 2]
    assert rec["ratio_mean"] == pytest.approx(0.50)
    # Across seeds, not the 0.5 each seed reported across its own draws.
    assert rec["ratio_std_seeds"] == pytest.approx(0.0816, abs=1e-3)
    assert rec["cos_std_seeds"] == pytest.approx(0.0163, abs=1e-3)
    assert rec["abar_mean"] == pytest.approx(0.32)
    assert fake_aggregate["eq4_residual_max"] == pytest.approx(3e-5)


def test_aggregate_refuses_an_aggregate_input():
    with pytest.raises(ValueError, match="already an aggregate"):
        gd.aggregate([{"aggregate": True, "variants": {}}])


def test_aggregate_refuses_a_different_variant_set():
    a = _fake_seed_blob(0, 0.4, 0.0, 0.3)
    b = _fake_seed_blob(1, 0.5, 0.0, 0.3)
    b["variants"] = {"advantage_clip": b["variants"]["baseline_clipped_ratio"]}
    with pytest.raises(ValueError, match="different variant set"):
        gd.aggregate([a, b])


# ---------------------------------------------------------------------------
# The measured quantities reach the manuscript as macros
# ---------------------------------------------------------------------------


def _macro_names(text: str) -> list[str]:
    return re.findall(r"\\newcommand\{\\([A-Za-z]*)\}", text)


def _macro_values(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        match = re.match(r"\\newcommand\{\\([A-Za-z]*)\}", line)
        if match:
            out[match.group(1)] = line.split("}{", 1)[1].rstrip("}")
    return out


@pytest.fixture(scope="module")
def suite() -> tuple[dict, float, dict]:
    results = {
        "baseline_rl": {
            "score": 0.4375,
            "score_std": 0.0612,
            "all_scores": [0.3875, 0.4875],
            "history": AblationHistory(effective_batch_size=[4608 / 2, 4608 / 5]),
        },
        "advantage_clip": {
            "score": 0.4125,
            "score_std": 0.01,
            "history": AblationHistory(effective_batch_size=[4608 / 3]),
        },
    }
    return results, 0.475, {"batch_size": 4608}


@pytest.fixture(scope="module")
def emitted(suite, tmp_path_factory) -> str:
    results, pretrained, config = suite
    out = tmp_path_factory.mktemp("tex") / "results.tex"
    write_tex_macros(results, pretrained, out, config=config)
    return out.read_text()


@pytest.fixture(scope="module")
def emitted_with_gdelta(suite, fake_aggregate, tmp_path_factory) -> str:
    results, pretrained, config = suite
    out = tmp_path_factory.mktemp("tex_gd") / "results.tex"
    write_tex_macros(
        results, pretrained, out, config=config, gdelta=fake_aggregate
    )
    return out.read_text()


def test_gdelta_macros_are_emitted_and_tagged_by_ablation(emitted_with_gdelta):
    """Measured quantities appear, tagged by ablation name not variant name."""
    names = set(_macro_names(emitted_with_gdelta))
    required = {
        "mhGdeltaNSeeds",
        "mhGdeltaBcSelfCos",
        "mhGdeltaRandomCosSd",
        "mhGdeltaEqFourResidual",
        # baseline_clipped_ratio tags as BaselineRl, matching mhScoreBaselineRl.
        "mhGdeltaCvABaselineRl",
        "mhGdeltaRatioBaselineRl",
        "mhGdeltaAbarBaselineRl",
        "mhGdeltaRatioShufBaselineRl",
    }
    assert required <= names, f"missing macros: {sorted(required - names)}"
    assert not any(n.endswith("BaselineClippedRatio") for n in names)


def test_gdelta_cv_a_does_not_displace_the_ess_derived_one(
    emitted, emitted_with_gdelta
):
    """The two CV_A macros are different quantities and must both survive.

    ``mhCvABaselineRl`` recovers CV_A from the ESS logged during training;
    ``mhGdeltaCvABaselineRl`` is measured on the measurement batches at the
    pretrained checkpoint. The manuscript quotes both, and they do not agree.
    """
    before = _macro_values(emitted)
    after = _macro_values(emitted_with_gdelta)

    assert {"mhCvABaselineRl", "mhGdeltaCvABaselineRl"} <= set(after)
    # Adding the measurement must not perturb a macro that was already defined.
    assert {k: after[k] for k in before} == before


def test_a_serialised_nan_survives_the_round_trip(suite, fake_aggregate, tmp_path):
    """ESS is NaN when a batch has no winning window, and JSON has no NaN.

    orjson writes it as `null`, so the table and the macros both have to read
    `None` back as NaN rather than crash on a value the measurement is
    entitled to report.
    """
    import copy

    import orjson

    from experiments.rl_finetuning.analysis.tables import make_gdelta_table

    agg = copy.deepcopy(fake_aggregate)
    agg["variants"]["baseline_clipped_ratio"]["ess_fraction_mean"] = float("nan")
    agg = orjson.loads(orjson.dumps(agg))
    assert agg["variants"]["baseline_clipped_ratio"]["ess_fraction_mean"] is None

    df = make_gdelta_table(agg)
    assert df["ESS_Fraction"].is_nan().all()

    results, pretrained, config = suite
    out = tmp_path / "results.tex"
    write_tex_macros(results, pretrained, out, config=config, gdelta=agg)
    assert "\\newcommand{\\mhGdeltaEssBaselineRl}{nan}" in out.read_text()


def test_gdelta_macros_omitted_when_not_measured(emitted):
    """A run with no measurement emits no gdelta macros at all."""
    assert not [n for n in _macro_names(emitted) if n.startswith("mhGdelta")]
