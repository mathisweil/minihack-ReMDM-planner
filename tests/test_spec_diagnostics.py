"""Diagnostics closed-form spec tests (step 8).

Sources: research/spec-ablations.md §3 (CKA per Kornblith 2019
eqs (4)-(5); PCGrad surgery metrics per Yu 2020; ESS per Kish 1965;
§3.4 exact permutation test + bootstrap CI; §3.5 JS divergence).
Expected values are hand-computed in the docstrings. The craftax twin
file carries the same CKA/ESS/surgery and significance assertions --
`write_significance_test` is byte-identical across the repos -- while
the action-distribution and merge diagnostics are minihack-specific
(spec-ablations §3.5).
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest
import torch

from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import _effective_batch_size
from experiments.rl_finetuning.analysis.action_distribution import (
    compute_js,
    compute_kl,
)
from experiments.rl_finetuning.analysis.report import (
    _HYPOTHESIS_GROUPS,
    _score_hypothesis,
)
from experiments.rl_finetuning.analysis.tables import (
    baseline_rl_score_of,
    metric_scale,
    verdict,
    write_significance_test,
)
from experiments.rl_finetuning.diagnostics.gradient import compute_surgery_metrics
from experiments.rl_finetuning.diagnostics.representation import _linear_cka
from experiments.rl_finetuning.run_ablations import _merge_result_files


def test_linear_cka_is_one_for_identical_and_corr_squared_for_1d():
    """Linear CKA (Kornblith 2019 eqs (4)-(5)): CKA(X, X) = 1 and for
    1-D features CKA = corr^2. Same derivation and numbers as the
    craftax twin: x=[1,2,3,4], y=[1,3,2,4] -> CKA = 0.64.
    """
    x = torch.tensor([[1.0, 0.5], [2.0, -1.0], [-0.5, 0.25], [0.0, 3.0]])
    assert _linear_cka(x, x) == pytest.approx(1.0, abs=1e-5)
    x1 = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y1 = torch.tensor([[1.0], [3.0], [2.0], [4.0]])
    assert _linear_cka(x1, y1) == pytest.approx(0.64, abs=1e-5)


def test_linear_cka_is_invariant_to_scaling_and_orthogonal_maps():
    """CKA(X, c X Q) = 1 for isotropic c and orthogonal Q
    (Kornblith 2019 §2.3)."""
    x = torch.tensor([[1.0, 0.5], [2.0, -1.0], [-0.5, 0.25], [0.0, 3.0]])
    theta = 0.3
    q = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ]
    )
    assert _linear_cka(x, 2.5 * (x @ q)) == pytest.approx(1.0, abs=1e-5)


def test_effective_sample_size_closed_form():
    """ESS = (sum w)^2 / sum w^2 (Kish 1965): w=[1,1,2] -> 16/6;
    uniform weights give N. Same numbers as the craftax twin."""
    assert _effective_batch_size(torch.tensor([1.0, 1.0, 2.0])) == pytest.approx(
        16 / 6, rel=1e-6
    )
    assert _effective_batch_size(torch.ones(7)) == pytest.approx(7.0, rel=1e-6)


def test_surgery_metrics_measure_removed_gradient_mass():
    """Same derivation as the craftax twin: leaf a [2,0]->[1,0], leaf b
    unchanged -> fraction 3/29, one conflicting tensor."""
    before = {"a": torch.tensor([2.0, 0.0]), "b": torch.tensor([3.0, 4.0])}
    after = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([3.0, 4.0])}
    frac, n_conf = compute_surgery_metrics(before, after)
    assert frac == pytest.approx(3 / 29, rel=1e-5)
    assert n_conf == 1


def test_kl_and_js_closed_forms():
    """KL and JS on hand-computable distributions (spec-ablations §3.5:
    JS(p,q) = KL(p||m)/2 + KL(q||m)/2, m = (p+q)/2, natural log).

    Derivation: p=[1,0], q=[0,1] -> m=[0.5,0.5], KL(p||m) = ln 2 ->
    JS = ln 2 (the eps=1e-10 smoothing perturbs this below 1e-4).
    KL(p,p) = JS(p,p) = 0; JS is symmetric.
    """
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert compute_kl(p, p) == pytest.approx(0.0, abs=1e-8)
    assert compute_js(p, p) == pytest.approx(0.0, abs=1e-8)
    assert compute_js(p, q) == pytest.approx(math.log(2), abs=1e-4)
    assert compute_js(p, q) == pytest.approx(compute_js(q, p), abs=1e-12)


def _grad_alignment_setup(tiny_cfg, perturb: float):
    """A model displaced `perturb` from its pretrained reference, and a batch."""
    import copy

    import torch

    from src.diffusion.schedules import get_schedule
    from src.models.denoiser import make_model

    torch.manual_seed(0)
    tiny_cfg._schedule_fn = get_schedule(tiny_cfg.noise_schedule)

    ref_model = make_model(tiny_cfg)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    model = copy.deepcopy(ref_model)
    for param in model.parameters():
        param.requires_grad = True
    if perturb:
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * perturb)

    batch = 8
    local = torch.randint(0, 1000, (batch, tiny_cfg.crop_size, tiny_cfg.crop_size))
    glob = torch.randint(0, 1000, (batch, tiny_cfg.map_h, tiny_cfg.map_w))
    x0 = torch.randint(0, tiny_cfg.action_dim, (batch, tiny_cfg.seq_len))
    return model, ref_model, local.long(), glob.long(), x0.long(), torch.device("cpu")


def test_grad_alignment_shares_one_draw_and_references_the_pretrained_params(tiny_cfg):
    """The RL and BC gradients come from one ``(z_t, t)`` draw, and the BC
    gradient is taken at the pretrained parameters (spec-ablations §3.2; the
    same definition as craftax's `make_grad_alignment_fn`).

    Derivation of the exact case: uniform advantages make the RL loss
    ``(per_sample * 1).mean()`` and the BC loss ``per_sample.mean()`` the
    same expression, so on one draw at one parameter point the two
    gradients are the same vector and the cosine is exactly 1. Anything
    less is the draw differing: at independent draws the metric is a
    Monte-Carlo estimate whose scatter is the size of the quantity, and it
    reports objective disagreement where there is none by construction.

    Displacing the model from the reference then drops the cosine below 1
    while nothing about the objectives has changed, which is what taking
    the BC gradient at a fixed pretrained reference means.
    """
    import torch

    from experiments.rl_finetuning.ablations.losses import _core_loss
    from experiments.rl_finetuning.diagnostics.gradient import (
        _at_reference_parameters,
        _collect_flat_grad,
        compute_grad_alignment,
    )

    model, ref_model, local, glob, x0, device = _grad_alignment_setup(tiny_cfg, 0.0)
    batch = x0.shape[0]
    uniform = torch.ones(batch)

    # One draw, one parameter point, one objective in two spellings.
    cos, rl_norm, bc_norm = compute_grad_alignment(
        model, ref_model, local, glob, x0, uniform, tiny_cfg, device
    )
    assert cos == pytest.approx(1.0, abs=1e-4)
    assert rl_norm == pytest.approx(bc_norm, rel=1e-5)

    def shipped_independent_draws() -> float:
        """What the metric was: a second draw, and the BC gradient at `model`."""
        model.train()
        model.zero_grad()
        _core_loss(model, local, glob, x0, uniform, tiny_cfg, device).backward()
        g_rl = _collect_flat_grad(model)
        model.zero_grad()
        _core_loss(model, local, glob, x0, None, tiny_cfg, device).backward()
        g_bc = _collect_flat_grad(model)
        model.zero_grad()
        return (torch.dot(g_rl, g_bc) / (g_rl.norm() * g_bc.norm() + 1e-10)).item()

    independent = [shipped_independent_draws() for _ in range(5)]
    assert max(independent) < 1.0 - 1e-3
    assert max(independent) - min(independent) > 1e-3

    # The reference is the pretrained point, not wherever the run has got to.
    model, ref_model, local, glob, x0, device = _grad_alignment_setup(tiny_cfg, 0.05)
    displaced, _, _ = compute_grad_alignment(
        model, ref_model, local, glob, x0, uniform, tiny_cfg, device
    )
    assert displaced < 1.0 - 1e-3

    # And the swap that gets it there puts every parameter back.
    before = torch.cat([p.detach().clone().reshape(-1) for p in model.parameters()])
    reference = torch.cat([p.detach().reshape(-1) for p in ref_model.parameters()])
    assert (before - reference).abs().max() > 1e-3
    with _at_reference_parameters(model, ref_model):
        inside = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        assert (inside - reference).abs().max() == pytest.approx(0.0, abs=1e-12)
    after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    assert (after - before).abs().max() == pytest.approx(0.0, abs=1e-12)


def test_the_significance_test_states_its_floor_and_corrects_for_selection(tmp_path):
    """The significance test is exact over all C(n_a+n_b, n_b) relabellings,
    reports the floor that enumeration imposes, and draws its null
    distribution over every candidate arm rather than over the one it picked
    (spec-ablations §3.4; both repos' experiments/README tables).

    Derivation, floor: every relabelling's complement negates each mean
    difference and so ties the statistic, which makes the count at least two
    -- p >= 2/C(6,3) = 0.100 at three seeds a side, for any data whatsoever.
    Baseline [0,0,0] against [1e6,1e6,1e6] therefore reports p = 0.100, and
    0.100 has to be reported as the floor rather than left to read as
    marginal significance.

    Derivation, selection: baseline [0,1,2,3] against [4,5,6,7] has an
    observed difference of 4, which only the two extreme partitions of the
    70 relabellings reach -- p = 2/70 = 0.029 while that arm is the only
    candidate. The null arm [-6,-2,2,6] scores no better than baseline but
    is spread widely enough that its own relabellings reach a statistic of 4
    another twelve times, and it is a candidate the maximum must range over,
    so p becomes 14/70 = 0.200. Selecting the arm from the same scores and
    then testing it uncorrected reports 0.029 either way.
    """
    write_significance_test(
        {
            "baseline_rl": {"all_scores": [0.0, 0.0, 0.0]},
            "kl_penalty": {"all_scores": [1e6, 1e6, 1e6]},
        },
        tmp_path,
    )
    text = (tmp_path / "significance_test.txt").read_text()
    assert "20 relabellings" in text
    assert "p = 0.100" in text
    assert "minimum attainable p at 3 baseline and 3 condition seeds: 0.100" in text
    assert "AT the floor" in text

    alone = tmp_path / "alone"
    write_significance_test(
        {
            "baseline_rl": {"all_scores": [0.0, 1.0, 2.0, 3.0]},
            "kl_penalty": {"all_scores": [4.0, 5.0, 6.0, 7.0]},
        },
        alone,
    )
    text = (alone / "significance_test.txt").read_text()
    assert "1 candidate arm " in text
    assert "p = 0.029" in text
    ci_line = next(line for line in text.splitlines() if "bootstrap" in line)
    assert float(ci_line.split("[")[1].split(",")[0]) > 0.0

    with_null_arm = tmp_path / "with_null_arm"
    write_significance_test(
        {
            "baseline_rl": {"all_scores": [0.0, 1.0, 2.0, 3.0]},
            "kl_penalty": {"all_scores": [4.0, 5.0, 6.0, 7.0]},
            "ewc": {"all_scores": [-6.0, -2.0, 2.0, 6.0]},
        },
        with_null_arm,
    )
    text = (with_null_arm / "significance_test.txt").read_text()
    assert "2 candidate arms" in text
    assert "p = 0.200" in text


def test_merge_concatenates_scores_and_recomputes_over_the_union(tmp_path):
    """--merge concatenates per-seed scores for the same ablation and
    recomputes score/score_std over the union; the merged
    pretrained_score is the mean of the inputs (spec-ablations §1.3).

    Derivation: files with all_scores [1,2] and [3] merge to [1,2,3]:
    score = 2.0, score_std = population std = sqrt(2/3) = 0.8165;
    pretrained (0.4, 0.6) -> 0.5.
    """
    def _file(name, scores, pretrained):
        payload = {
            "pretrained_score": pretrained,
            "config": {"batch_size": 1},
            "ablations": {
                "baseline_rl": {
                    "score": float(np.mean(scores)),
                    "score_std": float(np.std(scores)),
                    "all_scores": scores,
                    "history": {},
                }
            },
        }
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return str(path)

    merged, pretrained, _ = _merge_result_files(
        [_file("a.json", [1.0, 2.0], 0.4), _file("b.json", [3.0], 0.6)]
    )
    assert merged["baseline_rl"]["all_scores"] == [1.0, 2.0, 3.0]
    assert merged["baseline_rl"]["score"] == pytest.approx(2.0)
    assert merged["baseline_rl"]["score_std"] == pytest.approx(
        math.sqrt(2 / 3), rel=1e-6
    )
    assert pretrained == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Ablation-suite verdict rule (shared with the sibling repo, character for
# character; PARITY open question resolved 2026-08-17)
# ---------------------------------------------------------------------------


def test_verdict_labels_against_baseline_rl_at_metric_scale():
    """Labels are taken against `baseline_rl`, with thresholds that are
    fractions of the metric's magnitude: IMPROVEMENT above +5%, COLLAPSE
    below -10%, NEUTRAL between.

    Derivation at scale 10 (`baseline_rl` 10.0, pretrained 8.0, the order
    of magnitude of a Craftax episode-weighted mean return): the
    improvement bar is +0.5 and the collapse bar -1.0, so 10.6 improves,
    10.4 does not, 9.1 holds and 8.9 collapses.

    The last case is the one the absolute rule got wrong. Constructed to
    the recorded shape: an arm sitting 1.911 below `baseline_rl` read
    IMPROVEMENT under the old craftax rule, because +0.089 against
    pretrained cleared an absolute +0.05 bar.
    """
    assert verdict(10.6, 10.0, 8.0) == "IMPROVEMENT"
    assert verdict(10.4, 10.0, 8.0) == "NEUTRAL"
    assert verdict(9.1, 10.0, 8.0) == "NEUTRAL"
    assert verdict(8.9, 10.0, 8.0) == "COLLAPSE"
    assert verdict(10.0 - 1.911, 10.0, 8.0) == "COLLAPSE"


def test_verdict_reduces_to_the_absolute_rule_at_a_metric_scale_of_one():
    """At scale 1.0 the fractions are the absolute +0.05 / -0.10 they
    replace, and both comparisons are strict.

    This is the anchor for a bounded metric: a MiniHack win rate lives in
    [0, 1], so the rule that governed it is unchanged in form. With
    `baseline_rl` 0.0 and pretrained 1.0 the scale is exactly 1.0 and the
    delta is the score itself, so the boundaries are exact in float.
    """
    assert verdict(0.05, 0.0, 1.0) == "NEUTRAL"
    assert verdict(0.06, 0.0, 1.0) == "IMPROVEMENT"
    assert verdict(-0.10, 0.0, 1.0) == "NEUTRAL"
    assert verdict(-0.11, 0.0, 1.0) == "COLLAPSE"


def test_verdict_scale_is_the_larger_reference_and_one_is_required():
    """The scale is the larger reference score in absolute value, so a
    `baseline_rl` near zero cannot shrink the threshold to nothing; with
    both references at zero there is no scale and no label is defensible.

    Derivation: `baseline_rl` 0.0 with pretrained 8.0 gives scale 8.0, so
    the bars are +0.4 and -0.8, not +0.0 and -0.0.
    """
    assert metric_scale(0.0, 8.0) == 8.0
    assert metric_scale(10.0, 8.0) == 10.0
    assert metric_scale(-3.0, 1.0) == 3.0

    assert verdict(0.39, 0.0, 8.0) == "NEUTRAL"
    assert verdict(0.41, 0.0, 8.0) == "IMPROVEMENT"
    assert verdict(-0.79, 0.0, 8.0) == "NEUTRAL"
    assert verdict(-0.81, 0.0, 8.0) == "COLLAPSE"

    assert verdict(0.0, 0.0, 0.0) == "NEUTRAL"
    assert verdict(1.0, 0.0, 0.0) == "NEUTRAL"


def test_the_reference_arm_falls_back_to_the_pretrained_score():
    """A suite run without `baseline_rl` has no reference arm, so the
    pretrained score stands in and every delta is measured from it."""
    assert baseline_rl_score_of({"baseline_rl": {"score": 0.7}}, 0.5) == 0.7
    assert baseline_rl_score_of({"kl_penalty": {"score": 0.6}}, 0.5) == 0.5


# ---------------------------------------------------------------------------
# Hypothesis attribution: the evidence set and the recommendation must agree
# (shared with the sibling repo, character for character; S7-9, decided
# 2026-08-18)
# ---------------------------------------------------------------------------

# The six groups and their evidence sets, pinned to the same literal in both
# repos. `analysis/report.py` carried no test at all until now, and the two
# dicts drifting apart would put different numbers under one heading in a
# cross-repo table. Update both files together or not at all.
_EXPECTED_EVIDENCE_SETS = {
    "Catastrophic Forgetting": [
        "ewc", "frozen_backbone", "head_only", "kl_penalty", "llrd", "lora",
    ],
    "Gradient Conflict": ["gradient_surgery", "kl_penalty", "low_t"],
    "Signal Sparsity": [
        "bc_wins", "reward_filtering", "reward_model", "running_stats",
    ],
    "Distributional Shift": ["action_diversity", "mixed_replay"],
    "Mode Collapse": ["advantage_clip", "entropy_bonus", "normalized_adv"],
    "t-Bias": ["low_t", "t_curriculum"],
}


def _named_in(text: str, arm: str) -> bool:
    """Does *text* name *arm* by its registry name, in prose?

    Registry keys are snake_case and the recommendations write them as prose,
    so the separator is relaxed to space, underscore or hyphen: `low_t`
    appears as "low-t" and `entropy_bonus` as "entropy bonus". The word
    bounds are what keep this honest -- a bare substring test would find
    `ewc` inside any word containing those letters, and matching on the
    relaxed separator alone would miss the hyphenated forms entirely.
    """
    pattern = r"\b" + r"[ _\-]".join(re.escape(part) for part in arm.split("_")) + r"\b"
    return re.search(pattern, text, re.IGNORECASE) is not None


def test_every_arm_a_recommendation_names_is_in_its_own_evidence_set():
    """A hypothesis may not recommend an intervention whose ablation it
    excludes from the evidence that scores it.

    `Catastrophic Forgetting` recommended LoRA -- "or use LoRA to restrict
    the parameter update space" -- while omitting the `lora` arm from its
    `supporting_ablations`, in both repos identically. Not cosmetic:
    `_score_hypothesis` computes `evidence_score = n_supporting /
    max(n_tested, 1)` over that list, so the omission changes the ranking
    `diagnosis.md` and the hypothesis-verdict tables print. Author decision
    2026-08-18: drift, not scoping.

    Only recommendations that name a registered arm are constrained. That
    eight of the 25 arms are cited by no hypothesis at all is a separate,
    deliberately open question and is not asserted here.
    """
    offenders = {
        name: sorted(
            arm
            for arm in REGISTRY
            if _named_in(info["recommendation"], arm)
            and arm not in info["supporting_ablations"]
        )
        for name, info in _HYPOTHESIS_GROUPS.items()
    }
    offenders = {name: arms for name, arms in offenders.items() if arms}

    assert not offenders, (
        "hypotheses recommending an intervention whose arm they leave out of "
        f"their own evidence set: {offenders}"
    )


def test_the_hypothesis_evidence_sets_are_the_pinned_shared_ones():
    """The groups and their membership are identical across the two repos.

    Nothing else pins `_HYPOTHESIS_GROUPS`, and it is the input to every
    number in `diagnosis.md`'s hypothesis ranking, so silent drift here is
    invisible until two repos disagree in one table.
    """
    actual = {
        name: sorted(info["supporting_ablations"])
        for name, info in _HYPOTHESIS_GROUPS.items()
    }

    assert actual == _EXPECTED_EVIDENCE_SETS


def test_the_evidence_score_is_the_raw_quotient_in_both_repos():
    """`evidence_score` is the unrounded fraction; rounding is for display.

    minihack returned `round(evidence, 3)` and craftax the raw quotient, so
    the same inputs gave 0.3330 and 0.3333 under one field name. Every
    consumer already formats at the point of use -- `:.0%` in the report
    tables and `int(score * 5)` for the star rating -- so rounding inside the
    scorer bought nothing and cost cross-repo agreement.
    """
    results = {
        "baseline_rl": {"score": 10.0},
        "kl_penalty": {"score": 10.5},
        "ewc": {"score": 10.5},
        "llrd": {"score": 9.0},
        "lora": {"score": 9.0},
        "frozen_backbone": {"score": 9.0},
        "head_only": {"score": 9.0},
    }
    scored = _score_hypothesis(
        "Catastrophic Forgetting",
        _HYPOTHESIS_GROUPS["Catastrophic Forgetting"],
        results,
        8.0,
    )

    assert scored["n_supporting"] == 2
    assert scored["n_tested"] == 6
    assert scored["evidence_score"] == 2 / 6


def test_an_unregistered_supporting_arm_is_an_error():
    """A typo'd or retired arm name must not be scored as a smaller sample.

    `_score_hypothesis` skips arms absent from `results`, which is correct for
    a run that did not include them -- and indistinguishable from a name that
    can never appear. Left unguarded, renaming an arm silently lowers
    `n_tested` and moves every evidence score that cites it.
    """
    broken = dict(_HYPOTHESIS_GROUPS["Catastrophic Forgetting"])
    broken["supporting_ablations"] = ["ewc", "not_an_ablation"]

    with pytest.raises(KeyError, match="not_an_ablation"):
        _score_hypothesis("Catastrophic Forgetting", broken, {}, 0.0)
