"""Diagnostics closed-form spec tests (step 8).

Sources: research/spec-ablations.md §3 (CKA per Kornblith 2019
eqs (4)-(5); PCGrad surgery metrics per Yu 2020; ESS per Kish 1965;
§3.4 exact permutation test + bootstrap CI; §3.5 JS divergence).
Expected values are hand-computed in the docstrings. The craftax twin
file carries the same CKA/ESS/surgery assertions; the significance,
action-distribution and merge diagnostics are minihack-specific
(spec-ablations §3.4/§3.5; craftax has no significance tables).
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


def test_exact_permutation_test_and_bootstrap_ci(tmp_path):
    """The significance test is an exact two-sided permutation test over
    all C(n_a+n_b, n_b) relabellings plus a seed-0 bootstrap 95% CI
    (spec-ablations §3.4; mh experiments/README tables).

    Derivation: baseline scores [1,2,3] vs best [7,8,9]: observed mean
    difference 6; of the C(6,3)=20 relabellings only the two extreme
    partitions reach |diff| >= 6, so p = 2/20 = 0.100. Every bootstrap
    resample difference is positive, so the CI excludes 0.
    """
    results = {
        "baseline_rl": {"all_scores": [1.0, 2.0, 3.0]},
        "kl_penalty": {"all_scores": [7.0, 8.0, 9.0]},
    }
    write_significance_test(results, tmp_path)
    text = (tmp_path / "significance_test.txt").read_text()
    assert "20 relabellings" in text
    assert "p = 0.100" in text
    ci_line = next(line for line in text.splitlines() if "bootstrap" in line)
    lo = float(ci_line.split("[")[1].split(",")[0])
    assert lo > 0.0


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
