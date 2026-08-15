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

import numpy as np
import pytest
import torch

from experiments.rl_finetuning.ablations.training import _effective_batch_size
from experiments.rl_finetuning.analysis.action_distribution import (
    compute_js,
    compute_kl,
)
from experiments.rl_finetuning.analysis.tables import write_significance_test
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
