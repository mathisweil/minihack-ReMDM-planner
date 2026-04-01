"""Hypothesis attribution dashboard and diagnosis report.

Produces ``diagnosis.md`` -- a human-readable verdict that:
1. States the primary failure mode
2. Provides evidence from the ablation results
3. Ranks hypotheses by evidence strength
4. Recommends next experiments

Also generates ``diagnosis_decision_tree.png``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import AblationHistory

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

_DPI = 150

_HYPOTHESIS_GROUPS: dict[str, dict] = {
    "Catastrophic Forgetting": {
        "supporting_ablations": [
            "kl_penalty", "ewc", "llrd", "frozen_backbone", "head_only",
        ],
        "description": (
            "Pretrained representations are corrupted by RL gradients."
        ),
        "recommendation": (
            "Implement strong parameter regularisation (EWC + LLRD) "
            "or use LoRA to restrict update space."
        ),
    },
    "Gradient Conflict": {
        "supporting_ablations": [
            "gradient_surgery", "kl_penalty", "low_t",
        ],
        "description": (
            "RL and BC gradients point in conflicting directions."
        ),
        "recommendation": (
            "Apply PCGrad and investigate t-distribution bias."
        ),
    },
    "Signal Sparsity": {
        "supporting_ablations": [
            "bc_wins", "reward_filtering", "running_stats", "reward_model",
        ],
        "description": (
            "Returns are too sparse or noisy for useful training signal."
        ),
        "recommendation": (
            "Increase episodes per iteration, use reward shaping, "
            "or apply curriculum-based episode selection."
        ),
    },
    "Distributional Shift": {
        "supporting_ablations": ["mixed_replay", "action_diversity"],
        "description": (
            "Online data distribution diverges too far from "
            "offline pretraining distribution."
        ),
        "recommendation": (
            "Maintain large offline replay buffer or apply "
            "importance sampling corrections."
        ),
    },
    "Mode Collapse": {
        "supporting_ablations": [
            "entropy_bonus", "advantage_clip", "normalized_adv",
        ],
        "description": (
            "Model collapses to degenerate distribution, losing "
            "action diversity."
        ),
        "recommendation": (
            "Add strong entropy bonus and clip advantages."
        ),
    },
    "t-Bias": {
        "supporting_ablations": ["low_t", "t_curriculum"],
        "description": (
            "High-t gradients dominate and carry misleading signal."
        ),
        "recommendation": (
            "Restrict training to low-t regime or use t-curriculum."
        ),
    },
}


def _score_hypothesis(
    hyp_name: str,
    hyp_info: dict,
    results: dict[str, dict],
    pretrained_score: float,
) -> dict:
    """Score a hypothesis by supporting ablation success rate.

    An ablation "supports" the hypothesis if its score exceeds
    both the pretrained score and the baseline_rl score.

    Args:
        hyp_name: Hypothesis name.
        hyp_info: Dict with supporting_ablations, description, recommendation.
        results: Ablation results.
        pretrained_score: Pretrained score.

    Returns:
        Scoring dict with evidence_score, n_supporting, n_tested.
    """
    baseline_score = results.get("baseline_rl", {}).get(
        "score", pretrained_score,
    )
    threshold = max(pretrained_score, baseline_score)
    n_tested = 0
    n_supporting = 0

    for abl_name in hyp_info["supporting_ablations"]:
        if abl_name in results:
            n_tested += 1
            if results[abl_name]["score"] > threshold + 0.01:
                n_supporting += 1

    evidence = n_supporting / max(n_tested, 1)

    return {
        "hypothesis": hyp_name,
        "evidence_score": round(evidence, 3),
        "n_supporting": n_supporting,
        "n_tested": n_tested,
        "description": hyp_info["description"],
        "recommendation": hyp_info["recommendation"],
    }


def _plot_decision_tree(
    scored: list[dict],
    out_dir: Path,
) -> None:
    """Horizontal bar chart of hypothesis evidence scores.

    Args:
        scored: List of scored hypothesis dicts.
        out_dir: Output directory.
    """
    scored_sorted = sorted(scored, key=lambda x: x["evidence_score"])
    names = [s["hypothesis"] for s in scored_sorted]
    scores = [s["evidence_score"] for s in scored_sorted]

    with plt.rc_context({"figure.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.5)))
        colors = plt.cm.RdYlGn(np.array(scores))  # type: ignore[attr-defined]
        ax.barh(range(len(names)), scores, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Evidence Score")
        ax.set_title("Hypothesis Evidence (0 = no support, 1 = full support)")
        ax.set_xlim(0, 1.1)
        fig.tight_layout()
        path = out_dir / "diagnosis_decision_tree.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)


def generate_diagnosis_report(
    results: dict[str, dict],
    pretrained_score: float,
    out_dir: Path,
) -> None:
    """Generate ``diagnosis.md`` and decision tree plot.

    Args:
        results: Full ablation results dict.
        pretrained_score: Pretrained model eval score.
        out_dir: Output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = [
        _score_hypothesis(name, info, results, pretrained_score)
        for name, info in _HYPOTHESIS_GROUPS.items()
    ]
    scored.sort(key=lambda x: -x["evidence_score"])

    # Decision tree plot
    _plot_decision_tree(scored, out_dir)

    # Markdown report
    lines = [
        "# Ablation Diagnosis Report",
        "",
        "## Hypothesis Ranking",
        "",
    ]

    for i, s in enumerate(scored, 1):
        stars = "*" * max(1, int(s["evidence_score"] * 5))
        lines.append(
            f"### {i}. {s['hypothesis']} "
            f"[{stars}] ({s['evidence_score']:.0%})"
        )
        lines.append("")
        lines.append(f"**Description:** {s['description']}")
        lines.append("")
        lines.append(
            f"**Evidence:** {s['n_supporting']}/{s['n_tested']} "
            f"supporting ablations improved over baseline."
        )
        lines.append("")
        lines.append(f"**Recommendation:** {s['recommendation']}")
        lines.append("")

    # Summary of individual ablation scores
    lines.append("## Individual Ablation Scores")
    lines.append("")
    lines.append("| Ablation | Group | Score | Delta vs Pretrained |")
    lines.append("|---|---|---|---|")

    baseline_score = results.get("baseline_rl", {}).get(
        "score", pretrained_score,
    )
    for name in sorted(results.keys()):
        res = results[name]
        spec = REGISTRY.get(name)
        group = spec.group if spec else "?"
        delta = res["score"] - pretrained_score
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {name} | {group} | {res['score']:.4f} | "
            f"{sign}{delta:.4f} |"
        )

    lines.append("")
    lines.append(f"*Pretrained score: {pretrained_score:.4f}*")
    lines.append(f"*Baseline RL score: {baseline_score:.4f}*")

    report_path = out_dir / "diagnosis.md"
    report_path.write_text("\n".join(lines))
    logger.info("Saved %s", report_path)
