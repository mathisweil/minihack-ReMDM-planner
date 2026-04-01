"""Matplotlib figure generators for the ablation analysis suite.

``generate_all_plots`` accepts a results dict and output directory,
and writes all PNG files.

Style conventions:
- Font sizes: title=13, axis labels=11, ticks=9, legend=9
- Grid: alpha=0.3, linestyle="--"
- Pretrained baseline shown as dashed horizontal in comparison plots
- Group colours: Baseline=grey, A=blue, B=orange, C=green, D=red
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
_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}

_GROUP_COLORS = {
    "Baseline": "#9E9E9E",
    "A": "#2196F3",
    "B": "#FF9800",
    "C": "#4CAF50",
    "D": "#F44336",
}


def _group_color(name: str) -> str:
    """Return plot colour for an ablation by its registry group.

    Args:
        name: Ablation name.

    Returns:
        Hex colour string.
    """
    spec = REGISTRY.get(name)
    group = spec.group if spec else "Baseline"
    return _GROUP_COLORS.get(group, "#9E9E9E")


def _ema(values: list[float], alpha: float = 0.3) -> list[float]:
    """Exponential moving average smoothing.

    Args:
        values: Raw values.
        alpha: Smoothing factor (0 = no smooth, 1 = no memory).

    Returns:
        Smoothed list.
    """
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure and close.

    Args:
        fig: Matplotlib figure.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Per-ablation training curves
# ---------------------------------------------------------------------------


def plot_training_curve(
    name: str,
    history: AblationHistory,
    out_dir: Path,
) -> None:
    """Training loss and win-rate curve for a single ablation.

    Args:
        name: Ablation name.
        history: Training history.
        out_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{name}: Training Curve", fontsize=14)

        ax1.plot(history.iters, _ema(history.loss), color=_group_color(name))
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss (EMA)")

        if history.eval_iters:
            ax2.plot(
                history.eval_iters, history.eval_score,
                "o-", color=_group_color(name),
            )
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("ID Win Rate")
        ax2.set_title("Evaluation Win Rate")

        fig.tight_layout()
        _save(fig, out_dir / f"train_{name}.png")


# ---------------------------------------------------------------------------
# Score comparison bar chart
# ---------------------------------------------------------------------------


def plot_score_comparison(
    results: dict[str, dict],
    pretrained_score: float,
    out_dir: Path,
) -> None:
    """Bar chart comparing final scores across all ablations.

    Args:
        results: ``{name: {"score": float, "history": AblationHistory}}``.
        pretrained_score: Pretrained baseline score.
        out_dir: Output directory.
    """
    names = sorted(results.keys())
    scores = [results[n]["score"] for n in names]
    colors = [_group_color(n) for n in names]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 5))
        bars = ax.bar(range(len(names)), scores, color=colors)
        ax.axhline(pretrained_score, ls="--", color="black", label="Pretrained")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("ID Win Rate")
        ax.set_title("Final Score Comparison")
        ax.legend()
        fig.tight_layout()
        _save(fig, out_dir / "score_comparison.png")


# ---------------------------------------------------------------------------
# Gradient alignment over training
# ---------------------------------------------------------------------------


def plot_grad_alignment(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """Gradient alignment (cos sim) curves for all ablations.

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if h.grad_align_iters:
                ax.plot(
                    h.grad_align_iters, _ema(h.grad_align),
                    label=name, color=_group_color(name), alpha=0.8,
                )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cosine Similarity (RL vs BC)")
        ax.set_title("Gradient Alignment")
        ax.legend(ncol=2, fontsize=7)
        fig.tight_layout()
        _save(fig, out_dir / "grad_alignment.png")


# ---------------------------------------------------------------------------
# KL drift over training
# ---------------------------------------------------------------------------


def plot_repr_drift(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """KL divergence drift curves by t-range.

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
        titles = ["KL mean", "KL low-t", "KL mid-t", "KL high-t"]
        keys = [
            "repr_drift_kl", "repr_drift_kl_low_t",
            "repr_drift_kl_mid_t", "repr_drift_kl_high_t",
        ]
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if not h.repr_drift_iters:
                continue
            for ax, key in zip(axes, keys):
                vals = getattr(h, key)
                ax.plot(
                    h.repr_drift_iters, _ema(vals),
                    label=name, color=_group_color(name), alpha=0.7,
                )
        for ax, title in zip(axes, titles):
            ax.set_xlabel("Iteration")
            ax.set_title(title)
        axes[0].set_ylabel("KL(ref || cur)")
        axes[0].legend(ncol=2, fontsize=6)
        fig.suptitle("Representation Drift", fontsize=14)
        fig.tight_layout()
        _save(fig, out_dir / "repr_drift.png")


# ---------------------------------------------------------------------------
# CKA similarity
# ---------------------------------------------------------------------------


def plot_cka(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """CKA similarity curves.

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if h.cka_iters:
                ax.plot(
                    h.cka_iters, h.cka_similarity,
                    "o-", label=name, color=_group_color(name), alpha=0.8,
                )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("CKA")
        ax.set_title("CKA Similarity vs Pretrained")
        ax.legend(ncol=2, fontsize=7)
        fig.tight_layout()
        _save(fig, out_dir / "cka_similarity.png")


# ---------------------------------------------------------------------------
# t-bin gradient norms heatmap
# ---------------------------------------------------------------------------


def plot_t_bin_norms(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """Heatmap of per-t-bin gradient norms at the last checkpoint.

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    names = sorted(results.keys())
    data_rows: list[list[float]] = []
    bin_labels: list[str] = []

    for name in names:
        h: AblationHistory = results[name]["history"]
        if h.t_bin_norms:
            last = h.t_bin_norms[-1]
            if not bin_labels:
                bin_labels = list(last.keys())
            data_rows.append([last.get(k, 0.0) for k in bin_labels])
        else:
            data_rows.append([])

    valid = [r for r in data_rows if r]
    if not valid:
        return

    n_bins = len(valid[0])
    matrix = np.zeros((len(names), n_bins))
    for i, row in enumerate(data_rows):
        if row:
            matrix[i] = row

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title("Gradient Norms by t-bin (final iteration)")
        fig.colorbar(im, ax=ax, label="L2 Norm")
        fig.tight_layout()
        _save(fig, out_dir / "t_bin_norms.png")


# ---------------------------------------------------------------------------
# High-t / low-t gradient norm ratio
# ---------------------------------------------------------------------------


def plot_t_ratio(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """High-t / low-t gradient norm ratio over training.

    Ratio >> 1 indicates high-t gradients dominate (t-bias hypothesis).

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if h.t_analysis_iters and h.norm_low_t and h.norm_high_t:
                ratios = [
                    hi / (lo + 1e-10)
                    for hi, lo in zip(h.norm_high_t, h.norm_low_t)
                ]
                ax.plot(
                    h.t_analysis_iters, _ema(ratios),
                    "D-", label=name, color=_group_color(name),
                    alpha=0.8, markersize=4, linewidth=1.5,
                )
        ax.axhline(1.0, ls="--", color="black", alpha=0.5, label="Equal")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Ratio (>1 = high-t dominates)")
        ax.set_title("High-t / Low-t Gradient Norm Ratio")
        ax.legend(ncol=2, fontsize=7)
        fig.tight_layout()
        _save(fig, out_dir / "t_ratio.png")


# ---------------------------------------------------------------------------
# Win rate evolution
# ---------------------------------------------------------------------------


def plot_win_rate(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """Win rate over training for all ablations.

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if h.iters and h.win_rate:
                ax.plot(
                    h.iters, _ema(h.win_rate),
                    label=name, color=_group_color(name), alpha=0.8,
                )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Win Rate")
        ax.set_title("Online Win Rate (EMA)")
        ax.legend(ncol=2, fontsize=7)
        fig.tight_layout()
        _save(fig, out_dir / "win_rate.png")


# ---------------------------------------------------------------------------
# Group-level comparison (boxplot)
# ---------------------------------------------------------------------------


def plot_group_comparison(
    results: dict[str, dict],
    pretrained_score: float,
    out_dir: Path,
) -> None:
    """Boxplot of final scores grouped by ablation category.

    Args:
        results: Full results dict.
        pretrained_score: Pretrained score.
        out_dir: Output directory.
    """
    groups: dict[str, list[float]] = {}
    for name, res in results.items():
        spec = REGISTRY.get(name)
        g = spec.group if spec else "Baseline"
        groups.setdefault(g, []).append(res["score"])

    order = ["Baseline", "A", "B", "C", "D"]
    labels = [g for g in order if g in groups]
    data = [groups[g] for g in labels]
    colors = [_GROUP_COLORS[g] for g in labels]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.axhline(pretrained_score, ls="--", color="black", label="Pretrained")
        ax.set_ylabel("ID Win Rate")
        ax.set_title("Score Distribution by Group")
        ax.legend()
        fig.tight_layout()
        _save(fig, out_dir / "group_comparison.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_all_plots(
    results: dict[str, dict],
    pretrained_score: float,
    out_dir: Path,
) -> None:
    """Generate all analysis plots.

    Args:
        results: ``{name: {"score": float, "history": AblationHistory}}``.
        pretrained_score: Pretrained model eval score.
        out_dir: Output directory for PNGs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating plots in %s", out_dir)

    # Per-ablation training curves
    for name, res in results.items():
        plot_training_curve(name, res["history"], out_dir)

    # Cross-ablation comparisons
    plot_score_comparison(results, pretrained_score, out_dir)
    plot_grad_alignment(results, out_dir)
    plot_repr_drift(results, out_dir)
    plot_cka(results, out_dir)
    plot_t_bin_norms(results, out_dir)
    plot_t_ratio(results, out_dir)
    plot_win_rate(results, out_dir)
    plot_group_comparison(results, pretrained_score, out_dir)

    logger.info("All plots generated.")
