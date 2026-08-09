"""Matplotlib figure generators for the ablation analysis suite.

``generate_all_plots`` accepts a results dict and output directory,
and writes all PNG files.

Style conventions:
- Font sizes: title=13, axis labels=11, ticks=9, legend=9
- Grid: alpha=0.3, linestyle="--"
- Pretrained baseline shown as dashed horizontal in comparison plots
- Colorblind-safe group palette (Wong 2011)
- Each ablation gets a unique color + linestyle within its group
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.rl_finetuning.ablations.registry import (  # noqa: E402
    REGISTRY,
)
from experiments.rl_finetuning.ablations.training import (  # noqa: E402
    AblationHistory,
)

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

# Colorblind-safe group palette (Wong 2011).
_GROUP_COLORS = {
    "Baseline": "#999999",
    "A": "#0072B2",
    "B": "#E69F00",
    "C": "#CC79A7",
    "D": "#D55E00",
}

# Line style cycling within groups for visual distinction.
_LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
_MARKERS = ["o", "s", "D", "^", "v", "P", "X"]

# Cache: ablation name -> (color, linestyle, marker).
_ABLATION_STYLE_CACHE: dict[str, tuple[str, str, str]] = {}


def _build_ablation_styles() -> None:
    """Assign each ablation a unique (color, linestyle, marker)."""
    if _ABLATION_STYLE_CACHE:
        return
    groups: dict[str, list[str]] = {}
    for name, spec in REGISTRY.items():
        groups.setdefault(spec.group, []).append(name)
    for group, names in groups.items():
        color = _GROUP_COLORS.get(group, "#999999")
        for i, name in enumerate(sorted(names)):
            ls = _LINE_STYLES[i % len(_LINE_STYLES)]
            mk = _MARKERS[i % len(_MARKERS)]
            _ABLATION_STYLE_CACHE[name] = (color, ls, mk)


def _ablation_style(name: str) -> tuple[str, str, str]:
    """Return ``(color, linestyle, marker)`` for an ablation.

    Args:
        name: Ablation name.

    Returns:
        Tuple of hex colour, linestyle, marker string.
    """
    _build_ablation_styles()
    if name in _ABLATION_STYLE_CACHE:
        return _ABLATION_STYLE_CACHE[name]
    spec = REGISTRY.get(name)
    group = spec.group if spec else "Baseline"
    return _GROUP_COLORS.get(group, "#999999"), "-", "o"


def _group_color(name: str) -> str:
    """Return plot colour for an ablation by its registry group.

    Args:
        name: Ablation name.

    Returns:
        Hex colour string.
    """
    return _ablation_style(name)[0]


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
    # Replace None (from NaN -> JSON null -> None roundtrip) with 0.0
    clean = [v if v is not None else 0.0 for v in values]
    out = [clean[0]]
    for v in clean[1:]:
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
                history.eval_iters,
                history.eval_score,
                "o-",
                color=_group_color(name),
            )
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("ID Win Rate")
        ax2.set_title("Evaluation Win Rate")

        fig.tight_layout()
        _save(fig, out_dir / f"train_{name}.png")


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
        fig, ax = plt.subplots(figsize=(max(10.0, len(names) * 0.6), 5))
        ax.bar(range(len(names)), scores, color=colors)
        ax.axhline(pretrained_score, ls="--", color="black", label="Pretrained")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("ID Win Rate")
        ax.set_title("Final Score Comparison")
        ax.legend()
        fig.tight_layout()
        _save(fig, out_dir / "score_comparison.png")


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
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if h.grad_align_iters:
                c, ls, mk = _ablation_style(name)
                ax.plot(
                    h.grad_align_iters,
                    _ema(h.grad_align),
                    label=name,
                    color=c,
                    linestyle=ls,
                    alpha=0.8,
                    linewidth=1.5,
                )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cosine Similarity (RL vs BC)")
        ax.set_title("Gradient Alignment")
        ax.legend(
            ncol=3,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(0, -0.15),
        )
        fig.tight_layout()
        _save(fig, out_dir / "grad_alignment.png")


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
            "repr_drift_kl",
            "repr_drift_kl_low_t",
            "repr_drift_kl_mid_t",
            "repr_drift_kl_high_t",
        ]
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if not h.repr_drift_iters:
                continue
            c, ls, _ = _ablation_style(name)
            for ax, key in zip(axes, keys):
                vals = getattr(h, key)
                ax.plot(
                    h.repr_drift_iters,
                    _ema(vals),
                    label=name,
                    color=c,
                    linestyle=ls,
                    alpha=0.7,
                )
        for ax, title in zip(axes, titles):
            ax.set_xlabel("Iteration")
            ax.set_title(title)
        axes[0].set_ylabel("KL(ref || cur)")
        axes[0].legend(
            ncol=3,
            fontsize=6,
            loc="upper left",
            bbox_to_anchor=(0, -0.18),
        )
        fig.suptitle("Representation Drift", fontsize=14)
        fig.tight_layout()
        _save(fig, out_dir / "repr_drift.png")


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
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if h.cka_iters:
                c, ls, mk = _ablation_style(name)
                ax.plot(
                    h.cka_iters,
                    h.cka_similarity,
                    label=name,
                    color=c,
                    linestyle=ls,
                    marker=mk,
                    markersize=4,
                    alpha=0.8,
                )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("CKA")
        ax.set_title("CKA Similarity vs Pretrained")
        ax.legend(
            ncol=3,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(0, -0.15),
        )
        fig.tight_layout()
        _save(fig, out_dir / "cka_similarity.png")


def plot_t_bin_norms(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """Heatmap of per-t-bin gradient norms at the last checkpoint.

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    valid_names: list[str] = []
    data_rows: list[list[float]] = []
    bin_labels: list[str] = []

    for name in sorted(results.keys()):
        h: AblationHistory = results[name]["history"]
        if not h.t_bin_norms:
            continue
        last = h.t_bin_norms[-1]
        if not bin_labels:
            bin_labels = list(last.keys())
        data_rows.append([last.get(k, 0.0) for k in bin_labels])
        valid_names.append(name)

    if not data_rows:
        return

    n_bins = len(data_rows[0])
    matrix = np.array(data_rows)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(
            figsize=(10, max(4.0, len(valid_names) * 0.35)),
        )
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(valid_names)))
        ax.set_yticklabels(valid_names, fontsize=8)
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title("Gradient Norms by t-bin (final iteration)")
        fig.colorbar(im, ax=ax, label="L2 Norm")
        fig.tight_layout()
        _save(fig, out_dir / "t_bin_norms.png")


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
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if h.t_analysis_iters and h.norm_low_t and h.norm_high_t:
                ratios = [
                    hi / (lo + 1e-10) for hi, lo in zip(h.norm_high_t, h.norm_low_t)
                ]
                c, ls, mk = _ablation_style(name)
                ax.plot(
                    h.t_analysis_iters,
                    _ema(ratios),
                    label=name,
                    color=c,
                    linestyle=ls,
                    marker=mk,
                    alpha=0.8,
                    markersize=4,
                    linewidth=1.5,
                )
        ax.axhline(1.0, ls="--", color="black", alpha=0.5, label="Equal")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Ratio (>1 = high-t dominates)")
        ax.set_title("High-t / Low-t Gradient Norm Ratio")
        ax.legend(
            ncol=3,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(0, -0.15),
        )
        fig.tight_layout()
        _save(fig, out_dir / "t_ratio.png")


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
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if h.iters and h.win_rate:
                c, ls, _ = _ablation_style(name)
                ax.plot(
                    h.iters,
                    _ema(h.win_rate),
                    label=name,
                    color=c,
                    linestyle=ls,
                    alpha=0.8,
                    linewidth=1.5,
                )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Win Rate")
        ax.set_title("Online Win Rate (EMA)")
        ax.legend(
            ncol=3,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(0, -0.15),
        )
        fig.tight_layout()
        _save(fig, out_dir / "win_rate.png")


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
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.axhline(pretrained_score, ls="--", color="black", label="Pretrained")
        ax.set_ylabel("ID Win Rate")
        ax.set_title("Score Distribution by Group")
        ax.legend()
        fig.tight_layout()
        _save(fig, out_dir / "group_comparison.png")


def plot_gradient_conflict_map(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """Binary heatmap of gradient conflicts (cos_sim < 0).

    Rows are ablations, columns are diagnostic iterations. A red
    cell indicates the RL gradient conflicted with the BC gradient
    at that checkpoint.

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    all_iters: set[int] = set()
    for res in results.values():
        h: AblationHistory = res["history"]
        all_iters.update(h.grad_align_iters)
    if not all_iters:
        return

    sorted_iters = sorted(all_iters)
    iter_to_col = {it: j for j, it in enumerate(sorted_iters)}
    names = sorted(results.keys())
    matrix = np.full((len(names), len(sorted_iters)), np.nan)

    for i, name in enumerate(names):
        h = results[name]["history"]
        for it, val in zip(h.grad_align_iters, h.grad_align):
            matrix[i, iter_to_col[it]] = 1.0 if val < 0 else 0.0

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(
            figsize=(max(8, len(sorted_iters) * 0.4), max(4, len(names) * 0.35)),
        )
        cmap = plt.cm.RdYlGn_r.copy()  # type: ignore[attr-defined]
        cmap.set_bad(color="white")
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xticks(range(0, len(sorted_iters), max(1, len(sorted_iters) // 10)))
        ax.set_xticklabels(
            [
                str(sorted_iters[j])
                for j in range(
                    0,
                    len(sorted_iters),
                    max(1, len(sorted_iters) // 10),
                )
            ],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_xlabel("Iteration")
        ax.set_title("Gradient Conflict Map (red = cos_sim(RL, BC) < 0)")
        fig.colorbar(im, ax=ax, label="Conflict", ticks=[0, 1])
        fig.tight_layout()
        _save(fig, out_dir / "gradient_conflict_map.png")


def plot_score_delta(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """Sorted horizontal bar chart of score improvement over baseline_rl.

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    baseline_score = results.get("baseline_rl", {}).get("score", 0.0)
    names = sorted(
        results.keys(),
        key=lambda n: results[n]["score"] - baseline_score,
    )
    deltas = [results[n]["score"] - baseline_score for n in names]
    colors = [_group_color(n) for n in names]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(
            figsize=(8, max(4, len(names) * 0.35)),
        )
        ax.barh(range(len(names)), deltas, color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Delta vs Baseline RL")
        ax.set_title("Score Improvement Over Baseline RL (sorted)")
        fig.tight_layout()
        _save(fig, out_dir / "score_delta.png")


def plot_per_env_delta(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """Heatmap of per-environment win rate change (end minus start).

    Rows are ablations, columns are environments. Blue = improved,
    red = degraded.

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    env_names: list[str] = []
    for res in results.values():
        h: AblationHistory = res["history"]
        if h.per_env_win_rates and len(h.per_env_win_rates) >= 2:
            env_names = sorted(h.per_env_win_rates[0].keys())
            break
    if not env_names:
        return

    short_envs = [e.replace("MiniHack-", "").replace("-v0", "") for e in env_names]

    valid_names: list[str] = []
    data_rows: list[list[float]] = []
    for name in sorted(results.keys()):
        h = results[name]["history"]
        if h.per_env_win_rates and len(h.per_env_win_rates) >= 2:
            start = h.per_env_win_rates[0]
            end = h.per_env_win_rates[-1]
            data_rows.append([end.get(e, 0.0) - start.get(e, 0.0) for e in env_names])
            valid_names.append(name)

    if not data_rows:
        return

    matrix = np.array(data_rows)
    v_abs = max(float(np.abs(matrix).max()), 0.01)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(
            figsize=(max(8, len(env_names) * 1.5), max(4, len(valid_names) * 0.35)),
        )
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu",
            vmin=-v_abs,
            vmax=v_abs,
        )
        ax.set_yticks(range(len(valid_names)))
        ax.set_yticklabels(valid_names, fontsize=8)
        ax.set_xticks(range(len(short_envs)))
        ax.set_xticklabels(short_envs, rotation=45, ha="right", fontsize=9)
        ax.set_title("Per-Environment Win Rate Change (End - Start)")
        fig.colorbar(im, ax=ax, label="Delta Win Rate")
        fig.tight_layout()
        _save(fig, out_dir / "per_env_delta.png")


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
    plot_gradient_conflict_map(results, out_dir)
    plot_score_delta(results, out_dir)
    plot_per_env_delta(results, out_dir)

    logger.info("All plots generated.")
