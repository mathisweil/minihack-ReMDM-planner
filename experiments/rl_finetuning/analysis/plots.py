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
    """Exponential moving average smoothing for a list of scalars.

    A missing value -- ``None`` after a NaN has round-tripped through JSON,
    or a NaN itself -- is missing data, not a measurement of zero. It is
    carried through as NaN, which matplotlib draws as a **gap** in the
    line, and the average steps over it rather than being pulled towards
    zero by it: a run whose evaluation failed once would otherwise show a
    win rate collapsing to 0 and recovering, which is a finding the suite
    would report and nothing that happened.

    Args:
        values: Raw scalar values, possibly with ``None`` or NaN holes.
        alpha:  Smoothing factor (0=no smoothing, 1=no memory).

    Returns:
        Smoothed list of the same length, NaN wherever the input was
        missing, and NaN in the leading positions until the first real
        value arrives.
    """
    if not values:
        return []
    smoothed: list[float] = []
    state: float | None = None
    for v in values:
        if v is None or v != v:  # None, or NaN, which is not equal to itself
            smoothed.append(float("nan"))
            continue
        state = float(v) if state is None else alpha * float(v) + (1 - alpha) * state
        smoothed.append(state)
    return smoothed


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


def plot_ablation_curves(
    name: str,
    history: AblationHistory,
    pretrained_score: float,
    out_dir: Path,
) -> None:
    """2x3 grid of per-ablation training curves.

    Panels: eval score, training loss, online env score, KL drift,
    gradient alignment, gradient norms.

    Args:
        name: Ablation name.
        history: Training history.
        pretrained_score: Pretrained baseline score (dashed horizontal).
        out_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Training Curves: {name}", fontsize=14, fontweight="bold")
        color = _group_color(name)

        ax = axes[0, 0]
        if history.eval_iters:
            ax.plot(
                history.eval_iters,
                history.eval_score,
                "o-",
                color=color,
                linewidth=1.5,
                label="eval",
            )
        ax.axhline(
            pretrained_score,
            ls="--",
            color="black",
            alpha=0.6,
            label="pretrained",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("ID Win Rate")
        ax.set_title("Eval Win Rate vs Iteration")
        ax.legend()

        ax = axes[0, 1]
        if history.iters:
            ax.plot(
                history.iters,
                history.loss,
                color=color,
                alpha=0.4,
                linewidth=0.8,
                label="raw",
            )
            ax.plot(
                history.iters,
                _ema(history.loss),
                color=color,
                linewidth=1.5,
                label="EMA",
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss vs Iteration")
        ax.legend()

        ax = axes[0, 2]
        if history.env_score_iters:
            ax.plot(
                history.env_score_iters,
                _ema(history.env_score),
                color=color,
                linewidth=1.5,
            )
        ax.axhline(pretrained_score, ls="--", color="black", alpha=0.6)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_title("Online Env Score vs Iteration")

        ax = axes[1, 0]
        if history.repr_drift_iters:
            ax.plot(
                history.repr_drift_iters,
                history.repr_drift_kl,
                color=color,
                linewidth=1.5,
                label="mean",
            )
            if history.repr_drift_kl_low_t:
                ax.plot(
                    history.repr_drift_iters,
                    history.repr_drift_kl_low_t,
                    color=color,
                    alpha=0.5,
                    linestyle=":",
                    label="low-t",
                )
            if history.repr_drift_kl_high_t:
                ax.plot(
                    history.repr_drift_iters,
                    history.repr_drift_kl_high_t,
                    color=color,
                    alpha=0.5,
                    linestyle="-.",
                    label="high-t",
                )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Drift from Pretrained")
        ax.legend()

        ax = axes[1, 1]
        if history.grad_align_iters:
            ax.plot(
                history.grad_align_iters,
                history.grad_align,
                color=color,
                linewidth=1.5,
            )
            ax.axhline(0, ls="--", color="black", alpha=0.4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Gradient Alignment (cos sim vs BC)")
        ax.set_ylim(-1.1, 1.1)

        ax = axes[1, 2]
        if history.grad_align_iters:
            ax.plot(
                history.grad_align_iters,
                history.rl_grad_norm,
                color=color,
                linewidth=1.5,
                label="RL grad",
            )
            ax.plot(
                history.grad_align_iters,
                history.bc_grad_norm,
                color=color,
                linestyle="--",
                linewidth=1.2,
                label="BC grad",
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("L2 Norm")
        ax.set_title("Gradient Norms")
        ax.legend()

        fig.tight_layout()
        _save(fig, out_dir / f"curves_{name}.png")


def plot_per_layer_gradient_heatmap(
    name: str,
    history: AblationHistory,
    out_dir: Path,
) -> None:
    """Heatmap of per-layer gradient norms over training iterations.

    Args:
        name: Ablation name.
        history: Training history.
        out_dir: Output directory.
    """
    if not history.per_layer_norms:
        return

    all_keys = sorted({k for d in history.per_layer_norms for k in d})
    if not all_keys:
        return

    matrix = np.array(
        [[d.get(k, 0.0) for k in all_keys] for d in history.per_layer_norms]
    ).T
    iters = history.per_layer_iters or list(range(len(history.per_layer_norms)))

    with plt.rc_context(_STYLE):
        fig_h = max(4.0, min(18.0, len(all_keys) * 0.22))
        fig, ax = plt.subplots(figsize=(12, fig_h))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xticks(range(len(iters)))
        ax.set_xticklabels([str(i) for i in iters], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(all_keys)))
        ax.set_yticklabels(all_keys, fontsize=5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Layer")
        ax.set_title(f"Per-Layer Gradient Norms: {name}")
        fig.colorbar(im, ax=ax, label="L2 Norm")
        fig.tight_layout()
        _save(fig, out_dir / f"per_layer_grad_heatmap_{name}.png")


def plot_t_bin_grad_norms(
    name: str,
    history: AblationHistory,
    out_dir: Path,
) -> None:
    """Per-t-bin gradient norms over training for a single ablation.

    Args:
        name: Ablation name.
        history: Training history.
        out_dir: Output directory.
    """
    if not history.t_bin_norms:
        return

    bins = sorted({k for d in history.t_bin_norms for k in d})
    iters = history.t_analysis_iters or list(range(len(history.t_bin_norms)))
    cmap = matplotlib.colormaps["plasma"].resampled(max(len(bins), 1))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        for j, bin_key in enumerate(bins):
            vals = [d.get(bin_key, 0.0) for d in history.t_bin_norms]
            ax.plot(iters, vals, color=cmap(j), linewidth=1.2, alpha=0.8, label=bin_key)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("L2 Norm")
        ax.set_title(f"Per-t-Bin Gradient Norms: {name}")
        ax.legend(loc="upper right", ncol=3, fontsize=7)
        fig.tight_layout()
        _save(fig, out_dir / f"t_bin_grad_norms_{name}.png")


def plot_per_env_collapse_heatmap(
    name: str,
    history: AblationHistory,
    out_dir: Path,
) -> None:
    """Heatmap: rows=environments, cols=eval iterations, colour=win rate.

    One figure per ablation, showing which environments are lost first
    during collapse.

    Args:
        name: Ablation name.
        history: Training history.
        out_dir: Output directory.
    """
    if not history.per_env_win_rates:
        return

    env_names = sorted({k for d in history.per_env_win_rates for k in d})
    if not env_names:
        return

    short_envs = [e.replace("MiniHack-", "").replace("-v0", "") for e in env_names]
    n_evals = len(history.per_env_win_rates)
    matrix = np.array(
        [[d.get(e, 0.0) for d in history.per_env_win_rates] for e in env_names],
        dtype=np.float32,
    )
    iters = history.eval_iters or list(range(n_evals))

    with plt.rc_context(_STYLE):
        fig_h = max(3.0, len(env_names) * 0.45 + 1.5)
        fig, ax = plt.subplots(figsize=(max(10.0, n_evals * 0.5), fig_h))
        im = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            cmap="YlOrRd_r",
        )
        ax.set_yticks(range(len(env_names)))
        ax.set_yticklabels(short_envs, fontsize=8)
        step = max(1, n_evals // 10)
        ax.set_xticks(range(0, n_evals, step))
        ax.set_xticklabels(
            [str(iters[i]) for i in range(0, n_evals, step)],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_xlabel("Eval Iteration")
        ax.set_ylabel("Environment")
        ax.set_title(f"Per-Environment Collapse Heatmap: {name}")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Win Rate", fontsize=8)
        fig.tight_layout()
        _save(fig, out_dir / f"per_env_collapse_{name}.png")


def plot_final_score_comparison(
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
        _save(fig, out_dir / "final_score_comparison.png")


def plot_eval_scores_over_training(
    results: dict[str, dict],
    pretrained_score: float,
    out_dir: Path,
) -> None:
    """All ablation eval scores overlaid on the same axes.

    Args:
        results: Full results dict.
        pretrained_score: Pretrained baseline score (dashed horizontal).
        out_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axhline(
            pretrained_score,
            ls="--",
            color="black",
            linewidth=1.5,
            label="Pretrained",
        )
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if not h.eval_iters:
                continue
            c, ls, mk = _ablation_style(name)
            ax.plot(
                h.eval_iters,
                h.eval_score,
                label=name,
                color=c,
                linestyle=ls,
                marker=mk,
                markersize=3,
                alpha=0.8,
                linewidth=1.2,
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("ID Win Rate")
        ax.set_title("Eval Win Rate vs Iteration (all ablations)")
        ax.legend(
            ncol=3,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(0, -0.15),
        )
        fig.tight_layout()
        _save(fig, out_dir / "eval_scores_over_training.png")


def plot_gradient_alignment(
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
        _save(fig, out_dir / "gradient_alignment.png")


def plot_representation_drift(
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
            for ax, key in zip(axes, keys, strict=False):
                vals = getattr(h, key)
                ax.plot(
                    h.repr_drift_iters,
                    _ema(vals),
                    label=name,
                    color=c,
                    linestyle=ls,
                    alpha=0.7,
                )
        for ax, title in zip(axes, titles, strict=False):
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
        _save(fig, out_dir / "representation_drift.png")


def plot_cka_similarity(
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


def plot_t_bin_norms_heatmap(
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
        _save(fig, out_dir / "t_bin_norms_heatmap.png")


def plot_t_analysis(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """High-t / low-t gradient norm ratio and low-high cosine alignment.

    Left panel: ratio >> 1 indicates high-t gradients dominate (t-bias
    hypothesis). Right panel: cosine similarity between the low-t and
    high-t gradient directions; negative values indicate the two regimes
    pull against each other.

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if not h.t_analysis_iters:
                continue
            c, ls, mk = _ablation_style(name)
            if h.norm_low_t and h.norm_high_t:
                ratios = [
                    hi / (lo + 1e-10)
                    for hi, lo in zip(h.norm_high_t, h.norm_low_t, strict=False)
                ]
                axes[0].plot(
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
            if h.lowhigh_cos:
                axes[1].plot(
                    h.t_analysis_iters,
                    h.lowhigh_cos,
                    label=name,
                    color=c,
                    linestyle=ls,
                    alpha=0.8,
                    linewidth=1.2,
                )

        axes[0].axhline(1.0, ls="--", color="black", alpha=0.5, label="Equal")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Ratio (>1 = high-t dominates)")
        axes[0].set_title("High-t / Low-t Gradient Norm Ratio")

        axes[1].axhline(0.0, ls="--", color="black", alpha=0.4)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_title("Low-t / High-t Gradient Cosine Similarity")
        axes[1].set_ylim(-1.1, 1.1)

        axes[0].legend(
            ncol=3,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(0, -0.15),
        )
        fig.tight_layout()
        _save(fig, out_dir / "t_distribution_analysis.png")


def plot_return_distributions(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """Online win rate and effective batch size over training.

    Effective batch size is the number of trajectories with non-degenerate
    advantages; a collapse towards zero means most of the batch contributes
    no learning signal.

    Args:
        results: Full results dict.
        out_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, res in sorted(results.items()):
            h: AblationHistory = res["history"]
            if not h.iters:
                continue
            c, ls, _ = _ablation_style(name)
            if h.win_rate:
                axes[0].plot(
                    h.iters,
                    _ema(h.win_rate),
                    label=name,
                    color=c,
                    linestyle=ls,
                    alpha=0.8,
                    linewidth=1.5,
                )
            if h.effective_batch_size:
                axes[1].plot(
                    h.iters,
                    _ema(h.effective_batch_size),
                    label=name,
                    color=c,
                    linestyle=ls,
                    alpha=0.8,
                    linewidth=1.5,
                )

        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Win Rate")
        axes[0].set_title("Online Win Rate (EMA)")

        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Effective Batch Size")
        axes[1].set_title("Effective Batch Size (EMA)")

        axes[0].legend(
            ncol=3,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(0, -0.15),
        )
        fig.tight_layout()
        _save(fig, out_dir / "win_rate_and_effective_batch_size.png")


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
        for patch, c in zip(bp["boxes"], colors, strict=False):
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
        for it, val in zip(h.grad_align_iters, h.grad_align, strict=False):
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
        _save(fig, out_dir / "score_delta_over_baseline_rl.png")


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
    output_dir: Path,
) -> None:
    """Generate all analysis figures and save to ``output_dir/figures/``.

    Args:
        results: ``{name: {"score": float, "history": AblationHistory}}``.
        pretrained_score: Pretrained model eval score.
        output_dir: Root output directory; figures go in ``output_dir/figures/``.
    """
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating plots in %s", fig_dir)

    logger.info("Generating per-ablation training curves...")
    for name, res in results.items():
        plot_ablation_curves(name, res["history"], pretrained_score, fig_dir)
        plot_per_layer_gradient_heatmap(name, res["history"], fig_dir)
        plot_t_bin_grad_norms(name, res["history"], fig_dir)
        plot_per_env_collapse_heatmap(name, res["history"], fig_dir)

    logger.info("Generating aggregate comparison plots...")
    plot_final_score_comparison(results, pretrained_score, fig_dir)
    plot_eval_scores_over_training(results, pretrained_score, fig_dir)
    plot_score_delta(results, fig_dir)

    logger.info("Generating gradient analysis plots...")
    plot_gradient_alignment(results, fig_dir)
    plot_gradient_conflict_map(results, fig_dir)

    logger.info("Generating representation drift plots...")
    plot_representation_drift(results, fig_dir)
    plot_cka_similarity(results, fig_dir)

    logger.info("Generating timestep analysis plots...")
    plot_t_analysis(results, fig_dir)
    plot_t_bin_norms_heatmap(results, fig_dir)

    logger.info("Generating return / advantage plots...")
    plot_return_distributions(results, fig_dir)

    logger.info("Generating group comparison plots...")
    plot_group_comparison(results, pretrained_score, fig_dir)
    plot_per_env_delta(results, fig_dir)

    logger.info("All plots saved to %s", fig_dir)
