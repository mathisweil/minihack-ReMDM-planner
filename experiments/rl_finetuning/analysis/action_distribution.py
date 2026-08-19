"""Action distribution analysis: pre- vs post-RL fine-tuning comparison.

Compares action distributions of the ReMDM planner before (pretrained BC)
and after (RL fine-tuned) to characterise whether performance collapse
manifests as action distribution change (mode collapse) or subtler
representational degradation with similar action distributions.

Key diagnostic:
    - JS divergence < 0.05  -> representation drift, not behavioural change
    - JS divergence 0.05-0.15 -> mixed representational and behavioural shift
    - JS divergence >= 0.15 -> substantial mode collapse

Analysis outputs:
    1. Action frequency comparison (side-by-side bar charts)
    2. Delta and log-ratio probability change
    3. Distribution metrics dashboard (entropy, effective actions, divergences,
       Gini)
    4. Episode-level histograms (returns and lengths)
    5. Cumulative distribution curve with 80%/95% thresholds
    6. Action transition matrices (pre, post, difference heatmaps)
    7. Statistical tests (chi-squared, Mann-Whitney U)
    8. Paper-ready interpretation summary
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import orjson  # noqa: E402
import torch  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

from src.planners.collect import run_model_episode  # noqa: E402

logger = logging.getLogger(__name__)

MINIHACK_ACTIONS: dict[int, str] = {
    0: "N",
    1: "E",
    2: "S",
    3: "W",
    4: "NE",
    5: "SE",
    6: "SW",
    7: "NW",
    8: "UP",
    9: "DWN",
    10: "WAIT",
    11: "KICK",
}

_DPI = 150


def collect_action_statistics(
    model: torch.nn.Module,
    env_ids: list[str],
    num_episodes_per_env: int,
    cfg: SimpleNamespace,
    device: torch.device | str,
    diffusion_steps: int = 5,
    replan_every: int = 2,
    max_steps: int = 200,
) -> dict:
    """Collect action statistics from real model rollouts across environments.

    Runs ``num_episodes_per_env`` episodes per environment ID using
    ``run_model_episode`` and aggregates action counts, episode returns,
    lengths, and win indicators.

    Args:
        model: Denoising model (will be set to eval mode).
        env_ids: List of MiniHack environment registry IDs.
        num_episodes_per_env: Number of rollout episodes per environment.
        cfg: Config namespace (must contain ``replan_every``, ``seq_len``,
            etc.).
        device: Torch device for model inference.
        diffusion_steps: Number of reverse denoising steps.
        replan_every: Re-plan interval override. Stored into ``cfg``
            temporarily for the rollout.
        max_steps: Maximum steps per episode.

    Returns:
        Dict with keys ``"all_actions"`` (np.ndarray), ``"action_counts"``
        (Counter), ``"episode_returns"`` (np.ndarray),
        ``"episode_lengths"`` (np.ndarray), ``"episode_won"`` (np.ndarray).
    """
    all_actions: list[int] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_won: list[bool] = []

    # Temporarily override cfg fields for rollout parameters
    orig_replan = getattr(cfg, "replan_every", 16)
    orig_diff_steps = getattr(cfg, "diffusion_steps_eval", 10)
    cfg.replan_every = replan_every
    cfg.diffusion_steps_eval = diffusion_steps

    model.eval()
    try:
        for env_id in env_ids:
            logger.info("Rolling out in %s ...", env_id)
            for _ in range(num_episodes_per_env):
                with torch.no_grad():
                    result = run_model_episode(
                        model,
                        env_id,
                        cfg,
                        device,
                        max_steps=max_steps,
                    )
                actions = result["actions"].tolist()
                all_actions.extend(actions)
                episode_returns.append(result["total_reward"])
                episode_lengths.append(result["steps"])
                episode_won.append(result["won"])
    finally:
        cfg.replan_every = orig_replan
        cfg.diffusion_steps_eval = orig_diff_steps

    return {
        "all_actions": np.asarray(all_actions, dtype=np.int64),
        "action_counts": Counter(all_actions),
        "episode_returns": np.asarray(episode_returns, dtype=np.float64),
        "episode_lengths": np.asarray(episode_lengths, dtype=np.int64),
        "episode_won": np.asarray(episode_won, dtype=bool),
    }


def compute_entropy(probs: np.ndarray) -> float:
    """Compute base-2 Shannon entropy of a probability distribution.

    Args:
        probs: 1-D probability vector (non-negative, sums to 1).

    Returns:
        Entropy in bits. Zero-probability entries are ignored.
    """
    p = probs[probs > 0]
    return float(np.sum(-p * np.log2(p)))


def compute_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(p || q) with epsilon smoothing, natural log.

    Args:
        p: Reference distribution (1-D probability vector).
        q: Approximating distribution (1-D probability vector).

    Returns:
        KL divergence in nats (non-negative).
    """
    eps = 1e-10
    p_safe = (p + eps) / (p + eps).sum()
    q_safe = (q + eps) / (q + eps).sum()
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def compute_js(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two distributions.

    Uses ``compute_kl`` internally: ``JS(p,q) = 0.5*(KL(p||m) + KL(q||m))``
    where ``m = 0.5*(p + q)``.

    Args:
        p: First distribution (1-D probability vector).
        q: Second distribution (1-D probability vector).

    Returns:
        JS divergence in nats (non-negative, bounded by ln(2)).
    """
    m = 0.5 * (p + q)
    return 0.5 * (compute_kl(p, m) + compute_kl(q, m))


def compute_gini(probs: np.ndarray) -> float:
    """Compute the Gini coefficient of a probability distribution.

    0 = perfectly uniform, 1 = maximally concentrated on one action.

    Args:
        probs: 1-D probability vector (non-negative, sums to 1).

    Returns:
        Gini coefficient in [0, 1).
    """
    s = np.sort(probs)
    n = len(s)
    total = s.sum()
    if total == 0:
        return 0.0
    indices = np.arange(1, n + 1)
    return float((2.0 * np.sum(indices * s) - (n + 1) * total) / (n * total))


def action_transitions(actions: np.ndarray, n: int) -> np.ndarray:
    """Compute row-normalised action transition matrix P(next | current).

    Args:
        actions: 1-D integer array of sequential action indices.
        n: Number of distinct actions (matrix dimension).

    Returns:
        ``(n, n)`` transition matrix. Rows with zero total count are left
        as zeros (no division by zero).
    """
    t_mat = np.zeros((n, n), dtype=np.float64)
    if len(actions) < 2:
        return t_mat
    prev = actions[:-1]
    nxt = actions[1:]
    valid = (prev < n) & (nxt < n)
    np.add.at(t_mat, (prev[valid], nxt[valid]), 1)
    row_sums = t_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return t_mat / row_sums


def compute_all_metrics(
    pre_probs: np.ndarray,
    post_probs: np.ndarray,
    pre_stats: dict,
    post_stats: dict,
    action_dim: int,
) -> dict:
    """Compute comprehensive distribution comparison metrics.

    Args:
        pre_probs: Pre-RL action probability vector of length ``action_dim``.
        post_probs: Post-RL action probability vector of length ``action_dim``.
        pre_stats: Dict from ``collect_action_statistics`` (pre-RL).
        post_stats: Dict from ``collect_action_statistics`` (post-RL).
        action_dim: Number of discrete actions.

    Returns:
        Dict with string keys mapping to float metric values, covering
        entropy, KL, JS, TV distance, effective actions, Gini, mode
        probability, win rate, and mean return.
    """
    max_entropy = float(np.log2(action_dim))
    pre_entropy = compute_entropy(pre_probs)
    post_entropy = compute_entropy(post_probs)

    return {
        "Pre-RL Entropy": pre_entropy,
        "Post-RL Entropy": post_entropy,
        "Entropy Change": post_entropy - pre_entropy,
        "Max Possible Entropy": max_entropy,
        "Pre-RL Normalised Entropy": (
            pre_entropy / max_entropy if max_entropy > 0 else 0.0
        ),
        "Post-RL Normalised Entropy": (
            post_entropy / max_entropy if max_entropy > 0 else 0.0
        ),
        "KL(Post || Pre)": compute_kl(post_probs, pre_probs),
        "KL(Pre || Post)": compute_kl(pre_probs, post_probs),
        "JS Divergence": compute_js(pre_probs, post_probs),
        "TV Distance": 0.5 * float(np.sum(np.abs(pre_probs - post_probs))),
        "Pre-RL Effective Actions": float(np.sum(pre_probs > 0.01)),
        "Post-RL Effective Actions": float(np.sum(post_probs > 0.01)),
        "Pre-RL Gini": compute_gini(pre_probs),
        "Post-RL Gini": compute_gini(post_probs),
        "Pre-RL Mode Prob": float(pre_probs.max()),
        "Post-RL Mode Prob": float(post_probs.max()),
        "Pre-RL Win Rate": float(pre_stats["episode_won"].mean()),
        "Post-RL Win Rate": float(post_stats["episode_won"].mean()),
        "Pre-RL Mean Return": float(pre_stats["episode_returns"].mean()),
        "Post-RL Mean Return": float(post_stats["episode_returns"].mean()),
    }


def run_statistical_tests(
    pre_stats: dict,
    post_stats: dict,
    action_dim: int,
) -> dict:
    """Run chi-squared and Mann-Whitney U statistical tests.

    Chi-squared compares the two action count distributions as a 2 x A
    contingency table (with 1 pseudocount, so an action neither policy ever
    takes cannot empty a column). Mann-Whitney U compares episode return
    distributions.

    Both counts are **observed**, and the test has to treat them that way.
    A goodness-of-fit test -- ``scipy.stats.chisquare`` against the second
    sample rescaled to the first sample's total -- treats the second sample
    as a known expectation, which drops the sampling error in the thing it
    is comparing against and roughly doubles the statistic: measured over
    2000 draws with the *same* distribution in both samples, it rejects at
    alpha = 0.05 on 39.9 % of draws at 500 actions per sample and 42.2 % at
    5000, against a nominal 5 %. The contingency form estimates the shared
    expectation from both margins and measures 3.5 % and 4.0 % on the same
    draws. Degrees of freedom are A - 1 either way, so only the expectation
    changes.

    Args:
        pre_stats: Dict from ``collect_action_statistics`` (pre-RL).
        post_stats: Dict from ``collect_action_statistics`` (post-RL).
        action_dim: Number of discrete actions.

    Returns:
        Dict with ``"chi2"``, ``"chi2_p"``, ``"chi2_significant"``,
        ``"mannwhitney_u"``, ``"mannwhitney_p"``,
        ``"mannwhitney_significant"`` keys.
    """
    pre_counts = (
        np.array(
            [pre_stats["action_counts"].get(i, 0) for i in range(action_dim)],
            dtype=np.float64,
        )
        + 1.0
    )
    post_counts = (
        np.array(
            [post_stats["action_counts"].get(i, 0) for i in range(action_dim)],
            dtype=np.float64,
        )
        + 1.0
    )

    chi2, p_chi2 = scipy_stats.chi2_contingency(np.vstack([pre_counts, post_counts]))[:2]

    u_stat, p_ret = scipy_stats.mannwhitneyu(
        pre_stats["episode_returns"],
        post_stats["episode_returns"],
        alternative="two-sided",
    )

    return {
        "chi2": float(chi2),
        "chi2_p": float(p_chi2),
        "chi2_significant": bool(p_chi2 < 0.05),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(p_ret),
        "mannwhitney_significant": bool(p_ret < 0.05),
    }


def generate_action_distribution_plots(
    pre_probs: np.ndarray,
    post_probs: np.ndarray,
    pre_stats: dict,
    post_stats: dict,
    metrics: dict,
    action_dim: int,
    out_dir: Path,
    suffix: str = "",
) -> None:
    """Generate six diagnostic plots comparing pre- and post-RL distributions.

    Plots are saved as PNG at 150 DPI in ``out_dir``, each with ``suffix``
    appended to the stem:
        1. ``action_dist_comparison.png`` -- side-by-side bars
        2. ``probability_change.png`` -- delta and log-ratio bars
        3. ``distribution_metrics.png`` -- 2x2 dashboard
        4. ``episode_analysis.png`` -- return and length histograms
        5. ``cumulative_distribution.png`` -- cumulative sorted curve
        6. ``action_transitions.png`` -- pre, post, diff heatmaps

    Args:
        pre_probs: Pre-RL action probability vector.
        post_probs: Post-RL action probability vector.
        pre_stats: Dict from ``collect_action_statistics`` (pre-RL).
        post_stats: Dict from ``collect_action_statistics`` (post-RL).
        metrics: Dict from ``compute_all_metrics``.
        action_dim: Number of discrete actions.
        out_dir: Directory to write PNG files into (created if needed).
        suffix: Appended to each filename stem, e.g. ``"_kl_penalty"``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [MINIHACK_ACTIONS.get(i, str(i)) for i in range(action_dim)]
    short_labels = [lab[:5] for lab in labels]
    x = np.arange(action_dim)
    y_max = max(pre_probs.max(), post_probs.max()) * 1.15

    # ---- Figure 1: side-by-side action distribution bars ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x, pre_probs, color="steelblue", alpha=0.85)
    axes[0].set_title("Pre-RL (BC Only)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Action")
    axes[0].set_ylabel("Probability")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylim(0, y_max)
    axes[1].bar(x, post_probs, color="coral", alpha=0.85)
    axes[1].set_title(
        "Post-RL (After Fine-tuning)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xlabel("Action")
    axes[1].set_ylabel("Probability")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylim(0, y_max)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"action_dist_comparison{suffix}.png",
        dpi=_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---- Figure 2: delta and log-ratio analysis ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    delta = post_probs - pre_probs
    safe_ratio = np.where(
        pre_probs > 1e-6,
        post_probs / pre_probs,
        1e-10,
    )
    log_ratio = np.clip(np.log2(safe_ratio), -5.0, 5.0)

    colors_d = ["green" if d > 0 else "red" for d in delta]
    axes[0].bar(x, delta, color=colors_d, alpha=0.75)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(
        "Change in Action Probability (Post - Pre)",
        fontweight="bold",
    )
    axes[0].set_xlabel("Action")
    axes[0].set_ylabel("Delta Probability")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")

    colors_r = ["green" if r > 0 else "red" for r in log_ratio]
    axes[1].bar(x, log_ratio, color=colors_r, alpha=0.75)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(
        "Log2 Ratio of Action Probabilities (Post/Pre)",
        fontweight="bold",
    )
    axes[1].set_xlabel("Action")
    axes[1].set_ylabel("log2(Post/Pre)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"probability_change{suffix}.png",
        dpi=_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---- Figure 3: distribution metrics dashboard (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) Entropy
    ax = axes[0, 0]
    cats = ["Pre-RL", "Post-RL", "Max Possible"]
    vals = [
        metrics["Pre-RL Entropy"],
        metrics["Post-RL Entropy"],
        metrics["Max Possible Entropy"],
    ]
    bars = ax.bar(cats, vals, color=["steelblue", "coral", "gray"], alpha=0.85)
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Distribution Entropy", fontweight="bold")
    for bar, val in zip(bars, vals, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )

    # (0,1) Effective actions
    ax = axes[0, 1]
    vals2 = [
        metrics["Pre-RL Effective Actions"],
        metrics["Post-RL Effective Actions"],
    ]
    bars2 = ax.bar(
        ["Pre-RL", "Post-RL"],
        vals2,
        color=["steelblue", "coral"],
        alpha=0.85,
    )
    ax.axhline(
        action_dim,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Total actions",
    )
    ax.set_ylabel("# Actions")
    ax.set_title("Effective Actions (>1%)", fontweight="bold")
    ax.legend(fontsize=8)
    for bar, val in zip(bars2, vals2, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # (1,0) Divergence metrics
    ax = axes[1, 0]
    div_names = ["KL(Post||Pre)", "KL(Pre||Post)", "JS Div", "TV Dist"]
    div_vals = [
        metrics["KL(Post || Pre)"],
        metrics["KL(Pre || Post)"],
        metrics["JS Divergence"],
        metrics["TV Distance"],
    ]
    bars3 = ax.bar(div_names, div_vals, color="mediumpurple", alpha=0.85)
    ax.set_ylabel("Value")
    ax.set_title("Divergence Metrics", fontweight="bold")
    ax.set_xticklabels(div_names, rotation=15)
    for bar, val in zip(bars3, div_vals, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # (1,1) Gini coefficient
    ax = axes[1, 1]
    vals4 = [metrics["Pre-RL Gini"], metrics["Post-RL Gini"]]
    bars4 = ax.bar(
        ["Pre-RL", "Post-RL"],
        vals4,
        color=["steelblue", "coral"],
        alpha=0.85,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Gini Coefficient")
    ax.set_title(
        "Action Distribution Inequality\n(0=uniform, 1=collapsed)",
        fontweight="bold",
    )
    ax.axhline(0, color="green", linestyle="--", alpha=0.4, label="Uniform")
    ax.axhline(1, color="red", linestyle="--", alpha=0.4, label="Collapsed")
    ax.legend(fontsize=8)
    for bar, val in zip(bars4, vals4, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    fig.tight_layout()
    fig.savefig(
        out_dir / f"distribution_metrics{suffix}.png",
        dpi=_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---- Figure 4: episode-level histograms ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(
        pre_stats["episode_returns"],
        bins=20,
        alpha=0.6,
        label="Pre-RL",
        color="steelblue",
    )
    ax.hist(
        post_stats["episode_returns"],
        bins=20,
        alpha=0.6,
        label="Post-RL",
        color="coral",
    )
    ax.axvline(
        pre_stats["episode_returns"].mean(),
        color="steelblue",
        linestyle="--",
        lw=2,
    )
    ax.axvline(
        post_stats["episode_returns"].mean(),
        color="coral",
        linestyle="--",
        lw=2,
    )
    ax.set_xlabel("Episode Return")
    ax.set_ylabel("Count")
    ax.set_title("Episode Return Distribution", fontweight="bold")
    ax.legend()

    ax = axes[1]
    ax.hist(
        pre_stats["episode_lengths"],
        bins=20,
        alpha=0.6,
        label="Pre-RL",
        color="steelblue",
    )
    ax.hist(
        post_stats["episode_lengths"],
        bins=20,
        alpha=0.6,
        label="Post-RL",
        color="coral",
    )
    ax.axvline(
        pre_stats["episode_lengths"].mean(),
        color="steelblue",
        linestyle="--",
        lw=2,
    )
    ax.axvline(
        post_stats["episode_lengths"].mean(),
        color="coral",
        linestyle="--",
        lw=2,
    )
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution", fontweight="bold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        out_dir / f"episode_analysis{suffix}.png",
        dpi=_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---- Figure 5: cumulative distribution curve ----
    fig, ax = plt.subplots(figsize=(10, 5))
    pre_sorted = np.sort(pre_probs)[::-1]
    post_sorted = np.sort(post_probs)[::-1]
    pre_cum = np.cumsum(pre_sorted)
    post_cum = np.cumsum(post_sorted)
    xs = np.arange(1, action_dim + 1)
    ax.plot(
        xs,
        pre_cum,
        "o-",
        label="Pre-RL (BC)",
        color="steelblue",
        lw=2,
        markersize=6,
    )
    ax.plot(
        xs,
        post_cum,
        "s-",
        label="Post-RL",
        color="coral",
        lw=2,
        markersize=6,
    )
    for thresh, label in [(0.8, "80%"), (0.95, "95%")]:
        ax.axhline(thresh, color="gray", linestyle="--", alpha=0.5)
        ax.text(
            action_dim - 0.5,
            thresh + 0.01,
            label,
            fontsize=10,
            color="gray",
        )
    ax.set_xlabel("Number of Top Actions", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title(
        "Cumulative Action Probability (Sorted by Frequency)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"cumulative_distribution{suffix}.png",
        dpi=_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---- Figure 6: action transition matrices ----
    pre_t = action_transitions(pre_stats["all_actions"], action_dim)
    post_t = action_transitions(post_stats["all_actions"], action_dim)
    diff = post_t - pre_t
    v_max = max(np.abs(diff).max(), 1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im1 = axes[0].imshow(pre_t, cmap="Blues", aspect="auto")
    im2 = axes[1].imshow(post_t, cmap="Oranges", aspect="auto")
    im3 = axes[2].imshow(
        diff,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-v_max,
        vmax=v_max,
    )
    titles = [
        "Pre-RL Transitions",
        "Post-RL Transitions",
        "Difference (Post-Pre)",
    ]
    for ax_i, title, im in zip(axes, titles, [im1, im2, im3], strict=False):
        ax_i.set_title(title, fontweight="bold")
        ax_i.set_xticks(range(action_dim))
        ax_i.set_yticks(range(action_dim))
        ax_i.set_xticklabels(
            short_labels,
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax_i.set_yticklabels(short_labels, fontsize=8)
        ax_i.set_xlabel("Next Action")
        ax_i.set_ylabel("Current Action")
        plt.colorbar(im, ax=ax_i, fraction=0.046)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"action_transitions{suffix}.png",
        dpi=_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def interpret_results(metrics: dict) -> str:
    """Generate a paper-ready interpretation of the JS divergence finding.

    Thresholds:
        - JS < 0.05: small (supports representation drift hypothesis)
        - 0.05 <= JS < 0.15: moderate (mixed causes)
        - JS >= 0.15: large (mode collapse likely)

    Args:
        metrics: Dict from ``compute_all_metrics``.

    Returns:
        Multi-line interpretation string suitable for a diagnosis report.
    """
    js = metrics["JS Divergence"]
    if js < 0.05:
        return (
            f"JS Divergence = {js:.4f} -- SMALL -- "
            "action distributions nearly identical. "
            "Performance collapse NOT explained by action distribution "
            "change. Supports representation drift hypothesis."
        )
    if js < 0.15:
        return (
            f"JS Divergence = {js:.4f} -- MODERATE -- "
            "some distribution shift, both representational and "
            "behavioural changes likely."
        )
    return (
        f"JS Divergence = {js:.4f} -- LARGE -- "
        "substantial action distribution shift, mode collapse likely."
    )


def run_action_distribution_analysis(
    pre_model: torch.nn.Module,
    post_model: torch.nn.Module,
    cfg: SimpleNamespace,
    device: torch.device | str,
    out_dir: Path | str,
    num_episodes: int = 50,
    suffix: str = "",
    pre_stats: dict | None = None,
) -> dict:
    """Run the full action distribution analysis pipeline.

    Collects pre- and post-RL rollout statistics, computes divergence
    metrics and statistical tests, generates all diagnostic plots, and
    writes a JSON results file to ``out_dir``.

    Args:
        pre_model: Pre-RL (BC-only) model.
        post_model: Post-RL (fine-tuned) model.
        cfg: Config namespace (must contain ``id_envs``, ``action_dim``,
            etc.).
        device: Torch device.
        out_dir: Output directory for plots and results JSON.
        num_episodes: Episodes per environment for statistics collection.
        suffix: Appended to every output filename, e.g. ``"_kl_penalty"``.
            Lets one output directory hold results for many ablations.
        pre_stats: Pre-collected pre-RL statistics. When comparing many
            ablations against the same pretrained model, collect once with
            ``collect_action_statistics`` and pass it here rather than
            re-rolling the baseline for every ablation.

    Returns:
        Results dict containing metrics, probabilities, episode data,
        statistical tests, and interpretation string.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_ids: list[str] = cfg.id_envs
    action_dim: int = cfg.action_dim

    if pre_stats is None:
        logger.info("Collecting pre-RL action statistics ...")
        pre_stats = collect_action_statistics(
            pre_model,
            env_ids,
            num_episodes,
            cfg,
            device,
        )
    logger.info(
        "Pre-RL: %d actions, win=%.2f%%, return=%.3f",
        len(pre_stats["all_actions"]),
        pre_stats["episode_won"].mean() * 100,
        pre_stats["episode_returns"].mean(),
    )

    logger.info("Collecting post-RL action statistics ...")
    post_stats = collect_action_statistics(
        post_model,
        env_ids,
        num_episodes,
        cfg,
        device,
    )
    logger.info(
        "Post-RL: %d actions, win=%.2f%%, return=%.3f",
        len(post_stats["all_actions"]),
        post_stats["episode_won"].mean() * 100,
        post_stats["episode_returns"].mean(),
    )

    # Compute probability distributions from raw counts
    pre_freqs = np.array(
        [pre_stats["action_counts"].get(i, 0) for i in range(action_dim)],
        dtype=np.float64,
    )
    post_freqs = np.array(
        [post_stats["action_counts"].get(i, 0) for i in range(action_dim)],
        dtype=np.float64,
    )
    pre_total = pre_freqs.sum()
    post_total = post_freqs.sum()
    pre_probs = pre_freqs / pre_total if pre_total > 0 else pre_freqs
    post_probs = post_freqs / post_total if post_total > 0 else post_freqs

    # Metrics
    metrics = compute_all_metrics(
        pre_probs,
        post_probs,
        pre_stats,
        post_stats,
        action_dim,
    )

    # Statistical tests
    stat_tests = run_statistical_tests(pre_stats, post_stats, action_dim)

    # Plots
    generate_action_distribution_plots(
        pre_probs,
        post_probs,
        pre_stats,
        post_stats,
        metrics,
        action_dim,
        out_dir,
        suffix,
    )

    # Interpretation
    interpretation = interpret_results(metrics)
    logger.info("Interpretation: %s", interpretation)

    # Assemble results
    results: dict = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "statistical_tests": stat_tests,
        "pre_probs": pre_probs.tolist(),
        "post_probs": post_probs.tolist(),
        "pre_episode_returns": pre_stats["episode_returns"].tolist(),
        "post_episode_returns": post_stats["episode_returns"].tolist(),
        "pre_win_rate": float(pre_stats["episode_won"].mean()),
        "post_win_rate": float(post_stats["episode_won"].mean()),
        "action_names": {str(k): v for k, v in MINIHACK_ACTIONS.items()},
        "interpretation": interpretation,
    }

    # Write JSON via orjson
    json_path = out_dir / f"action_distribution_results{suffix}.json"
    json_path.write_bytes(
        orjson.dumps(results, option=orjson.OPT_INDENT_2),
    )
    logger.info("Results saved to %s", json_path)

    return results


def plot_js_comparison(
    js_by_ablation: dict[str, float],
    out_dir: Path,
) -> None:
    """Sorted bar chart of pre/post JS divergence across ablations.

    The dashed guides mark the interpretation thresholds: below 0.05 the
    behaviour is essentially unchanged and the collapse is representational;
    above 0.15 it is substantial mode collapse.

    Args:
        js_by_ablation: Mapping of ablation name to JS divergence.
        out_dir: Output directory.
    """
    if not js_by_ablation:
        return

    names = sorted(js_by_ablation, key=lambda n: js_by_ablation[n])
    values = [js_by_ablation[n] for n in names]

    with plt.rc_context({"figure.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(9, max(4.0, len(names) * 0.35)))
        ax.barh(range(len(names)), values, color="steelblue", alpha=0.85)
        ax.axvline(0.05, ls="--", color="green", alpha=0.7, label="0.05 (drift)")
        ax.axvline(0.15, ls="--", color="red", alpha=0.7, label="0.15 (collapse)")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("JS Divergence (pretrained vs fine-tuned)")
        ax.set_title("Action Distribution Shift by Ablation")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / "js_divergence_comparison.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)


def run_all_action_distribution_analyses(
    pretrained_model: torch.nn.Module,
    trained_models: Iterable[tuple[str, torch.nn.Module]],
    cfg: SimpleNamespace,
    device: torch.device | str,
    output_dir: Path,
    num_episodes: int = 10,
) -> dict[str, dict]:
    """Run action distribution analysis for every completed ablation.

    Figures and per-ablation JSON go in ``output_dir/figures/action_dist/``.
    The pretrained baseline is rolled out once and reused for every
    comparison, so cost scales with the number of ablations rather than
    twice that.

    Args:
        pretrained_model: Pretrained (BC-only) model.
        trained_models: Iterable of ``(ablation_name, model)`` pairs. Pass a
            generator that loads one checkpoint at a time to avoid holding
            every fine-tuned model in memory at once.
        cfg: Config namespace.
        device: Torch device.
        output_dir: Root output directory.
        num_episodes: Episodes per environment per model.

    Returns:
        Mapping of ablation name to that ablation's results dict.
    """
    fig_dir = output_dir / "figures" / "action_dist"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Collecting pretrained action statistics once (%d episodes/env) ...",
        num_episodes,
    )
    pre_stats = collect_action_statistics(
        pretrained_model,
        cfg.id_envs,
        num_episodes,
        cfg,
        device,
    )

    comparisons: dict[str, dict] = {}
    js_by_ablation: dict[str, float] = {}

    for name, model in trained_models:
        logger.info("Action distribution analysis: %s", name)
        try:
            res = run_action_distribution_analysis(
                pretrained_model,
                model,
                cfg,
                device,
                fig_dir,
                num_episodes=num_episodes,
                suffix=f"_{name}",
                pre_stats=pre_stats,
            )
        except Exception:
            logger.exception("Action distribution analysis failed for %s", name)
            continue
        comparisons[name] = res
        js = res["metrics"].get("JS Divergence")
        if js is not None:
            js_by_ablation[name] = float(js)

    plot_js_comparison(js_by_ablation, fig_dir)
    logger.info("Action distribution analysis saved to %s", fig_dir)
    return comparisons
