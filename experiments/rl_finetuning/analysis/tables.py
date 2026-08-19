"""Summary tables for the ablation analysis suite.

Produces polars DataFrames and exports CSV + LaTeX.

Adapted from Craftax reference — uses MiniHack ID win rate as
the primary metric instead of Craftax achievement score.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import AblationHistory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict rule (shared with the sibling repo, character for character)
# ---------------------------------------------------------------------------

# Both repos label against `baseline_rl`: a verdict answers "did this
# intervention beat the plain RL run", which is the question the suite asks.
# The pretrained score is reported alongside as its own delta column, so
# nothing is lost by not labelling against it.
#
# The thresholds are FRACTIONS of the metric's own magnitude, not absolute
# constants. Craftax scores an episode-weighted mean return of magnitude ~10
# and minihack a mean per-env-ID win rate in [0, 1], so one absolute
# +0.05/-0.10 pair was two different claims: at the recorded craftax scores
# every arm read IMPROVEMENT, `lora` included at -1.911 against `baseline_rl`.
# The fractions keep the original rule's asymmetry - the bar to call an
# improvement is half the drop needed to call a collapse.
IMPROVEMENT_FRACTION = 0.05
COLLAPSE_FRACTION = 0.10

# The scale is the larger of the two reference scores in absolute value. Both
# are on the metric's own scale by construction, and taking the larger keeps
# the threshold from vanishing when one reference sits near zero. At a
# minihack scale of exactly 1.0 the rule reproduces the absolute +0.05/-0.10
# it replaces.
_MIN_METRIC_SCALE = 1e-9


def metric_scale(baseline_rl_score: float, pretrained_score: float) -> float:
    """Magnitude the verdict thresholds are taken as a fraction of.

    Args:
        baseline_rl_score: The `baseline_rl` arm's final score.
        pretrained_score:  The pretrained model's score.

    Returns:
        The larger reference score in absolute value.
    """
    return max(abs(float(baseline_rl_score)), abs(float(pretrained_score)))


def verdict(
    score: float,
    baseline_rl_score: float,
    pretrained_score: float,
) -> str:
    """Label one ablation's final score against `baseline_rl`.

    Args:
        score:             The ablation's final score.
        baseline_rl_score: The `baseline_rl` arm's final score.
        pretrained_score:  The pretrained model's score.

    Returns:
        ``"IMPROVEMENT"``, ``"COLLAPSE"`` or ``"NEUTRAL"``. Scores with no
        reference scale to measure against are ``"NEUTRAL"``: no label is
        defensible there.
    """
    scale = metric_scale(baseline_rl_score, pretrained_score)
    if scale < _MIN_METRIC_SCALE:
        return "NEUTRAL"
    delta = float(score) - float(baseline_rl_score)
    if delta > IMPROVEMENT_FRACTION * scale:
        return "IMPROVEMENT"
    if delta < -COLLAPSE_FRACTION * scale:
        return "COLLAPSE"
    return "NEUTRAL"


def baseline_rl_score_of(results: dict[str, dict], pretrained_score: float) -> float:
    """The `baseline_rl` arm's score, falling back to the pretrained score.

    Args:
        results:          ``{name: {"score": float, ...}}``.
        pretrained_score: Used when the suite ran without `baseline_rl`.

    Returns:
        The reference score every verdict is taken against.
    """
    return results.get("baseline_rl", {}).get("score", pretrained_score)


def _df_to_latex(
    df: pl.DataFrame,
    caption: str = "",
    label: str = "",
) -> str:
    """Convert polars DataFrame to LaTeX tabular.

    Args:
        df: DataFrame.
        caption: Table caption.
        label: Table label.

    Returns:
        LaTeX string.
    """
    cols = df.columns
    col_spec = "l" + "r" * (len(cols) - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{c.replace('_', chr(92) + '_')}}}" for c in cols)
        + " \\\\",
        "\\midrule",
    ]
    for row in df.iter_rows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val).replace("_", r"\_"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _save_table(
    df: pl.DataFrame,
    path_stem: Path,
    caption: str = "",
    label: str = "",
) -> None:
    """Save DataFrame as CSV and LaTeX.

    Args:
        df: DataFrame.
        path_stem: Output path without extension.
        caption: LaTeX caption.
        label: LaTeX label.
    """
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(path_stem) + ".csv")
    tex = _df_to_latex(df, caption=caption, label=label)
    Path(str(path_stem) + ".tex").write_text(tex)
    logger.info("Saved %s.csv/.tex", path_stem)


def write_significance_test(results: dict[str, dict], out_dir: Path) -> None:
    """Baseline vs the strongest condition: max-statistic permutation test + bootstrap CI.

    Writes ``significance_test.txt``.

    The condition tested is picked from the same scores the test then
    evaluates, so an uncorrected pairwise test is not calibrated: over 25 null
    arms its false-positive rate against a nominal 0.05 measures 0.090 at four
    seeds and 0.207 at five.  The null distribution here is therefore that of
    the **maximum** absolute mean difference over every candidate arm rather
    than of one pre-chosen arm -- each relabelling of the pooled
    ``(baseline, condition)`` scores is applied to every arm and the maximum is
    recomputed.  Measured false-positive rate 0.048 at four seeds and 0.055 at
    five.  Because the maximum is over signed-symmetric statistics, the arm it
    selects is the one furthest from baseline in **either** direction; the
    best-scoring arm is reported alongside it.

    The test is exact: it enumerates all ``C(n_a + n_b, n_b)`` relabellings.
    Every relabelling's complement negates each difference and so ties the
    statistic, which puts a hard floor of ``2 / C(n_a + n_b, n_b)`` on the
    p-value -- 0.333 at two seeds per condition, **0.100 at three**, 0.029 at
    four, 0.008 at five.  At the shipped three seeds no data whatsoever can
    reach 0.05, so the floor is written into the output and flagged when the
    p-value is sitting on it, rather than leaving 0.100 to read as marginal
    significance.
    """
    base = results.get("baseline_rl")
    if not base or not base.get("all_scores"):
        return
    others = {
        n: r
        for n, r in results.items()
        if n != "baseline_rl" and len(r.get("all_scores", [])) >= 2
    }
    if not others or len(base["all_scores"]) < 2:
        return
    import itertools
    from collections import Counter

    a = [float(x) for x in base["all_scores"]]
    n_a = len(a)
    arms = {n: [float(x) for x in r["all_scores"]] for n, r in others.items()}
    # One relabelling scheme is shared by every arm, so the arms it ranges over
    # must agree on their seed count.  The largest such group is used; anything
    # outside it is named in the output rather than dropped silently.
    n_b = Counter(len(v) for v in arms.values()).most_common(1)[0][0]
    dropped = sorted(n for n, v in arms.items() if len(v) != n_b)
    arms = {n: v for n, v in arms.items() if len(v) == n_b}
    pooled = [a + v for v in arms.values()]

    def _max_abs_diff(sel: set[int]) -> float:
        """Largest |mean(selected) - mean(rest)| over every arm, for one relabelling."""
        return max(
            abs(
                float(
                    np.mean([p[i] for i in sel])
                    - np.mean([p[i] for i in range(len(p)) if i not in sel])
                )
            )
            for p in pooled
        )

    relabellings = list(itertools.combinations(range(n_a + n_b), n_b))
    total = len(relabellings)
    obs_stat = _max_abs_diff(set(range(n_a, n_a + n_b)))
    count = sum(1 for s in relabellings if _max_abs_diff(set(s)) >= obs_stat - 1e-12)
    p_perm = count / total
    p_floor = 2 / total

    mean_a = float(np.mean(a))
    tested = max(arms, key=lambda n: abs(float(np.mean(arms[n])) - mean_a))
    top = max(arms, key=lambda n: float(np.mean(arms[n])))
    b = arms[tested]
    obs = float(np.mean(b) - mean_a)

    rng = np.random.default_rng(0)
    boots = [
        float(np.mean(rng.choice(b, len(b))) - np.mean(rng.choice(a, len(a))))
        for _ in range(10000)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    at_floor = " <- p is AT the floor; no data at this seed count can go lower" * (
        p_perm <= p_floor + 1e-12
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "significance_test.txt").write_text(
        f"baseline_rl scores: {a}\n"
        f"highest-scoring condition: {top} (mean {float(np.mean(arms[top])):.4f})\n"
        f"tested condition, furthest from baseline in either direction: "
        f"{tested} scores: {b}\n"
        f"mean difference (tested - baseline): {obs:.4f}\n"
        f"max-statistic permutation test over {len(arms)} candidate "
        f"arm{'s' * (len(arms) != 1)} "
        f"(two-sided, exact, {total} relabellings): p = {p_perm:.3f}\n"
        f"minimum attainable p at {n_a} baseline and {n_b} condition seeds: "
        f"{p_floor:.3f}{at_floor}\n"
        f"bootstrap 95% CI of the difference (10000 resamples, seed 0): "
        f"[{lo:.4f}, {hi:.4f}]\n"
        + (
            f"arms excluded from the max (seed count != {n_b}): "
            f"{', '.join(dropped)}\n"
            if dropped
            else ""
        )
    )

def make_main_results_table(
    results: dict[str, dict],
    pretrained_score: float,
) -> pl.DataFrame:
    """Main results: Method | Group | Score | Delta Pretrained | Delta Baseline | Verdict.

    The Verdict column follows :func:`verdict`: labelled against
    `baseline_rl`, thresholds scaled to the metric.

    Args:
        results: ``{name: {"score": float, ...}}``.
        pretrained_score: Pretrained eval score.

    Returns:
        Polars DataFrame sorted by score descending.
    """
    baseline_score = baseline_rl_score_of(results, pretrained_score)

    rows: list[dict] = []
    for name, res in results.items():
        score = res["score"]
        spec = REGISTRY.get(name)
        group = spec.group if spec else "?"
        delta_pre = score - pretrained_score
        delta_bl = score - baseline_score
        label = verdict(score, baseline_score, pretrained_score)

        rows.append(
            {
                "Method": name,
                "Group": group,
                "Score": round(score, 4),
                "Score_Std": round(
                    float(res.get("score_std", 0.0)), 4
                ),  # popstd over seeds
                "Delta_Pretrained": round(delta_pre, 4),
                "Delta_Baseline": round(delta_bl, 4),
                "Verdict": label,
            }
        )

    return pl.DataFrame(rows).sort("Score", descending=True)


def make_group_summary_table(
    results: dict[str, dict],
    pretrained_score: float,
) -> pl.DataFrame:
    """Group-level summary: Group | N | Mean | Best | Worst | StdDev.

    Args:
        results: Ablation results dict.
        pretrained_score: Pretrained score.

    Returns:
        Polars DataFrame with one row per group.
    """
    groups: dict[str, list[float]] = {}
    for name, res in results.items():
        spec = REGISTRY.get(name)
        g = spec.group if spec else "?"
        groups.setdefault(g, []).append(res["score"])

    rows: list[dict] = []
    for g in ["Baseline", "A", "B", "C", "D"]:
        if g not in groups:
            continue
        scores = groups[g]
        rows.append(
            {
                "Group": g,
                "N": len(scores),
                "Mean": round(float(np.mean(scores)), 4),
                "Best": round(float(np.max(scores)), 4),
                "Worst": round(float(np.min(scores)), 4),
                "StdDev": round(float(np.std(scores)), 4),
            }
        )

    return pl.DataFrame(rows)


def make_gradient_analysis_table(
    results: dict[str, dict],
) -> pl.DataFrame:
    """Gradient alignment and representation drift analysis.

    Reports both the mean over training and the final recorded value, so a
    method that starts aligned and degrades is distinguishable from one that
    is misaligned throughout.

    Args:
        results: Ablation results dict.

    Returns:
        DataFrame sorted by final gradient alignment, descending.
    """
    rows: list[dict] = []
    for name, res in sorted(results.items()):
        h: AblationHistory = res["history"]
        if not (h.grad_align and h.rl_grad_norm and h.bc_grad_norm):
            continue

        aligns = h.grad_align
        drifts = h.repr_drift_kl
        trend = (
            "down"
            if (len(aligns) > 1 and aligns[-1] < aligns[0])
            else "up"
            if (len(aligns) > 1 and aligns[-1] > aligns[0])
            else "flat"
        )
        rows.append(
            {
                "Method": name,
                "Mean_Grad_Align": round(float(np.mean(aligns)), 4),
                "Final_Grad_Align": round(aligns[-1], 4),
                "Trend": trend,
                "RL_Norm": round(h.rl_grad_norm[-1], 4),
                "BC_Norm": round(h.bc_grad_norm[-1], 4),
                "Mean_KL_Drift": round(float(np.mean(drifts)), 6)
                if drifts
                else float("nan"),
                "Final_KL_Drift": round(drifts[-1], 6) if drifts else float("nan"),
            }
        )

    rows.sort(key=lambda r: r["Final_Grad_Align"], reverse=True)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def make_t_distribution_table(
    results: dict[str, dict],
) -> pl.DataFrame:
    """Timestep distribution analysis per ablation.

    A high-t dominant regime with negative low-high alignment is the
    signature of the t-bias hypothesis.

    Args:
        results: Ablation results dict.

    Returns:
        DataFrame with the high/low norm ratio, low-high cosine similarity
        and the dominant regime label per ablation.
    """
    rows: list[dict] = []
    for name, res in sorted(results.items()):
        h: AblationHistory = res["history"]
        if not h.norm_high_t:
            continue

        ratio = float(np.mean(h.norm_high_t)) / (float(np.mean(h.norm_low_t)) + 1e-10)
        dominant = "high-t" if ratio > 1.5 else "low-t" if ratio < 0.67 else "balanced"
        rows.append(
            {
                "Method": name,
                "HighLow_Ratio": round(ratio, 3),
                "LowHigh_Cos_Sim": round(float(np.mean(h.lowhigh_cos)), 4)
                if h.lowhigh_cos
                else float("nan"),
                "Dominant_Regime": dominant,
            }
        )

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def make_repr_drift_table(
    results: dict[str, dict],
) -> pl.DataFrame:
    """Final KL drift values at 4 t ranges.

    Args:
        results: Ablation results dict.

    Returns:
        DataFrame with KL values per ablation.
    """
    rows: list[dict] = []
    for name, res in sorted(results.items()):
        h: AblationHistory = res["history"]
        if h.repr_drift_kl:
            rows.append(
                {
                    "Method": name,
                    "KL_mean": round(h.repr_drift_kl[-1], 4),
                    "KL_low_t": round(h.repr_drift_kl_low_t[-1], 4),
                    "KL_mid_t": round(h.repr_drift_kl_mid_t[-1], 4),
                    "KL_high_t": round(h.repr_drift_kl_high_t[-1], 4),
                }
            )

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def make_per_env_table(
    results: dict[str, dict],
) -> pl.DataFrame:
    """Per-environment win rate at final eval checkpoint.

    Args:
        results: Ablation results dict.

    Returns:
        DataFrame with per-env win rates.
    """
    rows: list[dict] = []
    for name, res in sorted(results.items()):
        # Average per-seed final win rates when recorded;
        # fall back to the legacy single merged history otherwise.
        finals = [
            f["per_env_win_rates"]
            for f in res.get("per_seed_finals", [])
            if isinstance(f, dict) and isinstance(f.get("per_env_win_rates"), dict)
        ]
        if finals:
            row = {"Method": name}
            for k in sorted({k for f in finals for k in f}):
                vals = [f[k] for f in finals if k in f]
                row[k] = round(float(np.mean(vals)), 4)
            rows.append(row)
            continue
        h: AblationHistory = res["history"]
        if h.per_env_win_rates:
            row = {"Method": name}
            row.update(h.per_env_win_rates[-1])
            rows.append(row)

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def make_forgetting_analysis_table(
    results: dict[str, dict],
    pretrained_score: float,
    collapse_threshold: float = 0.1,
) -> pl.DataFrame:
    """Forgetting timeline: first collapse, min score, recovery.

    One function, character for character, in both repos. The two halves
    had drifted apart in five places; each is settled below, with the
    behaviour that was dropped named so the choice can be read back.

    **Collapse boundary** is multiplicative: ``pretrained * (1 -
    collapse_threshold)``, a tenth below pretrained. minihack subtracted an
    absolute 0.05 instead, which means something different on a Craftax
    achievement score than on a MiniHack win rate; the verdict rule was
    scaled to the metric on 2026-08-17 and this follows it.

    **Recovery** is judged from the first collapse onward -- did any later
    evaluation climb back to the boundary. An arm that never collapsed
    reports ``"N/A"``, not recovery: minihack asked only whether the final
    score cleared the boundary, which labels every healthy arm "recovered"
    from a collapse it never had.

    **The recovery score** is ``res["score"]``, the terminal evaluation,
    which is the quantity the main results table, the verdict rule and the
    hypothesis table all use. minihack read the last in-loop evaluation
    instead, so its ``Final_Score`` column and the ``Score`` column beside
    it could disagree.

    **An arm with no evaluation history still gets a row**, with a null
    minimum and no collapse. minihack skipped it, which silently shrinks
    the denominator of any count taken over this table.

    **Arms are visited in sorted order**, so the table is byte-reproducible
    across runs; craftax inherited dict order.

    Args:
        results:            ``{name: {"history": AblationHistory, "score": float}}``.
        pretrained_score:   Pretrained model eval score.
        collapse_threshold: Fraction below pretrained that counts as collapse.

    Returns:
        Polars DataFrame with one row per ablation, in sorted name order.
    """
    collapse_level = pretrained_score * (1 - collapse_threshold)
    rows: list[dict] = []
    for name, res in sorted(results.items()):
        history: AblationHistory = res["history"]
        evals = history.eval_score
        eval_iters = history.eval_iters

        first_collapse_iter = "never"
        min_score = round(min(evals), 4) if evals else None
        min_score_iter = eval_iters[evals.index(min(evals))] if evals else None
        recovery_score = round(res["score"], 4)
        recovered = "N/A"

        for i, (it, sc) in enumerate(zip(eval_iters, evals, strict=False)):
            if sc < collapse_level:
                first_collapse_iter = str(it)
                later_scores = evals[i + 1 :]
                recovered = (
                    "Y" if any(s >= collapse_level for s in later_scores) else "N"
                )
                break

        rows.append(
            {
                "Method": name,
                "First_Collapse_Iter": first_collapse_iter,
                "Min_Score": min_score,
                "Min_Score_Iter": min_score_iter,
                "Recovery_Score": recovery_score,
                "Recovered": recovered,
            }
        )

    return pl.DataFrame(rows)


def make_hypothesis_verdict_table(
    results: dict[str, dict],
    pretrained_score: float,
) -> pl.DataFrame:
    """Map each ablation to its hypothesis and render a verdict.

    The Verdict column follows :func:`verdict`, the same rule as
    :func:`make_main_results_table`.

    Args:
        results: Ablation results dict.
        pretrained_score: Pretrained eval score.

    Returns:
        DataFrame with Method, Group, Score, Verdict, Hypothesis.
    """
    baseline_score = baseline_rl_score_of(results, pretrained_score)

    rows: list[dict] = []
    for name, res in sorted(results.items()):
        spec = REGISTRY.get(name)
        if spec is None:
            continue
        score = res["score"]
        delta_bl = score - baseline_score
        label = verdict(score, baseline_score, pretrained_score)

        rows.append(
            {
                "Method": name,
                "Group": spec.group,
                "Score": round(score, 4),
                "Delta_Baseline": round(delta_bl, 4),
                "Verdict": label,
                "Hypothesis": spec.hypothesis,
            }
        )

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def generate_summary_tables(
    results: dict[str, dict],
    pretrained_score: float,
    output_dir: Path,
) -> dict[str, pl.DataFrame]:
    """Generate all summary tables and save to ``output_dir/tables/``.

    Args:
        results: Full ablation results dict.
        pretrained_score: Pretrained model eval score.
        output_dir: Root output directory; tables go in ``output_dir/tables/``.

    Returns:
        Dict mapping table name to polars DataFrame.
    """
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating tables in %s", tables_dir)

    tables: dict[str, pl.DataFrame] = {}

    tables["main_results"] = make_main_results_table(results, pretrained_score)
    _save_table(
        tables["main_results"],
        tables_dir / "main_results",
        caption="Main ablation results",
        label="tab:main-results",
    )
    write_significance_test(results, tables_dir)

    tables["group_summary"] = make_group_summary_table(results, pretrained_score)
    _save_table(
        tables["group_summary"],
        tables_dir / "group_summary",
        caption="Group-level summary",
        label="tab:group-summary",
    )

    grad_df = make_gradient_analysis_table(results)
    if grad_df.shape[0] > 0:
        tables["gradient_analysis"] = grad_df
        _save_table(
            grad_df,
            tables_dir / "gradient_analysis",
            caption="Gradient alignment and representation drift analysis",
            label="tab:gradient",
        )

    t_df = make_t_distribution_table(results)
    if t_df.shape[0] > 0:
        tables["t_distribution"] = t_df
        _save_table(
            t_df,
            tables_dir / "t_distribution",
            caption="Timestep distribution analysis",
            label="tab:t-dist",
        )

    repr_df = make_repr_drift_table(results)
    if repr_df.shape[0] > 0:
        tables["repr_drift"] = repr_df
        _save_table(
            repr_df,
            tables_dir / "repr_drift",
            caption="Representation drift (KL)",
            label="tab:repr-drift",
        )

    env_df = make_per_env_table(results)
    if env_df.shape[0] > 0:
        tables["per_env"] = env_df
        _save_table(
            env_df,
            tables_dir / "per_env",
            caption="Per-environment win rates",
            label="tab:per-env",
        )

    forg_df = make_forgetting_analysis_table(results, pretrained_score)
    if forg_df.shape[0] > 0:
        tables["forgetting_analysis"] = forg_df
        _save_table(
            forg_df,
            tables_dir / "forgetting_analysis",
            caption="Catastrophic forgetting timeline.",
            label="tab:forgetting",
        )

    hyp_df = make_hypothesis_verdict_table(results, pretrained_score)
    if hyp_df.shape[0] > 0:
        tables["hypothesis_verdict"] = hyp_df
        _save_table(
            hyp_df,
            tables_dir / "hypothesis_verdict",
            caption="Hypothesis verdicts",
            label="tab:hyp-verdicts",
        )

    logger.info("All tables saved to %s", tables_dir)
    return tables
