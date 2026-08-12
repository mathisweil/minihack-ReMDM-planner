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
    """C-002 (F-035): baseline vs best condition, exact permutation test + bootstrap CI.

    Writes ``significance_test.txt``. With three seeds per condition the
    permutation test is exact (C(6,3) = 20 relabellings).
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

    best = max(others, key=lambda n: float(np.mean(others[n]["all_scores"])))
    a = [float(x) for x in base["all_scores"]]
    b = [float(x) for x in others[best]["all_scores"]]
    obs = float(np.mean(b) - np.mean(a))
    pooled = a + b
    n_b = len(b)
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), n_b):
        grp_b = [pooled[i] for i in idx]
        grp_a = [pooled[i] for i in range(len(pooled)) if i not in idx]
        if abs(float(np.mean(grp_b) - np.mean(grp_a))) >= abs(obs) - 1e-12:
            count += 1
        total += 1
    p_perm = count / total
    rng = np.random.default_rng(0)
    boots = [
        float(np.mean(rng.choice(b, len(b))) - np.mean(rng.choice(a, len(a))))
        for _ in range(10000)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "significance_test.txt").write_text(
        f"baseline_rl scores: {a}\nbest condition: {best} scores: {b}\n"
        f"mean difference (best - baseline): {obs:.4f}\n"
        f"exact permutation test (two-sided, {total} relabellings): p = {p_perm:.3f}\n"
        f"bootstrap 95% CI of the difference (10000 resamples, seed 0): "
        f"[{lo:.4f}, {hi:.4f}]\n"
    )


def make_main_results_table(
    results: dict[str, dict],
    pretrained_score: float,
) -> pl.DataFrame:
    """Main results: Method | Group | Score | Delta Pretrained | Delta Baseline | Verdict.

    Args:
        results: ``{name: {"score": float, ...}}``.
        pretrained_score: Pretrained eval score.

    Returns:
        Polars DataFrame sorted by score descending.
    """
    baseline_score = results.get("baseline_rl", {}).get("score", pretrained_score)

    rows: list[dict] = []
    for name, res in results.items():
        score = res["score"]
        spec = REGISTRY.get(name)
        group = spec.group if spec else "?"
        delta_pre = score - pretrained_score
        delta_bl = score - baseline_score

        if delta_bl > 0.05:
            verdict = "IMPROVEMENT"
        elif delta_bl < -0.1:
            verdict = "COLLAPSE"
        else:
            verdict = "NEUTRAL"

        rows.append(
            {
                "Method": name,
                "Group": group,
                "Score": round(score, 4),
                "Score_Std": round(
                    float(res.get("score_std", 0.0)), 4
                ),  # C-002 (F-035): popstd over seeds
                "Delta_Pretrained": round(delta_pre, 4),
                "Delta_Baseline": round(delta_bl, 4),
                "Verdict": verdict,
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
        # C-002 (F-024): average per-seed final win rates when recorded;
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
    collapse_threshold: float = 0.05,
) -> pl.DataFrame:
    """Forgetting timeline: first collapse, min score, recovery.

    Collapse is defined as eval score dropping below
    ``pretrained_score - collapse_threshold``.

    Args:
        results: Ablation results dict.
        pretrained_score: Pretrained eval score.
        collapse_threshold: Drop from pretrained that counts as collapse.

    Returns:
        DataFrame with one row per ablation.
    """
    rows: list[dict] = []
    for name, res in sorted(results.items()):
        h: AblationHistory = res["history"]
        if not h.eval_iters or not h.eval_score:
            continue

        scores = h.eval_score
        iters = h.eval_iters
        min_score = min(scores)
        min_idx = scores.index(min_score)
        final_score = scores[-1]
        boundary = pretrained_score - collapse_threshold

        first_collapse = "never"
        for it, sc in zip(iters, scores, strict=False):
            if sc < boundary:
                first_collapse = str(it)
                break

        recovered = final_score >= boundary

        rows.append(
            {
                "Method": name,
                "First_Collapse": first_collapse,
                "Min_Score": round(min_score, 4),
                "Min_Score_Iter": iters[min_idx],
                "Final_Score": round(final_score, 4),
                "Recovered": recovered,
            }
        )

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def make_hypothesis_verdict_table(
    results: dict[str, dict],
    pretrained_score: float,
) -> pl.DataFrame:
    """Map each ablation to its hypothesis and render a verdict.

    Verdict thresholds match ``make_main_results_table``:
    IMPROVEMENT (delta > +0.05), COLLAPSE (delta < -0.1), else NEUTRAL.

    Args:
        results: Ablation results dict.
        pretrained_score: Pretrained eval score.

    Returns:
        DataFrame with Method, Group, Score, Verdict, Hypothesis.
    """
    baseline_score = results.get("baseline_rl", {}).get(
        "score",
        pretrained_score,
    )

    rows: list[dict] = []
    for name, res in sorted(results.items()):
        spec = REGISTRY.get(name)
        if spec is None:
            continue
        score = res["score"]
        delta_bl = score - baseline_score

        if delta_bl > 0.05:
            verdict = "IMPROVEMENT"
        elif delta_bl < -0.1:
            verdict = "COLLAPSE"
        else:
            verdict = "NEUTRAL"

        rows.append(
            {
                "Method": name,
                "Group": spec.group,
                "Score": round(score, 4),
                "Delta_Baseline": round(delta_bl, 4),
                "Verdict": verdict,
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
    write_significance_test(results, tables_dir)  # C-002 (F-035)

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
            caption="Forgetting analysis",
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
