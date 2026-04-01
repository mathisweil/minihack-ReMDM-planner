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
# LaTeX export
# ---------------------------------------------------------------------------


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
        " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\",
        "\\midrule",
    ]
    for row in df.iter_rows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
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


# ---------------------------------------------------------------------------
# Main results table
# ---------------------------------------------------------------------------


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

        if delta_bl > 0.02:
            verdict = "Improved"
        elif delta_bl < -0.02:
            verdict = "Degraded"
        else:
            verdict = "Neutral"

        rows.append({
            "Method": name,
            "Group": group,
            "Score": round(score, 4),
            "Delta_Pretrained": round(delta_pre, 4),
            "Delta_Baseline": round(delta_bl, 4),
            "Verdict": verdict,
        })

    return pl.DataFrame(rows).sort("Score", descending=True)


# ---------------------------------------------------------------------------
# Group summary table
# ---------------------------------------------------------------------------


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
        rows.append({
            "Group": g,
            "N": len(scores),
            "Mean": round(float(np.mean(scores)), 4),
            "Best": round(float(np.max(scores)), 4),
            "Worst": round(float(np.min(scores)), 4),
            "StdDev": round(float(np.std(scores)), 4),
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gradient diagnostics table
# ---------------------------------------------------------------------------


def make_gradient_diagnostics_table(
    results: dict[str, dict],
) -> pl.DataFrame:
    """Gradient diagnostics at final recorded iteration.

    Args:
        results: Ablation results dict.

    Returns:
        DataFrame with cos_sim, rl_norm, bc_norm per ablation.
    """
    rows: list[dict] = []
    for name, res in sorted(results.items()):
        h: AblationHistory = res["history"]
        if h.grad_align:
            rows.append({
                "Method": name,
                "Cos_Sim": round(h.grad_align[-1], 4),
                "RL_Norm": round(h.rl_grad_norm[-1], 4),
                "BC_Norm": round(h.bc_grad_norm[-1], 4),
            })

    return pl.DataFrame(rows) if rows else pl.DataFrame()


# ---------------------------------------------------------------------------
# Representation drift table
# ---------------------------------------------------------------------------


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
            rows.append({
                "Method": name,
                "KL_mean": round(h.repr_drift_kl[-1], 4),
                "KL_low_t": round(h.repr_drift_kl_low_t[-1], 4),
                "KL_mid_t": round(h.repr_drift_kl_mid_t[-1], 4),
                "KL_high_t": round(h.repr_drift_kl_high_t[-1], 4),
            })

    return pl.DataFrame(rows) if rows else pl.DataFrame()


# ---------------------------------------------------------------------------
# Per-env win rate table
# ---------------------------------------------------------------------------


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
        h: AblationHistory = res["history"]
        if h.per_env_win_rates:
            row: dict = {"Method": name}
            row.update(h.per_env_win_rates[-1])
            rows.append(row)

    return pl.DataFrame(rows) if rows else pl.DataFrame()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_summary_tables(
    results: dict[str, dict],
    pretrained_score: float,
    out_dir: Path,
) -> None:
    """Generate all summary tables.

    Args:
        results: Full ablation results dict.
        pretrained_score: Pretrained model eval score.
        out_dir: Output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating tables in %s", out_dir)

    main_df = make_main_results_table(results, pretrained_score)
    _save_table(
        main_df, out_dir / "main_results",
        caption="Main ablation results", label="tab:main-results",
    )

    group_df = make_group_summary_table(results, pretrained_score)
    _save_table(
        group_df, out_dir / "group_summary",
        caption="Group-level summary", label="tab:group-summary",
    )

    grad_df = make_gradient_diagnostics_table(results)
    if grad_df.shape[0] > 0:
        _save_table(
            grad_df, out_dir / "gradient_diagnostics",
            caption="Gradient diagnostics", label="tab:grad-diag",
        )

    repr_df = make_repr_drift_table(results)
    if repr_df.shape[0] > 0:
        _save_table(
            repr_df, out_dir / "repr_drift",
            caption="Representation drift (KL)", label="tab:repr-drift",
        )

    env_df = make_per_env_table(results)
    if env_df.shape[0] > 0:
        _save_table(
            env_df, out_dir / "per_env_win_rates",
            caption="Per-environment win rates", label="tab:per-env",
        )

    logger.info("All tables generated.")
