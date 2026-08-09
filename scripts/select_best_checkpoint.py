"""Select the best checkpoint of a run per the pre-specified rule.

Rule (RETRAIN_LOG.md, Best-checkpoint protocol): score every
checkpoint-time eval JSON (eval_iter{N}.json / eval_offline_step{N}.json)
by the unweighted mean win_rate over all ID and OOD environments; pick
the highest score; break ties by the smaller step N.

Usage: uv run python scripts/select_best_checkpoint.py <run_dir>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PATTERNS = (
    re.compile(r"^eval_iter(\d+)\.json$"),
    re.compile(r"^eval_offline_step(\d+)\.json$"),
)


def score_eval_json(path: Path) -> float:
    """Unweighted mean win_rate over every env in the id and ood sections."""
    results = json.loads(path.read_text())["results"]
    rates = [
        env_stats["win_rate"]
        for section in ("id", "ood")
        for env_stats in results.get(section, {}).values()
    ]
    if not rates:
        raise ValueError(f"{path} contains no per-env results")
    return sum(rates) / len(rates)


def select_best(run_dir: Path) -> tuple[int, float, list[tuple[int, float]]]:
    """Return (best_step, best_score, all (step, score) pairs sorted by step)."""
    scored: dict[int, float] = {}
    for f in run_dir.iterdir():
        for pat in _PATTERNS:
            m = pat.match(f.name)
            if m:
                scored[int(m.group(1))] = score_eval_json(f)
    if not scored:
        raise FileNotFoundError(
            f"No eval_iter*.json or eval_offline_step*.json in {run_dir}"
        )
    ranked = sorted(scored.items())
    best_step, best_score = max(ranked, key=lambda kv: (kv[1], -kv[0]))
    return best_step, best_score, ranked


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = Path(sys.argv[1])
    best_step, best_score, ranked = select_best(run_dir)
    print(f"{'step':>10}  {'mean win rate (ID+OOD)':>24}")
    for step, score in ranked:
        marker = "  <-- best" if step == best_step else ""
        print(f"{step:>10}  {score:>24.4f}{marker}")
    print(f"\nbest = {best_step}")


if __name__ == "__main__":
    main()
