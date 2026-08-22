"""Profile one RL fine-tuning ablation iteration by component.

Runs ``run_ablation`` unmodified and attributes wall-clock time to the
pieces that matter for the speed pass: environment construction, env
stepping, GPU sampling, window extraction, the host-to-device transfer,
the gradient step, each diagnostic, and evaluation.

Everything is measured by monkey-patching the callables ``training.py``
already uses, so the training loop itself is never edited and the
measured run computes exactly what a real run computes.

Usage:
    python scripts/profile_ablation.py --checkpoint PATH \
        --ablation baseline_rl --max-iter 51

Without ``--config`` this profiles ``configs/defaults.yaml`` merged with
``experiments/rl_finetuning/configs/ablations_final_minihack_gpu_24gb.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.rl_finetuning.ablations import training as T  # noqa: E402
from experiments.rl_finetuning.ablations.registry import REGISTRY  # noqa: E402
from experiments.rl_finetuning.run_ablations import (  # noqa: E402
    _load_yaml,
    _merge_to_namespace,
)
from src.envs import minihack_env  # noqa: E402
from src.models.denoiser import ModelEMA  # noqa: E402
from src.planners.inference import Evaluator  # noqa: E402

logger = logging.getLogger("profile_ablation")


class Acc:
    """Accumulating wall-clock timer with a call counter."""

    def __init__(self) -> None:
        self.total = 0.0
        self.calls = 0

    def add(self, dt: float) -> None:
        """Record one timed call of duration *dt* seconds."""
        self.total += dt
        self.calls += 1

    def reset(self) -> None:
        """Zero the accumulator."""
        self.total = 0.0
        self.calls = 0


ACC: dict[str, Acc] = defaultdict(Acc)
WINDOWS: list[int] = []


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _wrap(fn, key: str, sync: bool = False):
    """Return *fn* wrapped so its wall time accumulates under *key*."""

    def inner(*a, **kw):
        if sync:
            _sync()
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            if sync:
                _sync()
            ACC[key].add(time.perf_counter() - t0)

    return inner


def install_patches() -> None:
    """Monkey-patch the callables whose cost we want attributed."""
    # -- Collection internals -------------------------------------------
    # ``collect/env_acquire`` is the whole cost of getting a usable env,
    # whether that is a fresh construction or a pool hit;
    # ``collect/env_init`` is the construction alone, so the two diverge
    # once the pool starts serving.
    if hasattr(T, "acquire_env"):
        T.acquire_env = _wrap(T.acquire_env, "collect/env_acquire")
    else:
        T.make_env = _wrap(T.make_env, "collect/env_acquire")
    minihack_env.AdvancedObservationEnv.step = _wrap(
        minihack_env.AdvancedObservationEnv.step, "collect/env_step"
    )
    minihack_env.AdvancedObservationEnv.reset = _wrap(
        minihack_env.AdvancedObservationEnv.reset, "collect/env_reset"
    )
    minihack_env.AdvancedObservationEnv.close = _wrap(
        minihack_env.AdvancedObservationEnv.close, "collect/env_close"
    )
    minihack_env.AdvancedObservationEnv.__init__ = _wrap(
        minihack_env.AdvancedObservationEnv.__init__, "collect/env_init"
    )
    T.remdm_sample = _wrap(T.remdm_sample, "collect/gpu_sample", sync=True)
    T._extract_windows = _wrap(T._extract_windows, "collect/extract_windows")

    # -- Window count actually available to the gradient step ------------
    # ``compute_advantages`` is called once per iteration on every window
    # collected after the ablation's filters, so its input length is the
    # number the ``local_obs[:batch_size]`` slice draws from.
    _adv = T.compute_advantages

    def _adv_counting(returns, *a, **kw):
        WINDOWS.append(int(returns.shape[0]))
        return _adv(returns, *a, **kw)

    T.compute_advantages = _adv_counting

    # -- Model plumbing --------------------------------------------------
    ModelEMA.make_eval_model = _wrap(
        ModelEMA.make_eval_model, "train/make_eval_model", sync=True
    )

    # -- Diagnostics -----------------------------------------------------
    T.compute_grad_alignment = _wrap(
        T.compute_grad_alignment, "diag/grad_alignment", sync=True
    )
    T.compute_per_layer_grad_norms = _wrap(
        T.compute_per_layer_grad_norms, "diag/per_layer_norms_only", sync=True
    )
    T.compute_repr_drift = _wrap(T.compute_repr_drift, "diag/repr_drift", sync=True)
    T.compute_cka = _wrap(T.compute_cka, "diag/cka", sync=True)
    T.compute_t_analysis = _wrap(T.compute_t_analysis, "diag/t_analysis", sync=True)
    Evaluator.evaluate = _wrap(Evaluator.evaluate, "eval/evaluate", sync=True)


ITER_ROWS: list[dict] = []


def install_metric_capture() -> None:
    """Capture each iteration's metric dict plus the timer deltas.

    ``training.py`` stops its own ``speed/iter_time_sec`` clock straight
    after the gradient step, before the diagnostics and the eval, so it
    understates a diagnostic iteration badly. ``prof/wall_iter_sec`` is
    the gap between consecutive log calls, which is the real thing.
    """
    prev_total: dict[str, float] = {}
    prev_calls: dict[str, int] = {}
    last = [time.perf_counter()]
    seen_windows = [0]

    def _log(metrics: dict, step: int) -> None:
        now = time.perf_counter()
        row = dict(metrics)
        row["prof/wall_iter_sec"] = now - last[0]
        last[0] = now
        # First new advantage call of the iteration is the online batch;
        # mixed_replay adds a second call for the buffer sample.
        if len(WINDOWS) > seen_windows[0]:
            row["prof/windows_collected"] = WINDOWS[seen_windows[0]]
            seen_windows[0] = len(WINDOWS)
        for key, acc in ACC.items():
            row[f"prof/{key}"] = acc.total - prev_total.get(key, 0.0)
            row[f"prof/{key}_calls"] = acc.calls - prev_calls.get(key, 0)
            prev_total[key] = acc.total
            prev_calls[key] = acc.calls
        ITER_ROWS.append(row)

    T._wandb_log = _log


def main() -> None:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(_PROJECT_ROOT / "configs/defaults.yaml"))
    p.add_argument(
        "--ablations-config",
        default=str(
            _PROJECT_ROOT
            / "experiments/rl_finetuning/configs/ablations_final_minihack_gpu_24gb.yaml"
        ),
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ablation", default="baseline_rl")
    p.add_argument("--max-iter", type=int, default=51)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="Write per-iteration JSON here.")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override key=value (parsed as JSON, then as str).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    overrides: dict = {"max_iter": args.max_iter}
    for item in args.override:
        key, _, raw = item.partition("=")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw

    cfg = _merge_to_namespace(
        _load_yaml(args.config),
        _load_yaml(args.ablations_config),
        overrides,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    logger.info("TMPDIR=%s", os.environ.get("TMPDIR", "<unset>"))
    logger.info("config=%s", args.config)
    logger.info("ablations_config=%s", args.ablations_config)
    logger.info("checkpoint=%s", args.checkpoint)
    logger.info(
        "n_embd=%s n_head=%s n_layer=%s batch_size=%s episodes_per_iter=%s",
        cfg.n_embd,
        cfg.n_head,
        cfg.n_layer,
        cfg.batch_size,
        cfg.episodes_per_iter,
    )

    install_patches()
    install_metric_capture()

    spec = REGISTRY[args.ablation]
    t0 = time.perf_counter()
    _, final_score, _ = T.run_ablation(
        spec, cfg, args.checkpoint, device, seed=args.seed
    )
    wall = time.perf_counter() - t0

    print("\n" + "=" * 78)
    print(f"ablation={args.ablation} iters={args.max_iter} wall={wall:.1f}s")
    print(f"final_score={final_score:.4f}")
    print("=" * 78)
    print(f"{'component':38s} {'total_s':>10s} {'calls':>9s} {'ms/call':>10s}")
    for key in sorted(ACC):
        acc = ACC[key]
        per = 1000.0 * acc.total / acc.calls if acc.calls else 0.0
        print(f"{key:38s} {acc.total:10.2f} {acc.calls:9d} {per:10.3f}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "ablation": args.ablation,
                    "max_iter": args.max_iter,
                    "seed": args.seed,
                    "wall_sec": wall,
                    "final_score": final_score,
                    "config": {
                        "n_embd": cfg.n_embd,
                        "n_head": cfg.n_head,
                        "n_layer": cfg.n_layer,
                        "batch_size": cfg.batch_size,
                        "episodes_per_iter": cfg.episodes_per_iter,
                    },
                    "totals": {
                        k: {"total_sec": v.total, "calls": v.calls}
                        for k, v in ACC.items()
                    },
                    "iters": ITER_ROWS,
                },
                indent=1,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
