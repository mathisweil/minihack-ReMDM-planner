"""Data quality mixing experiment for masked discrete diffusion.

Proves that masked discrete diffusion is brittle to training data
quality degradation by measuring performance across oracle/self-generated
data mixing ratios.

Hypothesis:
    Performance degrades monotonically as the fraction of self-generated
    (suboptimal) data increases relative to oracle data.

Design:
    Test oracle fractions [1.0, 0.9, 0.7, 0.5, 0.0].
    1.0 (pure oracle) and 0.0 (pure RL) are known endpoints.
    Only intermediate fractions [0.9, 0.7, 0.5] are trained.

Ported from ``minihack_reference/experiments/mixing_experiment.py``.
"""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import orjson  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch import Tensor  # noqa: E402

from experiments.rl_finetuning.ablations.losses import (  # noqa: E402
    LossContext,
    make_loss_baseline,
)
from experiments.rl_finetuning.ablations.training import (  # noqa: E402
    MixedReplayBuffer,
    _extract_windows,
)
from src.diffusion.schedules import get_schedule  # noqa: E402
from src.models.denoiser import ModelEMA, make_model  # noqa: E402
from src.planners.collect import run_model_episode  # noqa: E402
from src.planners.inference import Evaluator  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_FRACTIONS: list[float] = [1.0, 0.9, 0.7, 0.5, 0.0]
_INTERMEDIATE_FRACTIONS: list[float] = [0.9, 0.7, 0.5]



def _collect_oracle_data(
    model: nn.Module,
    cfg: SimpleNamespace,
    device: torch.device,
    n_episodes: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collect episodes from the pretrained model and extract windows.

    Uses the same sliding-window extraction as the ablation training
    pipeline to produce training-ready tensors.

    Args:
        model: Pretrained model in eval mode.
        cfg: Config namespace.
        device: Torch device.
        n_episodes: Number of episodes to collect.

    Returns:
        Tuple of (local_obs ``[N,9,9]``, global_obs ``[N,21,79]``,
        x0 ``[N,H]``, returns ``[N]``).
    """
    all_local: list[Tensor] = []
    all_global: list[Tensor] = []
    all_x0: list[Tensor] = []
    all_returns: list[Tensor] = []

    env_ids = cfg.id_envs
    model.eval()

    for i in range(n_episodes):
        env_id = env_ids[i % len(env_ids)]
        ep = run_model_episode(
            model, env_id, cfg, device, stochastic=True,
        )
        lo, go, x0, ret = _extract_windows(
            ep, cfg.seq_len, cfg.pad_token,
        )
        if lo.shape[0] == 0:
            continue
        all_local.append(lo)
        all_global.append(go)
        all_x0.append(x0)
        all_returns.append(
            torch.full((lo.shape[0],), ret, dtype=torch.float32)
        )

    if not all_local:
        empty_lo = torch.empty(0, 9, 9, dtype=torch.long, device=device)
        empty_go = torch.empty(
            0, 21, 79, dtype=torch.long, device=device,
        )
        empty_x0 = torch.empty(
            0, cfg.seq_len, dtype=torch.long, device=device,
        )
        empty_ret = torch.empty(0, dtype=torch.float32, device=device)
        return empty_lo, empty_go, empty_x0, empty_ret

    return (
        torch.cat(all_local).to(device),
        torch.cat(all_global).to(device),
        torch.cat(all_x0).to(device),
        torch.cat(all_returns).to(device),
    )



def run_mixing_point(
    oracle_fraction: float,
    oracle_local: Tensor,
    oracle_global: Tensor,
    oracle_x0: Tensor,
    model_init_state: dict[str, Tensor],
    cfg: SimpleNamespace,
    device: torch.device,
    max_iter: int = 500,
    batch_size: int = 32,
    eval_every: int = 100,
    eval_episodes: int = 10,
) -> tuple[dict[str, list[float] | list[int]], float]:
    """Train one mixing point and return history + final win rate.

    Creates a fresh model from ``model_init_state``, builds a
    ``MixedReplayBuffer`` pre-filled with the oracle portion, then
    trains by collecting self-generated data each iteration.

    Args:
        oracle_fraction: Fraction in (0, 1) of buffer for oracle data.
        oracle_local: Oracle local observations ``[N, 9, 9]``.
        oracle_global: Oracle global observations ``[N, 21, 79]``.
        oracle_x0: Oracle action sequences ``[N, H]``.
        model_init_state: State dict to initialise the model.
        cfg: Config namespace.
        device: Torch device.
        max_iter: Number of training iterations.
        batch_size: Training batch size.
        eval_every: Iterations between evaluations.
        eval_episodes: Episodes per evaluation environment.

    Returns:
        Tuple of (history_dict, final_id_win_rate). History dict has
        keys ``"iter"``, ``"loss"``, ``"id_winrate"``,
        ``"id_winrate_iter"``.
    """
    model = make_model(cfg).to(device)
    model.load_state_dict(model_init_state)
    ema = ModelEMA(model, decay=getattr(cfg, "ema_decay", 0.999))

    schedule_fn = get_schedule(cfg.noise_schedule)
    cfg._schedule_fn = schedule_fn

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4,
    )

    ctx = LossContext(ref_model=None, schedule_fn=schedule_fn, cfg=cfg)
    loss_fn = make_loss_baseline(ctx)

    evaluator = Evaluator()
    buf_capacity = getattr(cfg, "mixed_replay_buffer_size", 10000)
    buf = MixedReplayBuffer(buf_capacity, cfg.seq_len, device)

    # Pre-fill oracle portion
    n_oracle = oracle_local.shape[0]
    if n_oracle > 0 and oracle_fraction > 0.0:
        n_fill = min(
            int(buf_capacity * oracle_fraction), n_oracle,
        )
        oracle_returns = torch.ones(
            n_oracle, dtype=torch.float32, device=device,
        )
        buf.push(
            oracle_local[:n_fill],
            oracle_global[:n_fill],
            oracle_x0[:n_fill],
            oracle_returns[:n_fill],
        )
        logger.info(
            "Pre-filled buffer with %d oracle windows (%.0f%% target)",
            n_fill, oracle_fraction * 100,
        )

    history: dict[str, list[float] | list[int]] = {
        "iter": [],
        "loss": [],
        "id_winrate": [],
        "id_winrate_iter": [],
    }

    env_ids = cfg.id_envs
    running_loss = 0.0
    n_log = 0

    for iteration in range(1, max_iter + 1):
        # Collect self-generated episode
        eval_model = ema.make_eval_model(model)
        env_id = env_ids[(iteration - 1) % len(env_ids)]
        ep = run_model_episode(
            eval_model, env_id, cfg, device, stochastic=True,
        )
        lo, go, x0, ret = _extract_windows(
            ep, cfg.seq_len, cfg.pad_token,
        )
        if lo.shape[0] > 0:
            returns_t = torch.full(
                (lo.shape[0],), ret, dtype=torch.float32,
            )
            buf.push(
                lo.to(device), go.to(device),
                x0.to(device), returns_t.to(device),
            )

        # Gradient step
        loss_val = 0.0
        if buf.size >= batch_size:
            b_lo, b_go, b_x0, b_ret = buf.sample(batch_size)
            # Advantage weights: clamp returns to [0.1, 5.0]
            adv = b_ret.clamp(min=0.0)
            batch_mean = adv.mean().clamp(min=1e-8)
            adv = (adv / batch_mean).clamp(0.1, 5.0)  # [B]

            model.train()
            optimizer.zero_grad()
            loss = loss_fn(
                model, b_lo, b_go, b_x0, adv, cfg, device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()
            ema.update(model)
            loss_val = loss.item()

        running_loss += loss_val
        n_log += 1

        if iteration % 10 == 0:
            avg_loss = running_loss / max(n_log, 1)
            history["iter"].append(iteration)
            history["loss"].append(avg_loss)
            running_loss = 0.0
            n_log = 0

        # Periodic evaluation
        if iteration % eval_every == 0:
            eval_model = ema.make_eval_model(model)
            results = evaluator.evaluate(
                cfg.id_envs, eval_model, eval_episodes, cfg, device,
            )
            id_wr = float(np.mean([
                v["win_rate"] for v in results.values()
            ]))
            history["id_winrate"].append(id_wr)
            history["id_winrate_iter"].append(iteration)
            logger.info(
                "  [Oracle-%.0f%%] iter=%d  loss=%.4f  id_wr=%.3f",
                oracle_fraction * 100, iteration, loss_val, id_wr,
            )

    # Final evaluation
    final_model = ema.make_eval_model(model)
    final_results = evaluator.evaluate(
        cfg.id_envs, final_model, eval_episodes, cfg, device,
    )
    final_id_wr = float(np.mean([
        v["win_rate"] for v in final_results.values()
    ]))
    logger.info(
        "  [Oracle-%.0f%%] FINAL id_win_rate: %.4f",
        oracle_fraction * 100, final_id_wr,
    )

    return history, final_id_wr



def check_monotonicity(
    fractions: list[float],
    rates: list[float],
) -> tuple[bool, list[str]]:
    """Check if win rates are non-increasing as oracle fraction decreases.

    Fractions are expected in descending order (e.g. [1.0, 0.9, ...]).

    Args:
        fractions: Oracle fractions in descending order.
        rates: Corresponding ID win rates.

    Returns:
        Tuple of (is_monotonic, list of violation description strings).
        Empty list if monotonic.
    """
    violations: list[str] = []
    for i in range(len(rates) - 1):
        if rates[i] < rates[i + 1]:
            violations.append(
                f"{fractions[i]:.0%} -> {fractions[i + 1]:.0%}: "
                f"{rates[i]:.2%} -> {rates[i + 1]:.2%}"
            )
    return len(violations) == 0, violations



_COLORS_BY_FRAC: dict[float, str] = {
    1.0: "#1B5E20",
    0.9: "#4CAF50",
    0.7: "#FF9800",
    0.5: "#F44336",
    0.0: "#B71C1C",
}


def generate_mixing_plots(
    fractions: list[float],
    rates: list[float],
    mixing_results: dict[float, dict],
    known_100: float,
    known_0: float,
    out_dir: Path,
) -> None:
    """Generate the 3-panel mixing experiment figure.

    Args:
        fractions: All oracle fractions (descending), including
            known endpoints.
        rates: Corresponding ID win rates.
        mixing_results: Dict mapping intermediate fractions to their
            result dicts (must have ``"history"`` key with
            ``"id_winrate"`` and ``"id_winrate_iter"`` lists).
        known_100: Known win rate for 100% oracle.
        known_0: Known win rate for 0% oracle.
        out_dir: Directory to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 0: Degradation curve
    ax = axes[0]
    frac_pct = [f * 100 for f in fractions]
    rate_pct = [r * 100 for r in rates]
    ax.plot(
        frac_pct, rate_pct, "ko-", linewidth=2, markersize=8, zorder=3,
    )
    for frac, rate in zip(fractions, rates):
        c = _COLORS_BY_FRAC.get(frac, "gray")
        ax.scatter(frac * 100, rate * 100, color=c, s=120, zorder=4)
        ax.annotate(
            f"{rate:.1%}",
            (frac * 100, rate * 100),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    ax.axhline(
        known_100 * 100, color="green", linestyle="--", alpha=0.5,
        label=f"Pure oracle ({known_100:.1%})",
    )
    ax.axhline(
        known_0 * 100, color="red", linestyle="--", alpha=0.5,
        label=f"Pure RL ({known_0:.1%})",
    )
    ax.set_xlabel("Oracle Data Fraction (%)")
    ax.set_ylabel("ID Win Rate (%)")
    ax.set_title(
        "Performance vs Oracle Data Fraction\n(Degradation Curve)"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-5, 105)
    y_ceil = min(100.0, known_100 * 100 * 1.3)
    ax.set_ylim(0, max(y_ceil, 10))
    ax.invert_xaxis()

    # Panel 1: ID win rate over training per fraction
    ax2 = axes[1]
    ax2.axhline(
        known_100 * 100, color="green", linestyle="--", linewidth=2,
        label=f"Pure oracle ({known_100:.1%})", alpha=0.7,
    )
    ax2.axhline(
        known_0 * 100, color="red", linestyle=":", linewidth=1.5,
        label=f"Pure RL ({known_0:.1%})", alpha=0.7,
    )
    for frac, info in sorted(
        mixing_results.items(), key=lambda kv: -kv[0],
    ):
        hist = info["history"]
        name = f"Oracle-{int(frac * 100)}%"
        wr_list = hist.get("id_winrate", [])
        it_list = hist.get("id_winrate_iter", [])
        if wr_list:
            ax2.plot(
                it_list,
                [r * 100 for r in wr_list],
                color=_COLORS_BY_FRAC.get(frac, "gray"),
                marker="o",
                linewidth=2,
                markersize=5,
                label=name,
            )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("ID Win Rate (%)")
    ax2.set_title("ID Win Rate Over Training\nby Oracle Fraction")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, max(y_ceil, 10))

    # Panel 2: Final ID win rate bar chart
    ax3 = axes[2]
    bar_labels = [
        f"{int(f * 100)}%\noracle" for f in fractions
    ]
    bar_colors = [_COLORS_BY_FRAC.get(f, "gray") for f in fractions]
    bars = ax3.bar(
        range(len(fractions)),
        rate_pct,
        color=bar_colors,
        alpha=0.85,
    )
    ax3.axhline(
        known_100 * 100, color="green", linestyle="--", alpha=0.5,
    )
    ax3.set_xticks(range(len(fractions)))
    ax3.set_xticklabels(bar_labels, fontsize=9)
    ax3.set_ylabel("Final ID Win Rate (%)")
    ax3.set_title("Final Performance by Data Mix")
    ax3.grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, rates):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plot_path = out_dir / "mixing_degradation_curve.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mixing plot: %s", plot_path)



def run_mixing_experiment(
    pretrained_checkpoint: str,
    cfg: SimpleNamespace,
    device: torch.device,
    out_dir: str | Path,
    oracle_fractions: list[float] | None = None,
    max_iter: int = 500,
    batch_size: int = 32,
    eval_every: int = 100,
    eval_episodes: int = 10,
) -> dict:
    """Run the full data quality mixing experiment.

    Loads a pretrained model, evaluates it as the 100% oracle baseline,
    collects an oracle dataset, trains each intermediate mixing point,
    evaluates the 0% oracle (pure RL) baseline, assembles the full
    degradation curve, checks monotonicity, and generates plots.

    Args:
        pretrained_checkpoint: Path to pretrained DAgger checkpoint.
        cfg: Config namespace.
        device: Torch device.
        out_dir: Output directory for plots and results JSON.
        oracle_fractions: Intermediate fractions to train. Defaults
            to ``[0.9, 0.7, 0.5]``.
        max_iter: Training iterations per mixing point.
        batch_size: Training batch size.
        eval_every: Iterations between evaluations.
        eval_episodes: Episodes per evaluation environment.

    Returns:
        Results dict with keys ``"oracle_fractions"``,
        ``"id_win_rates"``, ``"is_monotonic"``, ``"baseline_id"``,
        ``"pure_rl_id"``, ``"mixing_details"``.
    """
    if oracle_fractions is None:
        oracle_fractions = list(_INTERMEDIATE_FRACTIONS)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load pretrained model
    logger.info("Loading pretrained checkpoint: %s", pretrained_checkpoint)
    model = make_model(cfg).to(device)
    ckpt = torch.load(
        pretrained_checkpoint, map_location=device, weights_only=False,
    )
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    init_state = copy.deepcopy(model.state_dict())

    # Set up schedule on cfg
    schedule_fn = get_schedule(cfg.noise_schedule)
    cfg._schedule_fn = schedule_fn

    # Evaluate pretrained as 100% oracle baseline
    logger.info("Evaluating pretrained model (100%% oracle baseline)...")
    evaluator = Evaluator()
    baseline_results = evaluator.evaluate(
        cfg.id_envs, model, eval_episodes, cfg, device,
    )
    known_100 = float(np.mean([
        v["win_rate"] for v in baseline_results.values()
    ]))
    logger.info("100%% oracle baseline ID win rate: %.4f", known_100)

    # Collect oracle dataset from pretrained rollouts
    n_oracle_episodes = max(
        getattr(cfg, "mixing_oracle_episodes", 50), 10,
    )
    logger.info(
        "Collecting oracle dataset (%d episodes)...",
        n_oracle_episodes,
    )
    oracle_lo, oracle_go, oracle_x0, oracle_ret = _collect_oracle_data(
        model, cfg, device, n_oracle_episodes,
    )
    logger.info("Oracle dataset: %d windows", oracle_lo.shape[0])

    # Train intermediate mixing points
    mixing_results: dict[float, dict] = {}
    for frac in sorted(oracle_fractions, reverse=True):
        logger.info(
            "=" * 60 + "\nTraining mixing point: Oracle-%.0f%%\n"
            + "=" * 60,
            frac * 100,
        )
        history, final_id = run_mixing_point(
            oracle_fraction=frac,
            oracle_local=oracle_lo,
            oracle_global=oracle_go,
            oracle_x0=oracle_x0,
            model_init_state=init_state,
            cfg=cfg,
            device=device,
            max_iter=max_iter,
            batch_size=batch_size,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
        )
        mixing_results[frac] = {
            "final_id": final_id,
            "history": history,
        }

    # 0% oracle (pure RL) -- train with no oracle data
    logger.info("Training 0%% oracle (pure RL) baseline...")
    empty_lo = torch.empty(
        0, 9, 9, dtype=torch.long, device=device,
    )
    empty_go = torch.empty(
        0, 21, 79, dtype=torch.long, device=device,
    )
    empty_x0 = torch.empty(
        0, cfg.seq_len, dtype=torch.long, device=device,
    )
    rl_history, known_0 = run_mixing_point(
        oracle_fraction=0.0,
        oracle_local=empty_lo,
        oracle_global=empty_go,
        oracle_x0=empty_x0,
        model_init_state=init_state,
        cfg=cfg,
        device=device,
        max_iter=max_iter,
        batch_size=batch_size,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
    )
    mixing_results[0.0] = {
        "final_id": known_0,
        "history": rl_history,
    }

    # Assemble full degradation curve
    all_fractions = [1.0] + sorted(oracle_fractions, reverse=True) + [0.0]
    all_rates = (
        [known_100]
        + [mixing_results[f]["final_id"] for f in sorted(
            oracle_fractions, reverse=True,
        )]
        + [known_0]
    )

    is_monotonic, violations = check_monotonicity(
        all_fractions, all_rates,
    )
    if is_monotonic:
        logger.info(
            "CONFIRMED: Monotonic degradation across all fractions."
        )
    else:
        for v in violations:
            logger.info("Non-monotonic violation: %s", v)

    # Generate plots
    generate_mixing_plots(
        all_fractions, all_rates, mixing_results,
        known_100, known_0, out_path,
    )

    # Assemble results dict
    results: dict = {
        "oracle_fractions": all_fractions,
        "id_win_rates": [float(r) for r in all_rates],
        "is_monotonic": is_monotonic,
        "baseline_id": float(known_100),
        "pure_rl_id": float(known_0),
        "mixing_details": {
            str(frac): {
                "final_id": float(info["final_id"]),
                "history": {
                    k: [float(v) for v in vals]
                    for k, vals in info["history"].items()
                },
            }
            for frac, info in mixing_results.items()
        },
    }

    # Save results JSON
    results_path = out_path / "mixing_results.json"
    results_path.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    logger.info("Results saved: %s", results_path)

    return results
