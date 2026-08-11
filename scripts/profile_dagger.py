"""Profile DAgger iteration components to identify bottlenecks.

Runs a small number of DAgger iterations and reports per-component
timing breakdowns. Use this to decide which optimisations matter.

Usage:
    python scripts/profile_dagger.py [--config PATH] [--override key=value ...]

Without ``--config`` this profiles ``configs/defaults.yaml``, whose
performance knobs differ from the cluster configs (``use_amp`` and
``torch_compile`` are off there, and ``dagger_batch_size`` differs), so
pass the config the run will actually use.
"""

from __future__ import annotations

import os
import platform
import random
import sys
import time
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.buffer import ReplayBuffer
from src.config import load_config
from src.curriculum import DynamicCurriculum, efficiency_filter
from src.diffusion.forward import q_sample
from src.diffusion.loss import auxiliary_goal_loss, mdlm_loss
from src.diffusion.sampling import greedy_sample
from src.diffusion.schedules import get_schedule
from src.envs.minihack_env import collect_oracle_trajectory, make_env
from src.models.denoiser import ModelEMA, make_model

# ── Helpers ─────────────────────────────────────────────────────────────

NUM_PROFILE_ITERATIONS = 3


class Timer:
    """Accumulating wall-clock timer."""

    def __init__(self) -> None:
        self.total: float = 0.0
        self.calls: int = 0
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total += time.perf_counter() - self._start
        self.calls += 1
        self._start = None

    @property
    def per_call_ms(self) -> float:
        return (self.total / max(self.calls, 1)) * 1000


def print_device_info() -> None:
    """Print device and system information."""
    print("=" * 72)
    print("DEVICE & SYSTEM INFO")
    print("=" * 72)
    print(f"  Platform:      {platform.system()} {platform.release()}")
    print(f"  CPU count:     {os.cpu_count()}")
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  PyTorch:       {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA:          {torch.version.cuda}")
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  GPU memory:    {mem / 1e9:.1f} GB")
    else:
        print("  GPU:           None (CPU only)")
    print()


def print_table(
    rows: list[tuple[str, float, float, float]],
    total_time: float,
) -> None:
    """Print a formatted timing table.

    Args:
        rows: List of (name, total_s, per_call_ms, pct) tuples.
        total_time: Total iteration time for percentage calculation.
    """
    header = f"{'Component':<35} | {'Total (s)':>10} | {'Per-call (ms)':>14} | {'% of iter':>10}"
    print(header)
    print("-" * len(header))
    for name, total_s, per_call_ms, pct in rows:
        print(f"  {name:<33} | {total_s:>10.3f} | {per_call_ms:>14.1f} | {pct:>9.1f}%")
    print("-" * len(header))
    print(f"  {'TOTAL per iteration':<33} | {total_time:>10.3f} |               |     100.0%")
    print()


# ── Profiling logic ─────────────────────────────────────────────────────


def profile_model_rollout_detailed(
    model: nn.Module,
    env_id: str,
    cfg: SimpleNamespace,
    device: torch.device | str,
    seed: int,
) -> dict[str, Timer]:
    """Profile a single model rollout with sub-component timing.

    Args:
        model: Eval-mode model.
        env_id: MiniHack environment ID.
        cfg: Config namespace.
        device: Torch device.
        seed: RNG seed.

    Returns:
        Dict of named Timers for each sub-component.
    """
    timers = {
        "env_reset": Timer(),
        "env_step": Timer(),
        "model_forward": Timer(),
        "overhead": Timer(),
    }

    env = make_env(env_id, None, cfg)
    try:
        with timers["env_reset"]:
            (local, glb), _info = env.reset(seed=seed)

        plan = None
        step_in_plan = 0
        max_steps = 500

        model.eval()
        for step_idx in range(max_steps):
            if plan is None or step_in_plan >= cfg.replan_every:
                with timers["overhead"]:
                    local_t = torch.from_numpy(
                        local[np.newaxis]
                    ).long().to(device)
                    glb_t = torch.from_numpy(
                        glb[np.newaxis]
                    ).long().to(device)

                with timers["model_forward"]:
                    plan = greedy_sample(
                        model, local_t, glb_t, cfg, device,
                    )
                step_in_plan = 0

            assert plan is not None
            action: int = plan[0, step_in_plan].item()  # type: ignore[union-attr]
            action = max(0, min(action, cfg.action_dim - 1))
            step_in_plan += 1

            with timers["env_step"]:
                (local, glb), reward, terminated, truncated, info = env.step(
                    action,
                )

            if info.get("won", False) or terminated or truncated:
                break
    finally:
        env.close()
    return timers


def profile_gradient_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    ema: ModelEMA,
    cfg: SimpleNamespace,
    device: torch.device | str,
    schedule_fn: Callable,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> dict[str, Timer]:
    """Profile a single gradient step with sub-component timing.

    Args:
        model: Training model.
        optimizer: Torch optimizer.
        buffer: Replay buffer.
        ema: EMA tracker.
        cfg: Config namespace.
        device: Torch device.
        schedule_fn: Noise schedule function.
        scaler: Optional GradScaler for AMP.
        use_amp: Whether to use mixed-precision autocast.

    Returns:
        Dict of named Timers.
    """
    timers = {
        "buffer_sample": Timer(),
        "forward": Timer(),
        "loss": Timer(),
        "backward": Timer(),
        "optimizer_step": Timer(),
        "ema_update": Timer(),
    }

    model.train()

    with timers["buffer_sample"]:
        batch = buffer.sample(cfg.dagger_batch_size)

    if batch is None:
        return timers

    local_np, global_np, actions_np = batch
    local_t = torch.from_numpy(local_np).long().to(device)
    global_t = torch.from_numpy(global_np).long().to(device)
    actions_t = torch.from_numpy(actions_np).long().to(device)

    B = actions_t.shape[0]
    t = torch.rand(B, device=device).clamp(1e-5, 1.0 - 1e-5)
    zt = q_sample(actions_t, t, cfg.mask_token, cfg.pad_token, schedule_fn)
    t_discrete = (t * cfg.num_diffusion_steps).long().clamp(
        0, cfg.num_diffusion_steps - 1,
    )

    optimizer.zero_grad()
    with timers["forward"]:
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(local_t, global_t, zt, t_discrete)

    with timers["loss"]:
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss_diff = mdlm_loss(
                out["actions"], actions_t, zt, t,
                cfg.mask_token, cfg.pad_token, schedule_fn,
                weight_clip=cfg.loss_weight_clip,
                label_smoothing=cfg.label_smoothing,
            )
            loss_aux = torch.tensor(0.0, device=device)
            if "goal_pred" in out:
                loss_aux = auxiliary_goal_loss(out["goal_pred"], global_t)
            loss = loss_diff + cfg.aux_loss_weight * loss_aux

    with timers["backward"]:
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.dagger_grad_clip)

    with timers["optimizer_step"]:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    with timers["ema_update"]:
        ema.update(model)

    return timers


def run_profiling(cfg: SimpleNamespace) -> None:
    """Run the full profiling session.

    Args:
        cfg: Config namespace.
    """
    device = cfg.device
    print_device_info()

    # ── Setup ───────────────────────────────────────────────────────
    print("Setting up model, buffer, curriculum...")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = make_model(cfg).to(device)
    if getattr(cfg, "torch_compile", False) and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    ema = ModelEMA(model, decay=cfg.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.dagger_lr,
        weight_decay=cfg.weight_decay,
    )
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seq_len, cfg.pad_token)
    curriculum = DynamicCurriculum(
        cfg.id_envs, cfg.curriculum_queue_size, cfg.curriculum_preseed,
    )
    schedule_fn = get_schedule(cfg.noise_schedule)

    # Seed buffer with oracle data
    print("Seeding buffer with oracle trajectories...")
    for i, env_id in enumerate(cfg.id_envs):
        for s in range(3):
            traj = collect_oracle_trajectory(env_id, seed=i * 100 + s, cfg=cfg)
            if traj is not None:
                buffer.add(traj)
    print(f"Buffer seeded: {len(buffer)} windows\n")

    # Create an eval model for collection (on CPU to save GPU memory)
    eval_model = ema.make_eval_model(model)
    eval_model = eval_model.to("cpu")

    # ── Accumulators ────────────────────────────────────────────────
    t_curriculum = Timer()
    t_model_rollout = Timer()
    t_oracle_rollout = Timer()
    t_efficiency = Timer()
    t_buffer_add = Timer()
    t_gradient_steps = Timer()
    t_iteration_total = Timer()

    # Sub-component accumulators for model rollout
    t_model_env_reset = Timer()
    t_model_env_step = Timer()
    t_model_forward = Timer()
    t_model_overhead = Timer()

    # Sub-component accumulators for gradient step
    t_grad_buffer_sample = Timer()
    t_grad_forward = Timer()
    t_grad_loss = Timer()
    t_grad_backward = Timer()
    t_grad_optimizer = Timer()
    t_grad_ema = Timer()

    n_eps = getattr(cfg, "episodes_per_iteration", 10)
    n_grad_steps = getattr(cfg, "grad_steps_per_iteration", 100)

    # AMP setup for profiling
    _use_amp = (
        getattr(cfg, "use_amp", False) and str(device).startswith("cuda")
    )
    _scaler = (
        torch.amp.GradScaler("cuda", enabled=_use_amp) if _use_amp else None
    )

    print("=" * 72)
    print(f"PROFILING {NUM_PROFILE_ITERATIONS} DAgger ITERATIONS")
    print(f"  Episodes/iteration:  {n_eps}")
    print(f"  Grad steps/iteration: {n_grad_steps}")
    print(f"  Batch size:          {cfg.dagger_batch_size}")
    print(f"  Device:              {device}")
    print(f"  AMP:                 {_use_amp}")
    print("=" * 72)
    print()

    # ── Profile loop ────────────────────────────────────────────────
    for it in range(NUM_PROFILE_ITERATIONS):
        print(f"Iteration {it + 1}/{NUM_PROFILE_ITERATIONS}...")
        iter_start = time.perf_counter()

        # Collection phase: n_eps episodes
        for ep in range(n_eps):
            with t_curriculum:
                env_id = curriculum.sample_env()
            seed = random.randint(0, 2**31 - 1)

            # Model rollout with detailed sub-timing (CPU for collection)
            with t_model_rollout:
                sub_timers = profile_model_rollout_detailed(
                    eval_model, env_id, cfg, "cpu", seed,
                )
            # Accumulate sub-timers
            t_model_env_reset.total += sub_timers["env_reset"].total
            t_model_env_reset.calls += sub_timers["env_reset"].calls
            t_model_env_step.total += sub_timers["env_step"].total
            t_model_env_step.calls += sub_timers["env_step"].calls
            t_model_forward.total += sub_timers["model_forward"].total
            t_model_forward.calls += sub_timers["model_forward"].calls
            t_model_overhead.total += sub_timers["overhead"].total
            t_model_overhead.calls += sub_timers["overhead"].calls

            # Oracle rollout
            with t_oracle_rollout:
                oracle_result = collect_oracle_trajectory(
                    env_id, seed, cfg,
                )
            oracle_steps = (
                len(oracle_result["actions"]) if oracle_result else 999
            )

            # Efficiency filter
            model_won = False  # untrained model almost never wins
            model_steps = 500
            with t_efficiency:
                add = efficiency_filter(
                    model_won, model_steps, oracle_steps,
                    cfg.efficiency_multiplier,
                )

            # Buffer add
            if add and oracle_result is not None:
                with t_buffer_add:
                    buffer.add(oracle_result)

        # Free collection GPU memory before training phase
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Gradient steps phase
        with t_gradient_steps:
            for gs in range(n_grad_steps):
                sub_timers = profile_gradient_step(
                    model, optimizer, buffer, ema, cfg, device, schedule_fn,
                    scaler=_scaler, use_amp=_use_amp,
                )
                t_grad_buffer_sample.total += sub_timers["buffer_sample"].total
                t_grad_buffer_sample.calls += sub_timers["buffer_sample"].calls
                t_grad_forward.total += sub_timers["forward"].total
                t_grad_forward.calls += sub_timers["forward"].calls
                t_grad_loss.total += sub_timers["loss"].total
                t_grad_loss.calls += sub_timers["loss"].calls
                t_grad_backward.total += sub_timers["backward"].total
                t_grad_backward.calls += sub_timers["backward"].calls
                t_grad_optimizer.total += sub_timers["optimizer_step"].total
                t_grad_optimizer.calls += sub_timers["optimizer_step"].calls
                t_grad_ema.total += sub_timers["ema_update"].total
                t_grad_ema.calls += sub_timers["ema_update"].calls

        # Re-sync eval model for next iteration
        ema.apply_to(eval_model)
        eval_model.eval()

        iter_elapsed = time.perf_counter() - iter_start
        t_iteration_total.total += iter_elapsed
        t_iteration_total.calls += 1
        print(f"  -> {iter_elapsed:.2f}s")

    # ── Results ─────────────────────────────────────────────────────
    avg_iter = t_iteration_total.total / NUM_PROFILE_ITERATIONS

    # Compute per-iteration averages
    def avg(timer: Timer) -> float:
        return timer.total / NUM_PROFILE_ITERATIONS

    print()
    print("=" * 72)
    print("PROFILING RESULTS (averaged over {} iterations)".format(
        NUM_PROFILE_ITERATIONS,
    ))
    print("=" * 72)

    # Main component table
    main_rows = [
        (
            f"Model rollout ({n_eps} eps)",
            avg(t_model_rollout),
            t_model_rollout.per_call_ms / n_eps if t_model_rollout.calls else 0,
            avg(t_model_rollout) / avg_iter * 100,
        ),
        (
            f"Oracle rollout ({n_eps} eps)",
            avg(t_oracle_rollout),
            t_oracle_rollout.per_call_ms / n_eps if t_oracle_rollout.calls else 0,
            avg(t_oracle_rollout) / avg_iter * 100,
        ),
        (
            f"Gradient steps ({n_grad_steps})",
            avg(t_gradient_steps),
            t_gradient_steps.per_call_ms if t_gradient_steps.calls else 0,
            avg(t_gradient_steps) / avg_iter * 100,
        ),
        (
            "Curriculum sampling",
            avg(t_curriculum),
            t_curriculum.per_call_ms if t_curriculum.calls else 0,
            avg(t_curriculum) / avg_iter * 100,
        ),
        (
            "Efficiency filter",
            avg(t_efficiency),
            t_efficiency.per_call_ms if t_efficiency.calls else 0,
            avg(t_efficiency) / avg_iter * 100,
        ),
        (
            "Buffer add",
            avg(t_buffer_add),
            t_buffer_add.per_call_ms if t_buffer_add.calls else 0,
            avg(t_buffer_add) / avg_iter * 100,
        ),
    ]
    other_time = avg_iter - sum(r[1] for r in main_rows)
    main_rows.append((
        "Other overhead",
        other_time,
        0,
        other_time / avg_iter * 100,
    ))

    print("\n--- Main Components ---")
    print_table(main_rows, avg_iter)

    # Model rollout sub-breakdown
    model_total = avg(t_model_rollout)
    if model_total > 0:
        print("--- Model Rollout Breakdown ---")
        mr_rows = [
            (
                "  model.forward() (greedy sample)",
                avg(t_model_forward),
                t_model_forward.per_call_ms if t_model_forward.calls else 0,
                avg(t_model_forward) / model_total * 100,
            ),
            (
                "  env.step()",
                avg(t_model_env_step),
                t_model_env_step.per_call_ms if t_model_env_step.calls else 0,
                avg(t_model_env_step) / model_total * 100,
            ),
            (
                "  env.reset()",
                avg(t_model_env_reset),
                t_model_env_reset.per_call_ms if t_model_env_reset.calls else 0,
                avg(t_model_env_reset) / model_total * 100,
            ),
            (
                "  Tensor overhead",
                avg(t_model_overhead),
                t_model_overhead.per_call_ms if t_model_overhead.calls else 0,
                avg(t_model_overhead) / model_total * 100,
            ),
        ]
        print_table(mr_rows, model_total)

    # Gradient step sub-breakdown
    grad_total = avg(t_gradient_steps)
    if grad_total > 0:
        print("--- Gradient Step Breakdown ---")
        gs_rows = [
            (
                "  buffer.sample()",
                avg(t_grad_buffer_sample),
                t_grad_buffer_sample.per_call_ms if t_grad_buffer_sample.calls else 0,
                avg(t_grad_buffer_sample) / grad_total * 100,
            ),
            (
                "  model.forward()",
                avg(t_grad_forward),
                t_grad_forward.per_call_ms if t_grad_forward.calls else 0,
                avg(t_grad_forward) / grad_total * 100,
            ),
            (
                "  loss computation",
                avg(t_grad_loss),
                t_grad_loss.per_call_ms if t_grad_loss.calls else 0,
                avg(t_grad_loss) / grad_total * 100,
            ),
            (
                "  backward + grad clip",
                avg(t_grad_backward),
                t_grad_backward.per_call_ms if t_grad_backward.calls else 0,
                avg(t_grad_backward) / grad_total * 100,
            ),
            (
                "  optimizer.step()",
                avg(t_grad_optimizer),
                t_grad_optimizer.per_call_ms if t_grad_optimizer.calls else 0,
                avg(t_grad_optimizer) / grad_total * 100,
            ),
            (
                "  EMA update",
                avg(t_grad_ema),
                t_grad_ema.per_call_ms if t_grad_ema.calls else 0,
                avg(t_grad_ema) / grad_total * 100,
            ),
        ]
        print_table(gs_rows, grad_total)

    # ── Memory audit ────────────────────────────────────────────────
    if torch.cuda.is_available():
        print("=" * 72)
        print("MEMORY AUDIT")
        print("=" * 72)

        torch.cuda.reset_peak_memory_stats()

        # Single forward pass (collection-style, batch=1)
        model.eval()
        with torch.no_grad():
            local_t = torch.randint(
                0, 5999, (1, cfg.crop_size, cfg.crop_size),
                dtype=torch.long, device=device,
            )
            global_t = torch.randint(
                0, 5999, (1, cfg.map_h, cfg.map_w),
                dtype=torch.long, device=device,
            )
            seq = torch.full(
                (1, cfg.seq_len), cfg.mask_token,
                dtype=torch.long, device=device,
            )
            _ = model(local_t, global_t, seq, 50)
        peak_inference = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak GPU mem (inference, B=1):   {peak_inference:.1f} MB")

        torch.cuda.reset_peak_memory_stats()

        # Training forward + backward (batch=dagger_batch_size)
        model.train()
        B = min(cfg.dagger_batch_size, 1024)
        local_t = torch.randint(
            0, 5999, (B, cfg.crop_size, cfg.crop_size),
            dtype=torch.long, device=device,
        )
        global_t = torch.randint(
            0, 5999, (B, cfg.map_h, cfg.map_w),
            dtype=torch.long, device=device,
        )
        actions_t = torch.randint(
            0, cfg.action_dim, (B, cfg.seq_len),
            dtype=torch.long, device=device,
        )
        t = torch.rand(B, device=device).clamp(1e-5, 1.0 - 1e-5)
        zt = q_sample(actions_t, t, cfg.mask_token, cfg.pad_token, schedule_fn)
        t_discrete = (t * cfg.num_diffusion_steps).long().clamp(
            0, cfg.num_diffusion_steps - 1,
        )
        out = model(local_t, global_t, zt, t_discrete)
        loss = mdlm_loss(
            out["actions"], actions_t, zt, t,
            cfg.mask_token, cfg.pad_token, schedule_fn,
            weight_clip=cfg.loss_weight_clip,
        )
        loss.backward()
        peak_training = torch.cuda.max_memory_allocated() / 1e6
        total_gpu = torch.cuda.get_device_properties(0).total_memory / 1e6
        headroom = total_gpu - peak_training
        print(f"  Peak GPU mem (train, B={B}):  {peak_training:.1f} MB")
        print(f"  Total GPU memory:                {total_gpu:.1f} MB")
        print(f"  Headroom for larger batch:       {headroom:.1f} MB")
        print()
    else:
        print("\n(No GPU — skipping memory audit)\n")

    # ── Summary ─────────────────────────────────────────────────────
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    collection_time = avg(t_model_rollout) + avg(t_oracle_rollout)
    print(f"  Collection (model+oracle): {collection_time:.2f}s "
          f"({collection_time / avg_iter * 100:.1f}%)")
    print(f"  Training (grad steps):     {avg(t_gradient_steps):.2f}s "
          f"({avg(t_gradient_steps) / avg_iter * 100:.1f}%)")
    print(f"  Total per iteration:       {avg_iter:.2f}s")
    print()


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli_overrides = {}
    config_path: str | None = None
    args = [a for a in sys.argv[1:] if a != "--override"]
    pending_config = False
    for arg in args:
        if pending_config:
            config_path = arg
            pending_config = False
        elif arg == "--config":
            pending_config = True
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
        elif "=" in arg:
            k, v = arg.split("=", 1)
            cli_overrides[k] = v
    if pending_config:
        raise SystemExit("--config expects a path")

    # Use smoke-test-like settings for fast profiling
    # but keep realistic episode/grad counts
    cli_overrides.setdefault("use_wandb", "false")
    cli_overrides.setdefault("buffer_capacity", "500")

    cfg = load_config(config_path, cli_overrides=cli_overrides)
    run_profiling(cfg)
