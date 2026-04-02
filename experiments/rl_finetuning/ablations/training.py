"""Core training loop for RL fine-tuning ablations (PyTorch).

``run_ablation`` is the public entry point: it loads a pretrained
checkpoint, builds the loss/optimizer specified by the ``AblationSpec``,
collects rollouts with the diffusion model, and trains in a standard
Python loop (no JAX scan).

Adapted from Craftax JAX implementation.
Key differences:
- Eager PyTorch loop instead of ``jax.lax.scan``
- Rollouts via ``run_model_episode`` from ``src/planners/collect``
- Eval via ``Evaluator`` from ``src/planners/inference``
- ``ModelEMA`` from ``src/models/denoiser``
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import random
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from experiments.rl_finetuning.ablations.losses import (
    LossContext,
    estimate_fisher_diagonal,
)
from experiments.rl_finetuning.ablations.optimizers import (
    apply_lora_to_model,
    collect_gradients,
    apply_gradients,
    gradient_surgery,
    make_optimizer_lora,
    remove_lora_from_model,
)
from experiments.rl_finetuning.ablations.registry import AblationSpec
from experiments.rl_finetuning.diagnostics.gradient import (
    compute_grad_alignment,
    compute_per_layer_grad_norms,
    compute_surgery_metrics,
)
from experiments.rl_finetuning.diagnostics.representation import (
    compute_cka,
    compute_repr_drift,
)
from experiments.rl_finetuning.diagnostics.timestep import (
    compute_t_analysis,
)
from src.diffusion.schedules import get_schedule
from src.models.denoiser import ModelEMA, make_model
from src.planners.collect import run_model_episode
from src.planners.inference import Evaluator

logger = logging.getLogger(__name__)

_EPS: float = 1e-5


def _wandb_log(metrics: dict[str, float], step: int) -> None:
    """Log metrics to wandb if available, guarded against import/init failure.

    Always injects ``"iteration": step`` so that
    ``define_metric(ns, step_metric="iteration")`` works correctly.

    Args:
        metrics: Flat ``{namespace/key: value}`` dict.
        step: Global iteration index.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({**metrics, "iteration": step}, step=step)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# AblationHistory
# ---------------------------------------------------------------------------


@dataclass
class AblationHistory:
    """Typed training history for a single ablation run.

    All list fields are appended at their respective logging frequency.
    Serialisable via ``to_dict()`` / ``from_dict()``.
    """

    iters: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)

    env_score_iters: list[int] = field(default_factory=list)
    env_score: list[float] = field(default_factory=list)

    eval_iters: list[int] = field(default_factory=list)
    eval_score: list[float] = field(default_factory=list)

    grad_align_iters: list[int] = field(default_factory=list)
    grad_align: list[float] = field(default_factory=list)
    rl_grad_norm: list[float] = field(default_factory=list)
    bc_grad_norm: list[float] = field(default_factory=list)

    per_layer_iters: list[int] = field(default_factory=list)
    per_layer_norms: list[dict[str, float]] = field(default_factory=list)

    repr_drift_iters: list[int] = field(default_factory=list)
    repr_drift_kl: list[float] = field(default_factory=list)
    repr_drift_kl_low_t: list[float] = field(default_factory=list)
    repr_drift_kl_mid_t: list[float] = field(default_factory=list)
    repr_drift_kl_high_t: list[float] = field(default_factory=list)

    cka_iters: list[int] = field(default_factory=list)
    cka_similarity: list[float] = field(default_factory=list)

    t_analysis_iters: list[int] = field(default_factory=list)
    norm_low_t: list[float] = field(default_factory=list)
    norm_high_t: list[float] = field(default_factory=list)
    lowhigh_cos: list[float] = field(default_factory=list)
    t_bin_norms: list[dict[str, float]] = field(default_factory=list)

    win_rate: list[float] = field(default_factory=list)
    effective_batch_size: list[float] = field(default_factory=list)

    surgery_iters: list[int] = field(default_factory=list)
    surgery_fraction: list[float] = field(default_factory=list)
    surgery_n_conflicting: list[int] = field(default_factory=list)

    per_env_win_rates: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict.

        Returns:
            Dict with all list fields.
        """
        return {k: list(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "AblationHistory":
        """Reconstruct from a dict.

        Args:
            d: Dict from ``to_dict()``.

        Returns:
            ``AblationHistory`` instance.
        """
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# Replay buffer for mixed replay (simple ring buffer)
# ---------------------------------------------------------------------------


class MixedReplayBuffer:
    """Fixed-size ring buffer storing (local_obs, global_obs, x0, returns).

    Args:
        capacity: Maximum number of windows.
        seq_len: Action sequence length.
        device: Torch device.
    """

    def __init__(
        self, capacity: int, seq_len: int, device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.seq_len = seq_len
        self.device = device
        self._local = torch.zeros(capacity, 9, 9, dtype=torch.long, device=device)
        self._global = torch.zeros(capacity, 21, 79, dtype=torch.long, device=device)
        self._x0 = torch.zeros(capacity, seq_len, dtype=torch.long, device=device)
        self._returns = torch.zeros(capacity, dtype=torch.float32, device=device)
        self._write_idx = 0
        self._count = 0

    def push(
        self,
        local_obs: Tensor,
        global_obs: Tensor,
        x0: Tensor,
        returns: Tensor,
    ) -> None:
        """Push a batch of windows into the buffer.

        Args:
            local_obs: ``[N, 9, 9]``.
            global_obs: ``[N, 21, 79]``.
            x0: ``[N, H]``.
            returns: ``[N]``.
        """
        n = local_obs.shape[0]
        for i in range(n):
            idx = self._write_idx % self.capacity
            self._local[idx] = local_obs[i]
            self._global[idx] = global_obs[i]
            self._x0[idx] = x0[i]
            self._returns[idx] = returns[i]
            self._write_idx += 1
            self._count = min(self._count + 1, self.capacity)

    def sample(
        self, n: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample *n* windows uniformly with replacement.

        Args:
            n: Number of samples.

        Returns:
            Tuple of (local_obs, global_obs, x0, returns).
        """
        valid = max(self._count, 1)
        idx = torch.randint(0, valid, (n,), device=self.device)
        return self._local[idx], self._global[idx], self._x0[idx], self._returns[idx]

    @property
    def size(self) -> int:
        """Current number of stored windows."""
        return self._count


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


class RewardModel(nn.Module):
    """Lightweight MLP for learned return prediction.

    Args:
        obs_dim: Input dimensionality (flattened obs features).
        width: Hidden layer width.
        depth: Number of hidden layers.
    """

    def __init__(self, obs_dim: int, width: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Predict return from observation features.

        Args:
            x: ``[B, obs_dim]``.

        Returns:
            ``[B]`` predicted returns.
        """
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------


def compute_advantages(
    returns: Tensor,
    floor: float,
    cap: float,
    wins_only: bool,
    win_thresh: float,
    use_running_stats: bool,
    ema_decay: float,
    running_mean: float,
    running_std: float,
) -> tuple[Tensor, float, float]:
    """Compute per-sample advantage weights from returns.

    Args:
        returns: ``[N]`` raw episode/window returns.
        floor: Lower clip bound.
        cap: Upper clip bound.
        wins_only: If True, binary win mask (1 if return > thresh).
        win_thresh: Threshold for win detection.
        use_running_stats: If True, normalise with EMA stats.
        ema_decay: EMA decay factor.
        running_mean: Current EMA mean.
        running_std: Current EMA std.

    Returns:
        Tuple of (advantages ``[N]``, updated_mean, updated_std).
    """
    clipped = returns.clamp(min=0.0)
    batch_mean = clipped.mean().item()
    batch_std = clipped.std().item() + 1e-8

    if wins_only:
        adv = (returns > win_thresh).float()
    elif use_running_stats:
        new_mean = ema_decay * running_mean + (1.0 - ema_decay) * batch_mean
        new_std = ema_decay * running_std + (1.0 - ema_decay) * batch_std
        adv = ((clipped - new_mean) / new_std + 1.0).clamp(floor, cap)
        return adv, new_mean, new_std
    else:
        weights = clipped / (batch_mean + _EPS)
        adv = weights.clamp(floor, cap)

    return adv, batch_mean, batch_std


def _effective_batch_size(advantages: Tensor) -> float:
    """Effective batch size: (sum w)^2 / sum w^2.

    Args:
        advantages: ``[N]`` advantage weights.

    Returns:
        Effective batch size as float.
    """
    sum_w = advantages.sum()
    sum_w2 = (advantages ** 2).sum()
    return (sum_w ** 2 / sum_w2.clamp(min=1e-10)).item()


# ---------------------------------------------------------------------------
# Episode collection -> training windows
# ---------------------------------------------------------------------------


def _extract_windows(
    episode: dict,
    seq_len: int,
    pad_token: int,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Extract sliding windows from a single episode.

    Args:
        episode: Dict from ``run_model_episode`` with ``"local"``,
                 ``"global"``, ``"actions"``, ``"total_reward"`` keys.
        seq_len: Window length (plan horizon).
        pad_token: PAD token ID for short episodes.

    Returns:
        Tuple of (local_obs ``[W,9,9]``, global_obs ``[W,21,79]``,
        x0 ``[W,H]``, episode_return).
    """
    local_arr = torch.from_numpy(episode["local"]).long()   # [T, 9, 9]
    global_arr = torch.from_numpy(episode["global"]).long()  # [T, 21, 79]
    actions = torch.from_numpy(episode["actions"]).long()    # [T]
    T = actions.shape[0]
    ret = episode["total_reward"]

    if T == 0:
        return (
            torch.empty(0, 9, 9, dtype=torch.long),
            torch.empty(0, 21, 79, dtype=torch.long),
            torch.empty(0, seq_len, dtype=torch.long),
            ret,
        )

    # Pad if shorter than seq_len
    if T < seq_len:
        pad_len = seq_len - T
        actions = torch.cat([
            actions,
            torch.full((pad_len,), pad_token, dtype=torch.long),
        ])
        local_arr = torch.cat([
            local_arr,
            local_arr[-1:].expand(pad_len, -1, -1),
        ])
        global_arr = torch.cat([
            global_arr,
            global_arr[-1:].expand(pad_len, -1, -1),
        ])
        T = seq_len

    n_windows = T - seq_len + 1
    local_out = local_arr[:n_windows]       # [W, 9, 9]
    global_out = global_arr[:n_windows]     # [W, 21, 79]
    x0_out = actions.unfold(0, seq_len, 1)  # [W, H]

    return local_out, global_out, x0_out, ret


def collect_training_data(
    model: nn.Module,
    cfg: SimpleNamespace,
    device: torch.device,
    n_episodes: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collect episodes and extract training windows.

    Args:
        model: Eval-mode model for rollouts.
        cfg: Config namespace.
        device: Torch device.
        n_episodes: Number of episodes to collect.

    Returns:
        Tuple of (local_obs, global_obs, x0, returns) batched over
        all windows. Returns is per-window (all windows from same
        episode share the episode return).
    """
    all_local: list[Tensor] = []
    all_global: list[Tensor] = []
    all_x0: list[Tensor] = []
    all_returns: list[Tensor] = []
    total_wins = 0

    env_ids = cfg.id_envs
    for _ in range(n_episodes):
        env_id = random.choice(env_ids)
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
        all_returns.append(torch.full((lo.shape[0],), ret))
        if ep["won"]:
            total_wins += 1

    if not all_local:
        empty = torch.empty(0)
        return empty, empty, empty, empty

    local_obs = torch.cat(all_local).to(device)
    global_obs = torch.cat(all_global).to(device)
    x0 = torch.cat(all_x0).to(device)
    returns = torch.cat(all_returns).to(device)

    return local_obs, global_obs, x0, returns


# ---------------------------------------------------------------------------
# Reward model training step
# ---------------------------------------------------------------------------


def _train_reward_model(
    rm: RewardModel,
    rm_optim: torch.optim.Optimizer,
    local_obs: Tensor,
    global_obs: Tensor,
    returns: Tensor,
    n_steps: int,
) -> None:
    """Train the reward model on collected returns.

    Uses flattened global map as input features.

    Args:
        rm: Reward model.
        rm_optim: Reward model optimizer.
        local_obs: ``[N, 9, 9]``.
        global_obs: ``[N, 21, 79]``.
        returns: ``[N]`` targets.
        n_steps: Number of gradient steps.
    """
    features = global_obs.reshape(global_obs.shape[0], -1).float()  # [N, 21*79]
    rm.train()
    for _ in range(n_steps):
        rm_optim.zero_grad()
        preds = rm(features)
        loss = F.mse_loss(preds, returns)
        loss.backward()
        rm_optim.step()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_ablation(
    spec: AblationSpec,
    cfg: SimpleNamespace,
    checkpoint_path: str,
    device: torch.device,
    seed: int = 0,
    wandb_step_offset: int = 0,
) -> tuple[AblationHistory, float, nn.Module]:
    """Run one complete ablation experiment.

    Loads a pretrained checkpoint, sets up the ablation-specific loss
    and optimizer, collects rollouts, trains, and logs diagnostics.

    Args:
        spec: Ablation specification from the registry.
        cfg: Merged config namespace (defaults + ablation overrides).
        checkpoint_path: Path to pretrained DAgger checkpoint.
        device: Torch device.
        seed: Random seed.
        wandb_step_offset: Global step offset so that wandb steps are
            monotonically increasing across ablations/seeds.

    Returns:
        Tuple of (history, final_score, final_state_dict).
    """
    logger.info("=" * 60)
    logger.info("ABLATION: %s  [Group %s]", spec.name, spec.group)
    logger.info("  %s", spec.description)
    logger.info("=" * 60)

    # Log ablation metadata to wandb config
    try:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "ablation_name": spec.name,
                "ablation_group": spec.group,
                "ablation_description": spec.description,
                "seed": seed,
            }, allow_val_change=True)
    except Exception:
        pass

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load pretrained model
    raw_model = make_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["ema_state_dict"])

    ref_model = copy.deepcopy(raw_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Schedule function
    schedule_fn = get_schedule(cfg.noise_schedule)
    cfg._schedule_fn = schedule_fn

    # EMA (always uses raw_model; deepcopy breaks on compiled models)
    ema = ModelEMA(raw_model, decay=getattr(cfg, "ema_decay", 0.999))

    # EWC Fisher estimation
    fisher = None
    if spec.name == "ewc":
        n_batches = getattr(cfg, "ewc_fisher_batches", 20)
        logger.info("  Estimating Fisher diagonal (%d batches)...", n_batches)
        fisher_batches: list[tuple[Tensor, Tensor, Tensor]] = []
        eval_model = ema.make_eval_model(raw_model)
        for _ in range(n_batches):
            lo, go, x0, _ = collect_training_data(
                eval_model, cfg, device, n_episodes=1,
            )
            if lo.shape[0] > 0:
                bs = min(lo.shape[0], cfg.batch_size)
                fisher_batches.append((lo[:bs], go[:bs], x0[:bs]))
        fisher = estimate_fisher_diagonal(
            raw_model, schedule_fn, cfg, fisher_batches, device,
        )
        logger.info("  Fisher diagonal estimated.")

    # LoRA setup
    lora_params = None
    if spec.use_lora:
        lora_rank = getattr(cfg, "lora_rank", 8)
        lora_alpha = getattr(cfg, "lora_alpha", 16.0)
        lora_params = apply_lora_to_model(raw_model, lora_rank, lora_alpha)
        # Re-initialize EMA after LoRA changes the model's state_dict keys
        ema = ModelEMA(raw_model, decay=getattr(cfg, "ema_decay", 0.999))
        optimizer = make_optimizer_lora(cfg, lora_params)
    else:
        optimizer = spec.optimizer_factory(cfg, raw_model)

    # torch.compile: wrap for training only; shares parameters with raw_model
    if getattr(cfg, "torch_compile", False) and hasattr(torch, "compile"):
        logger.info("  Compiling model with torch.compile")
        model = torch.compile(raw_model, mode="default")
    else:
        model = raw_model

    # Loss context and function (ref_model is never compiled)
    ctx = LossContext(ref_model=ref_model, schedule_fn=schedule_fn, cfg=cfg)
    extra_kwargs = {}
    if fisher is not None:
        extra_kwargs["fisher"] = fisher
    loss_fn = spec.loss_factory(ctx, **extra_kwargs)

    # BC loss (for gradient surgery + alignment diagnostics)
    from experiments.rl_finetuning.ablations.losses import make_loss_baseline
    bc_loss_fn = make_loss_baseline(ctx)

    # Evaluator
    evaluator = Evaluator()

    # AMP (mixed precision)
    _use_amp = (
        getattr(cfg, "use_amp", False) and device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp)
    if _use_amp:
        logger.info("  AMP enabled for ablation training.")

    # Config values
    max_iter = getattr(cfg, "max_iter", 1000)
    batch_size = cfg.batch_size
    episodes_per_iter = getattr(cfg, "episodes_per_iter", 10)
    grad_steps = getattr(cfg, "grad_steps_per_iter", 1)
    max_grad_norm = getattr(cfg, "max_grad_norm", 1.0)

    # Diagnostic frequencies
    eval_every = getattr(cfg, "eval_every", 50)
    eval_episodes = getattr(cfg, "eval_episodes", 20)
    grad_align_every = getattr(cfg, "grad_align_every", 25)
    repr_drift_every = getattr(cfg, "repr_drift_every", 25)
    t_analysis_every = getattr(cfg, "t_analysis_every", 25)
    cka_every = getattr(cfg, "cka_every", 50)
    per_layer_every = getattr(cfg, "per_layer_every", 25)
    n_t_bins = getattr(cfg, "t_analysis_n_bins", 10)

    # Advantage params
    floor = getattr(cfg, "return_weight_floor", 0.1)
    cap = getattr(cfg, "return_weight_cap", 5.0)
    win_thresh = getattr(cfg, "win_threshold", 0.5)
    ema_decay_stats = getattr(cfg, "running_stats_ema_decay", 0.99)
    reward_filter_pct = getattr(cfg, "reward_filter_percentile", 75)

    # Mixed replay buffer
    replay_buf = None
    replay_ratio = getattr(cfg, "mixed_replay_ratio", 0.25)
    if spec.mixed_replay:
        buf_size = getattr(cfg, "mixed_replay_buffer_size", 10000)
        replay_buf = MixedReplayBuffer(buf_size, cfg.seq_len, device)

    # Reward model
    rm = None
    rm_optim = None
    rm_steps = getattr(cfg, "reward_model_train_steps", 50)
    if spec.reward_model_weighting:
        rm_width = getattr(cfg, "reward_model_width", 64)
        rm_depth = getattr(cfg, "reward_model_depth", 2)
        rm_lr = getattr(cfg, "reward_model_lr", 1e-3)
        obs_dim = 21 * 79
        rm = RewardModel(obs_dim, rm_width, rm_depth).to(device)
        rm_optim = torch.optim.Adam(rm.parameters(), lr=rm_lr)

    running_mean = 0.0
    running_std = 1.0
    history = AblationHistory()

    # Snapshot initial state for param drift tracking
    _init_state = {
        k: v.clone() for k, v in raw_model.state_dict().items()
        if v.is_floating_point()
    }

    # Log model param counts to wandb
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(
        p.numel() for p in raw_model.parameters() if p.requires_grad
    )
    try:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "model_total_params": total_params,
                "model_trainable_params": trainable_params,
            }, allow_val_change=True)
    except Exception:
        pass

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    for iteration in range(1, max_iter + 1):
        iter_start = time.perf_counter()
        cfg._current_iter = iteration

        # -- Collect episodes --
        collect_start = time.perf_counter()
        eval_model = ema.make_eval_model(raw_model)
        local_obs, global_obs, x0, returns = collect_training_data(
            eval_model, cfg, device, episodes_per_iter,
        )
        collect_time = time.perf_counter() - collect_start

        if local_obs.shape[0] == 0:
            logger.warning("  Iter %d: no data collected, skipping.", iteration)
            continue

        # -- Action diversity filter --
        if spec.action_diversity_filter:
            diverse = (x0 != x0[:, :1]).any(dim=1)
            local_obs = local_obs[diverse]
            global_obs = global_obs[diverse]
            x0 = x0[diverse]
            returns = returns[diverse]
            if local_obs.shape[0] == 0:
                continue

        # -- Reward filtering --
        if spec.reward_filtering:
            thresh = torch.quantile(returns, reward_filter_pct / 100.0)
            keep = returns >= thresh
            local_obs = local_obs[keep]
            global_obs = global_obs[keep]
            x0 = x0[keep]
            returns = returns[keep]
            if local_obs.shape[0] == 0:
                continue

        # -- Push to mixed replay buffer --
        if replay_buf is not None:
            replay_buf.push(local_obs, global_obs, x0, returns)

        # -- Reward model training --
        if rm is not None and rm_optim is not None:
            _train_reward_model(
                rm, rm_optim, local_obs, global_obs, returns, rm_steps,
            )
            with torch.no_grad():
                feats = global_obs.reshape(global_obs.shape[0], -1).float()
                returns = rm(feats)

        # -- Compute advantages --
        advantages, running_mean, running_std = compute_advantages(
            returns, floor, cap,
            wins_only=spec.wins_only,
            win_thresh=win_thresh,
            use_running_stats=spec.running_stats,
            ema_decay=ema_decay_stats,
            running_mean=running_mean,
            running_std=running_std,
        )

        # -- Shuffle and batch --
        n = local_obs.shape[0]
        perm = torch.randperm(n, device=device)
        local_obs = local_obs[perm]
        global_obs = global_obs[perm]
        x0 = x0[perm]
        advantages = advantages[perm]
        returns_perm = returns[perm]

        # -- Mixed replay: splice offline data --
        if replay_buf is not None and replay_buf.size > 0:
            n_offline = max(1, int(batch_size * replay_ratio))
            n_online = batch_size - n_offline
            buf_lo, buf_go, buf_x0, buf_ret = replay_buf.sample(n_offline)
            buf_adv, _, _ = compute_advantages(
                buf_ret, floor, cap,
                wins_only=False, win_thresh=win_thresh,
                use_running_stats=False, ema_decay=0.0,
                running_mean=0.0, running_std=1.0,
            )
            local_b = torch.cat([local_obs[:n_online], buf_lo])
            global_b = torch.cat([global_obs[:n_online], buf_go])
            x0_b = torch.cat([x0[:n_online], buf_x0])
            adv_b = torch.cat([advantages[:n_online], buf_adv])
        else:
            local_b = local_obs[:batch_size]
            global_b = global_obs[:batch_size]
            x0_b = x0[:batch_size]
            adv_b = advantages[:batch_size]

        # -- Gradient steps --
        train_start = time.perf_counter()
        model.train()
        g_rl: dict[str, Tensor] = {}
        g_proj: dict[str, Tensor] = {}
        loss_val = 0.0
        grad_norm_val = 0.0
        for _ in range(grad_steps):
            if spec.gradient_surgery:
                # PCGrad: compute RL grad, BC grad, project
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=_use_amp):
                    rl_loss = loss_fn(
                        model, local_b, global_b, x0_b, adv_b, cfg, device,
                    )
                scaler.scale(rl_loss).backward()
                scaler.unscale_(optimizer)
                g_rl = collect_gradients(model)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=_use_amp):
                    bc_loss = bc_loss_fn(
                        model, local_b, global_b, x0_b, None, cfg, device,
                    )
                scaler.scale(bc_loss).backward()
                scaler.unscale_(optimizer)
                g_bc = collect_gradients(model)

                g_proj = gradient_surgery(g_rl, g_bc)
                apply_gradients(model, g_proj)
                grad_norm_val = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                ).item()
                scaler.step(optimizer)
                scaler.update()
                loss_val = rl_loss.item()
            else:
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=_use_amp):
                    loss_val_t = loss_fn(
                        model, local_b, global_b, x0_b, adv_b, cfg, device,
                    )
                scaler.scale(loss_val_t).backward()
                scaler.unscale_(optimizer)
                grad_norm_val = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                ).item()
                scaler.step(optimizer)
                scaler.update()
                loss_val = loss_val_t.item()

            ema.update(raw_model)

        # -- Metrics --
        train_time = time.perf_counter() - train_start
        iter_time = time.perf_counter() - iter_start
        wr = (returns_perm > win_thresh).float().mean().item()
        eff_bs = _effective_batch_size(adv_b)
        mean_return = returns_perm.mean().item()

        history.iters.append(iteration)
        history.loss.append(loss_val)
        history.env_score_iters.append(iteration)
        history.env_score.append(mean_return)
        history.win_rate.append(wr)
        history.effective_batch_size.append(eff_bs)

        # Current LR
        current_lr = optimizer.param_groups[0]["lr"]

        # Build wandb metrics dict for this iteration
        wb_metrics: dict[str, float] = {
            "train/loss": loss_val,
            "train/learning_rate": current_lr,
            "train/grad_norm": grad_norm_val,
            "train/effective_batch_size": eff_bs,
            "online/win_rate": wr,
            "online/mean_return": mean_return,
            "speed/iter_time_sec": iter_time,
            "speed/collect_time_sec": collect_time,
            "speed/train_step_time_sec": train_time,
        }

        # GPU memory
        if torch.cuda.is_available():
            wb_metrics["speed/gpu_memory_mb"] = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            )

        # Global gate value
        if hasattr(raw_model, "global_gate"):
            gate_val = torch.sigmoid(raw_model.global_gate).item()
            wb_metrics["model/ema_gate_value"] = gate_val

        # Model health (every 10 iters)
        if iteration % 10 == 0:
            p_norm = sum(
                p.data.norm(2).item() ** 2 for p in raw_model.parameters()
            ) ** 0.5
            wb_metrics["model/param_norm"] = p_norm
            drift = sum(
                (p.data - _init_state[n]).norm(2).item() ** 2
                for n, p in raw_model.named_parameters()
                if n in _init_state
            ) ** 0.5
            wb_metrics["model/param_drift_from_init"] = drift

        # -- Gradient alignment --
        if iteration % grad_align_every == 0:
            cos, rl_n, bc_n = compute_grad_alignment(
                model, ref_model, local_b, global_b, x0_b,
                adv_b, cfg, device,
            )
            history.grad_align_iters.append(iteration)
            history.grad_align.append(cos)
            history.rl_grad_norm.append(rl_n)
            history.bc_grad_norm.append(bc_n)
            wb_metrics["diag/grad_alignment_cos"] = cos
            wb_metrics["diag/grad_alignment_rl_norm"] = rl_n
            wb_metrics["diag/grad_alignment_bc_norm"] = bc_n

            if spec.gradient_surgery:
                frac, n_conf = compute_surgery_metrics(g_rl, g_proj)
                history.surgery_iters.append(iteration)
                history.surgery_fraction.append(frac)
                history.surgery_n_conflicting.append(n_conf)
                wb_metrics["diag/surgery_frac"] = frac

        # -- Per-layer gradient norms --
        if iteration % per_layer_every == 0:
            model.train()
            model.zero_grad()
            with torch.amp.autocast("cuda", enabled=_use_amp):
                diag_loss = loss_fn(
                    model, local_b, global_b, x0_b, adv_b, cfg, device,
                )
            diag_loss.backward()
            norms = compute_per_layer_grad_norms(model)
            model.zero_grad()
            history.per_layer_iters.append(iteration)
            history.per_layer_norms.append(norms)

        # -- Representation drift --
        if iteration % repr_drift_every == 0:
            kl_m, kl_l, kl_mid, kl_h = compute_repr_drift(
                model, ref_model, local_b, global_b, x0_b, cfg, device,
            )
            history.repr_drift_iters.append(iteration)
            history.repr_drift_kl.append(kl_m)
            history.repr_drift_kl_low_t.append(kl_l)
            history.repr_drift_kl_mid_t.append(kl_mid)
            history.repr_drift_kl_high_t.append(kl_h)
            wb_metrics["diag/repr_drift_kl"] = kl_m

        # -- CKA --
        if iteration % cka_every == 0:
            cka_val = compute_cka(
                model, ref_model, local_b, global_b, x0_b, cfg, device,
            )
            history.cka_iters.append(iteration)
            history.cka_similarity.append(cka_val)
            wb_metrics["diag/cka_similarity"] = cka_val

        # -- t-analysis --
        if iteration % t_analysis_every == 0:
            bins, lh_cos, n_low, n_high = compute_t_analysis(
                model, local_b, global_b, x0_b, adv_b, cfg, device,
                n_bins=n_t_bins,
            )
            history.t_analysis_iters.append(iteration)
            bin_edges = [i / n_t_bins for i in range(n_t_bins + 1)]
            bin_dict = {
                f"t_{bin_edges[j]:.1f}-{bin_edges[j+1]:.1f}": bins[j]
                for j in range(n_t_bins)
            }
            history.t_bin_norms.append(bin_dict)
            history.norm_low_t.append(n_low)
            history.norm_high_t.append(n_high)
            history.lowhigh_cos.append(lh_cos)
            wb_metrics["diag/t_grad_norm_low"] = n_low
            wb_metrics["diag/t_grad_norm_high"] = n_high
            wb_metrics["diag/t_grad_cos_lohi"] = lh_cos

        # -- Evaluation --
        if iteration % eval_every == 0:
            eval_model = ema.make_eval_model(raw_model)
            results = evaluator.evaluate(
                cfg.id_envs, eval_model, eval_episodes, cfg, device,
            )
            id_wr = np.mean([
                v["win_rate"] for v in results.values()
            ])
            history.eval_iters.append(iteration)
            history.eval_score.append(float(id_wr))
            history.per_env_win_rates.append({
                k: v["win_rate"] for k, v in results.items()
            })
            wb_metrics["eval/id_win_rate"] = float(id_wr)
            for env_name, env_stats in results.items():
                short = env_name.replace("MiniHack-", "").replace("-v0", "")
                wb_metrics[f"eval/per_env/{short}/win_rate"] = env_stats["win_rate"]
                if "avg_steps" in env_stats:
                    wb_metrics[f"eval/per_env/{short}/avg_steps"] = env_stats["avg_steps"]
                if "avg_reward" in env_stats:
                    wb_metrics[f"eval/per_env/{short}/avg_reward"] = env_stats["avg_reward"]
            logger.info(
                "  [%s] iter=%d  loss=%.4f  id_win_rate=%.3f",
                spec.name, iteration, loss_val, id_wr,
            )

        # -- Emit all metrics to wandb --
        # Add ablation name prefix and use global step offset
        wb_metrics["iteration"] = wandb_step_offset + iteration
        wb_metrics["train/ablation_local_iter"] = iteration
        _wandb_log(wb_metrics, step=wandb_step_offset + iteration)

    # -- Final evaluation --
    final_model = ema.make_eval_model(raw_model)
    final_results = evaluator.evaluate(
        cfg.id_envs, final_model, eval_episodes, cfg, device,
    )
    final_score = float(np.mean([
        v["win_rate"] for v in final_results.values()
    ]))

    logger.info("  [%s] FINAL id_win_rate: %.4f", spec.name, final_score)

    # Clean up LoRA
    if spec.use_lora:
        remove_lora_from_model(raw_model)

    return history, final_score, raw_model
