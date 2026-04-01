"""Loss factory functions for all RL fine-tuning ablations (PyTorch).

Each factory returns a ``LossFn``::

    loss_fn(model, local_obs, global_obs, x0, advantages, cfg, device) -> scalar

All factories accept a ``LossContext`` that bundles shared context
(reference model, schedule functions, config) so the factories
themselves are pure and free of global state.

Adapted from Craftax JAX implementation to MiniHack PyTorch.
Key differences:
- MiniHack model: (local_obs, global_obs, zt, t_discrete) -> {"actions", "goal_pred"}
- t is sampled here, q_sample called here, model called here
- Per-sample loss computed manually (src/diffusion/loss.mdlm_loss returns global avg)
- Auxiliary goal loss always preserved unless explicitly ablated
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.diffusion.forward import q_sample
from src.diffusion.loss import auxiliary_goal_loss
from src.diffusion.schedules import alpha_prime

# Loss signature: (model, local_obs, global_obs, x0, advantages, cfg, device)
#   -> scalar loss
LossFn = Callable[
    [nn.Module, Tensor, Tensor, Tensor, Tensor | None, SimpleNamespace, torch.device],
    Tensor,
]

_EPS: float = 1e-5
_MAX_WEIGHT: float = 1000.0


@dataclass
class LossContext:
    """Shared context for all loss factory functions.

    Args:
        ref_model: Frozen pretrained model for regularisation losses.
        schedule_fn: alpha(t) noise schedule.
        cfg: Config namespace (lowercase keys).
    """

    ref_model: nn.Module | None
    schedule_fn: Callable[[Tensor], Tensor]
    cfg: SimpleNamespace


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _per_sample_masked_ce(
    logits: Tensor,
    x0: Tensor,
    zt: Tensor,
    mask_token: int,
    pad_token: int,
) -> Tensor:
    """Compute per-sample cross-entropy on masked, non-PAD positions.

    Args:
        logits: ``[B, H, V]`` model action logits.
        x0: ``[B, H]`` clean action sequences.
        zt: ``[B, H]`` noisy sequences.
        mask_token: MASK token ID.
        pad_token: PAD token ID.

    Returns:
        ``[B]`` per-sample mean cross-entropy on masked positions.
    """
    B, H, V = logits.shape
    is_masked = (zt == mask_token) & (x0 != pad_token)  # [B, H]

    safe_targets = x0.clamp(0, V - 1)  # [B, H]
    ce = F.cross_entropy(
        logits.reshape(-1, V),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape(B, H)  # [B, H]

    ce = ce * is_masked.float()  # [B, H]
    n_masked_per = is_masked.float().sum(dim=1).clamp(min=1.0)  # [B]
    return ce.sum(dim=1) / n_masked_per  # [B]


def _forward_and_loss(
    model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
    t_min: float = _EPS,
    t_max: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample t, mask, forward pass, compute per-sample loss + goal loss.

    Args:
        model: Current model.
        local_obs: ``[B, 9, 9]`` local crop.
        global_obs: ``[B, 21, 79]`` global map.
        x0: ``[B, H]`` clean action sequences.
        cfg: Config namespace.
        device: Torch device.
        t_min: Lower bound for uniform t sampling.
        t_max: Upper bound for uniform t sampling.

    Returns:
        Tuple of (per_sample_loss ``[B]``, aux_loss scalar, logits ``[B,H,V]``).
    """
    B = x0.shape[0]
    schedule_fn = cfg._schedule_fn

    # Sample continuous t in [t_min, t_max]
    t = torch.rand(B, device=device) * (t_max - t_min) + t_min  # [B]

    # Forward masking
    zt = q_sample(x0, t, cfg.mask_token, cfg.pad_token, schedule_fn)  # [B, H]

    # Convert to discrete timestep for model
    t_discrete = (t * cfg.num_diffusion_steps).long().clamp(
        0, cfg.num_diffusion_steps - 1
    )  # [B]

    # Model forward
    out = model(local_obs, global_obs, zt, t_discrete)
    logits = out["actions"]  # [B, H, A]
    goal_pred = out["goal_pred"]  # [B, 2]

    # Per-sample masked CE
    per_sample = _per_sample_masked_ce(
        logits, x0, zt, cfg.mask_token, cfg.pad_token
    )  # [B]

    # Auxiliary goal loss
    aux_loss = auxiliary_goal_loss(goal_pred, global_obs)

    return per_sample, aux_loss, logits


def _core_loss(
    model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    advantages: Tensor | None,
    cfg: SimpleNamespace,
    device: torch.device,
    t_min: float = _EPS,
    t_max: float = 1.0,
) -> Tensor:
    """Core loss: advantage-weighted ELBO + auxiliary goal loss.

    Args:
        model: Current model.
        local_obs: ``[B, 9, 9]``.
        global_obs: ``[B, 21, 79]``.
        x0: ``[B, H]`` clean actions.
        advantages: ``[B]`` advantage weights or None for uniform.
        cfg: Config namespace.
        device: Torch device.
        t_min: Lower t bound.
        t_max: Upper t bound.

    Returns:
        Scalar total loss.
    """
    per_sample, aux_loss, _ = _forward_and_loss(
        model, local_obs, global_obs, x0, cfg, device, t_min, t_max
    )

    if advantages is not None:
        loss = (per_sample * advantages).mean()
    else:
        loss = per_sample.mean()

    return loss + cfg.aux_loss_weight * aux_loss


def _kl_penalty(
    current_model: nn.Module,
    ref_model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
) -> Tensor:
    """KL divergence KL(current || pretrained) on masked positions.

    Args:
        current_model: Current fine-tuned model.
        ref_model: Frozen pretrained model.
        local_obs: ``[B, 9, 9]``.
        global_obs: ``[B, 21, 79]``.
        x0: ``[B, H]`` clean actions.
        cfg: Config namespace.
        device: Torch device.

    Returns:
        Scalar mean KL on masked positions.
    """
    B = x0.shape[0]
    schedule_fn = cfg._schedule_fn

    t = torch.rand(B, device=device) * (1.0 - _EPS) + _EPS  # [B]
    zt = q_sample(x0, t, cfg.mask_token, cfg.pad_token, schedule_fn)

    t_discrete = (t * cfg.num_diffusion_steps).long().clamp(
        0, cfg.num_diffusion_steps - 1
    )

    cur_out = current_model(local_obs, global_obs, zt, t_discrete)
    with torch.no_grad():
        ref_out = ref_model(local_obs, global_obs, zt, t_discrete)

    cur_logits = cur_out["actions"]  # [B, H, A]
    ref_logits = ref_out["actions"]  # [B, H, A]

    is_masked = (zt == cfg.mask_token).float()  # [B, H]

    cur_log = F.log_softmax(cur_logits, dim=-1)  # [B, H, A]
    ref_log = F.log_softmax(ref_logits, dim=-1)  # [B, H, A]
    cur_prob = cur_log.exp()

    kl = (cur_prob * (cur_log - ref_log)).sum(dim=-1)  # [B, H]
    kl_masked = (kl * is_masked).sum(dim=1) / is_masked.sum(dim=1).clamp(min=1.0)
    return kl_masked.mean()


def _entropy_bonus(
    model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
) -> Tensor:
    """Mean entropy of p_theta over masked positions.

    Args:
        model: Current model.
        local_obs: ``[B, 9, 9]``.
        global_obs: ``[B, 21, 79]``.
        x0: ``[B, H]``.
        cfg: Config namespace.
        device: Torch device.

    Returns:
        Scalar mean entropy.
    """
    B = x0.shape[0]
    schedule_fn = cfg._schedule_fn

    t = torch.rand(B, device=device) * (1.0 - _EPS) + _EPS
    zt = q_sample(x0, t, cfg.mask_token, cfg.pad_token, schedule_fn)
    t_discrete = (t * cfg.num_diffusion_steps).long().clamp(
        0, cfg.num_diffusion_steps - 1
    )

    out = model(local_obs, global_obs, zt, t_discrete)
    logits = out["actions"]  # [B, H, A]

    is_masked = (zt == cfg.mask_token).float()  # [B, H]

    log_probs = F.log_softmax(logits, dim=-1)  # [B, H, A]
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, H]

    return (entropy * is_masked).sum() / is_masked.sum().clamp(min=1.0)


def _ewc_penalty(
    fisher: dict[str, Tensor],
    model: nn.Module,
    ref_model: nn.Module,
) -> Tensor:
    """EWC penalty: lambda * sum(F_i * (theta_i - theta_i*)^2).

    Args:
        fisher: Fisher diagonal dict mapping param name -> tensor.
        model: Current model.
        ref_model: Pretrained anchor model.

    Returns:
        Scalar unweighted EWC penalty.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    ref_dict = dict(ref_model.named_parameters())
    for name, param in model.named_parameters():
        if name in fisher and name in ref_dict:
            penalty = penalty + (fisher[name] * (param - ref_dict[name]) ** 2).sum()
    return penalty


# ---------------------------------------------------------------------------
# Group A: Regularisation / Constraint Methods
# ---------------------------------------------------------------------------


def make_loss_baseline(ctx: LossContext) -> LossFn:
    """Standard return-weighted ELBO -- no modifications.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` implementing the baseline RL fine-tuning objective.
    """
    def loss_fn(
        model: nn.Module,
        local_obs: Tensor,
        global_obs: Tensor,
        x0: Tensor,
        advantages: Tensor | None,
        cfg: SimpleNamespace,
        device: torch.device,
    ) -> Tensor:
        return _core_loss(model, local_obs, global_obs, x0, advantages, cfg, device)

    return loss_fn


def make_loss_kl_penalty(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO + soft KL penalty vs pretrained.

    Args:
        ctx: Shared loss context (ref_model required).

    Returns:
        ``LossFn`` with additive KL penalty.
    """
    kl_coef = getattr(ctx.cfg, "kl_coef", 0.1)
    ref_model = ctx.ref_model

    def loss_fn(
        model: nn.Module,
        local_obs: Tensor,
        global_obs: Tensor,
        x0: Tensor,
        advantages: Tensor | None,
        cfg: SimpleNamespace,
        device: torch.device,
    ) -> Tensor:
        rl = _core_loss(model, local_obs, global_obs, x0, advantages, cfg, device)
        kl = _kl_penalty(model, ref_model, local_obs, global_obs, x0, cfg, device)
        return rl + kl_coef * kl

    return loss_fn


def make_loss_ewc(ctx: LossContext, fisher: dict[str, Tensor]) -> LossFn:
    """Return-weighted ELBO + EWC penalty using pre-computed Fisher diagonal.

    Args:
        ctx: Shared loss context.
        fisher: Fisher diagonal dict (pre-computed).

    Returns:
        ``LossFn`` with EWC regularisation.
    """
    ewc_lambda = getattr(ctx.cfg, "ewc_lambda", 100.0)
    ref_model = ctx.ref_model

    def loss_fn(
        model: nn.Module,
        local_obs: Tensor,
        global_obs: Tensor,
        x0: Tensor,
        advantages: Tensor | None,
        cfg: SimpleNamespace,
        device: torch.device,
    ) -> Tensor:
        rl = _core_loss(model, local_obs, global_obs, x0, advantages, cfg, device)
        penalty = _ewc_penalty(fisher, model, ref_model)
        return rl + ewc_lambda * penalty

    return loss_fn


def make_loss_trust_region_kl(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO + hard KL trust region via quadratic barrier.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` with hard KL trust region.
    """
    threshold = getattr(ctx.cfg, "trust_region_kl", 0.05)
    ref_model = ctx.ref_model

    def loss_fn(
        model: nn.Module,
        local_obs: Tensor,
        global_obs: Tensor,
        x0: Tensor,
        advantages: Tensor | None,
        cfg: SimpleNamespace,
        device: torch.device,
    ) -> Tensor:
        rl = _core_loss(model, local_obs, global_obs, x0, advantages, cfg, device)
        kl = _kl_penalty(model, ref_model, local_obs, global_obs, x0, cfg, device)
        violation = torch.clamp(kl - threshold, min=0.0)
        return rl + 1e4 * violation ** 2

    return loss_fn


def make_loss_mixed_replay(ctx: LossContext) -> LossFn:
    """Baseline loss; mixed replay batching handled at training loop level.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline.
    """
    return make_loss_baseline(ctx)


# ---------------------------------------------------------------------------
# Group B: Training Signal Modifications
# ---------------------------------------------------------------------------


def make_loss_bc_wins(ctx: LossContext) -> LossFn:
    """Uniform ELBO ignoring advantages (BC on wins).

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` with advantages zeroed out.
    """
    def loss_fn(
        model: nn.Module,
        local_obs: Tensor,
        global_obs: Tensor,
        x0: Tensor,
        advantages: Tensor | None,
        cfg: SimpleNamespace,
        device: torch.device,
    ) -> Tensor:
        return _core_loss(model, local_obs, global_obs, x0, None, cfg, device)

    return loss_fn


def make_loss_low_t(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO restricted to low-t regime.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` using only low-t samples.
    """
    t_max = getattr(ctx.cfg, "t_max_low", 0.2)

    def loss_fn(
        model: nn.Module,
        local_obs: Tensor,
        global_obs: Tensor,
        x0: Tensor,
        advantages: Tensor | None,
        cfg: SimpleNamespace,
        device: torch.device,
    ) -> Tensor:
        return _core_loss(
            model, local_obs, global_obs, x0, advantages, cfg, device,
            t_min=_EPS, t_max=t_max,
        )

    return loss_fn


def make_loss_t_curriculum(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO with annealing t range.

    The t range anneals from [t_start, 1.0] to [eps, t_end] over
    t_curriculum_steps iterations.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` with iteration-dependent t range. Uses
        ``cfg._current_iter`` (set by training loop) for progress.
    """
    t_start = getattr(ctx.cfg, "t_curriculum_start", 0.8)
    t_end = getattr(ctx.cfg, "t_curriculum_end", 0.2)
    steps = getattr(ctx.cfg, "t_curriculum_steps", 200)

    def loss_fn(
        model: nn.Module,
        local_obs: Tensor,
        global_obs: Tensor,
        x0: Tensor,
        advantages: Tensor | None,
        cfg: SimpleNamespace,
        device: torch.device,
    ) -> Tensor:
        current_iter = getattr(cfg, "_current_iter", 0)
        frac = min(current_iter / max(steps, 1), 1.0)
        t_min = t_start - frac * (t_start - _EPS)
        t_max = 1.0 - frac * (1.0 - t_end)
        t_min = max(t_min, _EPS)
        t_max = max(t_max, t_min + 0.05)
        t_max = min(t_max, 1.0)
        return _core_loss(
            model, local_obs, global_obs, x0, advantages, cfg, device,
            t_min=t_min, t_max=t_max,
        )

    return loss_fn


def make_loss_entropy_bonus(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO minus entropy bonus.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` with entropy regularisation.
    """
    entropy_coef = getattr(ctx.cfg, "entropy_coef", 0.01)

    def loss_fn(
        model: nn.Module,
        local_obs: Tensor,
        global_obs: Tensor,
        x0: Tensor,
        advantages: Tensor | None,
        cfg: SimpleNamespace,
        device: torch.device,
    ) -> Tensor:
        rl = _core_loss(model, local_obs, global_obs, x0, advantages, cfg, device)
        entropy = _entropy_bonus(model, local_obs, global_obs, x0, cfg, device)
        return rl - entropy_coef * entropy

    return loss_fn


def make_loss_gradient_surgery(ctx: LossContext) -> LossFn:
    """Baseline loss; PCGrad projection handled at training loop level.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline.
    """
    return make_loss_baseline(ctx)


def make_loss_advantage_clip(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO with PPO-style advantage clipping.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` with clipped advantages.
    """
    eps = getattr(ctx.cfg, "adv_clip_eps", 0.2)

    def loss_fn(
        model: nn.Module,
        local_obs: Tensor,
        global_obs: Tensor,
        x0: Tensor,
        advantages: Tensor | None,
        cfg: SimpleNamespace,
        device: torch.device,
    ) -> Tensor:
        if advantages is not None:
            advantages = advantages.clamp(1.0 - eps, 1.0 + eps)
        return _core_loss(model, local_obs, global_obs, x0, advantages, cfg, device)

    return loss_fn


def make_loss_normalized_adv(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO with group-normalised advantages (GRPO-style).

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` with std-normalised advantages.
    """
    def loss_fn(
        model: nn.Module,
        local_obs: Tensor,
        global_obs: Tensor,
        x0: Tensor,
        advantages: Tensor | None,
        cfg: SimpleNamespace,
        device: torch.device,
    ) -> Tensor:
        if advantages is not None:
            mean = advantages.mean()
            std = advantages.std()
            advantages = (advantages - mean) / (std + 1e-8)
        return _core_loss(model, local_obs, global_obs, x0, advantages, cfg, device)

    return loss_fn


# ---------------------------------------------------------------------------
# Group C: Architecture / Parameter Isolation -- loss is always baseline
# ---------------------------------------------------------------------------


def make_loss_frozen_backbone(ctx: LossContext) -> LossFn:
    """Baseline loss; backbone freezing handled at optimizer/mask level.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline.
    """
    return make_loss_baseline(ctx)


def make_loss_param_isolation(ctx: LossContext) -> LossFn:
    """Baseline loss; parameter isolation handled at optimizer/mask level.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline.
    """
    return make_loss_baseline(ctx)


# ---------------------------------------------------------------------------
# Group D: Reward / Data Quality -- loss is baseline; data transforms external
# ---------------------------------------------------------------------------


def make_loss_reward_quality(ctx: LossContext) -> LossFn:
    """Baseline loss; reward filtering and normalisation handled externally.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline.
    """
    return make_loss_baseline(ctx)


# ---------------------------------------------------------------------------
# Fisher diagonal estimation (for EWC)
# ---------------------------------------------------------------------------


def estimate_fisher_diagonal(
    model: nn.Module,
    schedule_fn: Callable[[Tensor], Tensor],
    cfg: SimpleNamespace,
    batches: list[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> dict[str, Tensor]:
    """Estimate Fisher information diagonal on held-out batches.

    F_i = E[(d log p / d theta_i)^2], averaged over batches.

    Args:
        model: Pretrained model (evaluation point).
        schedule_fn: alpha(t) noise schedule.
        cfg: Config namespace.
        batches: List of (local_obs, global_obs, x0) tuples.
        device: Torch device.

    Returns:
        Dict mapping parameter name -> Fisher diagonal tensor.
    """
    accumulator: dict[str, Tensor] = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
    }

    model.train()
    for local_obs, global_obs, x0 in batches:
        local_obs = local_obs.to(device)
        global_obs = global_obs.to(device)
        x0 = x0.to(device)

        model.zero_grad()
        loss = _core_loss(
            model, local_obs, global_obs, x0, None, cfg, device
        )
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulator[name] = accumulator[name] + param.grad.detach() ** 2

    n = max(len(batches), 1)
    return {name: acc / n for name, acc in accumulator.items()}
