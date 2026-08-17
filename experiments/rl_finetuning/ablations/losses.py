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
- Per-sample NELBO from src/diffusion/loss.mdlm_loss(reduction="none")
- Auxiliary goal loss always preserved unless explicitly ablated
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.diffusion.forward import q_sample
from src.diffusion.loss import auxiliary_goal_loss, mdlm_loss

# Loss signature: (model, local_obs, global_obs, x0, advantages, cfg, device)
#   -> scalar loss
LossFn = Callable[
    [nn.Module, Tensor, Tensor, Tensor, Tensor | None, SimpleNamespace, torch.device],
    Tensor,
]

_EPS: float = 1e-5


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


def _forward_and_loss(
    model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
    t_min: float = _EPS,
    t_max: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        Tuple of (per_sample_loss ``[B]``, aux_loss scalar,
        logits ``[B,H,V]``, zt ``[B,H]``, t_discrete ``[B]``).
    """
    B = x0.shape[0]
    schedule_fn = cfg._schedule_fn

    # Sample continuous t in [t_min, t_max]
    t = torch.rand(B, device=device) * (t_max - t_min) + t_min  # [B]

    # Forward masking
    zt = q_sample(x0, t, cfg.mask_token, cfg.pad_token, schedule_fn)  # [B, H]

    # Convert to discrete timestep for model
    t_discrete = (
        (t * cfg.num_diffusion_steps).long().clamp(0, cfg.num_diffusion_steps - 1)
    )  # [B]

    # Model forward
    out = model(local_obs, global_obs, zt, t_discrete)
    logits = out["actions"]  # [B, H, A]
    goal_pred = out["goal_pred"]  # [B, 2]

    # Per-sample NELBO: w(t) * sum_masked(CE) / L, delegated to
    # src.diffusion.loss.mdlm_loss exactly as the craftax suite
    # delegates to compute_loss (spec-method §3.1/§3.4; was step-8
    # finding S8-6: the suite dropped the NELBO weight and normalised
    # by the realised masked count).
    per_sample = mdlm_loss(
        logits,
        x0,
        zt,
        t,
        cfg.mask_token,
        cfg.pad_token,
        schedule_fn,
        weight_clip=getattr(cfg, "loss_weight_clip", 1000.0),
        label_smoothing=getattr(cfg, "label_smoothing", 0.0),
        reduction="none",
    )  # [B]

    # Auxiliary goal loss
    aux_loss = auxiliary_goal_loss(goal_pred, global_obs)

    return per_sample, aux_loss, logits, zt, t_discrete


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
    per_sample, aux_loss, _, _, _ = _forward_and_loss(
        model, local_obs, global_obs, x0, cfg, device, t_min, t_max
    )

    if advantages is not None:
        loss = (per_sample * advantages).mean()
    else:
        loss = per_sample.mean()

    return loss + cfg.aux_loss_weight * aux_loss


def _kl_from_logits(
    cur_logits: Tensor,
    ref_model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    zt: Tensor,
    t_discrete: Tensor,
    cfg: SimpleNamespace,
) -> Tensor:
    """KL(current || pretrained) on masked positions, reusing logits.

    Avoids a second gradient-tracked forward pass through the current
    model by accepting pre-computed logits from ``_forward_and_loss``.
    Only the frozen ref_model is run (under ``no_grad``).

    Args:
        cur_logits: ``[B, H, A]`` logits from the current model.
        ref_model: Frozen pretrained model.
        local_obs: ``[B, 9, 9]``.
        global_obs: ``[B, 21, 79]``.
        zt: ``[B, H]`` noisy sequence (same as used for cur_logits).
        t_discrete: ``[B]`` discrete timestep (same as above).
        cfg: Config namespace.

    Returns:
        Scalar mean KL on masked positions.
    """
    with torch.no_grad():
        ref_out = ref_model(local_obs, global_obs, zt, t_discrete)
    ref_logits = ref_out["actions"]  # [B, H, A]

    is_masked = (zt == cfg.mask_token).float()  # [B, H]

    cur_log = F.log_softmax(cur_logits, dim=-1)  # [B, H, A]
    ref_log = F.log_softmax(ref_logits, dim=-1)  # [B, H, A]
    cur_prob = cur_log.exp()

    kl = (cur_prob * (cur_log - ref_log)).sum(dim=-1)  # [B, H]
    kl_masked = (kl * is_masked).sum(dim=1) / is_masked.sum(dim=1).clamp(min=1.0)
    return kl_masked.mean()


def _entropy_from_logits(
    logits: Tensor,
    zt: Tensor,
    cfg: SimpleNamespace,
) -> Tensor:
    """Mean entropy from pre-computed logits on masked positions.

    Avoids a second gradient-tracked forward pass by reusing logits
    from ``_forward_and_loss``.

    Args:
        logits: ``[B, H, A]`` model logits.
        zt: ``[B, H]`` noisy sequence.
        cfg: Config namespace.

    Returns:
        Scalar mean entropy on masked positions.
    """
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

    Shares the model forward pass between the ELBO and KL terms to
    halve peak activation memory (1 gradient-tracked pass instead of 2).

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
        per_sample, aux_loss, logits, zt, t_discrete = _forward_and_loss(
            model,
            local_obs,
            global_obs,
            x0,
            cfg,
            device,
        )
        if advantages is not None:
            loss = (per_sample * advantages).mean()
        else:
            loss = per_sample.mean()
        loss = loss + cfg.aux_loss_weight * aux_loss

        kl = _kl_from_logits(
            logits,
            ref_model,
            local_obs,
            global_obs,
            zt,
            t_discrete,
            cfg,
        )
        return loss + kl_coef * kl

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

    Shares the model forward pass between the ELBO and KL terms to
    halve peak activation memory.

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
        per_sample, aux_loss, logits, zt, t_discrete = _forward_and_loss(
            model,
            local_obs,
            global_obs,
            x0,
            cfg,
            device,
        )
        if advantages is not None:
            loss = (per_sample * advantages).mean()
        else:
            loss = per_sample.mean()
        loss = loss + cfg.aux_loss_weight * aux_loss

        kl = _kl_from_logits(
            logits,
            ref_model,
            local_obs,
            global_obs,
            zt,
            t_discrete,
            cfg,
        )
        violation = torch.clamp(kl - threshold, min=0.0)
        return loss + 1e4 * violation**2

    return loss_fn


def make_loss_mixed_replay(ctx: LossContext) -> LossFn:
    """Baseline loss; mixed replay batching handled at training loop level.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline.
    """
    return make_loss_baseline(ctx)


def make_loss_bc_wins(ctx: LossContext) -> LossFn:
    """Uniform ELBO over the winning windows only (BC on wins).

    The training loop passes the binary win mask (return > win_threshold,
    from ``compute_advantages(wins_only=True)``) as ``advantages``.
    Rescaling the mask by ``B / n_wins`` turns ``_core_loss``'s batch
    mean into a uniform mean over the winning windows; a batch with no
    winning window contributes zero action loss (the auxiliary goal
    loss is kept, as in every other ablation). ``advantages=None`` is
    treated as all-wins (plain uniform BC).

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` averaging uniformly over winning windows.
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
        b = x0.shape[0]
        if advantages is None:
            win_mask = torch.ones(b, device=x0.device)
        else:
            win_mask = (advantages > 0).float()
        n_wins = win_mask.sum()
        scale = (b / n_wins.clamp(min=1.0)) * (n_wins > 0).float()
        return _core_loss(
            model, local_obs, global_obs, x0, win_mask * scale, cfg, device
        )

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
            model,
            local_obs,
            global_obs,
            x0,
            advantages,
            cfg,
            device,
            t_min=_EPS,
            t_max=t_max,
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
            model,
            local_obs,
            global_obs,
            x0,
            advantages,
            cfg,
            device,
            t_min=t_min,
            t_max=t_max,
        )

    return loss_fn


def make_loss_entropy_bonus(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO minus entropy bonus.

    Shares the model forward pass between the ELBO and entropy terms
    to halve peak activation memory.

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
        per_sample, aux_loss, logits, zt, _ = _forward_and_loss(
            model,
            local_obs,
            global_obs,
            x0,
            cfg,
            device,
        )
        if advantages is not None:
            loss = (per_sample * advantages).mean()
        else:
            loss = per_sample.mean()
        loss = loss + cfg.aux_loss_weight * aux_loss

        entropy = _entropy_from_logits(logits, zt, cfg)
        return loss - entropy_coef * entropy

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
    """Return-weighted ELBO with std-normalised advantages.

    Normalises advantages as (A - mean(A)) / (std(A) + eps) over the batch,
    unlike the simpler mean-normalisation used in the baseline.

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


def make_loss_reward_quality(ctx: LossContext) -> LossFn:
    """Baseline loss; reward filtering and normalisation handled externally.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline.
    """
    return make_loss_baseline(ctx)


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
        name: torch.zeros_like(param) for name, param in model.named_parameters()
    }

    _use_amp = getattr(cfg, "use_amp", False) and device.type == "cuda"
    model.train()
    for local_obs, global_obs, x0 in batches:
        local_obs = local_obs.to(device)
        global_obs = global_obs.to(device)
        x0 = x0.to(device)

        model.zero_grad()
        with torch.amp.autocast("cuda", enabled=_use_amp):
            loss = _core_loss(model, local_obs, global_obs, x0, None, cfg, device)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulator[name] = accumulator[name] + param.grad.detach() ** 2

    n = max(len(batches), 1)
    return {name: acc / n for name, acc in accumulator.items()}
