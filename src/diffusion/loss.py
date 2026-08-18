"""MDLM ELBO loss with SUBS parameterisation.

The craftax twin is src/diffusion/loss.py:compute_loss.

Computes continuous-time loss on masked positions only, with analytic
SUBS weighting clipped for numerical stability.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from src.diffusion.schedules import get_schedule_deriv_for

_MAX_WEIGHT: float = 1000.0  # matches loss_weight_clip default; craftax twin identical
_WEIGHT_DENOM_EPS: float = 1e-5  # floor for 1 - alpha_t; craftax _EPS identical


def mdlm_loss(
    logits: Tensor,
    x0: Tensor,
    zt: Tensor,
    t: Tensor,
    mask_token: int,
    pad_token: int,
    schedule_fn: Callable[[Tensor], Tensor],
    schedule_deriv_fn: Callable[[Tensor], Tensor] | None = None,
    weight_clip: float = _MAX_WEIGHT,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """Monte-Carlo estimate of the continuous-time MDLM NELBO.

    Per sample: ``w(t) * sum_masked(CE) / L`` with the analytic weight
    ``w(t) = -alpha'(t) / (1 - alpha_t)`` clipped at *weight_clip*, then
    the batch mean. This is the estimator stated by MDLM eq (10) and
    Shi et al. eq (4) under a constant per-token normalisation.
    ``reduction="none"`` returns the per-sample ``[B]`` vector instead
    of the batch mean (the ablation suite weighs samples by advantage,
    exactly as the craftax twin's ``compute_loss`` does internally).

    Replaces a flat average
    over all masked tokens in the batch — the MaskGIT loss of Shi et al.
    App. eq (28), which is not a likelihood bound — and the opt-in
    ``use_importance_weighting`` path, which divided by the realised
    masked count (a ``1/(1-alpha_t)`` distortion of the weight).

    Args:
        logits: Model output. Shape ``[B, L, vocab]``.
        x0: Clean action sequences. Shape ``[B, L]``, int64.
        zt: Noisy sequences. Shape ``[B, L]``, int64.
        t: Per-sample diffusion time in [0, 1]. Shape ``[B]``.
        mask_token: MASK token ID.
        pad_token: PAD token ID.
        schedule_fn: Noise schedule returning alpha(t).
        schedule_deriv_fn: Analytic d(alpha)/dt; resolved from
            *schedule_fn* via the registry when ``None``.
        weight_clip: Upper clamp for w(t) (default 1000).
        label_smoothing: Smoothing epsilon for cross-entropy.
        reduction: ``"mean"`` (default, scalar) or ``"none"`` (``[B]``).

    Returns:
        Scalar loss, or the per-sample ``[B]`` vector under
        ``reduction="none"``. Zero(s) when no masked positions exist, and
        differentiable in ``logits`` even then — see the mask comment below.
    """
    if logits.ndim != 3 or x0.shape != zt.shape or x0.shape != logits.shape[:2]:
        raise ValueError(
            "mdlm_loss expects logits [B, L, V] with x0/zt [B, L]; got "
            f"{tuple(logits.shape)}, {tuple(x0.shape)}, {tuple(zt.shape)}"
        )
    if reduction not in ("mean", "none"):
        raise ValueError(f"Unknown reduction: {reduction!r}")
    B, L, V = logits.shape

    # Mask: compute loss only on masked, non-PAD positions.
    #
    # An all-False mask is a legitimate draw, not an error: at a t where
    # alpha(t) is near 1 nothing gets masked. It is handled by the arithmetic
    # below rather than by an early return, because the value is not the only
    # thing that matters — the result has to stay differentiable in `logits`.
    # `ce * is_masked.float()` gives exactly zero while keeping `logits` in
    # the graph; a freshly allocated zero tensor gives the same number with
    # no graph, and any caller that back-propagates it raises "element 0 of
    # tensors does not require grad and does not have a grad_fn". That is
    # reachable whenever the caller's other loss terms cannot carry the graph
    # either, which is the case for every ablation that freezes the goal
    # head's input path. The craftax twin has always computed this zero
    # arithmetically.
    is_masked = (zt == mask_token) & (x0 != pad_token)  # [B, L]

    # Per-position cross-entropy
    # Clamp targets to valid vocab range — out-of-range positions (PAD,
    # MASK) will be zeroed out by is_masked anyway.
    safe_targets = x0.clamp(0, V - 1)  # [B, L]
    ce = F.cross_entropy(
        logits.reshape(-1, V),
        safe_targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    )  # [B*L]
    ce = ce.reshape(B, L)  # [B, L]

    # Zero out non-masked positions
    ce = ce * is_masked.float()  # [B, L]

    # NELBO weight w(t) = -alpha'(t) / (1 - alpha_t), analytic derivative
    if schedule_deriv_fn is None:
        schedule_deriv_fn = get_schedule_deriv_for(schedule_fn)
    alpha_t = schedule_fn(t)  # [B]
    w_t = (-schedule_deriv_fn(t)) / torch.clamp(
        1.0 - alpha_t,
        min=_WEIGHT_DENOM_EPS,
    )  # [B]
    w_t = torch.clamp(w_t, max=weight_clip)  # [B]

    # Constant per-token normalisation (1/L), NOT the realised masked count
    per_sample = w_t * ce.sum(dim=1) / L  # [B]
    if reduction == "none":
        return per_sample
    # `sum / max(B, 1)`, not `mean`: identical for every non-empty batch, and
    # zero rather than NaN for an empty one, which the removed early return
    # also happened to cover.
    return per_sample.sum() / max(B, 1)


def auxiliary_goal_loss(
    goal_pred: Tensor,
    global_obs: Tensor,
    pad_value: float = -1.0,
) -> Tensor:
    """MSE loss for auxiliary staircase-coordinate prediction.

    Args:
        goal_pred: Predicted normalised staircase coords. Shape ``[B, 2]``.
        global_obs: Full map glyphs. Shape ``[B, 21, 79]``, int.
        pad_value: Coordinate value used when staircase is not visible.

    Returns:
        Scalar MSE loss over samples where the staircase is visible.
        Exactly zero when no staircase is visible in the batch, whatever
        *goal_pred* holds, and differentiable in *goal_pred* whenever
        *goal_pred* itself is.
    """
    targets = find_staircase_from_glyphs(global_obs)  # [B, 2]
    targets = targets.to(goal_pred.device, dtype=goal_pred.dtype)

    # Only supervise where staircase is visible
    valid = targets[:, 0] != pad_value  # [B]
    if not valid.any():
        # An empty selection: exactly 0.0, in the graph, zero gradient.
        #
        # Three constraints meet here. The caller adds this term to the ELBO
        # term and back-propagates the sum, so a detached constant silently
        # drops this term from the graph -- that was `goal_pred.new_tensor(0.0)`,
        # removed in `0cfc632` because it left every arm with a frozen goal
        # head unable to back-propagate at all. Its replacement multiplied by
        # the empty `valid` mask, which keeps the graph but returns NaN for a
        # non-finite `goal_pred`, since `nan * False` is `nan` -- while the
        # supervised branch below excludes exactly those rows. Indexing the
        # same way that branch does satisfies all three: `goal_pred[valid]` is
        # empty, so the sum is exactly zero whatever `goal_pred` holds, and it
        # is still a function of `goal_pred`, so the graph survives.
        return goal_pred[valid].sum()

    diff = (goal_pred[valid] - targets[valid]) ** 2  # [N, 2]
    return diff.mean()


def find_staircase_from_glyphs(global_obs: Tensor) -> Tensor:
    """Locate the staircase '>' in the global glyph map.

    Searches for NLE staircase-down glyph (character code 62 = '>').
    Returns normalised (row/H, col/W) coordinates per batch element,
    or (-1, -1) when the staircase is not visible.

    Args:
        global_obs: Glyph map. Shape ``[B, H, W]`` or ``[H, W]``, int.

    Returns:
        Normalised coordinates. Shape ``[B, 2]`` (float32).
    """
    if global_obs.ndim == 2:
        global_obs = global_obs.unsqueeze(0)

    B, H, W = global_obs.shape
    # NLE staircase-down glyphs: ord('>') = 62, plus NLE tile variants
    # 2310 (S_dnstair), 2368 (S_dnstairs), 2383 (S_vodoor).
    is_stair = (
        (global_obs == 62)
        | (global_obs == 2310)
        | (global_obs == 2368)
        | (global_obs == 2383)
    )

    # Vectorised over the batch. The previous form looped over B
    # calling `is_stair[b].nonzero()`, and `nonzero` needs its output size
    # on the host, so every sample forced a device sync: 2048 syncs per
    # gradient step at `dagger_batch_size: 2048`, which dominated the step.
    #
    # `nonzero` returns indices in row-major order, so `positions[0]` is the
    # lowest flat index that is set. Taking the minimum flat index over the
    # masked positions reproduces that exactly, with no host round-trip.
    flat = is_stair.reshape(B, H * W)
    idx = torch.arange(H * W, device=global_obs.device, dtype=torch.int32)
    masked_idx = torch.where(flat, idx, torch.full_like(idx, H * W))
    first = masked_idx.min(dim=1).values  # [B]; == H*W when no staircase
    found = first < H * W

    row = (first // W).float() / max(1, H - 1)
    col = (first % W).float() / max(1, W - 1)
    coords = torch.stack(
        (
            torch.where(found, row, -1.0),
            torch.where(found, col, -1.0),
        ),
        dim=1,
    )
    return coords.to(torch.float32)
