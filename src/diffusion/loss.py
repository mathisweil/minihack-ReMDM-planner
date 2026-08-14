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
) -> Tensor:
    """Monte-Carlo estimate of the continuous-time MDLM NELBO.

    Per sample: ``w(t) * sum_masked(CE) / L`` with the analytic weight
    ``w(t) = -alpha'(t) / (1 - alpha_t)`` clipped at *weight_clip*, then
    the batch mean. This is the estimator stated by MDLM eq (10) and
    Shi et al. eq (4) under a constant per-token normalisation.

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

    Returns:
        Scalar loss. Returns ``0.0`` when no masked positions exist.
    """
    if logits.ndim != 3 or x0.shape != zt.shape or x0.shape != logits.shape[:2]:
        raise ValueError(
            "mdlm_loss expects logits [B, L, V] with x0/zt [B, L]; got "
            f"{tuple(logits.shape)}, {tuple(x0.shape)}, {tuple(zt.shape)}"
        )
    B, L, V = logits.shape

    # Mask: compute loss only on masked, non-PAD positions
    is_masked = (zt == mask_token) & (x0 != pad_token)  # [B, L]

    if not is_masked.any():
        return logits.new_tensor(0.0)

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
    per_sample = ce.sum(dim=1) / L  # [B]
    return (w_t * per_sample).mean()


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
        Returns ``0.0`` when no staircase is visible in the batch.
    """
    targets = find_staircase_from_glyphs(global_obs)  # [B, 2]
    targets = targets.to(goal_pred.device, dtype=goal_pred.dtype)

    # Only supervise where staircase is visible
    valid = targets[:, 0] != pad_value  # [B]
    if not valid.any():
        return goal_pred.new_tensor(0.0)

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
