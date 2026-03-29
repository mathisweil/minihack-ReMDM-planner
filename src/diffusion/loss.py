"""MDLM ELBO loss with SUBS parameterisation.

Ported from the Craftax JAX implementation (src/diffusion/loss.py).
Computes continuous-time loss on masked positions only, with analytic
SUBS weighting clipped for numerical stability.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from src.diffusion.schedules import alpha_prime


_MAX_WEIGHT: float = 1000.0


def mdlm_loss(
    logits: Tensor,
    x0: Tensor,
    zt: Tensor,
    t: Tensor,
    mask_token: int,
    pad_token: int,
    schedule_fn: Callable[[Tensor], Tensor],
    weight_clip: float = _MAX_WEIGHT,
    label_smoothing: float = 0.0,
    use_importance_weighting: bool = False,
) -> Tensor:
    """Compute masked diffusion loss.

    By default uses a simple masked cross-entropy average (matching the
    reference implementation).  When ``use_importance_weighting=True``,
    applies SUBS weighting ``w(t) = -alpha'(t) / (1 - alpha_t)``.

    Args:
        logits: Model output. Shape ``[B, L, vocab]``.
        x0: Clean action sequences. Shape ``[B, L]``, int64.
        zt: Noisy sequences. Shape ``[B, L]``, int64.
        t: Per-sample diffusion time in [0, 1]. Shape ``[B]``.
        mask_token: MASK token ID.
        pad_token: PAD token ID.
        schedule_fn: Noise schedule returning alpha(t).
        weight_clip: Upper clamp for SUBS weight (default 1000).
        label_smoothing: Smoothing epsilon for cross-entropy.
        use_importance_weighting: If ``True``, apply SUBS w(t) per sample.

    Returns:
        Scalar loss. Returns ``0.0`` when no masked positions exist.
    """
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

    # Global average over all masked positions (matches reference)
    n_masked_total = is_masked.float().sum().clamp(min=1.0)
    loss = ce.sum() / n_masked_total

    if use_importance_weighting:
        # SUBS weight: w_t = -alpha'(t) / (1 - alpha_t + eps)
        alpha_t = schedule_fn(t)  # [B]
        d_alpha = alpha_prime(t, schedule_fn)  # [B]
        w_t = (-d_alpha) / (1.0 - alpha_t + 1e-8)  # [B]
        w_t = w_t.clamp(0.0, weight_clip)  # [B]

        # Per-sample weighted loss (needed for SUBS)
        n_masked_per = is_masked.float().sum(dim=1).clamp(min=1.0)  # [B]
        per_sample = ce.sum(dim=1) / n_masked_per  # [B]
        loss = (per_sample * w_t).mean()

    return loss


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
    valid = (targets[:, 0] != pad_value)  # [B]
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

    coords = torch.full(
        (B, 2), -1.0, dtype=torch.float32, device=global_obs.device
    )
    for b in range(B):
        positions = is_stair[b].nonzero(as_tuple=False)  # [N, 2]
        if positions.shape[0] > 0:
            row = positions[0, 0].float() / max(1, H - 1)
            col = positions[0, 1].float() / max(1, W - 1)
            coords[b, 0] = row
            coords[b, 1] = col

    return coords
