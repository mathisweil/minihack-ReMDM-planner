"""ReMDM reverse denoising with remasking strategies.

Ported from the Craftax JAX implementation (src/diffusion/sampling.py).
Implements MaskGIT-style progressive unmasking with optional stochastic
remasking (ReMDM) using three strategy variants.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from src.diffusion.schedules import get_schedule

# NLE hazard glyph IDs and char codes (walls, locked doors, lava, water)
_HAZARD_GLYPHS: frozenset[int] = frozenset({2359, 2360, 2389, 2390})
_HAZARD_CHARS: frozenset[int] = frozenset(
    {ord("|"), ord("-"), ord("+"), ord("L"), ord("W")}
)
# Cardinal action → (dy, dx) offsets
_CARDINAL_OFFSETS: dict[int, tuple[int, int]] = {
    0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1),
}
_N_PHYSICS_CHECK = 8  # only inspect the first N plan positions


def _check_hazard(local_crop: np.ndarray, action: int) -> bool:
    """Return True if *action* from the agent's centre steps into a hazard.

    Args:
        local_crop: ``[crop_size, crop_size]`` glyph array.
        action: Cardinal action index (0=N, 1=E, 2=S, 3=W).

    Returns:
        ``True`` when the target cell contains a hazard glyph.
    """
    if action not in _CARDINAL_OFFSETS:
        return False
    cs = local_crop.shape[0]
    cy, cx = cs // 2, cs // 2
    dy, dx = _CARDINAL_OFFSETS[action]
    ny, nx = cy + dy, cx + dx
    if not (0 <= ny < cs and 0 <= nx < cs):
        return True
    glyph = int(local_crop[ny, nx])
    return glyph in _HAZARD_GLYPHS or glyph in _HAZARD_CHARS


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Zero out all but the top-k logits per position.

    Args:
        logits: Raw logits. Shape ``[..., V]``.
        k: Number of top entries to keep.

    Returns:
        Filtered logits with non-top-k set to ``-inf``.
    """
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    topk_vals, _ = logits.topk(k, dim=-1)  # [..., k]
    threshold = topk_vals[..., -1:]  # [..., 1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _compute_remask_prob(
    strategy: str,
    eta: float,
    sigma_max: float,
    confidence: Tensor | None,
) -> Tensor | float:
    """Compute per-token remasking probability.

    Args:
        strategy: One of ``"rescale"``, ``"cap"``, ``"conf"``.
        eta: Base remasking strength hyperparameter.
        sigma_max: ``1 - alpha_t(ratio)`` at current step.
        confidence: Per-token confidence scores. Shape ``[B, L]``.
            Required only for the ``"conf"`` strategy.

    Returns:
        Scalar or ``[B, L]`` tensor of remasking probabilities.
    """
    if strategy == "rescale":
        return eta * sigma_max
    if strategy == "cap":
        return min(eta, sigma_max)
    if strategy == "conf":
        assert confidence is not None, "conf strategy requires confidence"
        return eta * sigma_max * (1.0 - confidence)
    raise ValueError(f"Unknown remask strategy: {strategy}")


@torch.no_grad()
def remdm_sample(
    model: torch.nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    cfg: SimpleNamespace,
    device: torch.device | str,
    physics_aware: bool = True,
    blind_global: bool = False,
    return_analytics: bool = False,
    num_steps: int | None = None,
) -> Tensor | tuple[Tensor, list, list[float], list[int]]:
    """Generate action sequences via iterative ReMDM denoising.

    Args:
        model: Denoising model with forward signature
            ``(local_obs, global_obs, action_seq, t_discrete) -> dict``.
        local_obs: Local crop observations. Shape ``[B, 9, 9]``.
        global_obs: Global map observations. Shape ``[B, 21, 79]``.
        cfg: Config namespace with ``seq_len``, ``mask_token``,
            ``action_dim``, ``diffusion_steps_eval``, ``temperature``,
            ``top_k``, ``eta``, ``remask_strategy``, ``noise_schedule``.
        device: Torch device.
        physics_aware: If ``True``, soft-penalise hazardous cardinal actions
            by overriding their confidence to ``0.001`` before commitment
            ranking. Only checks the first ``_N_PHYSICS_CHECK`` positions.
        blind_global: If ``True``, zero out the global map observation
            (local-only ablation).
        return_analytics: If ``True``, also return per-step analytics as
            ``(seq, path_per_step, tracking_confidence, tracking_masked)``.
        num_steps: Override number of denoising steps (default uses
            ``cfg.diffusion_steps_eval``).

    Returns:
        When ``return_analytics=False`` (default): fully committed action
        sequence of shape ``[B, seq_len]``, int64, with no MASK tokens.

        When ``return_analytics=True``: tuple
        ``(seq, path_per_step, tracking_confidence, tracking_masked_count)``
        where ``path_per_step`` is a list of ``[seq_len]`` numpy arrays,
        ``tracking_confidence`` a list of per-step avg unmasked confidence
        floats, and ``tracking_masked_count`` a list of masked-token counts.
    """
    B = local_obs.shape[0]
    seq_len = cfg.seq_len
    mask_token = cfg.mask_token
    action_dim = cfg.action_dim
    K = num_steps if num_steps is not None else cfg.diffusion_steps_eval
    schedule_fn = get_schedule(cfg.noise_schedule)
    min_keep = max(1, int(seq_len * 0.10))  # Safety Net: always unmask ≥10%

    local_obs = local_obs.to(device)
    global_obs = global_obs.to(device)

    if blind_global:
        global_obs = torch.zeros_like(global_obs)

    # Pre-compute numpy local crops for physics checks (CPU, batch loop)
    local_np: np.ndarray | None = None  # [B, crop, crop]
    if physics_aware:
        local_np = local_obs.cpu().numpy()

    # Analytics buffers (only populated when return_analytics=True)
    path_per_step: list[np.ndarray] = []
    tracking_confidence: list[float] = []
    tracking_masked_count: list[int] = []

    # Start fully masked
    seq = torch.full(
        (B, seq_len), mask_token, dtype=torch.long, device=device
    )

    for k in range(1, K + 1):
        ratio = k / K
        # Pass as tensor (not Python int) to avoid torch.compile recompilation
        t_discrete = torch.full(
            (B,), int(cfg.num_diffusion_steps * (1.0 - ratio)),
            dtype=torch.long, device=device,
        )

        # Forward pass
        out = model(local_obs, global_obs, seq, t_discrete)
        logits = out["actions"]  # [B, seq_len, vocab]

        # Mask invalid action tokens (indices >= action_dim)
        logits[:, :, action_dim:] = float("-inf")

        # Temperature scaling
        logits = logits / cfg.temperature

        # Top-K filtering
        logits = top_k_filter(logits, cfg.top_k)

        # Sample predictions
        probs = F.softmax(logits, dim=-1)  # [B, seq_len, action_dim]
        preds = Categorical(probs=probs).sample()  # [B, seq_len]

        # Confidence: probability of the sampled token
        conf = probs.gather(
            -1, preds.unsqueeze(-1)
        ).squeeze(-1)  # [B, seq_len]

        # Physics softener: demote hazardous cardinal actions to conf=0.001
        if physics_aware and local_np is not None:
            preds_np = preds.cpu().numpy()  # [B, seq_len]
            conf_override = conf.clone()
            for b in range(B):
                crop_b = np.asarray(local_np[b])  # [crop, crop]
                for pos in range(min(_N_PHYSICS_CHECK, seq_len)):
                    action = int(preds_np[b, pos])
                    if _check_hazard(crop_b, action):
                        conf_override[b, pos] = 0.001
            conf = conf_override

        is_masked = seq == mask_token  # [B, seq_len]

        if k < K:
            # MaskGIT progressive unmasking with min-keep guarantee
            n_unmask = max(min_keep, max(1, int(seq_len * ratio)))

            # Set confidence of non-masked positions to -1 so they
            # are not selected for unmasking
            unmask_scores = conf.clone()
            unmask_scores[~is_masked] = -1.0

            # For each batch element, unmask top-confidence masked positions
            _, topk_indices = unmask_scores.topk(
                n_unmask, dim=-1
            )  # [B, n_unmask]

            # Build scatter mask for positions to unmask
            unmask_mask = torch.zeros_like(seq, dtype=torch.bool)
            unmask_mask.scatter_(1, topk_indices, True)
            unmask_mask = unmask_mask & is_masked  # only unmask masked pos

            seq = torch.where(unmask_mask, preds, seq)

            # ReMDM stochastic remasking of committed (non-masked) positions
            is_committed = seq != mask_token  # [B, seq_len]
            alpha_t_ratio = schedule_fn(
                torch.tensor(ratio, device=device)
            )
            sigma_max = (1.0 - alpha_t_ratio).item()

            remask_prob = _compute_remask_prob(
                cfg.remask_strategy, cfg.eta, sigma_max, conf
            )
            if isinstance(remask_prob, Tensor):
                do_remask = (
                    torch.rand_like(conf) < remask_prob
                ) & is_committed
            else:
                do_remask = (
                    torch.rand(B, seq_len, device=device) < remask_prob
                ) & is_committed
            seq = torch.where(do_remask, mask_token, seq)
        else:
            # Final step: commit all remaining MASK tokens
            seq = torch.where(is_masked, preds, seq)

        # Analytics tracking
        if return_analytics:
            path_per_step.append(seq[0].cpu().numpy().copy())
            still_masked = (seq[0] == mask_token)
            unmasked_conf = conf[0][~still_masked]
            avg_conf = (
                unmasked_conf.mean().item()
                if unmasked_conf.numel() > 0 else 0.0
            )
            tracking_confidence.append(avg_conf)
            tracking_masked_count.append(int(still_masked.sum().item()))

    assert (seq != mask_token).all(), (
        "remdm_sample produced MASK tokens in final output"
    )
    if return_analytics:
        return seq, path_per_step, tracking_confidence, tracking_masked_count
    return seq


@torch.no_grad()
def greedy_sample(
    model: torch.nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    cfg: SimpleNamespace,
    device: torch.device | str,
    blind_global: bool = False,
    num_steps: int | None = None,
) -> Tensor:
    """Greedy (argmax) MaskGIT sampling — no temperature, top-K, or remasking.

    Used by ``DataCollector`` during DAgger for deterministic rollouts,
    matching the reference ``run_model_episode`` behaviour.

    Args:
        model: Denoising model.
        local_obs: Shape ``[B, 9, 9]``.
        global_obs: Shape ``[B, 21, 79]``.
        cfg: Config namespace.
        device: Torch device.
        blind_global: Zero out global map (local-only ablation).

    Returns:
        Fully committed action sequence ``[B, seq_len]``, int64.
    """
    B = local_obs.shape[0]
    seq_len = cfg.seq_len
    mask_token = cfg.mask_token
    action_dim = cfg.action_dim
    K = num_steps if num_steps is not None else cfg.diffusion_steps_eval

    local_obs = local_obs.to(device)
    global_obs = global_obs.to(device)
    if blind_global:
        global_obs = torch.zeros_like(global_obs)

    seq = torch.full(
        (B, seq_len), mask_token, dtype=torch.long, device=device,
    )

    for k in range(1, K + 1):
        ratio = k / K
        t_discrete = torch.full(
            (B,), int(cfg.num_diffusion_steps * (1.0 - ratio)),
            dtype=torch.long, device=device,
        )

        out = model(local_obs, global_obs, seq, t_discrete)
        logits = out["actions"]  # [B, seq_len, vocab]

        # Mask invalid action tokens
        logits[:, :, action_dim:] = float("-inf")

        # Greedy: argmax over softmax (no temperature, no top-K)
        probs = F.softmax(logits, dim=-1)  # [B, seq_len, action_dim]
        confidences, preds = probs.max(dim=-1)  # [B, seq_len] each

        # MaskGIT progressive unmasking by confidence
        num_to_unmask = max(1, int(seq_len * ratio))
        is_masked = seq == mask_token  # [B, seq_len]

        # Score only masked positions for unmasking
        scores = confidences.clone()
        scores[~is_masked] = -1.0
        _, topk_idx = scores.topk(num_to_unmask, dim=-1)

        unmask_mask = torch.zeros_like(seq, dtype=torch.bool)
        unmask_mask.scatter_(1, topk_idx, True)
        unmask_mask = unmask_mask & is_masked

        seq = torch.where(unmask_mask, preds, seq)

        # No remasking in greedy mode

    # Force-commit any remaining masked tokens
    still_masked = seq == mask_token
    if still_masked.any():
        t_zero = torch.zeros(B, dtype=torch.long, device=device)
        out = model(local_obs, global_obs, seq, t_zero)
        logits = out["actions"]
        logits[:, :, action_dim:] = float("-inf")
        preds = logits.argmax(dim=-1)
        seq = torch.where(still_masked, preds, seq)

    return seq


def select_action(
    model: torch.nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    cfg: SimpleNamespace,
    device: torch.device | str,
    physics_aware: bool = True,
    blind_global: bool = False,
) -> int:
    """Sample a single action from a length-1 batch.

    Args:
        model: Denoising model.
        local_obs: Shape ``[9, 9]`` or ``[1, 9, 9]``.
        global_obs: Shape ``[21, 79]`` or ``[1, 21, 79]``.
        cfg: Config namespace.
        device: Torch device.
        physics_aware: Forward to ``remdm_sample``.
        blind_global: Forward to ``remdm_sample``.

    Returns:
        The first action of the generated plan (int).
    """
    if local_obs.ndim == 2:
        local_obs = local_obs.unsqueeze(0)
    if global_obs.ndim == 2:
        global_obs = global_obs.unsqueeze(0)
    seq = remdm_sample(
        model, local_obs, global_obs, cfg, device,
        physics_aware=physics_aware, blind_global=blind_global,
    )
    return seq[0, 0].item()
