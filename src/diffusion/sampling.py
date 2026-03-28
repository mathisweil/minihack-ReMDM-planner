"""ReMDM reverse denoising with remasking strategies.

Ported from the Craftax JAX implementation (src/diffusion/sampling.py).
Implements MaskGIT-style progressive unmasking with optional stochastic
remasking (ReMDM) using three strategy variants.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from src.diffusion.schedules import get_schedule


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
) -> Tensor:
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

    Returns:
        Fully committed action sequence. Shape ``[B, seq_len]``, int64.
        Guaranteed to contain no MASK tokens.
    """
    B = local_obs.shape[0]
    seq_len = cfg.seq_len
    mask_token = cfg.mask_token
    action_dim = cfg.action_dim
    K = cfg.diffusion_steps_eval
    schedule_fn = get_schedule(cfg.noise_schedule)

    local_obs = local_obs.to(device)
    global_obs = global_obs.to(device)

    # Start fully masked
    seq = torch.full(
        (B, seq_len), mask_token, dtype=torch.long, device=device
    )

    for k in range(1, K + 1):
        ratio = k / K
        t_discrete = int(cfg.num_diffusion_steps * (1.0 - ratio))

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

        is_masked = seq == mask_token  # [B, seq_len]

        if k < K:
            # MaskGIT progressive unmasking
            n_unmask = max(1, int(seq_len * ratio))

            # Set confidence of non-masked positions to -inf so they
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

    assert (seq != mask_token).all(), (
        "remdm_sample produced MASK tokens in final output"
    )
    return seq


def select_action(
    model: torch.nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    cfg: SimpleNamespace,
    device: torch.device | str,
) -> int:
    """Sample a single action from a length-1 batch.

    Args:
        model: Denoising model.
        local_obs: Shape ``[9, 9]`` or ``[1, 9, 9]``.
        global_obs: Shape ``[21, 79]`` or ``[1, 21, 79]``.
        cfg: Config namespace.
        device: Torch device.

    Returns:
        The first action of the generated plan (int).
    """
    if local_obs.ndim == 2:
        local_obs = local_obs.unsqueeze(0)
    if global_obs.ndim == 2:
        global_obs = global_obs.unsqueeze(0)
    seq = remdm_sample(model, local_obs, global_obs, cfg, device)
    return seq[0, 0].item()
