"""Timestep (t) distribution diagnostics for RL fine-tuning (PyTorch).

Analyses how gradient contributions and losses vary across the
continuous diffusion time t in [0, 1], partitioned into equal bins.

Adapted from Craftax JAX implementation.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from experiments.rl_finetuning.ablations.losses import _core_loss

_EPS: float = 1e-5
N_BINS: int = 10


def _flat_grad_at_range(
    model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    advantages: Tensor | None,
    cfg: SimpleNamespace,
    device: torch.device,
    t_min: float,
    t_max: float,
) -> Tensor:
    """Compute flattened gradient for a specific t range.

    Args:
        model: Current model (train mode).
        local_obs: ``[B, 9, 9]``.
        global_obs: ``[B, 21, 79]``.
        x0: ``[B, H]``.
        advantages: ``[B]`` or None.
        cfg: Config namespace.
        device: Torch device.
        t_min: Lower t bound.
        t_max: Upper t bound.

    Returns:
        1-D flattened gradient vector.
    """
    _use_amp = getattr(cfg, "use_amp", False) and device.type == "cuda"
    model.zero_grad()
    with torch.amp.autocast("cuda", enabled=_use_amp):
        loss = _core_loss(
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
    loss.backward()

    parts: list[Tensor] = []
    for param in model.parameters():
        if param.grad is not None:
            parts.append(param.grad.detach().reshape(-1))
        else:
            parts.append(torch.zeros(param.numel(), device=device))
    return torch.cat(parts)


def compute_t_analysis(
    model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    advantages: Tensor | None,
    cfg: SimpleNamespace,
    device: torch.device,
    n_bins: int = N_BINS,
) -> tuple[list[float], float, float, float]:
    """Per-t-bin gradient norms and low/high-t alignment.

    Partitions [0, 1] into ``n_bins`` equal bins, computes the RL
    loss gradient restricted to each bin, and reports norms plus
    the cosine similarity between low-t and high-t gradients.

    Args:
        model: Current model (train mode).
        local_obs: ``[B, 9, 9]``.
        global_obs: ``[B, 21, 79]``.
        x0: ``[B, H]``.
        advantages: ``[B]`` or None.
        cfg: Config namespace.
        device: Torch device.
        n_bins: Number of equal t bins.

    Returns:
        Tuple of:
        - bin_norms: list of ``n_bins`` per-bin gradient L2 norms.
        - low_high_cos: cosine sim between low-t and high-t grads.
        - norm_low_t: L2 norm of low-t gradient.
        - norm_high_t: L2 norm of high-t gradient.
    """
    model.train()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)

    args = (model, local_obs, global_obs, x0, advantages, cfg, device)

    bin_norms: list[float] = []
    for i in range(n_bins):
        t_lo = max(bin_edges[i].item(), _EPS)
        t_hi = max(bin_edges[i + 1].item(), t_lo + _EPS)
        flat = _flat_grad_at_range(*args, t_min=t_lo, t_max=t_hi)
        bin_norms.append(flat.norm().item())

    flat_low = _flat_grad_at_range(*args, t_min=_EPS, t_max=0.2)
    flat_high = _flat_grad_at_range(*args, t_min=0.8, t_max=1.0)

    norm_low = flat_low.norm().item()
    norm_high = flat_high.norm().item()
    cos = (torch.dot(flat_low, flat_high) / (norm_low * norm_high + 1e-10)).item()

    model.zero_grad()
    return bin_norms, cos, norm_low, norm_high
