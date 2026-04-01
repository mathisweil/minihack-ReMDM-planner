"""Gradient-space diagnostics for RL fine-tuning analysis (PyTorch).

Measures gradient alignment between RL and BC objectives, per-layer
gradient norms, and PCGrad surgery metrics.

Adapted from Craftax JAX implementation.  All functions are eager
PyTorch; no JIT compilation.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from experiments.rl_finetuning.ablations.losses import _core_loss


# ---------------------------------------------------------------------------
# Gradient alignment: cosine similarity between RL and BC gradients
# ---------------------------------------------------------------------------


@torch.no_grad()
def _collect_flat_grad(model: nn.Module) -> Tensor:
    """Flatten all parameter gradients into a single vector.

    Args:
        model: Model after a backward pass.

    Returns:
        1-D tensor of concatenated gradients (detached).
    """
    parts: list[Tensor] = []
    for param in model.parameters():
        if param.grad is not None:
            parts.append(param.grad.detach().reshape(-1))
        else:
            parts.append(torch.zeros(param.numel(), device=param.device))
    return torch.cat(parts)


def compute_grad_alignment(
    model: nn.Module,
    ref_model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    advantages: Tensor | None,
    cfg: SimpleNamespace,
    device: torch.device,
) -> tuple[float, float, float]:
    """Cosine similarity between RL and BC gradient vectors.

    Computes RL loss with advantages and BC loss without, then
    measures alignment of their full-model gradient vectors.

    Args:
        model: Current model (must be in train mode).
        ref_model: Pretrained model (unused here; kept for interface).
        local_obs: ``[B, 9, 9]``.
        global_obs: ``[B, 21, 79]``.
        x0: ``[B, H]`` clean actions.
        advantages: ``[B]`` advantage weights for RL loss.
        cfg: Config namespace with ``_schedule_fn``.
        device: Torch device.

    Returns:
        Tuple of (cosine_similarity, rl_grad_norm, bc_grad_norm).
    """
    model.train()

    # RL gradient
    model.zero_grad()
    rl_loss = _core_loss(
        model, local_obs, global_obs, x0, advantages, cfg, device,
    )
    rl_loss.backward()
    g_rl = _collect_flat_grad(model)

    # BC gradient (no advantage weighting)
    model.zero_grad()
    bc_loss = _core_loss(
        model, local_obs, global_obs, x0, None, cfg, device,
    )
    bc_loss.backward()
    g_bc = _collect_flat_grad(model)

    model.zero_grad()

    rl_norm = g_rl.norm().item()
    bc_norm = g_bc.norm().item()
    cos_sim = (
        torch.dot(g_rl, g_bc) / (rl_norm * bc_norm + 1e-10)
    ).item()

    return cos_sim, rl_norm, bc_norm


# ---------------------------------------------------------------------------
# Per-layer gradient norms
# ---------------------------------------------------------------------------


def compute_per_layer_grad_norms(
    model: nn.Module,
) -> dict[str, float]:
    """L2 gradient norm for each named parameter.

    Call after a ``loss.backward()``.

    Args:
        model: Model with populated ``.grad`` attributes.

    Returns:
        Dict mapping parameter name to L2 norm.
    """
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.detach().norm().item()
    return norms


# ---------------------------------------------------------------------------
# PCGrad surgery metrics
# ---------------------------------------------------------------------------


def compute_surgery_metrics(
    g_before: dict[str, Tensor],
    g_after: dict[str, Tensor],
) -> tuple[float, int]:
    """Measure gradient mass removed by PCGrad projection.

    Args:
        g_before: RL gradient dict before projection.
        g_after: RL gradient dict after projection.

    Returns:
        Tuple of (projected_mass_fraction, n_conflicting_params).
    """
    total_before = 0.0
    total_after = 0.0
    n_conflicting = 0

    for name in g_before:
        gb = g_before[name]
        ga = g_after.get(name, gb)
        sq_before = (gb * gb).sum().item()
        sq_after = (ga * ga).sum().item()
        total_before += sq_before
        total_after += sq_after
        if (gb * (gb - ga)).sum().item() > 0:
            n_conflicting += 1

    mass_removed = max(total_before - total_after, 0.0)
    fraction = mass_removed / max(total_before, 1e-10)
    return fraction, n_conflicting
