"""Noise schedule functions for MDLM diffusion.

Ported from the Craftax JAX implementation (src/diffusion/schedules.py).
All functions operate on PyTorch tensors and are pure (no global state).

Convention: alpha(t) is the fraction of tokens that remain *unmasked*.
  - alpha(0) = 1.0  (fully clean)
  - alpha(1) = 0.0  (fully masked)
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor


def linear_schedule(t: Tensor) -> Tensor:
    """Linear noise schedule: alpha(t) = 1 - t.

    Args:
        t: Diffusion time in [0, 1]. Any shape.

    Returns:
        Retention probability alpha_t, same shape as *t*.
    """
    return 1.0 - t


def cosine_schedule(t: Tensor) -> Tensor:
    """Cosine noise schedule: alpha(t) = cos(pi/2 * t)^2.

    Args:
        t: Diffusion time in [0, 1]. Any shape.

    Returns:
        Retention probability alpha_t, same shape as *t*.
    """
    return torch.cos(t * (math.pi / 2.0)) ** 2


_SCHEDULE_MAP: dict[str, Callable[[Tensor], Tensor]] = {
    "linear": linear_schedule,
    "cosine": cosine_schedule,
}


def get_schedule(name: str) -> Callable[[Tensor], Tensor]:
    """Look up a noise schedule by name.

    Args:
        name: One of ``"linear"`` or ``"cosine"``.

    Returns:
        The schedule function ``alpha(t)``.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in _SCHEDULE_MAP:
        raise KeyError(
            f"Unknown schedule '{name}'. "
            f"Available: {list(_SCHEDULE_MAP.keys())}"
        )
    return _SCHEDULE_MAP[name]


def alpha_prime(
    t: Tensor,
    schedule_fn: Callable[[Tensor], Tensor],
    eps: float = 1e-5,
) -> Tensor:
    """Numerical derivative d(alpha)/dt via central difference.

    Args:
        t: Diffusion time in [0, 1]. Any shape.
        schedule_fn: Noise schedule returning alpha(t).
        eps: Half-width for finite-difference stencil.

    Returns:
        Approximate derivative, same shape as *t*.
    """
    t_clamped = t.clamp(eps, 1.0 - eps)
    return (schedule_fn(t_clamped + eps) - schedule_fn(t_clamped - eps)) / (
        2.0 * eps
    )
