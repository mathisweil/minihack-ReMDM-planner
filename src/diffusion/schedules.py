"""Noise schedule functions for MDLM diffusion.

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
    """Cosine noise schedule: alpha(t) = cos(pi/2 * t).

    MDLM Appendix E.1 eq (92) ("Cosine"); the same function the craftax
    repo names "cosine".

    Args:
        t: Diffusion time in [0, 1]. Any shape.

    Returns:
        Retention probability alpha_t, same shape as *t*.
    """
    return torch.cos(t * (math.pi / 2.0))


def cosine_sq_schedule(t: Tensor) -> Tensor:
    """Cosine-squared noise schedule: alpha(t) = cos(pi/2 * t)^2.

    MDLM Appendix E.1 eq (91) ("Cosine Squared", after Nichol & Dhariwal).
    Previously registered under the name "cosine" in this repo; renamed so
    the label "cosine" denotes the same function in both repos.

    Args:
        t: Diffusion time in [0, 1]. Any shape.

    Returns:
        Retention probability alpha_t, same shape as *t*.
    """
    return torch.cos(t * (math.pi / 2.0)) ** 2


def linear_schedule_deriv(t: Tensor) -> Tensor:
    """Analytic d(alpha)/dt for the linear schedule."""
    return torch.full_like(t, -1.0)


def cosine_schedule_deriv(t: Tensor) -> Tensor:
    """Analytic d(alpha)/dt for the cosine schedule."""
    return -(math.pi / 2.0) * torch.sin(t * (math.pi / 2.0))


def cosine_sq_schedule_deriv(t: Tensor) -> Tensor:
    """Analytic d(alpha)/dt for the cosine-squared schedule."""
    return -(math.pi / 2.0) * torch.sin(t * math.pi)


_SCHEDULE_MAP: dict[str, Callable[[Tensor], Tensor]] = {
    "linear": linear_schedule,
    "cosine": cosine_schedule,
    "cosine_sq": cosine_sq_schedule,
}

_DERIV_BY_FN: dict[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]] = {
    linear_schedule: linear_schedule_deriv,
    cosine_schedule: cosine_schedule_deriv,
    cosine_sq_schedule: cosine_sq_schedule_deriv,
}


def get_schedule_deriv_for(
    schedule_fn: Callable[[Tensor], Tensor],
) -> Callable[[Tensor], Tensor]:
    """Analytic derivative for a registered schedule function.

    FIX-1 (ADJUDICATION B-1): the NELBO weight uses the analytic
    d(alpha)/dt as stated in MDLM eq (10) / Shi eq (4); the numerical
    stencil ``alpha_prime`` remains only for reference.

    Raises:
        KeyError: If *schedule_fn* is not a registered schedule.
    """
    if schedule_fn not in _DERIV_BY_FN:
        raise KeyError(
            "No analytic derivative registered for "
            f"{getattr(schedule_fn, '__name__', schedule_fn)!r}"
        )
    return _DERIV_BY_FN[schedule_fn]


def get_schedule(name: str) -> Callable[[Tensor], Tensor]:
    """Look up a noise schedule by name.

    Args:
        name: One of ``"linear"``, ``"cosine"``, ``"cosine_sq"``.

    Returns:
        The schedule function ``alpha(t)``.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in _SCHEDULE_MAP:
        raise KeyError(
            f"Unknown schedule '{name}'. Available: {list(_SCHEDULE_MAP.keys())}"
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
    return (schedule_fn(t_clamped + eps) - schedule_fn(t_clamped - eps)) / (2.0 * eps)
