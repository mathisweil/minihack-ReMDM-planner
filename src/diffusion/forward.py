"""Forward masking process q(z_t | x_0).

Shared pseudocode line 4 (METHOD_PARITY 2.1); the craftax twin is
src/diffusion/forward.py:forward_process.

Each token is independently replaced with mask_token with probability
sigma_t = 1 - alpha_t. PAD positions are never masked.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def q_sample(
    x0: Tensor,
    t: Tensor,
    mask_token: int,
    pad_token: int,
    schedule_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """Sample z_t from the forward masking process.

    Args:
        x0: Clean action sequences. Shape ``[B, L]``, dtype int64.
        t: Per-sample diffusion time in [0, 1]. Shape ``[B]``.
        mask_token: Integer ID of the MASK token.
        pad_token: Integer ID of the PAD token.
        schedule_fn: Noise schedule returning alpha(t).

    Returns:
        Noisy sequence z_t. Shape ``[B, L]``, dtype int64.
        PAD positions are preserved unchanged.
    """
    if x0.ndim != 2 or t.ndim != 1 or x0.shape[0] != t.shape[0]:
        raise ValueError(
            f"q_sample expects x0 [B, L] and t [B]; got {tuple(x0.shape)}, {tuple(t.shape)}"
        )
    alpha_t = schedule_fn(t)  # [B]
    sigma_t = 1.0 - alpha_t  # mask probability per sample
    sigma_t = sigma_t.unsqueeze(-1)  # [B, 1]

    # Independent Bernoulli masking per position
    mask_draws = torch.rand_like(x0, dtype=torch.float32)  # [B, L]
    do_mask = mask_draws < sigma_t  # [B, L]

    zt = torch.where(do_mask, mask_token, x0)

    # (A) benchmark-forced: PAD exists only in this repo (variable-length
    # oracle trajectories); craftax windows are fixed-length (METHOD_PARITY 2.1)
    # Restore PAD positions — never mask padding
    pad_mask = x0 == pad_token  # [B, L]
    return torch.where(pad_mask, pad_token, zt)

