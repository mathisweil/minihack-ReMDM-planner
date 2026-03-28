"""Forward masking process q(z_t | x_0).

Ported from the Craftax JAX implementation (src/diffusion/forward.py).
Each token is independently replaced with mask_token with probability
sigma_t = 1 - alpha_t. PAD positions are never masked.
"""

from __future__ import annotations

from typing import Callable

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
    alpha_t = schedule_fn(t)  # [B]
    sigma_t = 1.0 - alpha_t  # mask probability per sample
    sigma_t = sigma_t.unsqueeze(-1)  # [B, 1]

    # Independent Bernoulli masking per position
    mask_draws = torch.rand_like(x0, dtype=torch.float32)  # [B, L]
    do_mask = mask_draws < sigma_t  # [B, L]

    zt = torch.where(do_mask, mask_token, x0)

    # Restore PAD positions — never mask padding
    pad_mask = x0 == pad_token  # [B, L]
    zt = torch.where(pad_mask, pad_token, zt)

    return zt
