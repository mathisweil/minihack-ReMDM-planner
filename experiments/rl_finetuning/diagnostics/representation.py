"""Representation-space diagnostics: KL drift and CKA similarity (PyTorch).

KL drift measures how far the output distribution has moved from the
pretrained model across different t ranges.  CKA (Centred Kernel
Alignment) measures similarity of internal activations.

Adapted from Craftax JAX implementation.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.diffusion.forward import q_sample

_EPS: float = 1e-5


def _kl_at_range(
    model: nn.Module,
    ref_model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
    t_min: float,
    t_max: float,
) -> float:
    """Mean KL(ref || current) for t sampled in ``[t_min, t_max]``.

    Direction: KL(ref || cur) — measures how much the current model
    diverges from the pretrained reference.

    Args:
        model: Current fine-tuned model.
        ref_model: Frozen pretrained model.
        local_obs: ``[B, 9, 9]``.
        global_obs: ``[B, 21, 79]``.
        x0: ``[B, H]`` clean actions.
        cfg: Config with ``_schedule_fn``, ``mask_token``, etc.
        device: Torch device.
        t_min: Lower t bound.
        t_max: Upper t bound.

    Returns:
        Scalar mean KL divergence.
    """
    B = x0.shape[0]
    schedule_fn = cfg._schedule_fn

    t = torch.rand(B, device=device) * (t_max - t_min) + t_min  # [B]
    zt = q_sample(x0, t, cfg.mask_token, cfg.pad_token, schedule_fn)
    t_discrete = (
        (t * cfg.num_diffusion_steps)
        .long()
        .clamp(
            0,
            cfg.num_diffusion_steps - 1,
        )
    )

    cur_logits = model(local_obs, global_obs, zt, t_discrete)["actions"]
    ref_logits = ref_model(local_obs, global_obs, zt, t_discrete)["actions"]

    ref_log = F.log_softmax(ref_logits, dim=-1)  # [B, H, A]
    cur_log = F.log_softmax(cur_logits, dim=-1)  # [B, H, A]
    ref_prob = ref_log.exp()

    kl = (ref_prob * (ref_log - cur_log)).sum(dim=-1)  # [B, H]
    return kl.mean().item()


@torch.no_grad()
def compute_repr_drift(
    model: nn.Module,
    ref_model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """KL divergence drift from pretrained model at 4 t ranges.

    Args:
        model: Current fine-tuned model.
        ref_model: Frozen pretrained model.
        local_obs: ``[B, 9, 9]``.
        global_obs: ``[B, 21, 79]``.
        x0: ``[B, H]`` clean actions.
        cfg: Config namespace.
        device: Torch device.

    Returns:
        Tuple of (kl_mean, kl_low_t, kl_mid_t, kl_high_t).
    """
    model.eval()
    ref_model.eval()

    args = (model, ref_model, local_obs, global_obs, x0, cfg, device)
    kl_mean = _kl_at_range(*args, t_min=_EPS, t_max=1.0)
    kl_low = _kl_at_range(*args, t_min=_EPS, t_max=0.2)
    kl_mid = _kl_at_range(*args, t_min=0.3, t_max=0.7)
    kl_high = _kl_at_range(*args, t_min=0.8, t_max=1.0)

    return kl_mean, kl_low, kl_mid, kl_high


def _linear_cka(x: Tensor, y: Tensor) -> float:
    """Linear CKA between activation matrices.

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Uses the HSIC estimator with linear kernels.

    Args:
        x: ``[N, D_x]`` activations from current model.
        y: ``[N, D_y]`` activations from reference model.

    Returns:
        Scalar CKA in [0, 1]; 1 = identical representations.
    """
    n = x.shape[0]
    h = torch.eye(n, device=x.device) - torch.ones(n, n, device=x.device) / n

    kx = x @ x.T  # [N, N]
    ky = y @ y.T  # [N, N]

    hkx = h @ kx @ h  # centred
    hky = h @ ky @ h

    hsic_xy = (hkx * hky.T).sum()
    hsic_xx = (hkx * hkx.T).sum()
    hsic_yy = (hky * hky.T).sum()

    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)).item()


@torch.no_grad()
def compute_cka(
    model: nn.Module,
    ref_model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
) -> float:
    """CKA similarity between current and reference output representations.

    Uses mean-pooled logits over the sequence dimension as
    the representation vector.

    Args:
        model: Current fine-tuned model.
        ref_model: Frozen pretrained model.
        local_obs: ``[B, 9, 9]``.
        global_obs: ``[B, 21, 79]``.
        x0: ``[B, H]``.
        cfg: Config namespace.
        device: Torch device.

    Returns:
        Scalar CKA value.
    """
    model.eval()
    ref_model.eval()

    schedule_fn = cfg._schedule_fn
    cka_bs = min(
        getattr(cfg, "cka_batch_size", 64),
        x0.shape[0],
    )

    lo = local_obs[:cka_bs]
    go = global_obs[:cka_bs]
    acts = x0[:cka_bs]

    t = torch.rand(cka_bs, device=device) * 0.4 + 0.3  # [0.3, 0.7]
    zt = q_sample(acts, t, cfg.mask_token, cfg.pad_token, schedule_fn)
    t_discrete = (
        (t * cfg.num_diffusion_steps)
        .long()
        .clamp(
            0,
            cfg.num_diffusion_steps - 1,
        )
    )

    cur_logits = model(lo, go, zt, t_discrete)["actions"]  # [B, H, A]
    ref_logits = ref_model(lo, go, zt, t_discrete)["actions"]

    cur_repr = cur_logits.mean(dim=1)  # [B, A]
    ref_repr = ref_logits.mean(dim=1)

    return _linear_cka(cur_repr, ref_repr)


@torch.no_grad()
def compute_activation_norms(
    model: nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    x0: Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Activation norm statistics of model output logits.

    Args:
        model: Current model.
        local_obs: ``[B, 9, 9]``.
        global_obs: ``[B, 21, 79]``.
        x0: ``[B, H]``.
        cfg: Config namespace.
        device: Torch device.

    Returns:
        Tuple of (mean, std, p50, p90) logit norms.
    """
    model.eval()
    schedule_fn = cfg._schedule_fn
    B = x0.shape[0]

    t = torch.rand(B, device=device) * 0.4 + 0.3  # [0.3, 0.7]
    zt = q_sample(x0, t, cfg.mask_token, cfg.pad_token, schedule_fn)
    t_discrete = (
        (t * cfg.num_diffusion_steps)
        .long()
        .clamp(
            0,
            cfg.num_diffusion_steps - 1,
        )
    )

    logits = model(local_obs, global_obs, zt, t_discrete)["actions"]
    norms = logits.norm(dim=-1).mean(dim=-1)  # [B]

    mean = norms.mean().item()
    std = norms.std().item()
    p50 = norms.median().item()
    p90 = norms.quantile(0.9).item()
    return mean, std, p50, p90
