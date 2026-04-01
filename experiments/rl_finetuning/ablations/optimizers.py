"""Optimizer factory functions for RL fine-tuning ablations (PyTorch).

Each factory returns a ``torch.optim.Optimizer`` configured for a
specific ablation variant.  Factories are pure functions with no
global state.

LoRA parameter injection uses ``torch.nn.utils.parametrize`` to
augment attention weights without modifying the base model class.

Gradient surgery (PCGrad) operates on gradient dicts and is called
from the training loop, not inside the optimizer.

Adapted from Craftax JAX/optax implementation to PyTorch.
Key differences:
- PyTorch param groups replace optax.multi_transform (LLRD)
- requires_grad=False replaces optax.masked (frozen params)
- torch.nn.utils.parametrize replaces JAX functional param injection (LoRA)
- Per-tensor dict replaces JAX pytree (gradient surgery)
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import parametrize


# ---------------------------------------------------------------------------
# Standard AdamW
# ---------------------------------------------------------------------------


def make_optimizer_standard(
    cfg: SimpleNamespace,
    model: nn.Module,
) -> torch.optim.Optimizer:
    """AdamW baseline optimizer for all model parameters.

    Args:
        cfg: Config with ``lr``.
        model: Model whose parameters to optimise.

    Returns:
        AdamW optimizer.
    """
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=getattr(cfg, "lr", 3e-4),
        weight_decay=getattr(cfg, "weight_decay", 1e-4),
        eps=1e-5,
    )


# ---------------------------------------------------------------------------
# Group A: LLRD (Layer-wise Learning Rate Decay)
# ---------------------------------------------------------------------------


def _get_llrd_group(name: str) -> str:
    """Assign a learning-rate group label to a parameter name.

    Groups (fastest to slowest LR):
    - ``head``:       final output projection (``head.*``)
    - ``block_{i}``:  transformer layer at index *i* (0 = earliest)
    - ``obs_enc``:    everything else (observation encoders, embeddings)

    Args:
        name: Fully qualified parameter name
              (e.g. ``transformer.layers.2.linear1.weight``).

    Returns:
        Group label string.
    """
    if name.startswith("head."):
        return "head"
    if "transformer.layers." in name:
        idx_str = name.split("transformer.layers.")[1].split(".")[0]
        try:
            return f"block_{int(idx_str)}"
        except ValueError:
            return "head"
    return "obs_enc"


def make_optimizer_llrd(
    cfg: SimpleNamespace,
    model: nn.Module,
) -> torch.optim.Optimizer:
    """AdamW with Layer-wise Learning Rate Decay.

    LR at depth *d* from the output head = ``base_lr * decay^d``.

    - Head:                  base_lr (depth 0)
    - transformer layer N-1: base_lr * decay^1  (closest to head)
    - transformer layer 0:   base_lr * decay^N  (farthest)
    - Obs encoder:           base_lr * decay^(N+1)

    Args:
        cfg: Config with ``lr``, ``llrd_decay``, ``n_layer``.
        model: Model.

    Returns:
        AdamW with per-group learning rates.
    """
    base_lr = getattr(cfg, "lr", 3e-4)
    decay = getattr(cfg, "llrd_decay", 0.9)
    n_layers = getattr(cfg, "n_layer", 4)

    groups: dict[str, list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        label = _get_llrd_group(name)
        groups.setdefault(label, []).append(param)

    param_groups: list[dict] = []
    if "head" in groups:
        param_groups.append({"params": groups["head"], "lr": base_lr})
    for i in range(n_layers):
        key = f"block_{i}"
        if key in groups:
            depth = n_layers - i  # block_0 farthest from head
            param_groups.append({
                "params": groups[key],
                "lr": base_lr * (decay ** depth),
            })
    if "obs_enc" in groups:
        param_groups.append({
            "params": groups["obs_enc"],
            "lr": base_lr * (decay ** (n_layers + 1)),
        })

    wd = getattr(cfg, "weight_decay", 1e-4)
    return torch.optim.AdamW(param_groups, weight_decay=wd, eps=1e-5)


# ---------------------------------------------------------------------------
# Group A / C: Frozen backbone / parameter isolation
# ---------------------------------------------------------------------------


def make_optimizer_frozen(
    cfg: SimpleNamespace,
    model: nn.Module,
    frozen_fragments: list[str],
) -> torch.optim.Optimizer:
    """AdamW with specified parameter paths frozen.

    Parameters whose name contains ANY fragment from
    ``frozen_fragments`` have ``requires_grad`` set to False and are
    excluded from the optimizer.

    Args:
        cfg: Config with ``lr``.
        model: Model (modified in-place: requires_grad toggled).
        frozen_fragments: Name substrings identifying frozen params.

    Returns:
        AdamW for trainable parameters only.
    """
    trainable: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if any(frag in name for frag in frozen_fragments):
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable.append(param)

    if not trainable:
        dummy = torch.zeros(1, requires_grad=True)
        return torch.optim.AdamW([dummy], lr=0.0)

    return torch.optim.AdamW(
        trainable,
        lr=getattr(cfg, "lr", 3e-4),
        weight_decay=getattr(cfg, "weight_decay", 1e-4),
        eps=1e-5,
    )


# ---------------------------------------------------------------------------
# Frozen-path presets for specific ablations
# ---------------------------------------------------------------------------

# Freeze everything except the action head
FROZEN_BACKBONE: list[str] = [
    "embedding.",
    "cnn.",
    "global_embedding.",
    "global_cnn.",
    "global_pool.",
    "global_proj.",
    "global_gate",
    "goal_head.",
    "action_emb.",
    "timestep_emb.",
    "pos_emb.",
    "transformer.",
]

# Freeze everything except attention sublayers
FROZEN_EXCEPT_ATTENTION: list[str] = [
    "embedding.",
    "cnn.",
    "global_embedding.",
    "global_cnn.",
    "global_pool.",
    "global_proj.",
    "global_gate",
    "goal_head.",
    "action_emb.",
    "timestep_emb.",
    "pos_emb.",
    "linear1.",
    "linear2.",
    "norm2.",
    "head.",
]

# Freeze everything except FFN sublayers
FROZEN_EXCEPT_FFN: list[str] = [
    "embedding.",
    "cnn.",
    "global_embedding.",
    "global_cnn.",
    "global_pool.",
    "global_proj.",
    "global_gate",
    "goal_head.",
    "action_emb.",
    "timestep_emb.",
    "pos_emb.",
    "self_attn.",
    "norm1.",
    "head.",
]

# Freeze everything except the last transformer layer + head
FROZEN_EXCEPT_LAST_LAYER: list[str] = [
    "embedding.",
    "cnn.",
    "global_embedding.",
    "global_cnn.",
    "global_pool.",
    "global_proj.",
    "global_gate",
    "goal_head.",
    "action_emb.",
    "timestep_emb.",
    "pos_emb.",
    "transformer.layers.0.",
    "transformer.layers.1.",
    "transformer.layers.2.",
]


# ---------------------------------------------------------------------------
# Group A: LoRA via torch.nn.utils.parametrize
# ---------------------------------------------------------------------------


class _LoRAParametrization(nn.Module):
    """Low-rank parametrization for a weight matrix.

    Computes ``W_eff = W + (alpha / rank) * B @ A``.
    A is Gaussian-initialised, B is zero-initialised so the initial
    delta is zero (training starts from pretrained weights).

    Args:
        d_out: Output dimension of the weight matrix.
        d_in: Input dimension of the weight matrix.
        rank: LoRA rank *r*.
        alpha: LoRA alpha scaling factor.
    """

    def __init__(
        self, d_out: int, d_in: int, rank: int, alpha: float,
    ) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.02)  # [r, d_in]
        self.B = nn.Parameter(torch.zeros(d_out, rank))  # [d_out, r]
        self.scale = alpha / max(rank, 1)

    def forward(self, weight: Tensor) -> Tensor:
        """Return effective weight with LoRA delta.

        Args:
            weight: Original weight ``[d_out, d_in]``.

        Returns:
            ``W + scale * B @ A``, same shape as *weight*.
        """
        return weight + self.scale * (self.B @ self.A)


def apply_lora_to_model(
    model: nn.Module,
    rank: int,
    alpha: float,
) -> list[nn.Parameter]:
    """Add LoRA parametrizations to all attention layers in-place.

    Targets per transformer layer:
    - ``self_attn.in_proj_weight`` ``[3*d, d]`` (fused QKV)
    - ``self_attn.out_proj.weight`` ``[d, d]``

    Base model parameters are frozen; only LoRA A/B are trainable.

    Args:
        model: Model to augment (modified in-place).
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.

    Returns:
        Flat list of trainable LoRA nn.Parameter objects.
    """
    for p in model.parameters():
        p.requires_grad = False

    lora_params: list[nn.Parameter] = []

    for layer in model.transformer.layers:
        attn = layer.self_attn

        # in_proj_weight: [3*d_model, d_model]
        d_out_in, d_in_in = attn.in_proj_weight.shape
        lora_in = _LoRAParametrization(d_out_in, d_in_in, rank, alpha)
        parametrize.register_parametrization(attn, "in_proj_weight", lora_in)
        lora_params.extend([lora_in.A, lora_in.B])

        # out_proj.weight: [d_model, d_model]
        d_out_out, d_in_out = attn.out_proj.weight.shape
        lora_out = _LoRAParametrization(d_out_out, d_in_out, rank, alpha)
        parametrize.register_parametrization(attn.out_proj, "weight", lora_out)
        lora_params.extend([lora_out.A, lora_out.B])

    for p in lora_params:
        p.requires_grad = True

    return lora_params


def remove_lora_from_model(model: nn.Module) -> None:
    """Remove all LoRA parametrizations, baking deltas into weights.

    After calling this the model uses effective weights permanently
    and no longer carries LoRA parameters.

    Args:
        model: Model previously augmented with ``apply_lora_to_model``.
    """
    for layer in model.transformer.layers:
        attn = layer.self_attn
        if parametrize.is_parametrized(attn, "in_proj_weight"):
            parametrize.remove_parametrizations(attn, "in_proj_weight")
        if parametrize.is_parametrized(attn.out_proj, "weight"):
            parametrize.remove_parametrizations(attn.out_proj, "weight")


def make_optimizer_lora(
    cfg: SimpleNamespace,
    lora_params: list[nn.Parameter],
) -> torch.optim.Optimizer:
    """AdamW that only updates LoRA parameters.

    Call after ``apply_lora_to_model`` which returns the param list.

    Args:
        cfg: Config with ``lr``.
        lora_params: LoRA A/B parameters from ``apply_lora_to_model``.

    Returns:
        AdamW for LoRA parameters only.
    """
    return torch.optim.AdamW(
        lora_params,
        lr=getattr(cfg, "lr", 3e-4),
        weight_decay=getattr(cfg, "weight_decay", 1e-4),
        eps=1e-5,
    )


# ---------------------------------------------------------------------------
# Gradient surgery (PCGrad)
# ---------------------------------------------------------------------------


def collect_gradients(model: nn.Module) -> dict[str, Tensor]:
    """Snapshot current ``.grad`` for every parameter with a gradient.

    Args:
        model: Model after a backward pass.

    Returns:
        Dict mapping parameter name to detached gradient clone.
    """
    grads: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def apply_gradients(model: nn.Module, grads: dict[str, Tensor]) -> None:
    """Write gradient dict back into model ``.grad`` attributes.

    Args:
        model: Target model.
        grads: Gradient dict from ``collect_gradients`` or
               ``gradient_surgery``.
    """
    for name, param in model.named_parameters():
        if name in grads:
            param.grad = grads[name]


def gradient_surgery(
    g_rl: dict[str, Tensor],
    g_bc: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Project RL gradients to remove conflict with BC gradients (PCGrad).

    For each parameter tensor: if ``dot(g_rl, g_bc) < 0``, projects
    g_rl onto the plane orthogonal to g_bc.  Otherwise keeps g_rl
    unchanged.

    Args:
        g_rl: RL gradient dict ``{param_name: gradient}``.
        g_bc: BC gradient dict ``{param_name: gradient}``.

    Returns:
        Projected RL gradient dict.
    """
    projected: dict[str, Tensor] = {}
    for name, gr in g_rl.items():
        if name not in g_bc:
            projected[name] = gr
            continue
        gb = g_bc[name]
        dot = (gr * gb).sum()
        if dot < 0:
            norm_sq = (gb * gb).sum().clamp(min=1e-10)
            projected[name] = gr - (dot / norm_sq) * gb
        else:
            projected[name] = gr
    return projected
