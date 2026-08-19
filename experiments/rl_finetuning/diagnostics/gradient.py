"""Gradient-space diagnostics for RL fine-tuning analysis (PyTorch).

Measures gradient alignment between RL and BC objectives, per-layer
gradient norms, and PCGrad surgery metrics.

Adapted from Craftax JAX implementation.  All functions are eager
PyTorch; no JIT compilation.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from experiments.rl_finetuning.ablations.losses import _core_loss


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


def _canonical_name(name: str) -> str:
    """Parameter name with the wrappers' decorations removed.

    ``torch.compile`` prefixes ``_orig_mod.`` and
    ``torch.nn.utils.parametrize`` rewrites ``w`` to
    ``parametrizations.w.original``; the pretrained reference carries
    neither, so both are stripped before pairing parameters by name.

    Args:
        name: Name as ``named_parameters`` reports it.

    Returns:
        The name the uninstrumented module would report.
    """
    return (
        name.replace("_orig_mod.", "")
        .replace("parametrizations.", "")
        .replace(".original", "")
    )


@contextlib.contextmanager
def _at_reference_parameters(model: nn.Module, ref_model: nn.Module) -> Iterator[None]:
    """Hold ``model``'s parameters at ``ref_model``'s values for the block.

    Evaluating the BC gradient by loading the reference values into the
    current module -- rather than by differentiating ``ref_model`` itself --
    keeps both gradients in one parameter space and one ordering, so the
    cosine is well defined for every ablation, including the parametrized
    and the partly frozen ones.

    Parameters with no counterpart in the reference keep their current
    values. LoRA's A and B factors are the case that arises: the pretrained
    model has no analogue for them, and the sibling repo's ``params["base"]``
    reference leaves them out for the same reason.

    Args:
        model: Module whose parameters are swapped.
        ref_model: Pretrained module supplying the values.

    Yields:
        None, with the swap in force.
    """
    ref = {_canonical_name(n): p for n, p in ref_model.named_parameters()}
    saved: list[tuple[Tensor, Tensor]] = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            src = ref.get(_canonical_name(name))
            if src is not None and src.shape == param.shape:
                saved.append((param, param.detach().clone()))
                param.copy_(src)
    try:
        yield
    finally:
        with torch.no_grad():
            for param, value in saved:
                param.copy_(value)


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

    Computes RL loss with advantages and BC loss without, then measures
    alignment of their full-model gradient vectors.

    Both gradients are taken on the same batch **and the same ``(z_t, t)``
    draw**: ``_core_loss`` samples its timestep and its masking from the
    global generator, so the generator is rewound between the two backward
    passes. At independent draws the metric reports Monte-Carlo noise as
    objective disagreement. Measured over six trials on one fixed batch at
    the production architecture: two draws give a mean cosine of 0.858
    ranging over 0.104, where one draw gives 0.954 ranging over 0.063 -- a
    mean shift of 0.096. The sibling repo is hit far harder, reporting
    anti-alignment where the same-draw value is 0.98. Sharing the draw
    leaves the generator where one loss would have left it rather than
    where two would.

    The BC gradient is taken at the **pretrained** parameters, not at the
    current ones: a fixed reference is comparable across iterations and is
    the quantity the forgetting framing needs, and it is what the sibling
    repo has always measured. ``ref_model`` supplies them, held in place by
    :func:`_at_reference_parameters` for the BC pass alone.

    Args:
        model: Current model (must be in train mode).
        ref_model: Pretrained model, the reference the BC gradient is taken at.
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
    _use_amp = getattr(cfg, "use_amp", False) and device.type == "cuda"

    def _grad(weights: Tensor | None) -> Tensor:
        model.zero_grad()
        with torch.amp.autocast("cuda", enabled=_use_amp):
            loss = _core_loss(model, local_obs, global_obs, x0, weights, cfg, device)
        loss.backward()
        return _collect_flat_grad(model)

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    # RL gradient
    g_rl = _grad(advantages)

    # BC gradient: no advantage weighting, the pretrained parameters, and
    # the same draw the RL gradient saw.
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    with _at_reference_parameters(model, ref_model):
        g_bc = _grad(None)

    model.zero_grad()

    rl_norm = g_rl.norm().item()
    bc_norm = g_bc.norm().item()
    cos_sim = (torch.dot(g_rl, g_bc) / (rl_norm * bc_norm + 1e-10)).item()

    return cos_sim, rl_norm, bc_norm


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
