"""ReMDM reverse denoising with remasking strategies.

remdm_sample implements ReMDM Algorithm 1 (Wang et al.): Bernoulli
posterior unmasking with the Section 4.1 remasking schedules. Shared
pseudocode lines 8-12 (METHOD_PARITY 2.1); the craftax twin is
src/diffusion/sampling.py:sample_plan. greedy_sample is a separate
MaskGIT-style argmax decoder used only for DAgger collection (CH-6,
documented engineering choice).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from src.diffusion.schedules import get_schedule

# NLE hazard glyph IDs and char codes (walls, locked doors, lava, water)
_HAZARD_GLYPHS: frozenset[int] = frozenset({2359, 2360, 2389, 2390})
_HAZARD_CHARS: frozenset[int] = frozenset(
    {ord("|"), ord("-"), ord("+"), ord("L"), ord("W")}
)
# Cardinal action → (dy, dx) offsets
_CARDINAL_OFFSETS: dict[int, tuple[int, int]] = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}
_N_PHYSICS_CHECK = 8  # only inspect the first N plan positions

# Stability guards; values must match the craftax twin exactly.
_SIGMA_DENOM_EPS = 1e-8  # sigma_max and posterior denominators
# Demotion value for hazardous actions under the conf strategy (CH-6).
_HAZARD_DECODE_PROB = 0.001


def _check_hazard(local_crop: np.ndarray, action: int) -> bool:
    """Return True if *action* from the agent's centre steps into a hazard.

    Args:
        local_crop: ``[crop_size, crop_size]`` glyph array.
        action: Cardinal action index (0=N, 1=E, 2=S, 3=W).

    Returns:
        ``True`` when the target cell contains a hazard glyph.
    """
    if action not in _CARDINAL_OFFSETS:
        return False
    cs = local_crop.shape[0]
    cy, cx = cs // 2, cs // 2
    dy, dx = _CARDINAL_OFFSETS[action]
    ny, nx = cy + dy, cx + dx
    if not (0 <= ny < cs and 0 <= nx < cs):
        return True
    glyph = int(local_crop[ny, nx])
    return glyph in _HAZARD_GLYPHS or glyph in _HAZARD_CHARS


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    """Nucleus filtering (ReMDM Sec 5; CH-1 replaced top-k filtering).

    Keeps the smallest prefix of the descending-sorted distribution whose
    cumulative mass reaches ``top_p``; all other logits go to ``-inf``.
    Mirrors the craftax twin ``_nucleus_sample`` cutoff semantics.

    Args:
        logits: Raw logits. Shape ``[..., V]``.
        top_p: Nucleus threshold in (0, 1]; ``>= 1`` disables filtering.

    Returns:
        Filtered logits with out-of-nucleus entries set to ``-inf``.
    """
    if top_p is None or top_p >= 1.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
    cutoff = sorted_p.cumsum(dim=-1) - sorted_p  # exclusive cumsum
    remove_sorted = cutoff >= top_p
    remove = remove_sorted.gather(-1, sorted_idx.argsort(dim=-1))
    return logits.masked_fill(remove, float("-inf"))


def _compute_remask_prob(
    strategy: str,
    eta: float,
    sigma_max: float,
    psi: Tensor | None,
    committed: Tensor | None = None,
) -> Tensor | float:
    """Compute per-token remasking probability.

    FIX-3 (ADJUDICATION B-3): the ``conf`` strategy now consumes the
    stored decoding probability ``psi`` from the step each token was last
    unmasked (ReMDM Sec 4.1), not the current step's fresh confidence.

    Args:
        strategy: One of ``"rescale"``, ``"cap"``, ``"conf"``.
        eta: Base remasking strength hyperparameter.
        sigma_max: ReMDM eq 7 value ``min(1, (1 - alpha_s) / alpha_t)``,
            computed by the caller.
        psi: Stored decoding probabilities at last unmask. Shape
            ``[B, L]``, ``+inf`` at masked positions. Required only for
            the ``"conf"`` strategy.
        committed: Boolean mask of committed (non-masked) positions.
            Required only for the ``"conf"`` strategy.

    Returns:
        Scalar or ``[B, L]`` tensor of remasking probabilities.
    """
    if strategy == "rescale":
        return eta * sigma_max
    if strategy == "cap":
        return min(eta, sigma_max)
    if strategy == "conf":
        assert psi is not None, "conf strategy requires psi"
        assert committed is not None, "conf strategy requires the committed mask"
        # softmax(-psi) over committed positions, zero elsewhere,
        # scaled by eta * sigma_max: mirrors craftax ``sigma_conf``.
        neg = torch.where(
            committed,
            -psi,
            torch.tensor(float("-inf"), device=psi.device, dtype=psi.dtype),
        )
        any_committed = committed.any(dim=-1, keepdim=True)
        safe = torch.where(any_committed, neg, torch.zeros_like(neg))
        weights = torch.softmax(safe, dim=-1)
        return torch.where(
            committed, weights * (eta * sigma_max), torch.zeros_like(weights)
        )
    raise ValueError(f"Unknown remask strategy: {strategy}")


@torch.no_grad()
def remdm_sample(
    model: torch.nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    cfg: SimpleNamespace,
    device: torch.device | str,
    physics_aware: bool = True,
    blind_global: bool = False,
    return_analytics: bool = False,
    num_steps: int | None = None,
) -> Tensor | tuple[Tensor, list, list[float], list[int]]:
    """Generate action sequences via iterative ReMDM denoising.

    Args:
        model: Denoising model with forward signature
            ``(local_obs, global_obs, action_seq, t_discrete) -> dict``.
        local_obs: Local crop observations. Shape ``[B, 9, 9]``.
        global_obs: Global map observations. Shape ``[B, 21, 79]``.
        cfg: Config namespace with ``seq_len``, ``mask_token``,
            ``action_dim``, ``diffusion_steps_eval``, ``temperature``,
            ``top_p``, ``eta``, ``remask_strategy``, ``noise_schedule``.
        device: Torch device.
        physics_aware: If ``True``, soft-penalise hazardous cardinal actions
            by overriding their stored decoding probability to ``0.001`` so
            the ``conf`` strategy preferentially remasks them. Only checks
            the first ``_N_PHYSICS_CHECK`` positions.
        blind_global: If ``True``, zero out the global map observation
            (local-only ablation).
        return_analytics: If ``True``, also return per-step analytics as
            ``(seq, path_per_step, tracking_confidence, tracking_masked)``.
        num_steps: Override number of denoising steps (default uses
            ``cfg.diffusion_steps_eval``).

    Returns:
        When ``return_analytics=False`` (default): fully committed action
        sequence of shape ``[B, seq_len]``, int64, with no MASK tokens.

        When ``return_analytics=True``: tuple
        ``(seq, path_per_step, tracking_confidence, tracking_masked_count)``
        where ``path_per_step`` is a list of ``[seq_len]`` numpy arrays,
        ``tracking_confidence`` a list of per-step avg unmasked confidence
        floats, and ``tracking_masked_count`` a list of masked-token counts.
    """
    B = local_obs.shape[0]
    seq_len = cfg.seq_len
    mask_token = cfg.mask_token
    action_dim = cfg.action_dim
    K = num_steps if num_steps is not None else cfg.diffusion_steps_eval
    schedule_fn = get_schedule(cfg.noise_schedule)

    local_obs = local_obs.to(device)
    global_obs = global_obs.to(device)

    if blind_global:
        global_obs = torch.zeros_like(global_obs)

    # Pre-compute numpy local crops for physics checks (CPU, batch loop)
    local_np: np.ndarray | None = None  # [B, crop, crop]
    if physics_aware:
        local_np = local_obs.cpu().numpy()

    # Analytics buffers (only populated when return_analytics=True)
    path_per_step: list[np.ndarray] = []
    tracking_confidence: list[float] = []
    tracking_masked_count: list[int] = []

    # Start fully masked; psi stores the decoding probability at the step
    # each token was last unmasked (+inf while masked), per ReMDM Sec 4.1.
    seq = torch.full((B, seq_len), mask_token, dtype=torch.long, device=device)
    psi = torch.full((B, seq_len), float("inf"), device=device)

    # FIX-3 (ADJUDICATION B-3): ReMDM Algorithm 1 (Wang et al.). Masked
    # tokens unmask via independent Bernoulli draws from the approximate
    # posterior; committed tokens remask w.p. sigma from the Sec 4.1
    # schedule. Replaces the previous MaskGIT count-based unmasking and
    # its unsourced 10% min-keep floor. Shared pseudocode lines 8-12
    # (METHOD_PARITY 2.1); the craftax twin is sample_plan.
    for idx in range(K):
        t = (K - idx) / K
        s = (K - idx - 1) / K
        alpha_t = float(schedule_fn(torch.tensor(t)))
        alpha_s = float(schedule_fn(torch.tensor(s)))
        # Discrete conditioning bin for the learned timestep embedding
        # (B-9, free per MDLM Sec 3.5: time conditioning is optional).
        t_discrete = torch.full(
            (B,),
            min(int(t * cfg.num_diffusion_steps), cfg.num_diffusion_steps - 1),
            dtype=torch.long,
            device=device,
        )

        out = model(local_obs, global_obs, seq, t_discrete)
        logits = out["actions"]  # [B, seq_len, vocab]

        # Mask invalid action tokens (indices >= action_dim)
        logits[:, :, action_dim:] = float("-inf")

        logits = logits / cfg.temperature

        # Nucleus filtering (CH-1)
        logits = top_p_filter(logits, cfg.top_p)

        probs = F.softmax(logits, dim=-1)  # [B, seq_len, action_dim]
        preds = Categorical(probs=probs).sample()  # [B, seq_len]

        decode_prob = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)  # [B, seq_len]

        # Physics softener (unsourced engineering, default off; CH-6):
        # demote hazardous cardinal actions to decode_prob=0.001 so the
        # conf strategy preferentially remasks them.
        if physics_aware and local_np is not None:
            preds_np = preds.cpu().numpy()  # [B, seq_len]
            prob_override = decode_prob.clone()
            for b in range(B):
                crop_b = np.asarray(local_np[b])  # [crop, crop]
                for pos in range(min(_N_PHYSICS_CHECK, seq_len)):
                    action = int(preds_np[b, pos])
                    if _check_hazard(crop_b, action):
                        prob_override[b, pos] = _HAZARD_DECODE_PROB
            decode_prob = prob_override

        committed = seq != mask_token  # [B, seq_len]

        # Remasking probability sigma in [0, sigma_max] (ReMDM eq 7)
        sigma_max = min(1.0, (1.0 - alpha_s) / max(alpha_t, _SIGMA_DENOM_EPS))
        sigma = _compute_remask_prob(
            cfg.remask_strategy, cfg.eta, sigma_max, psi, committed
        )
        if not isinstance(sigma, Tensor):
            sigma = torch.full((B, seq_len), float(sigma), device=device)

        # Algorithm 1 posterior: masked tokens unmask w.p.
        # (alpha_s - (1 - sigma) alpha_t) / (1 - alpha_t)
        p_unmask = torch.clamp(
            (alpha_s - (1.0 - sigma) * alpha_t) / max(1.0 - alpha_t, _SIGMA_DENOM_EPS),
            0.0,
            1.0,
        )

        do_unmask = ~committed & (torch.rand(B, seq_len, device=device) < p_unmask)
        do_remask = committed & (torch.rand(B, seq_len, device=device) < sigma)

        seq = torch.where(do_unmask, preds, seq)
        seq = torch.where(do_remask, mask_token, seq)
        psi = torch.where(do_unmask, decode_prob, psi)
        psi = torch.where(do_remask, torch.full_like(psi, float("inf")), psi)

        # Analytics tracking
        if return_analytics:
            path_per_step.append(seq[0].cpu().numpy().copy())
            still_masked = seq[0] == mask_token
            unmasked_prob = psi[0][~still_masked]
            avg_conf = unmasked_prob.mean().item() if unmasked_prob.numel() > 0 else 0.0
            tracking_confidence.append(avg_conf)
            tracking_masked_count.append(int(still_masked.sum().item()))

    # Final greedy cleanup for any remaining masks (as in the craftax
    # twin); replaces the previous commit-all step and assertion.
    still_masked = seq == mask_token
    if still_masked.any():
        t_zero = torch.zeros(B, dtype=torch.long, device=device)
        out = model(local_obs, global_obs, seq, t_zero)
        logits = out["actions"]
        logits[:, :, action_dim:] = float("-inf")
        seq = torch.where(still_masked, logits.argmax(dim=-1), seq)

    if return_analytics:
        return seq, path_per_step, tracking_confidence, tracking_masked_count
    return seq


@torch.no_grad()
def greedy_sample(
    model: torch.nn.Module,
    local_obs: Tensor,
    global_obs: Tensor,
    cfg: SimpleNamespace,
    device: torch.device | str,
    blind_global: bool = False,
    num_steps: int | None = None,
) -> Tensor:
    """Greedy (argmax) MaskGIT sampling — no temperature, top-K, or remasking.

    Used by ``DataCollector`` during DAgger for deterministic rollouts,
    matching the reference ``run_model_episode`` behaviour.

    Args:
        model: Denoising model.
        local_obs: Shape ``[B, 9, 9]``.
        global_obs: Shape ``[B, 21, 79]``.
        cfg: Config namespace.
        device: Torch device.
        blind_global: Zero out global map (local-only ablation).

    Returns:
        Fully committed action sequence ``[B, seq_len]``, int64.
    """
    B = local_obs.shape[0]
    seq_len = cfg.seq_len
    mask_token = cfg.mask_token
    action_dim = cfg.action_dim
    K = num_steps if num_steps is not None else cfg.diffusion_steps_eval

    local_obs = local_obs.to(device)
    global_obs = global_obs.to(device)
    if blind_global:
        global_obs = torch.zeros_like(global_obs)

    seq = torch.full(
        (B, seq_len),
        mask_token,
        dtype=torch.long,
        device=device,
    )

    for k in range(1, K + 1):
        ratio = k / K
        t_discrete = torch.full(
            (B,),
            int(cfg.num_diffusion_steps * (1.0 - ratio)),
            dtype=torch.long,
            device=device,
        )

        out = model(local_obs, global_obs, seq, t_discrete)
        logits = out["actions"]  # [B, seq_len, vocab]

        # Mask invalid action tokens
        logits[:, :, action_dim:] = float("-inf")

        # Greedy: argmax over softmax (no temperature, no top-K)
        probs = F.softmax(logits, dim=-1)  # [B, seq_len, action_dim]
        confidences, preds = probs.max(dim=-1)  # [B, seq_len] each

        # MaskGIT progressive unmasking by confidence
        num_to_unmask = max(1, int(seq_len * ratio))
        is_masked = seq == mask_token  # [B, seq_len]

        # Score only masked positions for unmasking
        scores = confidences.clone()
        scores[~is_masked] = -1.0
        _, topk_idx = scores.topk(num_to_unmask, dim=-1)

        unmask_mask = torch.zeros_like(seq, dtype=torch.bool)
        unmask_mask.scatter_(1, topk_idx, True)
        unmask_mask = unmask_mask & is_masked

        seq = torch.where(unmask_mask, preds, seq)

        # No remasking in greedy mode

    # Force-commit any remaining masked tokens
    still_masked = seq == mask_token
    if still_masked.any():
        t_zero = torch.zeros(B, dtype=torch.long, device=device)
        out = model(local_obs, global_obs, seq, t_zero)
        logits = out["actions"]
        logits[:, :, action_dim:] = float("-inf")
        preds = logits.argmax(dim=-1)
        seq = torch.where(still_masked, preds, seq)

    return seq
