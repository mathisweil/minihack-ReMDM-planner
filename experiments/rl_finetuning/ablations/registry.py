"""AblationSpec registry: all 25 RL fine-tuning ablations.

Usage::

    from experiments.rl_finetuning.ablations.registry import REGISTRY, AblationSpec
    spec = REGISTRY["ewc"]

Adapted from Craftax JAX implementation to MiniHack PyTorch.
Key differences:
- OptimizerFactory signature: ``(cfg, model) -> torch.optim.Optimizer``
- Frozen-path fragments use PyTorch parameter names
  (e.g. ``transformer.layers.0.self_attn.in_proj_weight``)
- LoRA handled via ``apply_lora_to_model`` + ``make_optimizer_lora``
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.nn as nn

from experiments.rl_finetuning.ablations.losses import (
    LossFn,
    make_loss_advantage_clip,
    make_loss_baseline,
    make_loss_bc_wins,
    make_loss_entropy_bonus,
    make_loss_ewc,
    make_loss_frozen_backbone,
    make_loss_gradient_surgery,
    make_loss_kl_penalty,
    make_loss_low_t,
    make_loss_mixed_replay,
    make_loss_normalized_adv,
    make_loss_param_isolation,
    make_loss_reward_quality,
    make_loss_t_curriculum,
    make_loss_trust_region_kl,
)
from experiments.rl_finetuning.ablations.optimizers import (
    FROZEN_BACKBONE,
    FROZEN_EXCEPT_ATTENTION,
    FROZEN_EXCEPT_FFN,
    make_optimizer_frozen,
    make_optimizer_llrd,
    make_optimizer_standard,
)

# Optimizer factory signature: (cfg, model) -> Optimizer
OptimizerFactory = Callable[[SimpleNamespace, nn.Module], "torch.optim.Optimizer"]

# Loss factory signature: (ctx, **extra) -> LossFn
LossFactory = Callable[..., LossFn]


@dataclass
class AblationSpec:
    """Specification for a single RL fine-tuning ablation.

    Args:
        name: Short identifier used in CLI and output paths.
        group: Ablation group: ``"Baseline"``, ``"A"``, ``"B"``,
               ``"C"``, or ``"D"``.
        description: One-line human-readable description.
        hypothesis: What failure mode this ablation tests.
        loss_factory: Callable ``(ctx, **extra) -> LossFn``.
        optimizer_factory: Callable ``(cfg, model) -> Optimizer``.
        use_lora: If True, apply LoRA instead of full fine-tuning.
        wins_only: If True, pre-filter batch to win windows.
        gradient_surgery: If True, apply PCGrad to RL vs BC gradients.
        mixed_replay: If True, mix offline buffer into each batch.
        t_curriculum: If True, anneal t range over training.
        reward_filtering: If True, discard low-return windows.
        running_stats: If True, normalise advantages with running EMA.
        action_diversity_filter: If True, discard degenerate plans.
        reward_model_weighting: If True, weight with learned reward model.
    """

    name: str
    group: str
    description: str
    hypothesis: str
    loss_factory: LossFactory
    optimizer_factory: OptimizerFactory = field(
        default_factory=lambda: make_optimizer_standard,
    )
    use_lora: bool = False
    wins_only: bool = False
    gradient_surgery: bool = False
    mixed_replay: bool = False
    t_curriculum: bool = False
    reward_filtering: bool = False
    running_stats: bool = False
    action_diversity_filter: bool = False
    reward_model_weighting: bool = False


def _std_opt(cfg: SimpleNamespace, model: nn.Module) -> torch.optim.Optimizer:
    """Standard AdamW for all parameters."""
    return make_optimizer_standard(cfg, model)


def _llrd_opt(cfg: SimpleNamespace, model: nn.Module) -> torch.optim.Optimizer:
    """AdamW with layer-wise learning rate decay."""
    return make_optimizer_llrd(cfg, model)


def _frozen_backbone_opt(
    cfg: SimpleNamespace,
    model: nn.Module,
) -> torch.optim.Optimizer:
    """Freeze backbone, train only the action head."""
    return make_optimizer_frozen(cfg, model, FROZEN_BACKBONE)


def _head_only_opt(
    cfg: SimpleNamespace,
    model: nn.Module,
) -> torch.optim.Optimizer:
    """Freeze everything except the action head."""
    return make_optimizer_frozen(cfg, model, FROZEN_BACKBONE)


def _attention_only_opt(
    cfg: SimpleNamespace,
    model: nn.Module,
) -> torch.optim.Optimizer:
    """Freeze everything except attention sublayers."""
    return make_optimizer_frozen(cfg, model, FROZEN_EXCEPT_ATTENTION)


def _ffn_only_opt(
    cfg: SimpleNamespace,
    model: nn.Module,
) -> torch.optim.Optimizer:
    """Freeze everything except FFN sublayers."""
    return make_optimizer_frozen(cfg, model, FROZEN_EXCEPT_FFN)


def _layer_ablation_top_n_opt(n: int) -> OptimizerFactory:
    """Factory: freeze all transformer layers except the top *n*."""

    def _opt(
        cfg: SimpleNamespace,
        model: nn.Module,
    ) -> torch.optim.Optimizer:
        n_layers = getattr(cfg, "n_layer", 4)
        frozen: list[str] = [
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
        ]
        for i in range(n_layers - n):
            frozen.append(f"transformer.layers.{i}.")
        return make_optimizer_frozen(cfg, model, frozen)

    return _opt


REGISTRY: dict[str, AblationSpec] = {
    # -- Baseline ----------------------------------------------------------
    "baseline_rl": AblationSpec(
        name="baseline_rl",
        group="Baseline",
        description="Return-weighted ELBO -- no modifications",
        hypothesis=("Diagnoses whether the RL signal alone causes collapse"),
        loss_factory=make_loss_baseline,
        optimizer_factory=_std_opt,
    ),
    # -- Group A: Regularisation / Constraint Methods ----------------------
    "kl_penalty": AblationSpec(
        name="kl_penalty",
        group="A",
        description="Return-weighted ELBO + soft KL penalty vs pretrained",
        hypothesis=(
            "If this helps: catastrophic forgetting is the primary cause; "
            "soft regularisation suffices"
        ),
        loss_factory=make_loss_kl_penalty,
        optimizer_factory=_std_opt,
    ),
    "ewc": AblationSpec(
        name="ewc",
        group="A",
        description=(
            "ELBO + Elastic Weight Consolidation (Fisher diagonal regularisation)"
        ),
        hypothesis=(
            "If EWC helps: forgetting pretrained representations is the proximate cause"
        ),
        loss_factory=make_loss_ewc,
        optimizer_factory=_std_opt,
    ),
    "llrd": AblationSpec(
        name="llrd",
        group="A",
        description="Baseline ELBO with Layer-wise Learning Rate Decay",
        hypothesis=(
            "If LLRD helps: deep gradient flow into early layers "
            "corrupts representations"
        ),
        loss_factory=make_loss_baseline,
        optimizer_factory=_llrd_opt,
    ),
    "lora": AblationSpec(
        name="lora",
        group="A",
        description=(
            "Baseline ELBO with LoRA adaptation (rank-r attention projections only)"
        ),
        hypothesis=(
            "If LoRA works: too many unconstrained degrees of freedom cause collapse"
        ),
        loss_factory=make_loss_baseline,
        optimizer_factory=_std_opt,
        use_lora=True,
    ),
    "mixed_replay": AblationSpec(
        name="mixed_replay",
        group="A",
        description=("Baseline ELBO with offline data mixed into online batches"),
        hypothesis=(
            "If mixed replay helps: online data distribution alone is too corrupted"
        ),
        loss_factory=make_loss_mixed_replay,
        optimizer_factory=_std_opt,
        mixed_replay=True,
    ),
    "trust_region_kl": AblationSpec(
        name="trust_region_kl",
        group="A",
        description=("Baseline ELBO + hard KL trust region via quadratic barrier"),
        hypothesis=(
            "If hard constraint helps: soft KL is insufficient -- "
            "a hard boundary is needed"
        ),
        loss_factory=make_loss_trust_region_kl,
        optimizer_factory=_std_opt,
    ),
    # -- Group B: Training Signal Modifications ----------------------------
    "t_curriculum": AblationSpec(
        name="t_curriculum",
        group="B",
        description=("ELBO with t range annealed from high-t to low-t over training"),
        hypothesis=("If curriculum helps: ordering of learning signals matters"),
        loss_factory=make_loss_t_curriculum,
        optimizer_factory=_std_opt,
        t_curriculum=True,
    ),
    "entropy_bonus": AblationSpec(
        name="entropy_bonus",
        group="B",
        description=("Baseline ELBO minus entropy bonus (encourages action diversity)"),
        hypothesis=(
            "If entropy bonus helps: collapse is mode-collapse; not a gradient problem"
        ),
        loss_factory=make_loss_entropy_bonus,
        optimizer_factory=_std_opt,
    ),
    "gradient_surgery": AblationSpec(
        name="gradient_surgery",
        group="B",
        description=(
            "PCGrad: RL gradient projected to remove conflict with BC gradient"
        ),
        hypothesis=(
            "If PCGrad helps: gradients are conflicting and resolvable by projection"
        ),
        loss_factory=make_loss_gradient_surgery,
        optimizer_factory=_std_opt,
        gradient_surgery=True,
    ),
    "advantage_clip": AblationSpec(
        name="advantage_clip",
        group="B",
        description=(
            "Baseline ELBO with PPO-style advantage clipping to [1-eps, 1+eps]"
        ),
        hypothesis=(
            "If clipping helps: large advantage magnitudes destabilise training"
        ),
        loss_factory=make_loss_advantage_clip,
        optimizer_factory=_std_opt,
    ),
    "normalized_adv": AblationSpec(
        name="normalized_adv",
        group="B",
        description=(
            "Baseline ELBO with (A - mean) / (std + eps) advantage normalisation"
        ),
        hypothesis=(
            "If std normalisation helps: simple mean normalisation is too loose"
        ),
        loss_factory=make_loss_normalized_adv,
        optimizer_factory=_std_opt,
    ),
    "bc_wins": AblationSpec(
        name="bc_wins",
        group="B",
        description=("Uniform ELBO on win windows only (no advantage weighting)"),
        hypothesis=("If BC on wins helps: the return weighting is the specific cause"),
        loss_factory=make_loss_bc_wins,
        optimizer_factory=_std_opt,
        wins_only=True,
    ),
    "low_t": AblationSpec(
        name="low_t",
        group="B",
        description=("Return-weighted ELBO restricted to low-t (fine-detail) regime"),
        hypothesis=("If low-t helps: high-t (coarse-structure) gradients are biased"),
        loss_factory=make_loss_low_t,
        optimizer_factory=_std_opt,
    ),
    # -- Group C: Architecture / Parameter Isolation -----------------------
    "frozen_backbone": AblationSpec(
        name="frozen_backbone",
        group="C",
        description=(
            "Baseline ELBO with all params frozen except the final output head"
        ),
        hypothesis=(
            "If frozen backbone helps: deep gradient flow into backbone causes collapse"
        ),
        loss_factory=make_loss_frozen_backbone,
        optimizer_factory=_frozen_backbone_opt,
    ),
    "head_only": AblationSpec(
        name="head_only",
        group="C",
        description=("Baseline ELBO updating only the final linear projection"),
        hypothesis=(
            "If head-only works: backbone representations are fine; "
            "only decision boundary needs updating"
        ),
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_head_only_opt,
    ),
    "attention_only": AblationSpec(
        name="attention_only",
        group="C",
        description=(
            "Baseline ELBO updating only attention weights (Q/K/V/O); FFN frozen"
        ),
        hypothesis=(
            "If attention-only works: model needs routing updates, not feature updates"
        ),
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_attention_only_opt,
    ),
    "ffn_only": AblationSpec(
        name="ffn_only",
        group="C",
        description=("Baseline ELBO updating only FFN layers; attention frozen"),
        hypothesis=(
            "If FFN-only works: stored knowledge (FFN as memory) "
            "needs updating; not attention"
        ),
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_ffn_only_opt,
    ),
    "layer_ablation_top1": AblationSpec(
        name="layer_ablation_top1",
        group="C",
        description=("Baseline ELBO updating only the top-1 transformer block"),
        hypothesis=(
            "Minimal unfrozen depth needed; collapse depth correlates "
            "with gradient flow depth"
        ),
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_layer_ablation_top_n_opt(1),
    ),
    "layer_ablation_top2": AblationSpec(
        name="layer_ablation_top2",
        group="C",
        description=("Baseline ELBO updating only the top-2 transformer blocks"),
        hypothesis=(
            "Minimal unfrozen depth needed; collapse depth correlates "
            "with gradient flow depth"
        ),
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_layer_ablation_top_n_opt(2),
    ),
    "layer_ablation_top3": AblationSpec(
        name="layer_ablation_top3",
        group="C",
        description=("Baseline ELBO updating only the top-3 transformer blocks"),
        hypothesis=(
            "Minimal unfrozen depth needed; collapse depth correlates "
            "with gradient flow depth"
        ),
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_layer_ablation_top_n_opt(3),
    ),
    # -- Group D: Reward / Data Quality ------------------------------------
    "reward_filtering": AblationSpec(
        name="reward_filtering",
        group="D",
        description=(
            "Baseline ELBO trained only on top-75th-percentile return windows"
        ),
        hypothesis=("If filtering helps: noisy/low-return data poisons gradients"),
        loss_factory=make_loss_reward_quality,
        optimizer_factory=_std_opt,
        reward_filtering=True,
    ),
    "running_stats": AblationSpec(
        name="running_stats",
        group="D",
        description=(
            "Baseline ELBO with EMA running mean/std for advantage normalisation"
        ),
        hypothesis=(
            "If running stats help: batch normalisation is too noisy for small batches"
        ),
        loss_factory=make_loss_reward_quality,
        optimizer_factory=_std_opt,
        running_stats=True,
    ),
    "action_diversity": AblationSpec(
        name="action_diversity",
        group="D",
        description=("Baseline ELBO with degenerate (all-same-action) plans discarded"),
        hypothesis=("If diversity filtering helps: degenerate plans corrupt training"),
        loss_factory=make_loss_reward_quality,
        optimizer_factory=_std_opt,
        action_diversity_filter=True,
    ),
    "reward_model": AblationSpec(
        name="reward_model",
        group="D",
        description=(
            "Baseline ELBO with advantages re-weighted by a learned MLP reward model"
        ),
        hypothesis=(
            "If reward model helps: raw returns are too sparse; "
            "learned model smooths signal"
        ),
        loss_factory=make_loss_reward_quality,
        optimizer_factory=_std_opt,
        reward_model_weighting=True,
    ),
}
