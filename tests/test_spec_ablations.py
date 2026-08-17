"""Per-ablation behavioural spec tests (step 8).

One deterministic behavioural test per ablation mechanism of
research/spec-ablations.md §2, with expected values from the pinned
sources (SPG, Jaques 2017, Kirkpatrick 2017, Sun 2019, Hu 2021,
Yu 2020, Kim 2025) or derivations written in the docstrings. The
group-C trainable-set tests reuse the step-7 reproduction method
(requires_grad partition after the registry optimizer factory).
xfail(strict=True) marks canonical-vs-implemented disagreements from
the defect register or the step-8 findings list.

The craftax twin file carries the same mechanisms in its framework.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.rl_finetuning.ablations import registry
from experiments.rl_finetuning.ablations.losses import (
    LossContext,
    _core_loss,
    _ewc_penalty,
    make_loss_advantage_clip,
    make_loss_baseline,
    make_loss_bc_wins,
    make_loss_entropy_bonus,
    make_loss_ewc,
    make_loss_kl_penalty,
    make_loss_low_t,
    make_loss_normalized_adv,
    make_loss_t_curriculum,
    make_loss_trust_region_kl,
)
from experiments.rl_finetuning.ablations.optimizers import (
    apply_lora_to_model,
    gradient_surgery,
    make_optimizer_frozen,
    make_optimizer_llrd,
)
from experiments.rl_finetuning.ablations.training import (
    RewardModel,
    _train_reward_model,
    compute_advantages,
)
from src.diffusion.schedules import cosine_schedule, get_schedule
from src.models.denoiser import make_model

V, L = 4, 8  # real action vocabulary and window length for loss tests
B = 4


def _model_cfg(**over) -> SimpleNamespace:
    base = {
        "action_dim": V, "n_embd": 32, "n_head": 2, "n_layer": 2,
        "seq_len": L, "num_diffusion_steps": 10, "use_global_stream": True,
        "dropout": 0.0, "global_gate_init": -3.0, "n_global_tokens": 4,
        "lr": 1e-3, "weight_decay": 0.0,
    }
    base.update(over)
    return SimpleNamespace(**base)


def _cfg(**over) -> SimpleNamespace:
    base = {
        "mask_token": V, "pad_token": V + 1, "num_diffusion_steps": 1000,
        "aux_loss_weight": 0.0, "_schedule_fn": get_schedule("linear"),
    }
    base.update(over)
    return SimpleNamespace(**base)


class _FixedLogitsModel(nn.Module):
    """Stub returning constant logits (a [V] tensor) at every position."""

    def __init__(self, logits: torch.Tensor, seq_len: int = L):
        super().__init__()
        self.logits = logits
        self.seq_len = seq_len
        self.recorded_t: list[np.ndarray] = []

    def forward(self, local_obs, global_obs, seq, t_discrete):
        self.recorded_t.append(np.asarray(t_discrete))
        out = self.logits.expand(seq.shape[0], self.seq_len, -1).clone()
        return {"actions": out, "goal_pred": torch.zeros(seq.shape[0], 2)}


_UNIFORM = torch.zeros(V)
_P_LOGITS = torch.log(torch.tensor([0.7, 0.1, 0.1, 0.1]))
_P1_LOGITS = torch.log(torch.tensor([0.55, 0.15, 0.15, 0.15]))
_KL_PQ = 0.7 * math.log(2.8) + 0.3 * math.log(0.4)
_KL_P1Q = 0.55 * math.log(2.2) + 0.45 * math.log(0.6)
_KL_L = 512  # long window: P(a row has no masked position) ~ 1/513


def _batch(b=B, seq_len=L):
    g = torch.Generator().manual_seed(1)
    x0 = torch.randint(0, V, (b, seq_len), generator=g)
    local = torch.zeros(b, 9, 9, dtype=torch.long)
    glob = torch.zeros(b, 21, 79, dtype=torch.long)
    return local, glob, x0


def _ctx(model=None, cfg=None):
    return LossContext(ref_model=model, schedule_fn=get_schedule("linear"),
                       cfg=cfg or _cfg())


# ---------------------------------------------------------------------------
# baseline_rl and the advantage pipeline (spec-ablations §2 baseline row)
# ---------------------------------------------------------------------------


def test_compute_advantages_standard_branch_closed_form():
    """weight = clip(max(R,0)/(mean(max(R,0))+eps), 0.1, 5.0).

    Source: spec-ablations §2 baseline_rl effective params. Derivation:
    returns [0,1,2,3] -> clipped mean 1.5 -> raw weights [0, 2/3, 4/3, 2]
    -> floor lifts the first to 0.1. Same numbers as the craftax twin.
    """
    adv, mean, _ = compute_advantages(
        torch.tensor([0.0, 1.0, 2.0, 3.0]), 0.1, 5.0, wins_only=False,
        win_thresh=0.5, use_running_stats=False, ema_decay=0.99,
        running_mean=0.0, running_std=1.0,
    )
    assert np.allclose(adv.numpy(), [0.1, 2 / 3, 4 / 3, 2.0], atol=1e-4)
    assert mean == pytest.approx(1.5, abs=1e-6)


def test_compute_advantages_wins_only_branch_is_a_binary_mask():
    """wins_only: adv = 1[return > win_thresh] (spec-ablations §2
    bc_wins row, win_threshold 0.5)."""
    adv, _, _ = compute_advantages(
        torch.tensor([0.0, 0.5, 0.51, 3.0]), 0.1, 5.0, wins_only=True,
        win_thresh=0.5, use_running_stats=False, ema_decay=0.99,
        running_mean=0.0, running_std=1.0,
    )
    assert adv.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_compute_advantages_running_stats_branch_closed_form():
    """running_stats: EMA of batch mean/std, adv = clip((w-mu)/sigma + 1,
    0.1, 5.0) (spec-ablations §2 running_stats row).

    Derivation with ema_decay d=0.5, prior mean 0 / std 1, batch
    [0,1,2,3]: new_mean = 0.75; the batch std is the POPULATION std
    sqrt(1.25) = 1.11803 (spec-ablations §2 step-9 amendment resolving
    parity finding S8-7; same numbers as the craftax twin), so
    new_std = 0.5 + 0.5*1.11803 = 1.05902 and
    adv_i = clip((w_i-0.75)/1.05902 + 1, 0.1, 5).
    """
    adv, mean, std = compute_advantages(
        torch.tensor([0.0, 1.0, 2.0, 3.0]), 0.1, 5.0, wins_only=False,
        win_thresh=0.5, use_running_stats=True, ema_decay=0.5,
        running_mean=0.0, running_std=1.0,
    )
    new_std = 0.5 * 1.0 + 0.5 * (math.sqrt(1.25) + 1e-8)
    expected = np.clip((np.array([0, 1, 2, 3.0]) - 0.75) / new_std + 1.0, 0.1, 5.0)
    assert np.allclose(adv.numpy(), expected, atol=1e-4)
    assert mean == pytest.approx(0.75, abs=1e-6)
    assert std == pytest.approx(new_std, abs=1e-5)


def test_baseline_loss_is_linear_in_the_advantages():
    """The advantage weight is a per-sample multiplier on the loss
    (SPG eq (5) positive branch), so doubling every advantage doubles
    the loss under the same RNG state."""
    model = _FixedLogitsModel(_UNIFORM)
    cfg = _cfg()
    local, glob, x0 = _batch()
    loss_fn = make_loss_baseline(_ctx(cfg=cfg))
    adv = torch.tensor([0.5, 1.0, 1.5, 2.0])
    torch.manual_seed(0)
    l1 = float(loss_fn(model, local, glob, x0, adv, cfg, "cpu"))
    torch.manual_seed(0)
    l2 = float(loss_fn(model, local, glob, x0, 2.0 * adv, cfg, "cpu"))
    assert l2 == pytest.approx(2.0 * l1, rel=1e-6)


# ---------------------------------------------------------------------------
# bc_wins (defect §8.5)
# ---------------------------------------------------------------------------


def test_bc_wins_averages_uniformly_over_winning_windows():
    """Canonical bc_wins ('Uniform ELBO on win windows', win = return >
    win_threshold, spec-ablations §2; was defect §8.5): a batch with no
    winning window carries no action-loss signal (0 with the auxiliary
    goal loss weighted 0), and an all-winning batch reduces to the
    plain uniform ELBO. Win masks come from the pipeline's own
    compute_advantages(wins_only=True).
    """
    model = _FixedLogitsModel(_UNIFORM)
    cfg = _cfg()
    local, glob, x0 = _batch()

    def mask(returns):
        m, _, _ = compute_advantages(
            torch.tensor(returns), 0.1, 5.0, wins_only=True, win_thresh=0.5,
            use_running_stats=False, ema_decay=0.99,
            running_mean=0.0, running_std=1.0,
        )
        return m

    torch.manual_seed(0)
    lose = float(
        make_loss_bc_wins(_ctx(cfg=cfg))(
            model, local, glob, x0, mask([0.0, 0.1, 0.2, 0.3]), cfg, "cpu"
        )
    )
    assert lose == 0.0
    torch.manual_seed(0)
    all_wins = float(
        make_loss_bc_wins(_ctx(cfg=cfg))(
            model, local, glob, x0, mask([1.0, 2.0, 3.0, 4.0]), cfg, "cpu"
        )
    )
    torch.manual_seed(0)
    uniform = float(
        make_loss_baseline(_ctx(cfg=cfg))(
            model, local, glob, x0, torch.ones(B), cfg, "cpu"
        )
    )
    assert all_wins == pytest.approx(uniform, abs=0.0)


# ---------------------------------------------------------------------------
# advantage_clip / normalized_adv (spec-ablations §2)
# ---------------------------------------------------------------------------


def test_advantage_clip_clips_the_weight_to_the_documented_band():
    """advantage_clip clips the return-weight to [1-eps, 1+eps]
    (eps=0.2): equals the baseline loss on manually clipped advantages
    under the same RNG state (same construction as the craftax twin)."""
    model = _FixedLogitsModel(_UNIFORM)
    cfg = _cfg(adv_clip_eps=0.2)
    local, glob, x0 = _batch()
    adv = torch.tensor([10.0, 0.0, 1.0, 1.1])
    torch.manual_seed(0)
    got = float(
        make_loss_advantage_clip(_ctx(cfg=cfg))(model, local, glob, x0, adv, cfg, "cpu")
    )
    torch.manual_seed(0)
    want = float(
        make_loss_baseline(_ctx(cfg=cfg))(
            model, local, glob, x0, adv.clamp(0.8, 1.2), cfg, "cpu"
        )
    )
    assert got == pytest.approx(want, abs=0.0)


def test_normalized_adv_standardises_over_the_batch():
    """normalized_adv applies (A - mean)/(std + 1e-8) over the batch
    (What Matters C67; spec-ablations §2)."""
    model = _FixedLogitsModel(_UNIFORM)
    cfg = _cfg()
    local, glob, x0 = _batch()
    adv = torch.tensor([10.0, 0.0, 1.0, 1.1])
    norm = (adv - adv.mean()) / (adv.std() + 1e-8)
    torch.manual_seed(0)
    got = float(
        make_loss_normalized_adv(_ctx(cfg=cfg))(model, local, glob, x0, adv, cfg, "cpu")
    )
    torch.manual_seed(0)
    want = float(
        make_loss_baseline(_ctx(cfg=cfg))(model, local, glob, x0, norm, cfg, "cpu")
    )
    assert got == pytest.approx(want, abs=0.0)


# ---------------------------------------------------------------------------
# kl_penalty / trust_region_kl (Jaques 2017 family)
# ---------------------------------------------------------------------------


def test_kl_penalty_adds_coef_times_the_closed_form_kl():
    """kl_penalty adds kl_coef * KL(current || pretrained) on masked
    positions (Jaques 2017 eqs (2)-(4)). Derivation: constant
    per-position logits give masked-position KL = KL(p||q) =
    0.7 ln 2.8 + 0.3 ln 0.4 = 0.445846 (same numbers as the craftax
    twin). The coefficient difference 0.3-0.1 isolates 0.2*KL.
    """
    cur = _FixedLogitsModel(_P_LOGITS, seq_len=_KL_L)
    ref = _FixedLogitsModel(_UNIFORM, seq_len=_KL_L)
    local, glob, x0 = _batch(seq_len=_KL_L)
    losses = {}
    for coef in (0.1, 0.3):
        cfg = _cfg(kl_coef=coef)
        torch.manual_seed(0)
        losses[coef] = float(
            make_loss_kl_penalty(_ctx(ref, cfg))(
                cur, local, glob, x0, torch.ones(B), cfg, "cpu"
            )
        )
    assert (losses[0.3] - losses[0.1]) / 0.2 == pytest.approx(_KL_PQ, rel=1e-3)


def test_trust_region_barrier_is_zero_below_and_quadratic_above():
    """trust_region_kl adds a quadratic barrier c*max(KL-delta,0)^2
    (delta=0.05, spec-ablations §2). Below threshold (KL=0) the barrier
    is exactly 0; above it, barriers at KL levels 0.445846 and 0.203781
    satisfy the quadratic ratio (0.395846/0.153781)^2 = 6.6262 (same
    derivation as the craftax twin; pins the form, not the project c).
    """
    ref = _FixedLogitsModel(_UNIFORM, seq_len=_KL_L)
    local, glob, x0 = _batch(seq_len=_KL_L)
    cfg = _cfg(trust_region_kl=0.05)

    def barrier(cur_logits):
        cur = _FixedLogitsModel(cur_logits, seq_len=_KL_L)
        torch.manual_seed(0)
        total = float(
            make_loss_trust_region_kl(_ctx(ref, cfg))(
                cur, local, glob, x0, torch.ones(B), cfg, "cpu"
            )
        )
        torch.manual_seed(0)
        rl = float(
            make_loss_baseline(_ctx(ref, cfg))(
                cur, local, glob, x0, torch.ones(B), cfg, "cpu"
            )
        )
        return total - rl

    assert barrier(_UNIFORM) == pytest.approx(0.0, abs=1e-6)
    b0, b1 = barrier(_P_LOGITS), barrier(_P1_LOGITS)
    assert b0 > 0 and b1 > 0
    want_ratio = ((_KL_PQ - 0.05) / (_KL_P1Q - 0.05)) ** 2
    assert b0 / b1 == pytest.approx(want_ratio, rel=2e-2)


# ---------------------------------------------------------------------------
# ewc (Kirkpatrick 2017 eq (3), lambda-reparameterised per SOURCES.md)
# ---------------------------------------------------------------------------


def test_ewc_penalty_closed_form_and_factory_scaling():
    """EWC adds lambda * sum_i F_i (theta_i - theta*_i)^2 (Kirkpatrick
    2017 eq (3); lambda folds the paper's 1/2, documented deviation).

    Derivation: a 2-parameter linear layer with theta=[3,5],
    theta*=[1,1], F=[1,2] gives penalty 1*4 + 2*16 = 36; with
    ewc_lambda=100 the factory loss exceeds the same-seed baseline
    loss by exactly 3600.
    """

    class _Tiny(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(w))

        def forward(self, local_obs, global_obs, seq, t):
            out = torch.zeros(seq.shape[0], L, V)
            return {"actions": out, "goal_pred": torch.zeros(seq.shape[0], 2)}

    theta, ref = _Tiny([3.0, 5.0]), _Tiny([1.0, 1.0])
    fisher = {"w": torch.tensor([1.0, 2.0])}
    assert float(_ewc_penalty(fisher, theta, ref)) == pytest.approx(36.0)

    cfg = _cfg(ewc_lambda=100.0)
    local, glob, x0 = _batch()
    torch.manual_seed(0)
    got = float(
        make_loss_ewc(_ctx(ref, cfg), fisher)(
            theta, local, glob, x0, torch.ones(B), cfg, "cpu"
        )
    )
    torch.manual_seed(0)
    rl = float(_core_loss(theta, local, glob, x0, torch.ones(B), cfg, "cpu"))
    assert got - rl == pytest.approx(3600.0, rel=1e-5)


# ---------------------------------------------------------------------------
# entropy_bonus (standard tier, cf. Mnih 2016)
# ---------------------------------------------------------------------------


def test_entropy_bonus_subtracts_coef_times_the_closed_form_entropy():
    """entropy_bonus subtracts entropy_coef * H(p_theta) on masked
    positions. Derivation: constant p=[0.7,0.1,0.1,0.1] gives
    H = -(0.7 ln 0.7 + 0.3 ln 0.1) = 0.940448 (same numbers as the
    craftax twin); the coefficient difference isolates -0.02*H.
    """
    cur = _FixedLogitsModel(_P_LOGITS, seq_len=_KL_L)
    local, glob, x0 = _batch(seq_len=_KL_L)
    losses = {}
    for coef in (0.01, 0.03):
        cfg = _cfg(entropy_coef=coef)
        torch.manual_seed(0)
        losses[coef] = float(
            make_loss_entropy_bonus(_ctx(cfg=cfg))(
                cur, local, glob, x0, torch.ones(B), cfg, "cpu"
            )
        )
    entropy = -(0.7 * math.log(0.7) + 0.3 * math.log(0.1))
    assert (losses[0.01] - losses[0.03]) / 0.02 == pytest.approx(entropy, rel=1e-3)


# ---------------------------------------------------------------------------
# low_t / t_curriculum (Kim 2025; spec-ablations §2)
# ---------------------------------------------------------------------------


def test_low_t_restricts_sampling_to_the_low_noise_regime():
    """low_t trains only on t in [eps, t_max_low=0.2]: every discrete
    timestep handed to the model is <= 0.2 * num_diffusion_steps."""
    model = _FixedLogitsModel(_UNIFORM)
    cfg = _cfg(t_max_low=0.2)
    local, glob, x0 = _batch(b=64)
    torch.manual_seed(0)
    make_loss_low_t(_ctx(cfg=cfg))(model, local, glob, x0, None, cfg, "cpu")
    t = np.concatenate(model.recorded_t)
    assert t.max() <= 200 and t.min() >= 0


def test_t_curriculum_anneals_high_noise_to_low_noise():
    """t_curriculum anneals the t window from [0.8, 1.0] to [eps, 0.2]
    linearly over 200 iterations (Kim 2025, simplified linear anneal;
    t_start=0.8, t_end=0.2, steps=200 per spec-ablations §1.6).

    Expected discrete windows (num_diffusion_steps=1000): iter 0 ->
    [800, 999]; iter 100 -> [400, 600]; iter >= 200 -> [0, 200].
    """
    for it, (lo, hi) in [(0, (800, 999)), (100, (400, 600)), (200, (0, 200))]:
        model = _FixedLogitsModel(_UNIFORM)
        cfg = _cfg(t_curriculum_start=0.8, t_curriculum_end=0.2,
                   t_curriculum_steps=200, _current_iter=it)
        local, glob, x0 = _batch(b=64)
        torch.manual_seed(0)
        make_loss_t_curriculum(_ctx(cfg=cfg))(model, local, glob, x0, None, cfg, "cpu")
        t = np.concatenate(model.recorded_t)
        assert t.min() >= lo, (it, t.min(), lo)
        assert t.max() <= hi, (it, t.max(), hi)


# ---------------------------------------------------------------------------
# reward_filtering / action_diversity (spec-ablations §2, step-9 seams)
# ---------------------------------------------------------------------------


def test_window_returns_are_per_window_not_per_episode():
    """A window's return is the reward sum over exactly the actions it
    trains on (author decision 2026-08-16, PARITY "Ablation-suite data
    source and return definition"), not the episode total broadcast to
    every window - two windows through different states must not be
    credited alike. The craftax twin sums the same span.

    Derivation: rewards 0..9, seq_len 4 -> window w sums
    w + (w+1) + (w+2) + (w+3) = 4w + 6, i.e. 6, 10, 14, ... The
    episode total is 45, which no window equals.
    """
    import numpy as np

    from experiments.rl_finetuning.ablations.training import _extract_windows

    T, seq_len = 10, 4
    ep = {
        "local": np.zeros((T, 9, 9), dtype=np.int16),
        "global": np.zeros((T, 21, 79), dtype=np.int16),
        "actions": np.arange(T, dtype=np.int64),
        "rewards": np.arange(T, dtype=np.float32),
        "total_reward": float(np.arange(T).sum()),
    }

    _, _, x0, rets = _extract_windows(ep, seq_len=seq_len, pad_token=13)

    assert rets.shape == (T - seq_len + 1,) == (x0.shape[0],)
    expected = [4 * w + 6 for w in range(T - seq_len + 1)]
    assert rets.tolist() == pytest.approx(expected)
    assert 45.0 not in rets.tolist(), "episode total leaked into a window"


def test_padded_window_return_excludes_the_padding():
    """Padded steps earn nothing, so a short episode's single window
    scores the real rewards only."""
    import numpy as np

    from experiments.rl_finetuning.ablations.training import _extract_windows

    ep = {
        "local": np.zeros((3, 9, 9), dtype=np.int16),
        "global": np.zeros((3, 21, 79), dtype=np.int16),
        "actions": np.arange(3, dtype=np.int64),
        "rewards": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "total_reward": 6.0,
    }

    _, _, x0, rets = _extract_windows(ep, seq_len=8, pad_token=13)

    assert x0.shape == (1, 8)
    assert rets.tolist() == pytest.approx([6.0])


def test_reward_filter_keeps_strictly_above_the_percentile():
    """reward_filtering keeps windows with return STRICTLY above the
    batch percentile (spec-ablations §2, step-9 amendment: same
    boundary in both repos; was the PARITY '>= with ties keeps all'
    divergence).

    Derivation: returns 1..8, 75th percentile (linear interpolation) =
    6.25 -> keep {7, 8}. All-equal returns: percentile == value, so a
    strict > keeps nothing (the >= rule kept everything).
    """
    from experiments.rl_finetuning.ablations.training import reward_filter_mask

    keep = reward_filter_mask(torch.arange(1.0, 9.0), 75)
    assert keep.tolist() == [False] * 6 + [True, True]
    assert reward_filter_mask(torch.full((5,), 2.0), 75).sum() == 0


def test_action_diversity_discards_degenerate_plans():
    """action_diversity keeps only windows with more than one distinct
    action (spec-ablations §2)."""
    from experiments.rl_finetuning.ablations.training import action_diversity_mask

    x0 = torch.tensor([[1, 1, 1, 1], [1, 2, 1, 1], [0, 0, 0, 0]])
    assert action_diversity_mask(x0).tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# llrd (Sun 2019: eta_{k-1} = xi * eta_k, top-down from the head)
# ---------------------------------------------------------------------------


def test_llrd_learning_rates_decay_geometrically_from_the_head():
    """LLRD gives the head base_lr and each transformer layer at depth
    d from the top base_lr * decay^d; the observation encoders sit
    below the lowest layer (Sun 2019; decay 0.9, spec-ablations §2).

    With n_layer=2: head=base; layers.1=base*0.9; layers.0=base*0.81;
    everything else base*0.729. Param-group LRs are inspected directly.
    """
    cfg = _model_cfg(llrd_decay=0.9)
    model = make_model(cfg)
    opt = make_optimizer_llrd(cfg, model)
    lr_of_param = {}
    for group in opt.param_groups:
        for p in group["params"]:
            lr_of_param[id(p)] = group["lr"]

    def expected_lr(name: str) -> float:
        if name.startswith("head."):
            return 1e-3
        if "transformer.layers.1." in name:
            return 1e-3 * 0.9
        if "transformer.layers.0." in name:
            return 1e-3 * 0.81
        return 1e-3 * 0.9**3

    for name, p in model.named_parameters():
        assert lr_of_param[id(p)] == pytest.approx(expected_lr(name), rel=1e-9), name


# ---------------------------------------------------------------------------
# lora (Hu 2021 eq (3): W_eff = W + (alpha/r) B A, B zero-initialised)
# ---------------------------------------------------------------------------


def test_lora_delta_is_zero_at_init_scaled_by_alpha_over_r_and_isolated():
    """LoRA per Hu 2021 eq (3)/§4: B=0 at init (effective weight equals
    the pretrained weight exactly), the delta is (alpha/r)*B@A
    (recomputed with NumPy), and only the A/B factors are trainable.
    """
    rank, alpha = 8, 16.0
    cfg = _model_cfg()
    model = make_model(cfg)
    attn = model.transformer.layers[0].self_attn
    before = attn.in_proj_weight.detach().clone()

    lora_params = apply_lora_to_model(model, rank, alpha)
    assert torch.equal(attn.in_proj_weight, before), "init delta must be zero"

    for name, p in model.named_parameters():
        is_lora = "parametrizations" in name and (
            name.endswith(".A") or name.endswith(".B")
        )
        assert p.requires_grad == is_lora, name
    assert len(lora_params) == 2 * 2 * cfg.n_layer  # A and B per in/out per layer

    par = attn.parametrizations.in_proj_weight[0]
    with torch.no_grad():
        par.B.fill_(1.0)
    delta = (attn.in_proj_weight - before).detach().numpy()
    want = (alpha / rank) * (par.B.detach().numpy() @ par.A.detach().numpy())
    assert np.allclose(delta, want, atol=1e-5)
    assert np.linalg.matrix_rank(delta) <= rank


# ---------------------------------------------------------------------------
# gradient_surgery (Yu 2020 Alg 1, one-sided per SOURCES.md)
# ---------------------------------------------------------------------------


def test_pcgrad_projection_closed_form_and_one_sidedness():
    """PCGrad closed form, same numbers as the craftax twin:
    g_rl=[1,0], g_bc=[-1,1] -> projected [0.5, 0.5], orthogonal to
    g_bc; non-conflicting gradients pass through unchanged."""
    out = gradient_surgery(
        {"w": torch.tensor([1.0, 0.0])}, {"w": torch.tensor([-1.0, 1.0])}
    )
    assert np.allclose(out["w"].numpy(), [0.5, 0.5], atol=1e-6)
    assert float(out["w"] @ torch.tensor([-1.0, 1.0])) == pytest.approx(0.0, abs=1e-6)
    out2 = gradient_surgery(
        {"w": torch.tensor([1.0, 0.0])}, {"w": torch.tensor([1.0, 1.0])}
    )
    assert np.allclose(out2["w"].numpy(), [1.0, 0.0])


# ---------------------------------------------------------------------------
# reward_model (spec-ablations §2: MLP obs -> return, MSE)
# ---------------------------------------------------------------------------


def test_reward_model_learns_a_linear_return_map():
    """The reward model regresses returns from flattened map features
    with MSE (spec-ablations §2 reward_model row). 50 steps on a fixed
    linear target must cut the MSE by more than half."""
    torch.manual_seed(0)
    rm = RewardModel(obs_dim=21 * 79, width=64, depth=2)
    opt = torch.optim.Adam(rm.parameters(), lr=1e-3)
    glob = torch.randint(0, 3, (64, 21, 79))
    returns = glob.reshape(64, -1).float().mean(dim=1) * 2.0 + 1.0
    feats = glob.reshape(64, -1).float()
    with torch.no_grad():
        loss0 = float(torch.nn.functional.mse_loss(rm(feats), returns))
    _train_reward_model(rm, opt, torch.zeros(64, 9, 9), glob, returns, n_steps=50)
    with torch.no_grad():
        loss1 = float(torch.nn.functional.mse_loss(rm(feats), returns))
    assert loss1 < 0.5 * loss0


# ---------------------------------------------------------------------------
# Group C: trainable-parameter sets (defect §8.3 + step-8 findings)
# step-7 reproduction method: requires_grad partition after the factory
# ---------------------------------------------------------------------------


def _trainable_names(ablation: str) -> frozenset[str]:
    cfg = _model_cfg()
    model = make_model(cfg)
    registry.REGISTRY[ablation].optimizer_factory(cfg, model)
    return frozenset(n for n, p in model.named_parameters() if p.requires_grad)


def _all_names() -> frozenset[str]:
    return frozenset(n for n, _ in make_model(_model_cfg()).named_parameters())


def test_frozen_backbone_trains_the_head_and_token_embeddings():
    """Canonical set (spec-ablations §2, step-9 amendment): the action
    head plus the token-interface embeddings (action, timestep and
    positional embeddings); the backbone (obs streams incl. goal head,
    transformer stack, all norms) is frozen."""
    expected = frozenset(
        n
        for n in _all_names()
        if n.startswith(("action_emb.", "timestep_emb.", "pos_emb.", "head."))
    )
    assert _trainable_names("frozen_backbone") == expected


def test_head_only_is_a_distinct_intervention_from_frozen_backbone():
    """head_only trains exactly the final action projection - a strict
    subset of frozen_backbone's set (spec-ablations §2, step-9
    amendment; was defect §8.3: exact duplicates)."""
    head = _trainable_names("head_only")
    assert head == {"head.weight", "head.bias"}
    assert head < _trainable_names("frozen_backbone")


def test_attention_only_trains_only_the_attention_projections():
    """Canonical set (spec-ablations §2, step-9 amendment): exactly the
    attention projections Q/K/V/O; norms and head frozen (was step-8
    finding S8-4: norm1 trainable)."""
    expected = frozenset(n for n in _all_names() if ".self_attn." in n)
    assert _trainable_names("attention_only") == expected


def test_ffn_only_trains_only_the_ffn_layers():
    """Canonical set (spec-ablations §2, step-9 amendment): exactly the
    FFN linears in each encoder layer; norms and head frozen (was
    step-8 finding S8-5: norm2 trainable)."""
    expected = frozenset(
        n for n in _all_names() if ".linear1." in n or ".linear2." in n
    )
    assert _trainable_names("ffn_only") == expected


@pytest.mark.parametrize("top_n", [1, 2])
def test_layer_ablation_trains_only_the_top_layers_and_head(top_n):
    """Docs: 'Train only the top-k transformer block(s) (+ head)'
    (spec-ablations §2). With n_layer=2 the top-1 set is
    transformer.layers.1 (whole layer) plus the head; top-2 adds
    layers.0. The minihack implementation conforms.
    """
    kept = {f"transformer.layers.{i}." for i in range(2 - top_n, 2)}
    expected = frozenset(
        n for n in _all_names() if any(n.startswith(k) for k in kept)
    ) | {"head.weight", "head.bias"}
    assert _trainable_names(f"layer_ablation_top{top_n}") == expected


def test_a_frozen_fragment_list_matching_everything_is_an_error():
    """An all-frozen partition raises instead of silently training nothing.

    `make_optimizer_frozen` returned `AdamW([dummy], lr=0.0)` on an empty
    trainable set, with no warning and no counter, so a fragment list that
    matched every parameter would have reported a completed run whose
    weights never moved (F-4). Not reachable from the shipped registry --
    every group-C arm keeps between 2 and 26 tensors trainable -- which is
    why the branch needs a guard rather than a caller.
    """
    cfg = _model_cfg()
    model = make_model(cfg)

    with pytest.raises(ValueError, match="no trainable parameter"):
        make_optimizer_frozen(cfg, model, [""])


# ---------------------------------------------------------------------------
# Suite loss estimator (cross-repo twin; NELBO per spec-method §3.1/§3.4)
# ---------------------------------------------------------------------------


def test_suite_loss_uses_the_nelbo_weight_and_per_token_normalisation():
    """The suite's per-sample loss must be the NELBO estimator
    w(t) * sum_masked(CE) / L (spec-method §3.1/§3.4; spec-ablations §2
    baseline row: 'return-weighted ELBO').

    Same construction and expected value as the craftax twin: cosine
    schedule, uniform logits, t ~ U(eps, 0.2) via the low_t factory,
    unit advantages: E[per-sample] = ln V * E_t[-alpha'(t)] =
    ln V * (1 - cos(0.1 pi))/0.2 = 0.244715 ln V. Statistical bound
    0.03 ln V (~5 sigma, derivation in the craftax twin; was step-8
    finding S8-6: the suite dropped w(t) and normalised per masked
    count).
    """
    b = 16384
    model = _FixedLogitsModel(_UNIFORM)
    cfg = _cfg(t_max_low=0.2, _schedule_fn=cosine_schedule)
    g = torch.Generator().manual_seed(2)
    x0 = torch.randint(0, V, (b, L), generator=g)
    local = torch.zeros(b, 9, 9, dtype=torch.long)
    glob = torch.zeros(b, 21, 79, dtype=torch.long)
    torch.manual_seed(0)
    loss = float(
        make_loss_low_t(_ctx(cfg=cfg))(model, local, glob, x0, torch.ones(b), cfg, "cpu")
    )
    expected = math.log(V) * (1 - math.cos(0.1 * math.pi)) / 0.2
    assert abs(loss - expected) < 0.03 * math.log(V)
