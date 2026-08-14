"""Literature-anchored specification tests.

Every expected value here is derived from a primary source (cited
per test; full references below) or from a derivation written out in
the docstring - never from the current output of this or the sibling repo. The craftax repo
carries the same assertions with the same inputs and tolerances wherever
the mathematics is parameter-free and shared.

Tolerances: closed-form checks use atol=1e-6 (an order of magnitude above
float32 round-off, which the parity probes measured at <4e-8 on these
functions). Statistical checks state their sampling distribution and use a
4-sigma bound with the derivation in the docstring.

References:
- MDLM: Sahoo et al., "Simple and Effective Masked Diffusion Language
  Models", NeurIPS 2024. arXiv:2406.07524.
- Shi: Shi et al., "Simplified and Generalized Masked Diffusion for
  Discrete Data", NeurIPS 2024. arXiv:2406.04329.
- ReMDM: Wang et al., "Remasking Discrete Diffusion Models with
  Inference-Time Scaling", NeurIPS 2025. arXiv:2503.00307.
- MaskGIT: Chang et al., "MaskGIT: Masked Generative Image
  Transformer", CVPR 2022. arXiv:2202.04200.
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic
  Models", ICML 2021. arXiv:2102.09672.
- Holtzman et al., "The Curious Case of Neural Text Degeneration",
  ICLR 2020. arXiv:1904.09751.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from src.diffusion.forward import q_sample
from src.diffusion.loss import mdlm_loss
from src.diffusion.sampling import _compute_remask_prob, remdm_sample, top_p_filter
from src.diffusion.schedules import (
    cosine_schedule,
    cosine_schedule_deriv,
    cosine_sq_schedule,
    cosine_sq_schedule_deriv,
    get_schedule,
    linear_schedule,
    linear_schedule_deriv,
)

ATOL = 1e-6
T_GRID = torch.tensor([0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Noise schedules: closed forms and derivatives
# ---------------------------------------------------------------------------


def test_linear_schedule_closed_form():
    """alpha(t) = 1 - t; alpha'(t) = -1.

    Source: MDLM (Sahoo et al.) App E.1 eq (90) family (log-linear alpha_t
    = 1 - t); ReMDM (Wang et al.) Sec 3 uses the same convention
    alpha(0)=1, alpha(1)=0.
    """
    expected = torch.tensor([1.0, 2.0 / 3.0, 0.5, 1.0 / 3.0, 0.0], dtype=torch.float64)
    assert torch.allclose(linear_schedule(T_GRID), expected, atol=ATOL)
    assert torch.allclose(
        linear_schedule_deriv(T_GRID), torch.full_like(T_GRID, -1.0), atol=ATOL
    )


def test_cosine_schedule_closed_form():
    """alpha(t) = cos(pi t / 2); alpha'(t) = -(pi/2) sin(pi t / 2).

    Source: MDLM App E.1 eq (92) ("Cosine"): sigma(t) = -log cos(pi/2 (1-t))
    i.e. alpha = cos(pi/2 (1-t)) on MDLM's reversed time axis, equal to
    cos(pi t / 2) under this repo's alpha(0)=1 orientation. Values:
    cos(pi/6) = sqrt(3)/2, cos(pi/4) = sqrt(2)/2, cos(pi/3) = 1/2.
    """
    expected = torch.tensor(
        [1.0, math.sqrt(3) / 2, math.sqrt(2) / 2, 0.5, 0.0], dtype=torch.float64
    )
    assert torch.allclose(cosine_schedule(T_GRID), expected, atol=ATOL)
    expected_d = torch.tensor(
        [0.0, -math.pi / 4, -(math.pi / 2) * math.sqrt(2) / 2,
         -(math.pi / 2) * math.sqrt(3) / 2, -math.pi / 2],
        dtype=torch.float64,
    )
    assert torch.allclose(cosine_schedule_deriv(T_GRID), expected_d, atol=ATOL)


def test_cosine_sq_schedule_closed_form():
    """alpha(t) = cos^2(pi t / 2); alpha'(t) = -(pi/2) sin(pi t).

    Source: MDLM App E.1 eq (91) ("Cosine Squared"), attributed to Nichol &
    Dhariwal (their eq for alpha-bar with s=0). cos^2 at the grid:
    [1, 3/4, 1/2, 1/4, 0]; derivative via 2 cos(x)(-sin(x))(pi/2) =
    -(pi/2) sin(pi t).
    """
    expected = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float64)
    assert torch.allclose(cosine_sq_schedule(T_GRID), expected, atol=ATOL)
    expected_d = torch.tensor(
        [0.0, -(math.pi / 2) * math.sqrt(3) / 2, -math.pi / 2,
         -(math.pi / 2) * math.sqrt(3) / 2, 0.0],
        dtype=torch.float64,
    )
    assert torch.allclose(cosine_sq_schedule_deriv(T_GRID), expected_d, atol=ATOL)


def test_schedule_registry_names_follow_mdlm_e1():
    """The label "cosine" must denote MDLM eq (92), not eq (91).

    Source: MDLM App E.1; ADJUDICATION B-6 (label collision fixed by
    FIX-5). Guards against the two repos' "cosine" diverging again.
    """
    t = torch.tensor([0.5], dtype=torch.float64)
    assert torch.allclose(get_schedule("cosine")(t),
                          torch.tensor([math.sqrt(2) / 2], dtype=torch.float64),
                          atol=ATOL)
    assert torch.allclose(get_schedule("cosine_sq")(t),
                          torch.tensor([0.5], dtype=torch.float64), atol=ATOL)


# ---------------------------------------------------------------------------
# Forward corruption q(z_t | x_0)
# ---------------------------------------------------------------------------


def test_forward_marginal_endpoints_and_pad():
    """q(z_t|x) = Cat(alpha_t x + (1-alpha_t) m): t=0 identity, t=1 all-MASK;
    PAD positions are never corrupted.

    Source: MDLM Sec 3.2.1 (forward masking marginal); PAD exclusion is the
    benchmark-forced extension documented in METHOD_PARITY 2.1. Endpoints
    are deterministic: at t=1 the mask draw u<1 always holds (u in [0,1));
    at t=0 u<0 never holds.
    """
    torch.manual_seed(0)
    x0 = torch.randint(0, 12, (4, 64))
    x0[:, -8:] = 13  # PAD tail
    t0 = torch.zeros(4)
    t1 = torch.ones(4)

    z0 = q_sample(x0, t0, mask_token=12, pad_token=13, schedule_fn=linear_schedule)
    assert torch.equal(z0, x0), "t=0 must leave the sequence unchanged"

    z1 = q_sample(x0, t1, mask_token=12, pad_token=13, schedule_fn=linear_schedule)
    real = x0 != 13
    assert torch.all(z1[real] == 12), "t=1 must mask every non-PAD position"
    assert torch.all(z1[~real] == 13), "PAD positions must never be masked"


def test_forward_marginal_rate_matches_one_minus_alpha():
    """Empirical mask rate at t=0.5 (linear) is 1-alpha = 0.5 within 4 sigma.

    Source: MDLM Sec 3.2.1. N = 200*64 = 12800 independent Bernoulli(0.5)
    draws; sigma = sqrt(0.25/12800) = 0.00442; bound = 4 sigma = 0.0177.
    """
    torch.manual_seed(0)
    x0 = torch.randint(0, 12, (200, 64))
    t = torch.full((200,), 0.5)
    zt = q_sample(x0, t, mask_token=12, pad_token=13, schedule_fn=linear_schedule)
    rate = (zt == 12).float().mean().item()
    assert abs(rate - 0.5) < 0.0177


# ---------------------------------------------------------------------------
# Loss: NELBO estimator
# ---------------------------------------------------------------------------


def _uniform_logits_case(B=2, L=8, V=12):
    logits = torch.zeros(B, L, V)
    x0 = torch.randint(0, V, (B, L), generator=torch.Generator().manual_seed(0))
    return logits, x0


def test_loss_all_masked_uniform_logits_t1():
    """Loss = w(1) * log V with everything masked and uniform logits.

    Source: MDLM eq (10) integrand alpha'_t/(1-alpha_t) * CE summed over
    masked positions, per-token normalised. Derivation: uniform logits give
    CE = log V at every position; all L positions masked so sum/L = log V;
    linear schedule w(1) = -alpha'(1)/(1-alpha(1)) = 1/1 = 1.
    """
    logits, x0 = _uniform_logits_case()
    zt = torch.full_like(x0, 12)
    t = torch.ones(2)
    loss = mdlm_loss(logits, x0, zt, t, 12, 13, linear_schedule)
    assert abs(loss.item() - math.log(12)) < ATOL


def test_loss_denominator_is_per_token_not_per_masked():
    """Half-masked at t=0.5 (linear): loss = w(0.5)*logV*(4/8) = log V.

    Source: MDLM eq (8)/(10); Shi eq (4). The equations contain no division
    by the realised masked count; dividing by it (the pre-FIX-1 opt-in form
    and the pre-FIX-1 craftax form) would return 2*log V here, off by the
    factor L/n_masked = 2. This is the regression test for ADJUDICATION
    B-1's denominator finding.
    """
    logits, x0 = _uniform_logits_case()
    zt = x0.clone()
    zt[:, :4] = 12  # exactly half masked
    t = torch.full((2,), 0.5)
    loss = mdlm_loss(logits, x0, zt, t, 12, 13, linear_schedule)
    assert abs(loss.item() - math.log(12)) < ATOL


def test_loss_weight_clip_bound():
    """w(t) is clipped at weight_clip as t -> 0.

    Source: the divergence of w(t)=1/t as t->0 is a property of MDLM
    eq (10) under the linear schedule; the finite bound (1000, with the
    denominator floored at 1e-5) is this codebase's documented stability
    policy (_MAX_WEIGHT), shared with the craftax repo. Derivation: at
    t=1e-6, 1-alpha=1e-6 floors to 1e-5 giving w=1e5, clipped to 1000;
    half-masked uniform-logits loss = 1000 * log V * 0.5.
    """
    logits, x0 = _uniform_logits_case()
    zt = x0.clone()
    zt[:, :4] = 12
    t = torch.full((2,), 1e-6)
    loss = mdlm_loss(logits, x0, zt, t, 12, 13, linear_schedule)
    assert abs(loss.item() - 1000 * math.log(12) * 0.5) < 1e-3


def test_loss_excludes_pad_and_empty_mask_is_zero():
    """PAD positions contribute nothing; no masked positions -> loss 0.

    Source: MDLM Sec 3.2.3 (loss over masked positions only); PAD handling
    is the benchmark-forced extension. Derivation: 6 of 8 positions are
    real and masked at t=1 -> loss = 1 * log V * 6/8. The all-unmasked case
    has an empty diffusion term.
    """
    logits, x0 = _uniform_logits_case()
    x0[:, -2:] = 13
    zt = torch.full_like(x0, 12)
    t = torch.ones(2)
    loss = mdlm_loss(logits, x0, zt, t, 12, 13, linear_schedule)
    assert abs(loss.item() - math.log(12) * 6 / 8) < ATOL

    clean = mdlm_loss(logits, x0, x0.clone(), t, 12, 13, linear_schedule)
    assert clean.item() == 0.0


# ---------------------------------------------------------------------------
# Reverse step: remasking schedules and the sigma bound
# ---------------------------------------------------------------------------


def test_sigma_strategies_closed_form_and_bound():
    """sigma_max = min(1, (1-alpha_s)/alpha_t); rescale = eta*sigma_max;
    cap = min(eta, sigma_max); every sigma <= sigma_max.

    Source: ReMDM eq (7) and Sec 4.1 (Max-Capped and Rescaled schedules).
    Grid: linear schedule, K=10 reverse steps, eta=0.5.
    """
    eta = 0.5
    for k in range(1, 10):
        alpha_t = 1 - k / 10
        alpha_s = 1 - (k + 1) / 10
        sigma_max = min(1.0, (1 - alpha_s) / alpha_t)
        rescale = _compute_remask_prob("rescale", eta, sigma_max, None)
        cap = _compute_remask_prob("cap", eta, sigma_max, None)
        assert abs(rescale - eta * sigma_max) < ATOL
        assert abs(cap - min(eta, sigma_max)) < ATOL
        assert rescale <= sigma_max + ATOL and cap <= sigma_max + ATOL


def test_conf_strategy_softmax_of_stored_psi():
    """sigma_conf(l) = softmax(-psi)_l * eta * sigma_max over committed
    positions, zero at masked ones; lower psi => higher remask probability.

    Source: ReMDM Sec 4.1 (Confidence-Based Schedule): eta_conf =
    exp(-psi_l)/sum exp(-psi_l'), with psi the decoding probability stored
    when the token was last unmasked (FIX-3 corrected this repo to use the
    stored value). Sum over committed positions = eta * sigma_max.
    """
    eta, sigma_max = 0.5, 0.8
    psi = torch.tensor([[0.9, 0.2, float("inf"), 0.5]])
    committed = torch.tensor([[True, True, False, True]])
    sigma = _compute_remask_prob("conf", eta, sigma_max, psi, committed)
    assert sigma[0, 2].item() == 0.0
    assert sigma[0, 1] > sigma[0, 3] > sigma[0, 0], "lower psi must remask more"
    assert abs(sigma[0, committed[0]].sum().item() - eta * sigma_max) < 1e-5
    assert torch.all(sigma <= sigma_max + ATOL)


# ---------------------------------------------------------------------------
# Reverse chain behaviour (ReMDM Algorithm 1) via a deterministic stub
# ---------------------------------------------------------------------------


class _StubModel(torch.nn.Module):
    """Position-dependent peaked logits: argmax token = position % V."""

    def __init__(self, seq_len: int, v: int):
        super().__init__()
        self.seq_len, self.v = seq_len, v

    def forward(self, local_obs, global_obs, action_seq, t_discrete):
        B = action_seq.shape[0]
        logits = torch.full((B, self.seq_len, self.v), -5.0)
        for pos in range(self.seq_len):
            logits[:, pos, pos % self.v] = 5.0
        return {"actions": logits}


def _stub_cfg(**over):
    cfg = dict(
        seq_len=64, mask_token=12, action_dim=12, num_diffusion_steps=100,
        diffusion_steps_eval=4, temperature=1.0, top_p=1.0, eta=0.0,
        remask_strategy="rescale", noise_schedule="linear", crop_size=9,
        map_h=21, map_w=79,
    )
    cfg.update(over)
    return SimpleNamespace(**cfg)


def test_carryover_committed_tokens_persist_when_sigma_zero():
    """With sigma = 0 a committed token is never changed or remasked.

    Source: ReMDM Algorithm 1, z_t != m branch: Cat(z_s; (1-sigma) x_theta
    + sigma m) with x_theta carrying over unmasked inputs (MDLM Sec 3.2.3,
    Carry-Over Unmasking). With sigma=0 (eta=0 rescale) the branch is the
    identity. Uses the sampler's analytics trace across a 4-step chain.
    """
    torch.manual_seed(0)
    cfg = _stub_cfg()
    model = _StubModel(cfg.seq_len, cfg.action_dim)
    local = torch.zeros(2, 9, 9, dtype=torch.long)
    glob = torch.zeros(2, 21, 79, dtype=torch.long)
    _, path, _, _ = remdm_sample(
        model, local, glob, cfg, "cpu", physics_aware=False,
        return_analytics=True,
    )
    for earlier, later in zip(path, path[1:]):
        committed = earlier != cfg.mask_token
        assert (later[committed] == earlier[committed]).all(), (
            "a committed token changed or was remasked despite sigma=0"
        )


def test_posterior_unmask_rate_first_step():
    """First reverse step (t=1 -> s=1/2, sigma=0, linear) unmasks each
    masked token independently with p = (alpha_s - alpha_t)/(1 - alpha_t)
    = 0.5.

    Source: ReMDM Algorithm 1 approximate posterior, z_t = m branch.
    Statistical: 128*64 = 8192 Bernoulli(0.5) draws; sigma = sqrt(0.25/8192)
    = 0.0055; bound = 4 sigma = 0.0221. The pre-FIX-3 MaskGIT count rule
    would deterministically unmask exactly ceil(L/2) per row and, at the
    old first step (k=1/K), only L/K tokens - both outside this bound.
    """
    import numpy as np

    torch.manual_seed(0)
    cfg = _stub_cfg(seq_len=64, diffusion_steps_eval=2)
    model = _StubModel(cfg.seq_len, cfg.action_dim)
    local = torch.zeros(1, 9, 9, dtype=torch.long)
    glob = torch.zeros(1, 21, 79, dtype=torch.long)

    # The analytics trace records row 0 only, so run 128 single-row chains
    # (independent draws from the shared global RNG): 128*64 = 8192 tokens.
    seqs = []
    for _ in range(128):
        _, path, _, _ = remdm_sample(
            model, local, glob, cfg, "cpu", physics_aware=False,
            return_analytics=True,
        )
        seqs.append(path[0])  # state after the first reverse step
    rate = float(np.mean([(s != cfg.mask_token).mean() for s in seqs]))
    assert abs(rate - 0.5) < 0.0221


# ---------------------------------------------------------------------------
# Nucleus filtering
# ---------------------------------------------------------------------------


def test_top_p_filter_known_distribution():
    """Nucleus keeps the smallest prefix with cumulative mass >= p.

    Source: ReMDM Sec 5 adopts nucleus sampling (Holtzman et al.): the
    candidate set is the smallest prefix of the descending-sorted
    distribution whose cumulative probability reaches p. For probs
    [0.5, 0.3, 0.15, 0.05]: p=0.9 keeps {0,1,2} (cumulative 0.95 >= 0.9
    reached at the third token); p=0.5 keeps {0} alone.
    """
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    kept_09 = top_p_filter(logits, 0.9).isfinite()
    assert kept_09.tolist() == [[True, True, True, False]]
    kept_05 = top_p_filter(logits, 0.5).isfinite()
    assert kept_05.tolist() == [[True, False, False, False]]
    # p >= 1 disables filtering
    assert top_p_filter(logits, 1.0).isfinite().all()
