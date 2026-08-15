"""Literature-anchored specification tests — step-8 gap closure.

Complements tests/test_method_spec.py with the surfaces the step-8 audit
found untested in this repo: the reverse-posterior/remasking/carry-over
chain through remdm_sample, the final greedy cleanup, decode temperature
and label smoothing. Every expected value derives from a cited source or
a derivation written in the docstring — never from the current output of
this or the sibling repo. The craftax twin file carries the same
assertions with the same inputs and tolerances.

References as in tests/test_method_spec.py (MDLM arXiv:2406.07524;
ReMDM arXiv:2503.00307; Holtzman arXiv:1904.09751).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.diffusion.loss import mdlm_loss
from src.diffusion.sampling import remdm_sample
from src.diffusion.schedules import linear_schedule, linear_schedule_deriv


class _TimeCodedModel(nn.Module):
    """Stub whose argmax token encodes the decode time bin.

    token 0 for t_discrete > 90, token 1 for 50 < t_discrete <= 90,
    token 2 otherwise (num_diffusion_steps = 100 in the cfg below).
    Logit margin 30 makes categorical sampling deterministic to ~1e-13.
    """

    def __init__(self, seq_len: int, vocab: int):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab

    def forward(self, local_obs, global_obs, seq, t_discrete):
        td = int(t_discrete[0])
        idx = 0 if td > 90 else (1 if td > 50 else 2)
        logits = torch.full((seq.shape[0], self.seq_len, self.vocab), -30.0)
        logits[:, :, idx] = 30.0
        return {"actions": logits, "goal_pred": torch.zeros(seq.shape[0], 2)}


def _chain_cfg(eta: float) -> SimpleNamespace:
    return SimpleNamespace(
        seq_len=32, mask_token=3, action_dim=3, diffusion_steps_eval=3,
        temperature=1.0, top_p=1.0, eta=eta, remask_strategy="rescale",
        noise_schedule="linear", num_diffusion_steps=100,
    )


@pytest.mark.parametrize("eta", [0.0, 0.5])
def test_posterior_remask_carryover_token_distribution(eta):
    """Final token distribution matches the ReMDM Alg 1 posterior chain.

    Source: ReMDM eq (6) masked branch, eq (7) sigma_max, Sec 4.1
    rescale sigma = eta * sigma_max, MDLM Sec 3.2.3 carry-over.

    Derivation (linear schedule, K=3, grid t=1,2/3,1/3 with s=2/3,1/3,0;
    the stub decodes token 0 at t=1, token 1 at t=2/3, token 2 at
    t<=1/3):
      step 1 (t=1):   alpha_t=0, alpha_s=1/3, p_unmask=1/3 -> token 0.
      step 2 (t=2/3): alpha_t=1/3, alpha_s=2/3, sigma_max=1, sigma=eta;
                      committed remask w.p. eta; masked unmask w.p.
                      (2/3-(1-eta)/3)/(2/3) = (1+eta)/2 -> token 1.
      step 3 (t=1/3): alpha_s=1 -> sigma_max=0 (no remask), p_unmask=1
                      -> every remaining mask becomes token 2.
    Hence P(0) = (1-eta)/3, P(1) = (1+eta)/3, P(2) = 1/3.
    eta=0 additionally pins carry-over (sigma=0: committed tokens are
    never re-decided). Same derivation and bound as the craftax twin:
    N = 512*32 = 16384; max sigma = 0.0039; bound 0.02 = 5.1 sigma.
    """
    B, L = 512, 32
    cfg = _chain_cfg(eta)
    model = _TimeCodedModel(L, 4)
    seq = remdm_sample(
        model, torch.zeros(B, 9, 9), torch.zeros(B, 21, 79), cfg, "cpu",
        physics_aware=False,
    )
    freq = np.bincount(seq.numpy().ravel(), minlength=4) / (B * L)
    expected = np.array([(1 - eta) / 3, (1 + eta) / 3, 1 / 3, 0.0])
    assert np.all(np.abs(freq - expected) < 0.02), (freq, expected)


def test_final_cleanup_commits_all_remaining_masks():
    """With zero denoising steps, the final greedy cleanup decodes every
    position at t=0 (argmax), leaving no MASK token.

    Source: spec-method 4.9 (final-step commit of remaining masks; the
    craftax twin asserts the same). The stub decodes token 2 at t=0,
    so the output must be all-2.
    """
    B, L = 8, 16
    cfg = _chain_cfg(0.0)
    cfg.seq_len = L
    seq = remdm_sample(
        _TimeCodedModel(L, 4), torch.zeros(B, 9, 9), torch.zeros(B, 21, 79),
        cfg, "cpu", physics_aware=False, num_steps=0,
    )
    assert (seq.numpy() == 2).all()


class _FixedLogitsModel(nn.Module):
    """Stub returning fixed logits [0, 1] over two real actions."""

    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len

    def forward(self, local_obs, global_obs, seq, t_discrete):
        logits = torch.zeros(seq.shape[0], self.seq_len, 3)
        logits[:, :, 1] = 1.0
        logits[:, :, 2] = 0.0  # masked out by the sampler (>= action_dim)
        return {"actions": logits, "goal_pred": torch.zeros(seq.shape[0], 2)}


def test_decode_temperature_scales_logits_before_sampling():
    """Sampling frequencies follow softmax(logits / temperature).

    Source: spec-method 5.2. Derivation: logits [0, 1] at temperature
    0.5 give softmax([0, 2]) = [0.1192, 0.8808] over the two real
    actions; a single denoising step (K=1: t=1, s=0, p_unmask=1)
    commits every position in one draw. Statistical: 8192 draws;
    sigma = sqrt(0.8808*0.1192/8192) = 0.00358; bound 0.0143 = 4 sigma.
    Same numbers as the craftax twin.
    """
    B, L = 256, 32  # 8192 positions
    cfg = SimpleNamespace(
        seq_len=L, mask_token=2, action_dim=2, diffusion_steps_eval=1,
        temperature=0.5, top_p=1.0, eta=0.0, remask_strategy="rescale",
        noise_schedule="linear", num_diffusion_steps=100,
    )
    seq = remdm_sample(
        _FixedLogitsModel(L), torch.zeros(B, 9, 9), torch.zeros(B, 21, 79),
        cfg, "cpu", physics_aware=False,
    )
    p1 = math.exp(2) / (1 + math.exp(2))
    assert abs(seq.float().mean().item() - p1) < 0.0143


def test_label_smoothing_matches_closed_form():
    """Smoothed CE = -[(1-eps)+eps/V] log p_true - (eps/V) sum log p_other.

    Source: spec-method 3.7 (smoothing target (1-eps)*onehot + eps/V,
    which is exactly torch.nn.functional.cross_entropy's label_smoothing
    semantics; eps=0 is the exact ELBO). Derivation: model probs
    [0.7,0.1,0.1,0.1], true class 0, V=4, eps=0.3: coefficient on
    -log 0.7 is (1-0.3)+0.3/4 = 0.775; each other class gets 0.075.
    t pinned to 1 (linear, w=1, everything masked) makes the loss equal
    that CE exactly. Same inputs and expectations as the craftax twin.
    """
    B, L, V = 4, 8, 4
    logits = torch.log(torch.tensor([0.7, 0.1, 0.1, 0.1])).expand(B, L, V).clone()
    x0 = torch.zeros(B, L, dtype=torch.long)
    zt = torch.full((B, L), 5, dtype=torch.long)  # mask_token=5, pad=6
    t = torch.ones(B)
    expected = -0.775 * math.log(0.7) - 0.075 * 3 * math.log(0.1)
    got = mdlm_loss(
        logits, x0, zt, t, mask_token=5, pad_token=6,
        schedule_fn=linear_schedule, schedule_deriv_fn=linear_schedule_deriv,
        label_smoothing=0.3,
    )
    assert abs(float(got) - expected) < 1e-5
    got0 = mdlm_loss(
        logits, x0, zt, t, mask_token=5, pad_token=6,
        schedule_fn=linear_schedule, schedule_deriv_fn=linear_schedule_deriv,
        label_smoothing=0.0,
    )
    assert abs(float(got0) - (-math.log(0.7))) < 1e-5
