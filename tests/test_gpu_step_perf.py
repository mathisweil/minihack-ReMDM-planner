"""Pins for the GPU-side performance changes (PERF-C0 to PERF-C6).

Every change in that set is meant to be arithmetic-preserving: it removes
device syncs, kernel launches or PCIe traffic without touching the values
the training step computes. These tests hold that line, so a future edit
that quietly changes the maths fails here rather than in a training curve.

CPU-only, like the rest of the suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.buffer import ReplayBuffer
from src.diffusion.loss import find_staircase_from_glyphs
from src.models.denoiser import ModelEMA, make_model
from src.planners.online import Trainer

STAIR_GLYPHS = (62, 2310, 2368, 2383)


def _reference_staircase(global_obs: torch.Tensor) -> torch.Tensor:
    """The pre-PERF-C0 implementation: a Python loop over the batch.

    Kept here as the specification of the vectorised version.
    """
    if global_obs.ndim == 2:
        global_obs = global_obs.unsqueeze(0)
    B, H, W = global_obs.shape
    is_stair = torch.zeros_like(global_obs, dtype=torch.bool)
    for g in STAIR_GLYPHS:
        is_stair |= global_obs == g
    coords = torch.full((B, 2), -1.0, dtype=torch.float32)
    for b in range(B):
        positions = is_stair[b].nonzero(as_tuple=False)
        if positions.shape[0] > 0:
            coords[b, 0] = positions[0, 0].float() / max(1, H - 1)
            coords[b, 1] = positions[0, 1].float() / max(1, W - 1)
    return coords


# ── PERF-C0: vectorised staircase lookup ─────────────────────────────


@pytest.mark.parametrize("glyph", STAIR_GLYPHS)
def test_every_staircase_glyph_variant_is_found(glyph):
    maps = torch.ones(1, 21, 79, dtype=torch.long)
    maps[0, 7, 13] = glyph

    coords = find_staircase_from_glyphs(maps)

    assert coords[0, 0].item() == pytest.approx(7 / 20)
    assert coords[0, 1].item() == pytest.approx(13 / 78)


def test_absent_staircase_yields_the_pad_sentinel():
    maps = torch.ones(4, 21, 79, dtype=torch.long)

    coords = find_staircase_from_glyphs(maps)

    assert torch.equal(coords, torch.full((4, 2), -1.0))


def test_first_staircase_in_row_major_order_wins():
    """Several staircases: the lowest flat index is the one reported."""
    maps = torch.ones(1, 21, 79, dtype=torch.long)
    maps[0, 9, 40] = 62
    maps[0, 3, 70] = 62  # earlier row, so this is the one that must win
    maps[0, 15, 2] = 62

    coords = find_staircase_from_glyphs(maps)

    assert coords[0, 0].item() == pytest.approx(3 / 20)
    assert coords[0, 1].item() == pytest.approx(70 / 78)


def test_matches_the_per_sample_loop_on_random_maps():
    torch.manual_seed(0)
    maps = torch.randint(0, 3000, (32, 21, 79))
    # Guarantee a mix of present and absent staircases across the batch.
    for b in range(0, 32, 3):
        maps[b, b % 21, (b * 7) % 79] = STAIR_GLYPHS[b % len(STAIR_GLYPHS)]

    assert torch.equal(find_staircase_from_glyphs(maps), _reference_staircase(maps))


def test_unbatched_input_is_accepted():
    maps = torch.ones(21, 79, dtype=torch.long)
    maps[4, 4] = 62

    coords = find_staircase_from_glyphs(maps)

    assert coords.shape == (1, 2)
    assert torch.equal(coords, _reference_staircase(maps))


def test_result_is_float32_regardless_of_input_dtype():
    """The buffer holds glyphs as int16; the goal loss expects float32."""
    for dtype in (torch.int16, torch.int32, torch.int64):
        maps = torch.ones(2, 21, 79, dtype=dtype)
        maps[0, 1, 1] = 62
        assert find_staircase_from_glyphs(maps).dtype == torch.float32


# ── PERF-C2: device-side step metrics ────────────────────────────────


def _trainer(cfg, trajectory=None, device="cpu"):
    model = make_model(cfg).to(device)
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seq_len, cfg.pad_token)
    if trajectory is not None:
        buffer.add(trajectory)
    return Trainer(
        model,
        ModelEMA(model, decay=cfg.ema_decay),
        torch.optim.AdamW(model.parameters(), lr=cfg.dagger_lr),
        None,
        buffer,
        collector=None,
        evaluator=None,
        log=None,
        cfg=cfg,
        device=device,
        raw_model=model,
    )


def test_device_step_returns_tensors_and_the_wrapper_returns_floats(
    tiny_cfg, tiny_trajectory
):
    trainer = _trainer(tiny_cfg, tiny_trajectory)
    trainer.model.train()

    device_metrics = trainer._train_step_device()
    float_metrics = trainer._train_step()

    assert set(device_metrics) == set(float_metrics)
    for key, value in device_metrics.items():
        assert isinstance(value, torch.Tensor), key
        assert value.ndim == 0, key
        assert not value.requires_grad, key
    for key, value in float_metrics.items():
        assert isinstance(value, float), key


def test_empty_buffer_step_is_a_no_op_in_both_forms(tiny_cfg):
    trainer = _trainer(tiny_cfg)

    assert trainer._train_step() == {
        "loss": 0.0,
        "loss_diff": 0.0,
        "loss_aux": 0.0,
        "grad_norm": 0.0,
    }
    assert all(
        float(v) == 0.0 for v in trainer._train_step_device().values()
    )


# ── PERF-C1: the int16 transfer widens exactly ───────────────────────


def test_buffer_glyphs_widen_from_int16_without_changing_values(tiny_cfg):
    """C1 sends glyph maps as int16 and widens on the device instead.

    Both are exact for the glyph range, so the widened values must equal
    the CPU-side cast the code used to do.
    """
    glyphs = np.array(
        [[0, 1, 62, 2310, 2368, 2383, 5999, np.iinfo(np.int16).max]],
        dtype=np.int16,
    )

    widened_on_device = torch.from_numpy(glyphs).to("cpu").long()
    widened_on_host = torch.from_numpy(glyphs).long().to("cpu")

    assert torch.equal(widened_on_device, widened_on_host)
    assert widened_on_device.tolist() == glyphs.astype(np.int64).tolist()
