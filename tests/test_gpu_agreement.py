"""GPU-gated agreement and restore tests (step 8).

Sources: spec-training §5 (recipe runs with use_amp and torch_compile
enabled - their numerics must agree with the plain fp32 eager path),
spec-config §6.3/§6.4 (published checkpoint layout). Skipped cleanly
without CUDA or without the downloaded released checkpoints. The
craftax twin checks CUDA-vs-CPU agreement of its JAX loss/sampler and
the released PPO-expert restore.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.diffusion.loss import mdlm_loss
from src.diffusion.schedules import get_schedule
from src.models.denoiser import make_model
from tests.conftest import requires_cuda

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HF_ONLINE = (
    PROJECT_ROOT
    / "checkpoints/hf/checkpoints/online/Minihack-Online-Diffusion-DAgger-100M"
)


def _recipe_cfg():
    """The shipped recipe config (the model's stream geometry is coupled
    to the real crop/map dims, so tests use the real architecture)."""
    from src.config import load_config

    return load_config(None, {"use_wandb": False})


def _batch(cfg, device, b=8):
    g = torch.Generator().manual_seed(0)
    local = torch.randint(0, 100, (b, 9, 9), generator=g).to(device)
    glob = torch.randint(0, 100, (b, 21, 79), generator=g).to(device)
    zt = torch.randint(0, cfg.action_dim + 1, (b, cfg.seq_len), generator=g).to(device)
    t = torch.full((b,), 5, dtype=torch.long, device=device)
    return local, glob, zt, t


@requires_cuda
def test_amp_forward_agrees_with_fp32():
    """The recipe trains under AMP autocast (use_amp: true,
    spec-training §5); the autocast forward must agree with the fp32
    forward on identical inputs within half-precision tolerance
    (fp16 has ~1e-3 relative epsilon; bound 2e-2 on the logits).
    """
    torch.manual_seed(0)
    cfg = _recipe_cfg()
    model = make_model(cfg).cuda().eval()
    local, glob, zt, t = _batch(cfg, "cuda")
    with torch.no_grad():
        full = model(local, glob, zt, t)["actions"].float()
        with torch.amp.autocast("cuda"):
            amp = model(local, glob, zt, t)["actions"].float()
    denom = full.abs().max().clamp(min=1.0)
    assert float((full - amp).abs().max() / denom) < 2e-2


@requires_cuda
def test_loss_agrees_between_cuda_and_cpu():
    """mdlm_loss on identical explicit inputs (no RNG inside) agrees
    between CPU and CUDA within float32 reassociation tolerance
    (rel 1e-4): the estimator's mathematics is backend-invariant
    (spec-method §3.1/§3.4)."""
    torch.manual_seed(0)
    cfg = SimpleNamespace(action_dim=6, mask_token=6, pad_token=7, seq_len=16)
    b = 8
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(b, cfg.seq_len, cfg.action_dim + 2, generator=g)
    x0 = torch.randint(0, cfg.action_dim, (b, cfg.seq_len), generator=g)
    zt = torch.where(
        torch.rand(b, cfg.seq_len, generator=g) < 0.5, cfg.mask_token, x0
    )
    t = torch.rand(b, generator=g)
    args = {"mask_token": cfg.mask_token, "pad_token": cfg.pad_token,
            "schedule_fn": get_schedule("cosine")}
    cpu = float(mdlm_loss(logits, x0, zt, t, **args))
    gpu = float(
        mdlm_loss(logits.cuda(), x0.cuda(), zt.cuda(), t.cuda(), **args)
    )
    assert gpu == pytest.approx(cpu, rel=1e-4)


@requires_cuda
@pytest.mark.slow
def test_compiled_forward_agrees_with_eager(monkeypatch):
    """The recipe trains with torch_compile: true (spec-training §5);
    the compiled forward must agree with eager on identical inputs
    (rel 1e-3, allowing TF32-class matmul rounding differences in
    inductor kernels).

    The conftest isolation fixture blanks CUDA_VISIBLE_DEVICES (the
    parent process keeps its pre-initialised CUDA context, but
    inductor's triton compile workers are fresh subprocesses and would
    see no GPU), so the variable is restored for this test.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    torch.manual_seed(0)
    cfg = _recipe_cfg()
    model = make_model(cfg).cuda().eval()
    local, glob, zt, t = _batch(cfg, "cuda")
    with torch.no_grad():
        eager = model(local, glob, zt, t)["actions"].float()
        compiled = torch.compile(model)
        out = compiled(local, glob, zt, t)["actions"].float()
    denom = eager.abs().max().clamp(min=1.0)
    assert float((eager - out).abs().max() / denom) < 1e-3


@requires_cuda
@pytest.mark.skipif(
    not (_HF_ONLINE / "iter563.pth").exists(),
    reason="released checkpoints not downloaded to checkpoints/hf/",
)
def test_released_checkpoint_restores_and_runs_on_gpu():
    """The released DAgger checkpoint restores on GPU with its own
    config snapshot and produces a finite forward pass (spec-config
    §6.3: evaluate with the checkpoint's own config snapshot; step-7
    live-service check)."""
    import yaml

    snap = yaml.safe_load((_HF_ONLINE / "config.yaml").read_text())
    cfg = SimpleNamespace(**snap)
    model = make_model(cfg).cuda().eval()
    ckpt = torch.load(_HF_ONLINE / "iter563.pth", map_location="cuda",
                      weights_only=False)
    assert "model_state_dict" in ckpt
    model.load_state_dict(ckpt["model_state_dict"])
    b = 2
    local = torch.zeros(b, 9, 9, dtype=torch.long, device="cuda")
    glob = torch.zeros(b, 21, 79, dtype=torch.long, device="cuda")
    zt = torch.full((b, cfg.seq_len), cfg.mask_token, dtype=torch.long,
                    device="cuda")
    t = torch.zeros(b, dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = model(local, glob, zt, t)["actions"]
    assert bool(torch.isfinite(out).all())
