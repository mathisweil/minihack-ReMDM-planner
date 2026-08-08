"""Shared fixtures for the smoke suite.

Everything here is CPU-only, seeded, and confined to pytest tmp directories.
No test reads a real dataset or checkpoint, and none touches the network.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TINY_ENV = "MiniHack-Room-Random-5x5-v0"

CLI_TIMEOUT = 120

# Shrunken model, diffusion and loop settings. Keys mirror configs/defaults.yaml
# so the real loader and the real code paths are exercised unchanged.
TINY_OVERRIDES: dict = {
    "id_envs": [TINY_ENV],
    "ood_envs": [TINY_ENV],
    "n_embd": 32,
    "n_head": 2,
    "n_layer": 1,
    "seq_len": 8,
    # replan_every must not exceed seq_len or rollouts index past the plan.
    "replan_every": 8,
    "num_diffusion_steps": 4,
    "diffusion_steps_eval": 2,
    "diffusion_steps_collect": 2,
    "buffer_capacity": 32,
    "dagger_batch_size": 4,
    "offline_batch_size": 4,
    "total_timesteps": 1,
    "id_eval_every_timesteps": 10**9,
    "ood_eval_every_timesteps": 10**9,
    "checkpoint_every_timesteps": 10**9,
    "episodes_per_iteration": 1,
    "grad_steps_per_iteration": 1,
    "eval_episodes_per_env": 1,
    "checkpoint_eval_episodes": 1,
    "num_collection_workers": 0,
    "collect_episodes_per_env": 1,
    "collect_num_workers": 1,
    "use_wandb": False,
    "save_policy": False,
    "torch_compile": False,
    "use_amp": False,
    "device": "cpu",
    "seed": 0,
}


# ── Optional dependency / hardware gates ─────────────────────────────


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


requires_minihack = pytest.mark.skipif(
    not (_importable("minihack") and _importable("nle")),
    reason=(
        "MiniHack/NLE not installed - environment rollouts unavailable. "
        "Install with `uv sync` to enable these tests."
    ),
)

requires_cuda = pytest.mark.skipif(
    not _cuda_available(),
    reason="No CUDA device available - GPU-only code path skipped.",
)


# ── Module discovery (used to parametrise the import tests) ──────────


def discover_modules(package: str) -> list[str]:
    """Return *package* and every importable submodule name beneath it."""
    pkg = importlib.import_module(package)
    names = [package]
    names += [m.name for m in pkgutil.walk_packages(pkg.__path__, f"{package}.")]
    return sorted(names)


# ── Global isolation ─────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _isolate_side_effects(tmp_path_factory):
    """Redirect library scratch dirs into tmp and disable outbound calls."""
    sandbox = tmp_path_factory.mktemp("env_isolation")
    # Matplotlib spends ~15s rebuilding its font cache on a cold directory, so
    # this one lives in a stable spot under the system temp dir rather than in
    # the per-run sandbox. Still outside the repo and outside the user's home.
    mpl_cache = Path(tempfile.gettempdir()) / "remdm-smoke-mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DIR"] = str(sandbox / "wandb")
    os.environ["WANDB_SILENT"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DEVICE"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield


@pytest.fixture(autouse=True)
def _seeded():
    """Fix every RNG before each test."""
    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


# ── Configs ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def real_cfg() -> SimpleNamespace:
    """The unmodified production config, forced onto CPU."""
    from src.config import load_config

    cfg = load_config("configs/defaults.yaml")
    cfg.device = "cpu"
    return cfg


@pytest.fixture
def tiny_cfg() -> SimpleNamespace:
    """Production config loaded for real, then shrunk to toy dimensions."""
    from src.config import load_config

    cfg = load_config("configs/defaults.yaml")
    for key, value in TINY_OVERRIDES.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def tiny_config_file(tmp_path: Path) -> Path:
    """Write the shrunken config to disk for CLI entry-point tests."""
    payload = dict(TINY_OVERRIDES)
    payload["checkpoint_dir"] = str(tmp_path / "checkpoints")
    payload["collect_output"] = str(tmp_path / "dataset.pt")
    payload["baselines_output_dir"] = str(tmp_path / "baselines")
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


# ── Synthetic data ───────────────────────────────────────────────────


@pytest.fixture
def tiny_batch(tiny_cfg):
    """A synthetic model-input batch: (local_obs, global_obs, actions)."""
    import torch

    batch = 4
    local = torch.randint(0, 1000, (batch, tiny_cfg.crop_size, tiny_cfg.crop_size))
    glob = torch.randint(0, 1000, (batch, tiny_cfg.map_h, tiny_cfg.map_w))
    actions = torch.randint(0, tiny_cfg.action_dim, (batch, tiny_cfg.seq_len))
    return local.long(), glob.long(), actions.long()


@pytest.fixture
def tiny_trajectory(tiny_cfg) -> dict:
    """A synthetic oracle trajectory in the on-disk dataset format."""
    import numpy as np

    steps = 12
    return {
        "local": np.random.randint(
            0, 1000, (steps, tiny_cfg.crop_size, tiny_cfg.crop_size), dtype=np.int16
        ),
        "global": np.random.randint(
            0, 1000, (steps, tiny_cfg.map_h, tiny_cfg.map_w), dtype=np.int16
        ),
        "actions": np.random.randint(
            0, tiny_cfg.action_dim, (steps,), dtype=np.int64
        ),
        "env_id": TINY_ENV,
    }


@pytest.fixture
def tiny_dataset_file(tmp_path: Path, tiny_trajectory) -> Path:
    """A two-trajectory dataset written in the format run_offline expects."""
    import torch

    path = tmp_path / "dataset.pt"
    torch.save({"trajectories": [tiny_trajectory, tiny_trajectory]}, path)
    return path


@pytest.fixture
def tiny_checkpoint_file(tmp_path: Path, tiny_cfg) -> Path:
    """A checkpoint of an untrained tiny model, in the production format."""
    import torch

    from src.models.denoiser import ModelEMA, make_model

    model = make_model(tiny_cfg)
    ema = ModelEMA(model, decay=tiny_cfg.ema_decay)
    path = tmp_path / "tiny_checkpoint.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
        },
        path,
    )
    return path


# ── CLI helper ───────────────────────────────────────────────────────


def run_cli(*args: str, timeout: int = CLI_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a project entry point in a CPU-only, offline subprocess."""
    env = dict(os.environ)
    env.update(
        {
            "DEVICE": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "WANDB_MODE": "disabled",
            "PYTHONWARNINGS": "ignore",
            "PYTHONPATH": str(PROJECT_ROOT),
        }
    )
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def assert_cli_ok(result: subprocess.CompletedProcess) -> None:
    """Fail with the captured output when an entry point exits non-zero."""
    if result.returncode != 0:
        raise AssertionError(
            f"exit code {result.returncode}\n"
            f"--- stdout ---\n{result.stdout[-3000:]}\n"
            f"--- stderr ---\n{result.stderr[-3000:]}"
        )
