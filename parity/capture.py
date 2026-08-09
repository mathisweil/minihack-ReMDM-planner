"""Capture golden-output fingerprints for every trained checkpoint.

Writes parity/reference/*. Run once per approved baseline; parity/check.py
then verifies the working tree against these references. Never re-run this
to make a failing check pass.

Usage:
    uv run python parity/capture.py [--force]

Fingerprints:
    1. forward_<name>.npz/.json  raw + EMA logits on a fixed synthetic batch
    2. eval_<name>.json          3-episode inference metrics, fixed seed
    3. train_offline.json        8-grad-step loss trajectory + param checksum
    4. schema_<name>.txt         checkpoint key structure, shapes, dtypes
    5. tolerances.json           derived from an observed variability probe
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parity.fingerprint_lib import (  # noqa: E402
    PROJECT_ROOT, REFERENCE_DIR, array_stats, git_commit, run_main,
    save_json, sha256_array, sha256_file, sha256_tree,
)

CHECKPOINTS = [
    ("iter5", "checkpoints/iter5.pth", "checkpoints/config_iter5.yaml"),
    ("iter7", "checkpoints/iter7.pth", "checkpoints/config_iter7.yaml"),
    ("iter8", "checkpoints/iter8.pth", "checkpoints/config_iter8.yaml"),
    (
        "dagger123M",
        "checkpoints/online/Minihack-OnlineDiffusion-DAgger-123M/iter600.pth",
        "checkpoints/online/Minihack-OnlineDiffusion-DAgger-123M/config_iter600.yaml",
    ),
    (
        "bc82M",
        "checkpoints/offline/Minihack-OfflineDiffusion-BC-82M/offline_step40000.pth",
        "checkpoints/offline/Minihack-OfflineDiffusion-BC-82M/config_offline_step40000.yaml",
    ),
]

EVAL_ENV = "MiniHack-Room-Random-5x5-v0"
EVAL_EPISODES = 3
EVAL_SEED = 1234
TRAIN_GRAD_STEPS = 8
TRAIN_SEED = 0


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fixed_batch(cfg) -> dict[str, torch.Tensor]:
    """Deterministic synthetic model inputs; also saved to reference/."""
    g = torch.Generator().manual_seed(0)
    local = torch.randint(0, 6000, (4, cfg.crop_size, cfg.crop_size), generator=g)
    glob = torch.randint(0, 6000, (4, cfg.map_h, cfg.map_w), generator=g)
    actions = torch.randint(0, cfg.action_dim, (4, cfg.seq_len), generator=g)
    mask_pattern = torch.arange(cfg.seq_len) % 2 == 0
    actions[:, mask_pattern] = cfg.mask_token
    t = torch.tensor([0, 25, 50, 99]) % cfg.num_diffusion_steps
    return {"local": local, "global": glob, "actions": actions, "t": t}


def load_model_pair(ckpt_path: str, cfg_path: str):
    """Return (raw model, EMA-eval model, cfg) for a checkpoint."""
    from src.config import load_config
    from src.models.denoiser import ModelEMA, make_model

    cfg = load_config(cfg_path, {"device": "cpu"})
    ckpt = torch.load(PROJECT_ROOT / ckpt_path, map_location="cpu", weights_only=False)

    model = make_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ema = ModelEMA(model, decay=cfg.ema_decay)
    ema.load_state_dict(ckpt["ema_state_dict"])
    ema_model = ema.make_eval_model(model)
    ema_model.eval()
    return model, ema_model, cfg, ckpt


def capture_forward(name: str, ckpt_path: str, cfg_path: str) -> None:
    model, ema_model, cfg, _ = load_model_pair(ckpt_path, cfg_path)
    batch = fixed_batch(cfg)

    arrays: dict[str, np.ndarray] = {
        k: v.numpy() for k, v in batch.items()
    }
    with torch.no_grad():
        for tag, m in [("raw", model), ("ema", ema_model)]:
            out = m(batch["local"], batch["global"], batch["actions"], batch["t"])
            for key, val in out.items():
                arrays[f"{tag}_{key}"] = val.numpy()

    np.savez(REFERENCE_DIR / f"forward_{name}.npz", **arrays)
    meta = {
        "git_commit": git_commit(),
        "checkpoint": ckpt_path,
        "checkpoint_sha256": sha256_file(PROJECT_ROOT / ckpt_path),
        "outputs": {
            k: {**array_stats(v), "sha256": sha256_array(v)}
            for k, v in arrays.items() if k.startswith(("raw_", "ema_"))
        },
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    save_json(REFERENCE_DIR / f"forward_{name}.json", meta)
    print(f"forward_{name}: {meta['n_params']:,} params")


def run_eval(ckpt_path: str, cfg_path: str) -> dict:
    out_json = Path(tempfile.mkdtemp(prefix="parity-eval-")) / "eval.json"
    proc = run_main(
        [
            "--mode", "inference", "--config", cfg_path,
            "--checkpoint", ckpt_path, "--envs", EVAL_ENV,
            "--episodes", str(EVAL_EPISODES), "--seed", str(EVAL_SEED),
            "--override", "use_wandb=false", "--output", str(out_json),
        ],
        env_extra={"DEVICE": "cpu"},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"eval failed for {ckpt_path}:\n{proc.stderr[-2000:]}")
    results = json.loads(out_json.read_text())["results"][EVAL_ENV]
    shutil.rmtree(out_json.parent, ignore_errors=True)
    return results


def make_tiny_dataset(path: Path) -> None:
    """Committed synthetic dataset in the run_offline on-disk format."""
    rs = np.random.RandomState(0)
    trajs = []
    for _ in range(4):
        steps = 48
        trajs.append({
            "local": rs.randint(0, 1000, (steps, 9, 9)).astype(np.int16),
            "global": rs.randint(0, 1000, (steps, 21, 79)).astype(np.int16),
            "actions": rs.randint(0, 12, (steps,)).astype(np.int64),
            "env_id": EVAL_ENV,
        })
    torch.save({"trajectories": trajs}, path)


def run_train(dataset: Path) -> dict:
    """Short offline training from fixed seed; loss trajectory + checksums.

    Mirrors run_offline's pipeline (same loader, buffer, trainer) but keeps
    the trainer's returned loss history, which run_offline discards.
    """
    from src.buffer import ReplayBuffer
    from src.config import load_config
    from src.models.denoiser import ModelEMA, make_model, try_compile
    from src.planners.inference import Evaluator
    from src.planners.logging import Logger
    from src.planners.offline import load_offline_dataset, make_offline_trainer

    cfg = load_config(None, {"device": "cpu"})
    cfg.use_wandb = False
    cfg.seed = TRAIN_SEED
    cfg.offline_total_grad_steps = TRAIN_GRAD_STEPS
    cfg.offline_batch_size = 32
    cfg.offline_log_every = 1
    cfg.id_eval_every_timesteps = 10**12
    cfg.ood_eval_every_timesteps = 10**12
    cfg.checkpoint_every_timesteps = 0
    cfg.save_policy = False

    _seed_all(TRAIN_SEED)
    data = load_offline_dataset(str(dataset), cfg)
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seq_len, cfg.pad_token)
    buffer.load_offline_data(data, cfg.id_envs)

    raw_model = make_model(cfg).to(cfg.device)
    model = try_compile(raw_model, cfg)
    ema = ModelEMA(raw_model, decay=cfg.ema_decay)
    log = Logger(cfg)
    train_fn = make_offline_trainer(cfg)
    result = train_fn(
        model, ema, buffer, cfg, cfg.device, log=log,
        raw_model=raw_model, resume_state=None, evaluator=Evaluator(),
        id_envs=cfg.id_envs, ood_envs=cfg.ood_envs,
    )
    log.finish()

    params = {k: v.detach().numpy() for k, v in raw_model.state_dict().items()}
    return {
        "loss_history": [float(x) for x in result["loss_history"]],
        "final_loss": float(result["final_loss"]),
        "param_checksum": sha256_tree(params),
        "param_stats": array_stats(
            np.concatenate([p.ravel() for p in params.values()])
        ),
    }


def schema_text(ckpt_path: str) -> str:
    ckpt = torch.load(PROJECT_ROOT / ckpt_path, map_location="cpu", weights_only=False)
    lines: list[str] = []

    def walk(obj, prefix: str) -> None:
        if isinstance(obj, torch.Tensor):
            lines.append(f"{prefix}: Tensor {tuple(obj.shape)} {obj.dtype}")
        elif isinstance(obj, dict):
            for k in sorted(obj, key=str):
                walk(obj[k], f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, (list, tuple)):
            lines.append(f"{prefix}: {type(obj).__name__} len={len(obj)}")
        else:
            lines.append(f"{prefix}: {type(obj).__name__}")

    walk(ckpt, "")
    return "\n".join(lines) + "\n"


def capture_schema(name: str, ckpt_path: str) -> None:
    text = schema_text(ckpt_path)
    (REFERENCE_DIR / f"schema_{name}.txt").write_text(text)
    print(f"schema_{name}: {len(text.splitlines())} entries")


def main() -> None:
    force = "--force" in sys.argv
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    marker = REFERENCE_DIR / "MANIFEST.json"
    if marker.exists() and not force:
        sys.exit("References already exist; pass --force to overwrite.")

    torch.set_num_threads(4)

    for name, ckpt, cfg in CHECKPOINTS:
        capture_forward(name, ckpt, cfg)
        capture_schema(name, ckpt)

    # Evaluation fingerprints + variability probe (iter5 run twice)
    evals: dict[str, dict] = {}
    for name, ckpt, cfg in CHECKPOINTS:
        evals[name] = run_eval(ckpt, cfg)
        print(f"eval_{name}: {evals[name]}")
    probe_eval = run_eval(*[(c, y) for n, c, y in CHECKPOINTS if n == "iter5"][0])
    eval_deltas = {
        k: abs(float(evals["iter5"][k]) - float(probe_eval[k]))
        for k in ("win_rate", "avg_reward", "avg_steps")
    }
    print(f"eval variability probe (iter5, rerun): {eval_deltas}")
    for name in evals:
        save_json(REFERENCE_DIR / f"eval_{name}.json", {
            "git_commit": git_commit(),
            "env": EVAL_ENV, "episodes": EVAL_EPISODES, "seed": EVAL_SEED,
            "metrics": evals[name],
        })

    # Short-training fingerprint + variability probe (run twice)
    dataset = REFERENCE_DIR / "tiny_dataset.pt"
    if not dataset.exists() or force:
        make_tiny_dataset(dataset)
    t1 = run_train(dataset)
    t2 = run_train(dataset)
    train_delta = max(
        abs(a - b) for a, b in zip(t1["loss_history"], t2["loss_history"])
    )
    bit_repro = t1["param_checksum"] == t2["param_checksum"]
    print(f"train variability probe: max loss delta={train_delta:.3g}, "
          f"param checksums identical={bit_repro}")
    save_json(REFERENCE_DIR / "train_offline.json", {
        "git_commit": git_commit(),
        "seed": TRAIN_SEED, "grad_steps": TRAIN_GRAD_STEPS,
        "dataset_sha256": sha256_file(dataset),
        **t1,
    })

    # Tolerances from observed variability (never loosened by hand).
    eval_delta_max = max(eval_deltas.values())
    tolerances = {
        "forward_atol": 1e-7,
        "eval_win_rate_atol": max(4 * eval_deltas["win_rate"], 1e-9),
        "eval_reward_atol": max(4 * eval_deltas["avg_reward"], 1e-9),
        "eval_steps_atol": max(4 * eval_deltas["avg_steps"], 1e-9),
        "train_loss_atol": max(4 * train_delta, 1e-9),
        "train_bit_reproducible": bool(bit_repro),
        "eval_observed_delta_max": eval_delta_max,
        "train_observed_delta": train_delta,
    }
    save_json(REFERENCE_DIR / "tolerances.json", tolerances)

    save_json(marker, {
        "git_commit": git_commit(),
        "checkpoints": {
            n: sha256_file(PROJECT_ROOT / c) for n, c, _ in CHECKPOINTS
        },
    })
    print("capture complete")


if __name__ == "__main__":
    main()
