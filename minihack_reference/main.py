"""
DAgger training entry point. Mode A (Smoke Test) or Mode B (Main Run).

Usage:
  python main.py --mode smoke
  python main.py --mode main
  python main.py --mode main --no-warm-start   # Official main run: random init, fresh folder
  python main.py --mode main --resume path/to/generalization_iter2000.pth   # Resume from checkpoint

Mode A (Smoke Test): buffer_capacity=50, 30 iterations, ~1500 grad steps.
Mode B (Main Run): buffer_capacity=10000, 8000 iterations, staggered eval.
"""

import argparse
import glob
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import (
    ID_ENVS,
    BUFFER_CAPACITY,
    SMOKE_TEST_BUFFER_CAPACITY,
    HYPERPARAMS,
    ACTION_DIM,
    DATASET_VERSION,
    OFFLINE_CKPT,
    MAIN_RUN_MAX_ITERATIONS,
)
from model import LocalDiffusionPlannerWithGlobal
from trainer import Trainer
from utils import load_dataset, load_dataset_metadata, upload_generalization_to_hub


def main():
    parser = argparse.ArgumentParser(description="DAgger training: smoke or main run")
    parser.add_argument("--mode", choices=["smoke", "main"], default="main",
                        help="smoke: 30 iter, buffer=50; main: 8000 iter, buffer=10000")
    parser.add_argument("--no-warm-start", action="store_true",
                        help="Main run: start from random init (valid zero-shot). No checkpoint loaded.")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from (e.g. generalization_iter2000.pth). Overrides --no-warm-start.")
    parser.add_argument("--data", default=None,
                        help="Path to offline dataset .pt; omit to load from HF (config.DATASET_VERSION)")
    parser.add_argument("--save-dir", default="./checkpoints/dagger",
                        help="Checkpoint save directory")
    args = parser.parse_args()

    if args.mode == "smoke":
        buffer_capacity = SMOKE_TEST_BUFFER_CAPACITY
    else:
        buffer_capacity = BUFFER_CAPACITY

    run_id = None
    if args.mode == "main":
        if args.resume:
            args.save_dir = os.path.dirname(os.path.abspath(args.resume))
            run_id = os.path.basename(args.save_dir.rstrip("/"))
            print(f"Resuming from {args.resume} | save_dir={args.save_dir}")
        elif args.no_warm_start and args.save_dir == "./checkpoints/dagger":
            run_id = datetime.now().strftime("%Y%m%d_%H%M")
            args.save_dir = f"./checkpoints/main_run_v018_{run_id}"
            print(f"Main run clean start: {args.save_dir}")
        else:
            run_id = os.path.basename(args.save_dir.rstrip("/")) or datetime.now().strftime("%Y%m%d_%H%M")
        if args.no_warm_start and not args.resume:
            # Pre-flight: wipe old artifacts in save-dir
            os.makedirs(args.save_dir, exist_ok=True)
            for name in ["EMA_weights.pth", "optimizer.pt"]:
                f = os.path.join(args.save_dir, name)
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Pre-flight removed: {f}")
            for f in glob.glob(os.path.join(args.save_dir, "buffer*.pkl")):
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Pre-flight removed: {f}")

    # Load offline data
    if args.data and os.path.exists(args.data):
        data_path = args.data
        metadata = None
        meta_path = args.data.replace(".pt", "_meta.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path) as f:
                metadata = json.load(f)
    else:
        try:
            data_path = load_dataset(DATASET_VERSION)
            metadata = load_dataset_metadata(DATASET_VERSION)
        except Exception as e:
            if args.mode == "smoke":
                print("Smoke test: creating minimal synthetic buffer (no offline data)")
                import numpy as np
                from config import HYPERPARAMS
                seq_len = HYPERPARAMS["seq_len"]
                data_path = [
                    ((np.zeros((9, 9), dtype=np.int16), np.zeros((21, 79), dtype=np.int16)),
                     tuple([0] * seq_len))
                    for _ in range(20)
                ]
                metadata = {"samples_per_env": {"MiniHack-Room-Random-5x5-v0": 20}}
            else:
                print("No --data provided and could not load default dataset:", e)
                print("Use --data path/to/oracle_demos_XXX.pt")
                sys.exit(1)

    # Build model
    model = LocalDiffusionPlannerWithGlobal(
        action_dim=ACTION_DIM,
        n_embd=256,
        n_head=4,
        n_layer=4,
        n_global_tokens=8,
    ).to(HYPERPARAMS["device"])

    # Optional: warm start from checkpoint (skip when --no-warm-start or --resume)
    if args.resume:
        # Trainer will load from resume path
        pass
    elif args.mode == "main" and args.no_warm_start:
        print("Main run --no-warm-start: using randomly initialized model (no checkpoint loaded).")
    else:
        warm_path = os.path.join(os.path.dirname(__file__), "checkpoints", OFFLINE_CKPT, "inference_weights.pth")
        if not os.path.exists(warm_path):
            try:
                from utils import load_checkpoint
                weights = load_checkpoint(OFFLINE_CKPT, "inference_weights.pth")
                if isinstance(weights, dict) and "ema_state_dict" in weights:
                    model.load_state_dict(weights["ema_state_dict"], strict=False)
                elif hasattr(weights, "keys"):
                    model.load_state_dict(weights, strict=False)
                print(f"Warm start from {OFFLINE_CKPT}")
            except Exception:
                print("No warm start; training from scratch")

    trainer = Trainer(
        model=model,
        data_path=data_path,
        metadata=metadata,
        buffer_capacity=buffer_capacity,
        lr=3e-5,
        aux_loss_weight=0.5,
        save_dir=args.save_dir,
        hub_run_id=run_id if args.mode == "main" else None,
        hub_token=os.environ.get("HF_TOKEN"),
        resume_from=args.resume,
    )

    ema_model = trainer.run(mode=args.mode)

    os.makedirs(args.save_dir, exist_ok=True)
    curriculum_state = {
        "env_ids": trainer.curriculum.env_ids,
        "queue_size": trainer.curriculum.queue_size,
        "_queues": {e: list(q) for e, q in trainer.curriculum._queues.items()},
    }
    rng_states = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    ckpt_path = os.path.join(args.save_dir, "dagger_final.pth")
    torch.save({
        "model_state_dict": trainer.model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "curriculum_state": curriculum_state,
        "iteration": MAIN_RUN_MAX_ITERATIONS,
        "rng_states": rng_states,
        "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
        "scaler_state_dict": trainer.scaler.state_dict() if trainer.scaler else None,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")

    # Upload to HF generalization/ (no Drive)
    if args.mode == "main" and run_id:
        hub_token = os.environ.get("HF_TOKEN")
        if hub_token:
            upload_generalization_to_hub(args.save_dir, run_id, token=hub_token)
        else:
            print("Set HF_TOKEN to upload to Hugging Face generalization/")


if __name__ == "__main__":
    main()
