"""
ReMDM MiniHack Inference — evaluate a trained model from the command line.

Usage:
  python inference.py --checkpoint path/to/checkpoint.pth \
      --envs MiniHack-Room-Random-5x5-v0 MiniHack-MazeWalk-9x9-v0 \
      --episodes 50

  python inference.py --checkpoint path/to/checkpoint.pth \
      --des environments/situation_1_bottleneck.des \
      --episodes 20 --no-ema

  python inference.py --checkpoint path/to/checkpoint.pth \
      --envs MiniHack-Room-Random-5x5-v0 --output results.json
"""

import argparse
import sys
import os
import json

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import HYPERPARAMS, ACTION_DIM
from model import LocalDiffusionPlannerWithGlobal
from utils import evaluate_envs


def load_model(checkpoint_path, use_ema=True):
    """Load LocalDiffusionPlannerWithGlobal from checkpoint."""
    model = LocalDiffusionPlannerWithGlobal(
        action_dim=ACTION_DIM, n_embd=256, n_head=4, n_layer=4, n_global_tokens=8,
    ).to(HYPERPARAMS["device"])

    ckpt = torch.load(checkpoint_path, map_location=HYPERPARAMS["device"],
                      weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        key = 'ema_state_dict' if use_ema and 'ema_state_dict' in ckpt else 'model_state_dict'
        model.load_state_dict(ckpt[key])
        step = ckpt.get('global_step', '?')
        weight_type = 'EMA' if 'ema' in key else 'training'
        print(f"Loaded {weight_type} weights (step {step})")
    else:
        model.load_state_dict(ckpt)
        print("Loaded legacy weights")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ReMDM model on MiniHack environments")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--envs", type=str, nargs="+", default=None,
                        help="MiniHack environment IDs")
    parser.add_argument("--des", type=str, nargs="+", default=None,
                        help="Paths to .des scenario files")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per environment (default: 50)")
    parser.add_argument("--diffusion-steps", type=int, default=5,
                        help="Diffusion denoising steps (default: 5)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--no-ema", action="store_true",
                        help="Use training weights instead of EMA")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    if not args.envs and not args.des:
        parser.error("Provide at least one of --envs or --des")

    model = load_model(args.checkpoint, use_ema=not args.no_ema)

    print(f"\n{'=' * 60}")
    print(f"  ReMDM Evaluation | {args.episodes} episodes/env | "
          f"diffusion={args.diffusion_steps} | seed={args.seed}")
    print(f"{'=' * 60}")

    results = evaluate_envs(
        model,
        env_ids=args.envs,
        des_files=args.des,
        episodes=args.episodes,
        diffusion_steps=args.diffusion_steps,
        base_seed=args.seed,
        max_steps=args.max_steps,
    )

    if args.output:
        out = []
        for r in results:
            out.append({k: v for k, v in r.items() if k != 'details'})
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
