"""
Central configuration — hyperparameters, tokens, and constants.
All modules import from here.
"""

import os
import torch

# Device: use DEVICE env to override (e.g. cpu, cuda:1 for multi-GPU)
_default_device = "cuda" if torch.cuda.is_available() else "cpu"
HYPERPARAMS = {
    "lr": 1e-4,
    "batch_size": 256,
    "seq_len": 64,
    "device": os.environ.get("DEVICE", _default_device),
    "crop_size": 9,
    "ema_decay": 0.999,
}

ACTION_DIM = 12
MASK_TOKEN = 12
PAD_TOKEN = 13
ACTION_NAMES = ["N", "E", "S", "W", "NE", "SE", "SW", "NW", "UP", "DWN", "WAIT", "KICK"]

# DAgger / Curriculum settings
ID_ENVS = [
    "MiniHack-Room-Random-5x5-v0",
    "MiniHack-Room-Random-15x15-v0",
    "MiniHack-Corridor-R2-v0",
    "MiniHack-MazeWalk-9x9-v0",
]
OOD_ENVS = [
    "MiniHack-Room-Dark-15x15-v0",
    "MiniHack-Corridor-R5-v0",
    "MiniHack-MazeWalk-45x19-v0",
]

# Data sources (HF remdm-minihack/ReMDM-MiniHack)
DATASET_VERSION = "v017_local_baseline_v2_fixed"
OFFLINE_CKPT = "v017_local_baseline_3b"

BUFFER_CAPACITY = 10_000
EPISODES_PER_ITERATION = 10
GRAD_STEPS_PER_ITERATION = 50
EFFICIENCY_MULTIPLIER = 1.5
CURRICULUM_QUEUE_SIZE = 100

# Run modes
SMOKE_TEST_BUFFER_CAPACITY = 50
SMOKE_TEST_MAX_ITERATIONS = 30
SMOKE_EVAL_EPISODES = 5  # Fast eval for smoke test (vs 100 for main)
MAIN_RUN_MAX_ITERATIONS = 8000
ID_EVAL_EVERY_ITERATIONS = 100
OOD_EVAL_EVERY_ITERATIONS = 500
CHECKPOINT_EVERY_ITERATIONS = 1000
CHECKPOINT_EVAL_EPISODES = 20  # 20 eps per env at checkpoint eval
EVAL_EPISODES_PER_ENV = 50
