import argparse
import os
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import minihack
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

# Assuming your custom env is saved in a file called custom_env.py
from src.envs.minihack_env import AdvancedObservationEnv 

# ── 1. SB3 Compatibility Wrapper ───────────────────────────────────────

class SB3MiniHackWrapper(gym.Wrapper): # Changed to gym.Wrapper to intercept step()
    """Wraps AdvancedObservationEnv to make it compatible with SB3's MultiInputPolicy."""
    def __init__(self, env):
        super().__init__(env)
        local_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Dict({
            "local": gym.spaces.Box(low=0, high=6000, shape=local_shape, dtype=np.int16),
            "global": gym.spaces.Box(low=0, high=6000, shape=(21, 79), dtype=np.int16)
        })

    def reset(self, **kwargs):
        obs_tuple, info = self.env.reset(**kwargs)
        obs_dict = {"local": obs_tuple[0], "global": obs_tuple[1]}
        return obs_dict, info

    def step(self, action):
        obs_tuple, reward, terminated, truncated, info = self.env.step(action)
        obs_dict = {"local": obs_tuple[0], "global": obs_tuple[1]}
        
        # Tell SB3 that "won" means "success" so it tracks our Win Rate!
        if "won" in info:
            info["is_success"] = info["won"]
            
        return obs_dict, reward, terminated, truncated, info

# ── 2. Clean Evaluation Logging ────────────────────────────────────────

class PrefixedEvalCallback(EvalCallback):
    """Custom EvalCallback to prevent metric names from colliding in WandB."""
    def __init__(self, eval_env, prefix, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.prefix = prefix

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        
        if self.evaluations_results:
            # 1. Log Mean Reward
            mean_reward = np.mean(self.evaluations_results[-1])
            self.logger.record(f"{self.prefix}/mean_reward", mean_reward)
            
            # 2. Log Average Steps (Episode Length)
            avg_steps = np.mean(self.evaluations_length[-1])
            self.logger.record(f"{self.prefix}/avg_steps", avg_steps)
            
        # 3. Log Win Rate
        if self.evaluations_successes:
            win_rate = np.mean(self.evaluations_successes[-1])
            self.logger.record(f"{self.prefix}/win_rate", win_rate)
            
        return continue_training

# ── 3. Environment Factory ─────────────────────────────────────────────

def make_env(env_id, cfg, log_dir):
    """Creates, wraps, and monitors a single environment instance."""
    def _init():
        os.makedirs(log_dir, exist_ok=True)
        env = AdvancedObservationEnv(env_id, des_file=None, cfg=cfg)
        env = SB3MiniHackWrapper(env)
        env = Monitor(env, log_dir)
        return env
    return _init

# ── 4. Main Training Loop ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "a2c"])
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    args = parser.parse_args()

    # Environment definitions
    ID_ENVS = [
        'MiniHack-Room-Random-5x5-v0', 
        'MiniHack-Room-Random-15x15-v0', 
        'MiniHack-Corridor-R2-v0', 
        'MiniHack-MazeWalk-9x9-v0'
    ]
    OOD_ENVS = [
        "MiniHack-Room-Dark-15x15-v0",
        "MiniHack-Corridor-R5-v0",
        "MiniHack-MazeWalk-45x19-v0"
    ]

    # Mimic your DAgger config
    cfg = SimpleNamespace(
        crop_size=9, action_dim=8, pad_token=0, map_h=21, map_w=79
    )

    run = wandb.init(
        project="remdm-baselines",
        name=f"{args.algo.upper()}-MultiTask",
        sync_tensorboard=True
    )

    log_dir = f"./logs/{run.id}"

    # --- Setup Multi-Task Training Environment ---
    # We create 8 parallel workers (2 for each of the 4 ID maps)
    train_env_fns = [make_env(env_id, cfg, log_dir) for env_id in ID_ENVS * 2]
    train_env = SubprocVecEnv(train_env_fns)

    # --- Setup Evaluation Callbacks ---
    callbacks = [WandbCallback(model_save_path=f"models/{run.id}")]
    
    # Create an ID EvalCallback for each training map
    for env_id in ID_ENVS:
        # Strip redundant text to avoid SB3's 36-char truncation limit
        short_name = env_id.replace("MiniHack-", "").replace("-v0", "")
        
        eval_env = SubprocVecEnv([make_env(env_id, cfg, f"{log_dir}/eval_id/{env_id}")])
        callbacks.append(PrefixedEvalCallback(
            eval_env, 
            prefix=f"ID/{short_name}",  # Shortened prefix!
            best_model_save_path=f'./models/{run.id}/best_{env_id}/',
            log_path=f'./logs/{run.id}/eval_id/{env_id}/',
            eval_freq=args.eval_freq // train_env.num_envs,
            n_eval_episodes=50,
            deterministic=True
        ))

    # Create an OOD EvalCallback for each testing map
    for env_id in OOD_ENVS:
        short_name = env_id.replace("MiniHack-", "").replace("-v0", "")
        
        eval_env = SubprocVecEnv([make_env(env_id, cfg, f"{log_dir}/eval_ood/{env_id}")])
        callbacks.append(PrefixedEvalCallback(
            eval_env, 
            prefix=f"OOD/{short_name}", # Shortened prefix!
            best_model_save_path=None, 
            log_path=f'./logs/{run.id}/eval_ood/{env_id}/',
            eval_freq=args.eval_freq // train_env.num_envs,
            n_eval_episodes=50,
            deterministic=True
        ))

    callback_list = CallbackList(callbacks)

    # --- Initialize Algorithm ---
    print(f"Initializing {args.algo.upper()} with MultiInputPolicy...")
    if args.algo == "ppo":
        model = PPO("MultiInputPolicy", train_env, verbose=1, tensorboard_log=f"runs/{run.id}")
    elif args.algo == "dqn":
        model = DQN("MultiInputPolicy", train_env, verbose=1, tensorboard_log=f"runs/{run.id}", buffer_size=100_000)
    elif args.algo == "a2c":
        model = A2C("MultiInputPolicy", train_env, verbose=1, tensorboard_log=f"runs/{run.id}")

    # --- Train ---
    print(f"Training across {len(ID_ENVS)} ID maps...")
    model.learn(total_timesteps=args.timesteps, callback=callback_list)

    # --- Save and Cleanup ---
    model.save(f"models/{run.id}/{args.algo}_final")
    train_env.close()
    wandb.finish()
    print("Baseline training complete!")

if __name__ == "__main__":
    main()