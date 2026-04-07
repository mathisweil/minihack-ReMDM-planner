import argparse
import os
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import minihack
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
import wandb
from wandb.integration.sb3 import WandbCallback
from src.envs.minihack_env import AdvancedObservationEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.data import Dataset, DataLoader

from src.envs.minihack_env import collect_oracle_trajectory

# ── 1. SB3 Compatibility Wrapper ───────────────────────────────────────

class SB3MiniHackWrapper(gym.Wrapper): 
    """Wraps AdvancedObservationEnv to make it compatible with SB3's MultiInputPolicy."""
    def __init__(self, env):
        super().__init__(env)
        local_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Dict({
            "local": gym.spaces.Box(low=0, high=6000, shape=(1, *local_shape), dtype=np.int16),
            "global": gym.spaces.Box(low=0, high=6000, shape=(1, 21, 79), dtype=np.int16)
        })

    def reset(self, **kwargs):
        obs_tuple, info = self.env.reset(**kwargs)
        
        # Add the channel dimension! (9, 9) -> (1, 9, 9)
        obs_dict = {
            "local": np.expand_dims(obs_tuple[0], axis=0),
            "global": np.expand_dims(obs_tuple[1], axis=0)
        }
        return obs_dict, info

    def step(self, action):
        obs_tuple, reward, terminated, truncated, info = self.env.step(action)
        
        # Add the channel dimension!
        obs_dict = {
            "local": np.expand_dims(obs_tuple[0], axis=0),
            "global": np.expand_dims(obs_tuple[1], axis=0)
        }
        
        # Tell SB3 that "won" means "success" so it tracks our Win Rate!
        if "won" in info:
            info["is_success"] = info["won"]
            
        return obs_dict, reward, terminated, truncated, info

class MiniHackCNN(BaseFeaturesExtractor):
    """Custom CNN that can handle the specific 9x9 and 21x79 grid sizes."""
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # CNN for the 1x9x9 local crop
        self.local_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # CNN for the 1x21x79 global map
        self.global_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_loc = torch.zeros(1, *observation_space["local"].shape)
            dummy_glob = torch.zeros(1, *observation_space["global"].shape)
            n_flatten = self.local_cnn(dummy_loc).shape[1] + self.global_cnn(dummy_glob).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        loc = self.local_cnn(observations["local"].float())
        glob = self.global_cnn(observations["global"].float())
        return self.linear(torch.cat([loc, glob], dim=1))

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
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "a2c", "ppo-rnn", "bc"])
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

    # Mimic your DAgger config exactly!
    cfg = SimpleNamespace(
        crop_size=9, action_dim=12, pad_token=13, map_h=21, map_w=79
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
    policy_kwargs = {
        "features_extractor_class": MiniHackCNN,
        "features_extractor_kwargs": {"features_dim": 256},
    }
    # --- Initialize Algorithm ---
    print(f"Initializing {args.algo.upper()} with Custom MiniHack CNN...")
    
    if args.algo == "bc":
        # 1. Collect Expert Data using your Oracle
        print("Collecting Oracle Demonstrations...")
        all_loc, all_glob, all_acts = [], [], []
        
        for env_id in ID_ENVS:
            for seed in range(50):
                traj_dict = collect_oracle_trajectory(env_id, seed, cfg)
                if traj_dict is not None:
                    # Add channel dimension: (T, 9, 9) -> (T, 1, 9, 9)
                    all_loc.append(np.expand_dims(traj_dict["local"], axis=1))
                    all_glob.append(np.expand_dims(traj_dict["global"], axis=1))
                    all_acts.append(traj_dict["actions"])
        
        # Flatten all episodes into one massive dataset
        loc_arr = np.concatenate(all_loc, axis=0)
        glob_arr = np.concatenate(all_glob, axis=0)
        acts_arr = np.concatenate(all_acts, axis=0)
        
        # 2. Build Custom PyTorch DataLoader (Bypasses imitation's dict bugs!)
        class MiniHackBCDataset(Dataset):
            def __init__(self, loc, glob, acts):
                self.loc = torch.tensor(loc, dtype=torch.float32)
                self.glob = torch.tensor(glob, dtype=torch.float32)
                self.acts = torch.tensor(acts, dtype=torch.int64)
                
            def __len__(self):
                return len(self.acts)
                
            def __getitem__(self, idx):
                # imitation's internal training loop expects a dictionary with "obs" and "acts"
                return {
                    "obs": {
                        "local": self.loc[idx],
                        "global": self.glob[idx]
                    },
                    "acts": self.acts[idx]
                }

        batch_size = 256
        bc_dataset = MiniHackBCDataset(loc_arr, glob_arr, acts_arr)
        bc_dataloader = DataLoader(bc_dataset, batch_size=batch_size, shuffle=True)
        
        # 3. Setup the BC Policy
        # 3. Setup the BC Policy
        print(f"Collected {len(acts_arr)} total expert transitions. Training BC natively...")
        policy = ActorCriticPolicy(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            lr_schedule=lambda _: 1e-3,
            features_extractor_class=MiniHackCNN,
            features_extractor_kwargs={"features_dim": 256},
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # 4. Native PyTorch Behavioral Cloning Loop (Bypasses imitation entirely!)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        n_epochs = 10
        
        policy.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            
            for batch in bc_dataloader:
                # Move tensors to the correct device (GPU/CPU)
                obs = {k: v.to(policy.device) for k, v in batch["obs"].items()}
                acts = batch["acts"].to(policy.device)
                
                # SB3 policies have a built-in method to evaluate actions
                # It returns: (values, log_prob, entropy)
                _, log_prob, _ = policy.evaluate_actions(obs, acts)
                
                # Behavioral Cloning minimizes the Negative Log-Likelihood of expert actions
                loss = -log_prob.mean()
                
                # Standard PyTorch gradient update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(bc_dataloader)
            print(f"Epoch {epoch+1:02d}/{n_epochs} | BC Loss: {avg_loss:.4f}")
        
        model = policy # Save reference for your evaluation block below
        
        # 4. Manual Evaluation for BC (Win Rate & Avg Steps)
        print("\n" + "="*60)
        print(" FINAL EVALUATION: BEHAVIORAL CLONING")
        print("="*60)
        
        eval_eps = 50
        
        for split_name, env_list in [("ID", ID_ENVS), ("OOD", OOD_ENVS)]:
            print(f"\n--- {split_name} Environments ---")
            for env_id in env_list:
                short_name = env_id.replace("MiniHack-", "").replace("-v0", "")
                eval_env = SubprocVecEnv([make_env(env_id, cfg, f"{log_dir}/eval_{split_name.lower()}/{env_id}")])
                obs = eval_env.reset()
                
                wins = 0
                total_steps = 0
                episodes_completed = 0
                
                while episodes_completed < eval_eps:
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = eval_env.step(action)
                    
                    if dones[0]:
                        episodes_completed += 1
                        if infos[0].get("won", False):
                            wins += 1
                        total_steps += infos[0]["episode"]["l"]
                
                eval_env.close()
                
                win_rate = (wins / eval_eps) * 100
                avg_steps = total_steps / eval_eps
                
                print(f"{short_name:25} | Win Rate: {win_rate:5.1f}% | Avg Steps: {avg_steps:5.1f}")
                
                # Log to W&B matching the RL callback format
                wandb.log({
                    f"{split_name}/{short_name}/win_rate": win_rate,
                    f"{split_name}/{short_name}/avg_steps": avg_steps
                })
        print("="*60 + "\n")

    else:
        # Standard SB3 RL Training Block
        if args.algo == "ppo":
            model = PPO("MultiInputPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"runs/{run.id}")
        elif args.algo == "ppo-rnn":
            model = RecurrentPPO("MultiInputLstmPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"runs/{run.id}")
        elif args.algo == "dqn":
            model = DQN("MultiInputPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"runs/{run.id}", buffer_size=100_000)
        elif args.algo == "a2c":
            model = A2C("MultiInputPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"runs/{run.id}")

        print(f"Training across {len(ID_ENVS)} ID maps...")
        model.learn(total_timesteps=args.timesteps, callback=callback_list)

    # --- Save and Cleanup ---
    model_to_save = model if args.algo != "bc" else policy
    model_to_save.save(f"models/{run.id}/{args.algo}_final")
    if args.algo != "bc":
        train_env.close()
    wandb.finish()
    print("Baseline training complete!")

if __name__ == "__main__":
    main()