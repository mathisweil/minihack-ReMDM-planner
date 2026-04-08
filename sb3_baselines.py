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
import math

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

# ── 2. Decision Transformer ────────────────────────────────────────────

class MiniHackStateEncoder(nn.Module):
    """CNN encoder for MiniHack dict observations -> embedding."""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.local_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.global_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute flattened sizes
        with torch.no_grad():
            dummy_loc = torch.zeros(1, 1, 9, 9)
            dummy_glob = torch.zeros(1, 1, 21, 79)
            local_flat = self.local_cnn(dummy_loc).shape[1]
            global_flat = self.global_cnn(dummy_glob).shape[1]
        
        self.proj = nn.Linear(local_flat + global_flat, embed_dim)
    
    def forward(self, local_obs, global_obs):
        # local_obs: (B, T, 1, 9, 9) or (B, 1, 9, 9)
        # global_obs: (B, T, 1, 21, 79) or (B, 1, 21, 79)
        if local_obs.dim() == 5:
            B, T = local_obs.shape[:2]
            local_obs = local_obs.view(B * T, *local_obs.shape[2:])
            global_obs = global_obs.view(B * T, *global_obs.shape[2:])
            reshape = True
        else:
            B, T = local_obs.shape[0], 1
            reshape = False
        
        loc_feat = self.local_cnn(local_obs.float())
        glob_feat = self.global_cnn(global_obs.float())
        combined = torch.cat([loc_feat, glob_feat], dim=-1)
        out = self.proj(combined)
        
        if reshape:
            out = out.view(B, T, -1)
        return out


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for MiniHack.
    
    Sequence: (R_0, s_0, a_0, R_1, s_1, a_1, ..., R_t, s_t, ?) -> a_t
    """
    def __init__(
        self,
        n_actions=12,
        embed_dim=128,
        n_heads=4,
        n_layers=3,
        context_len=30,
        max_ep_len=200,
        dropout=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.n_actions = n_actions
        
        # Embeddings
        self.state_encoder = MiniHackStateEncoder(embed_dim)
        self.action_embed = nn.Embedding(n_actions + 1, embed_dim)  # +1 for padding/mask
        self.return_embed = nn.Linear(1, embed_dim)
        
        # Positional embeddings (for timesteps within episode)
        self.pos_embed = nn.Embedding(max_ep_len, embed_dim)
        
        # Token type embeddings (return, state, action)
        self.token_type_embed = nn.Embedding(3, embed_dim)
        
        self.embed_ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Action prediction head (predicts from state tokens)
        self.action_head = nn.Linear(embed_dim, n_actions)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, returns_to_go, local_obs, global_obs, actions, timesteps, attention_mask=None):
        """
        Args:
            returns_to_go: (B, T, 1)
            local_obs: (B, T, 1, 9, 9)
            global_obs: (B, T, 1, 21, 79)
            actions: (B, T) - action indices
            timesteps: (B, T) - position within episode
            attention_mask: (B, T) - 1 for real tokens, 0 for padding
        
        Returns:
            action_logits: (B, T, n_actions)
        """
        B, T = returns_to_go.shape[:2]
        device = returns_to_go.device
        
        # Embed each modality
        rtg_embed = self.return_embed(returns_to_go)  # (B, T, D)
        state_embed = self.state_encoder(local_obs, global_obs)  # (B, T, D)
        action_embed = self.action_embed(actions)  # (B, T, D)
        
        # Add positional embeddings
        pos_embed = self.pos_embed(timesteps)  # (B, T, D)
        
        rtg_embed = rtg_embed + pos_embed + self.token_type_embed.weight[0]
        state_embed = state_embed + pos_embed + self.token_type_embed.weight[1]
        action_embed = action_embed + pos_embed + self.token_type_embed.weight[2]
        
        # Interleave: (R_0, s_0, a_0, R_1, s_1, a_1, ...)
        # Shape: (B, 3*T, D)
        stacked = torch.stack([rtg_embed, state_embed, action_embed], dim=2)
        stacked = stacked.view(B, 3 * T, self.embed_dim)
        
        stacked = self.embed_ln(stacked)
        stacked = self.dropout(stacked)
        
        # Causal mask
        seq_len = 3 * T
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Attention mask for padding
        if attention_mask is not None:
            # Expand attention mask: (B, T) -> (B, 3*T)
            expanded_mask = attention_mask.unsqueeze(-1).repeat(1, 1, 3).view(B, 3 * T)
            # Convert to additive mask for transformer (0 -> -inf for masked positions)
            key_padding_mask = (expanded_mask == 0)
        else:
            key_padding_mask = None
        
        # Transformer forward
        hidden = self.transformer(stacked, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        
        # Extract state token representations (positions 1, 4, 7, ...)
        state_hidden = hidden[:, 1::3, :]  # (B, T, D)
        
        # Predict actions
        action_logits = self.action_head(state_hidden)  # (B, T, n_actions)
        
        return action_logits
    
    @torch.no_grad()
    def get_action(self, returns_to_go, local_obs, global_obs, actions, timesteps):
        """
        Get action for the current timestep (last position in sequence).
        Used during evaluation.
        """
        self.eval()
        action_logits = self.forward(returns_to_go, local_obs, global_obs, actions, timesteps)
        # Return action for the last timestep
        return action_logits[:, -1, :].argmax(dim=-1)


class DTDataset(Dataset):
    """Dataset for Decision Transformer training."""
    def __init__(self, trajectories, context_len=30, max_ep_len=200, n_actions=12):
        """
        Args:
            trajectories: List of dicts with keys:
                - local: (T, 1, 9, 9)
                - global: (T, 1, 21, 79)  
                - actions: (T,)
                - rewards: (T,)
                - returns_to_go: (T,)
        """
        self.context_len = context_len
        self.max_ep_len = max_ep_len
        self.n_actions = n_actions
        self.trajectories = trajectories
        
        # Compute dataset size (all valid starting positions)
        self.indices = []
        for traj_idx, traj in enumerate(trajectories):
            traj_len = len(traj["actions"])
            for start in range(traj_len):
                self.indices.append((traj_idx, start))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_idx, start = self.indices[idx]
        traj = self.trajectories[traj_idx]
        
        traj_len = len(traj["actions"])
        end = min(start + self.context_len, traj_len)
        actual_len = end - start
        
        # Extract sequence
        local = traj["local"][start:end].copy()
        glob = traj["global"][start:end].copy()
        actions = traj["actions"][start:end].copy()
        rtg = traj["returns_to_go"][start:end].copy()
        timesteps = np.arange(start, end)
        
        # CRITICAL: Clamp timesteps to valid range for positional embedding
        timesteps = np.clip(timesteps, 0, self.max_ep_len - 1)
        
        # CRITICAL: Clamp actions to valid range (0 to n_actions-1)
        actions = np.clip(actions, 0, self.n_actions - 1)
        
        # Pad if necessary
        pad_len = self.context_len - actual_len
        if pad_len > 0:
            local = np.pad(local, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
            glob = np.pad(glob, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
            actions = np.pad(actions, (0, pad_len), mode='constant', constant_values=0)  # pad with valid action
            rtg = np.pad(rtg, (0, pad_len), mode='constant')
            timesteps = np.pad(timesteps, (0, pad_len), mode='constant')  # pad with 0
        
        # Attention mask
        attention_mask = np.zeros(self.context_len)
        attention_mask[:actual_len] = 1
        
        return {
            "local": torch.tensor(local, dtype=torch.float32),
            "global": torch.tensor(glob, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "returns_to_go": torch.tensor(rtg, dtype=torch.float32).unsqueeze(-1),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32)
        }


def evaluate_dt(model, env_id, cfg, log_dir, target_return, context_len, n_episodes=50, device="cuda", max_ep_len=200):
    """Evaluate Decision Transformer on an environment."""
    from src.envs.minihack_env import AdvancedObservationEnv
    
    env = AdvancedObservationEnv(env_id, des_file=None, cfg=cfg)
    env = SB3MiniHackWrapper(env)
    
    model.eval()
    wins = 0
    total_steps = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        # Initialize context buffers
        local_hist = []
        global_hist = []
        action_hist = []
        rtg_hist = []
        timestep_hist = []
        
        current_rtg = target_return
        t = 0
        
        while not done and t < 200:
            # Add current observation to history
            local_hist.append(obs["local"])
            global_hist.append(obs["global"])
            rtg_hist.append(current_rtg)
            # CRITICAL: Clamp timestep to valid range for positional embedding
            timestep_hist.append(min(t, max_ep_len - 1))
            
            # Prepare input tensors (take last context_len)
            ctx_len = min(len(local_hist), context_len)
            
            local_in = np.stack(local_hist[-ctx_len:], axis=0)
            global_in = np.stack(global_hist[-ctx_len:], axis=0)
            rtg_in = np.array(rtg_hist[-ctx_len:])
            ts_in = np.array(timestep_hist[-ctx_len:])
            
            # For actions, we need ctx_len actions but we only have ctx_len-1 at the start
            if len(action_hist) < ctx_len:
                # Pad with zeros at the beginning
                act_in = np.zeros(ctx_len, dtype=np.int64)
                if len(action_hist) > 0:
                    act_in[-len(action_hist):] = action_hist[-ctx_len:]
            else:
                act_in = np.array(action_hist[-ctx_len:])
            
            # Convert to tensors
            local_t = torch.tensor(local_in, dtype=torch.float32).unsqueeze(0).to(device)
            global_t = torch.tensor(global_in, dtype=torch.float32).unsqueeze(0).to(device)
            rtg_t = torch.tensor(rtg_in, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            act_t = torch.tensor(act_in, dtype=torch.long).unsqueeze(0).to(device)
            ts_t = torch.tensor(ts_in, dtype=torch.long).unsqueeze(0).to(device)
            
            # Get action
            with torch.no_grad():
                action = model.get_action(rtg_t, local_t, global_t, act_t, ts_t).item()
            
            # Clamp action to valid range
            action = max(0, min(action, cfg.action_dim - 1))
            action_hist.append(action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update RTG
            current_rtg = current_rtg - reward
            t += 1
        
        if info.get("won", False):
            wins += 1
        total_steps += t
    
    env.close()
    return wins / n_episodes, total_steps / n_episodes


# ── 3. Clean Evaluation Logging ────────────────────────────────────────

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

# ── 4. Environment Factory ─────────────────────────────────────────────

def make_env(env_id, cfg, log_dir):
    """Creates, wraps, and monitors a single environment instance."""
    def _init():
        os.makedirs(log_dir, exist_ok=True)
        env = AdvancedObservationEnv(env_id, des_file=None, cfg=cfg)
        env = SB3MiniHackWrapper(env)
        env = Monitor(env, log_dir)
        return env
    return _init

# ── 5. Main Training Loop ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "a2c", "ppo-rnn", "bc", "dt"])
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    # Multi-seed support
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="List of seeds to run (e.g., --seeds 0 1 2)")
    parser.add_argument("--n_seeds", type=int, default=None, help="Number of seeds starting from 0 (alternative to --seeds)")
    # DT-specific args
    parser.add_argument("--dt_epochs", type=int, default=20)
    parser.add_argument("--dt_context_len", type=int, default=30)
    parser.add_argument("--dt_embed_dim", type=int, default=128)
    parser.add_argument("--dt_n_layers", type=int, default=3)
    parser.add_argument("--dt_n_heads", type=int, default=4)
    parser.add_argument("--dt_lr", type=float, default=1e-4)
    parser.add_argument("--dt_batch_size", type=int, default=64)
    args = parser.parse_args()
    
    # Handle seed specification
    if args.n_seeds is not None:
        seeds = list(range(args.n_seeds))
    else:
        seeds = args.seeds
    
    print(f"Running {len(seeds)} seed(s): {seeds}")

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Collect results across seeds
    all_seed_results = []
    
    for seed_idx, seed in enumerate(seeds):
        print("\n" + "="*70)
        print(f" SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print("="*70)
        
        # Set all random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        run = wandb.init(
            project="remdm-baselines",
            name=f"{args.algo.upper()}-MultiTask-seed{seed}",
            group=f"{args.algo.upper()}-MultiTask",  # Group runs for aggregation
            config={"seed": seed, **vars(args)},
            sync_tensorboard=True,
            reinit=True  # Allow multiple wandb.init() calls
        )

        log_dir = f"./logs/{run.id}"
        os.makedirs(f"models/{run.id}", exist_ok=True)
        
        seed_results = {"seed": seed}

        # --- Decision Transformer Training ---
        if args.algo == "dt":
            print("="*60)
            print(f" DECISION TRANSFORMER TRAINING (seed={seed})")
            print("="*60)
        
        # 1. Collect Oracle Demonstrations with rewards
        print("\nCollecting Oracle Demonstrations...")
        trajectories = []
        
        for env_id in ID_ENVS:
            for seed in range(100):  # More data for DT
                traj_dict = collect_oracle_trajectory(env_id, seed, cfg)
                if traj_dict is not None:
                    T = len(traj_dict["actions"])
                    
                    # Compute rewards (1 for reaching goal, 0 otherwise)
                    # Assuming last step is the goal
                    rewards = np.zeros(T)
                    rewards[-1] = 1.0  # Sparse reward at end
                    
                    # Compute returns-to-go
                    rtg = np.zeros(T)
                    rtg[-1] = rewards[-1]
                    for t in range(T - 2, -1, -1):
                        rtg[t] = rewards[t] + rtg[t + 1]
                    
                    trajectories.append({
                        "local": np.expand_dims(traj_dict["local"], axis=1),  # (T, 1, 9, 9)
                        "global": np.expand_dims(traj_dict["global"], axis=1),  # (T, 1, 21, 79)
                        "actions": traj_dict["actions"],
                        "rewards": rewards,
                        "returns_to_go": rtg
                    })
        
        print(f"Collected {len(trajectories)} successful trajectories")
        total_transitions = sum(len(t["actions"]) for t in trajectories)
        print(f"Total transitions: {total_transitions}")
        
        # Trajectory length statistics
        traj_lengths = [len(t["actions"]) for t in trajectories]
        print(f"Trajectory lengths: min={min(traj_lengths)}, max={max(traj_lengths)}, mean={np.mean(traj_lengths):.1f}")
        
        # Compute statistics for target return
        all_returns = [t["returns_to_go"][0] for t in trajectories]
        max_return = max(all_returns)
        mean_return = np.mean(all_returns)
        print(f"Return stats: max={max_return:.2f}, mean={mean_return:.2f}")
        
        # 2. Create Dataset and DataLoader
        dt_dataset = DTDataset(
            trajectories, 
            context_len=args.dt_context_len,
            max_ep_len=200,
            n_actions=cfg.action_dim
        )
        dt_dataloader = DataLoader(
            dt_dataset, 
            batch_size=args.dt_batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # 3. Initialize Model
        model = DecisionTransformer(
            n_actions=cfg.action_dim,
            embed_dim=args.dt_embed_dim,
            n_heads=args.dt_n_heads,
            n_layers=args.dt_n_layers,
            context_len=args.dt_context_len,
            max_ep_len=200
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # 4. Training Loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.dt_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.dt_epochs)
        
        print(f"\nTraining for {args.dt_epochs} epochs...")
        
        for epoch in range(args.dt_epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            
            for batch in dt_dataloader:
                local = batch["local"].to(device)
                glob = batch["global"].to(device)
                actions = batch["actions"].to(device)
                rtg = batch["returns_to_go"].to(device)
                timesteps = batch["timesteps"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Forward pass
                action_logits = model(rtg, local, glob, actions, timesteps, attention_mask)
                
                # Compute loss (cross-entropy on all non-padded positions)
                # Shift actions by 1 since we predict a_t from s_t
                target_actions = actions.clone()
                
                # Flatten for loss
                logits_flat = action_logits.view(-1, cfg.action_dim)
                targets_flat = target_actions.view(-1)
                mask_flat = attention_mask.view(-1)
                
                # Masked cross-entropy
                loss = nn.functional.cross_entropy(logits_flat, targets_flat, reduction='none')
                loss = (loss * mask_flat).sum() / mask_flat.sum()
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            avg_loss = total_loss / n_batches
            
            # Log to wandb
            wandb.log({
                "train/loss": avg_loss,
                "train/lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1
            })
            
            print(f"Epoch {epoch+1:02d}/{args.dt_epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
            # 5. Evaluation
            print("\n" + "="*60)
            print(f" FINAL EVALUATION: DECISION TRANSFORMER (seed={seed})")
            print("="*60)
            
            target_return = max_return  # Condition on achieving max return
            print(f"Evaluating with target return = {target_return:.2f}")
            
            for split_name, env_list in [("ID", ID_ENVS), ("OOD", OOD_ENVS)]:
                print(f"\n--- {split_name} Environments ---")
                for env_id in env_list:
                    short_name = env_id.replace("MiniHack-", "").replace("-v0", "")
                    
                    win_rate, avg_steps = evaluate_dt(
                        model, env_id, cfg, log_dir, 
                        target_return=target_return,
                        context_len=args.dt_context_len,
                        n_episodes=50,
                        device=device
                    )
                    
                    print(f"{short_name:25} | Win Rate: {win_rate*100:5.1f}% | Avg Steps: {avg_steps:5.1f}")
                    
                    # Store results for aggregation
                    seed_results[f"{split_name}/{short_name}/win_rate"] = win_rate * 100
                    seed_results[f"{split_name}/{short_name}/avg_steps"] = avg_steps
                    
                    wandb.log({
                        f"{split_name}/{short_name}/win_rate": win_rate * 100,
                        f"{split_name}/{short_name}/avg_steps": avg_steps
                    })
            
            print("="*60 + "\n")
            
            # Save model
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "n_actions": cfg.action_dim,
                    "embed_dim": args.dt_embed_dim,
                    "n_heads": args.dt_n_heads,
                    "n_layers": args.dt_n_layers,
                    "context_len": args.dt_context_len
                }
            }, f"models/{run.id}/dt_final.pt")
            
            all_seed_results.append(seed_results)
            wandb.finish()
            print(f"Decision Transformer seed {seed} complete!")
            continue  # Continue to next seed

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
                for traj_seed in range(50):  # Renamed to avoid shadowing loop var
                    traj_dict = collect_oracle_trajectory(env_id, traj_seed, cfg)
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
            print(f"Collected {len(acts_arr)} total expert transitions. Training BC natively...")
            policy = ActorCriticPolicy(
                observation_space=train_env.observation_space,
                action_space=train_env.action_space,
                lr_schedule=lambda _: 1e-3,
                features_extractor_class=MiniHackCNN,
                features_extractor_kwargs={"features_dim": 256},
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # 4. Native PyTorch Behavioral Cloning Loop
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
            n_epochs = 10
            
            policy.train()
            for epoch in range(n_epochs):
                total_loss = 0.0
                
                for batch in bc_dataloader:
                    obs = {k: v.to(policy.device) for k, v in batch["obs"].items()}
                    acts = batch["acts"].to(policy.device)
                    _, log_prob, _ = policy.evaluate_actions(obs, acts)
                    loss = -log_prob.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                avg_loss = total_loss / len(bc_dataloader)
                print(f"Epoch {epoch+1:02d}/{n_epochs} | BC Loss: {avg_loss:.4f}")
            
            model = policy  # Save reference for evaluation
            
            # 5. Manual Evaluation for BC (Win Rate & Avg Steps)
            print("\n" + "="*60)
            print(f" FINAL EVALUATION: BEHAVIORAL CLONING (seed={seed})")
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
                    
                    # Store results for aggregation
                    seed_results[f"{split_name}/{short_name}/win_rate"] = win_rate
                    seed_results[f"{split_name}/{short_name}/avg_steps"] = avg_steps
                    
                    wandb.log({
                        f"{split_name}/{short_name}/win_rate": win_rate,
                        f"{split_name}/{short_name}/avg_steps": avg_steps
                    })
            print("="*60 + "\n")
            
            # Save BC model and cleanup
            policy.save(f"models/{run.id}/bc_final_seed{seed}")
            all_seed_results.append(seed_results)
            wandb.finish()
            print(f"BC Seed {seed} complete!")

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
            
            # Save and Cleanup for RL
            model.save(f"models/{run.id}/{args.algo}_final_seed{seed}")
            train_env.close()
            
            all_seed_results.append(seed_results)
            wandb.finish()
            print(f"RL Seed {seed} complete!")

    # --- Aggregate Results Across Seeds ---
    if len(seeds) > 1 and len(all_seed_results) > 0:
        print("\n" + "="*70)
        print(" AGGREGATED RESULTS ACROSS ALL SEEDS")
        print("="*70)
        
        # Collect all metric keys (excluding 'seed')
        metric_keys = [k for k in all_seed_results[0].keys() if k != "seed"]
        
        if not metric_keys:
            print("Note: No per-environment metrics stored (RL uses callbacks for logging)")
        else:
            # Create aggregation summary
            agg_results = {}
            for key in metric_keys:
                values = [r[key] for r in all_seed_results if key in r]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    agg_results[key] = {"mean": mean_val, "std": std_val, "values": values}
            
            # Print aggregated results grouped by environment
            print(f"\nResults across {len(seeds)} seeds: {seeds}\n")
            
            # Group by ID/OOD
            for split in ["ID", "OOD"]:
                print(f"--- {split} Environments ---")
                env_metrics = {}
                for key, stats in agg_results.items():
                    if key.startswith(f"{split}/"):
                        parts = key.split("/")
                        env_name = parts[1]
                        metric_name = parts[2]
                        if env_name not in env_metrics:
                            env_metrics[env_name] = {}
                        env_metrics[env_name][metric_name] = stats
                
                for env_name, metrics in sorted(env_metrics.items()):
                    win_rate = metrics.get("win_rate", {})
                    avg_steps = metrics.get("avg_steps", {})
                    print(f"{env_name:25} | Win Rate: {win_rate.get('mean', 0):5.1f}% ± {win_rate.get('std', 0):4.1f} | Avg Steps: {avg_steps.get('mean', 0):5.1f} ± {avg_steps.get('std', 0):4.1f}")
                print()
            
            # Save aggregated results to file
            import json
            agg_output = {
                "algorithm": args.algo,
                "seeds": seeds,
                "n_seeds": len(seeds),
                "per_seed_results": all_seed_results,
                "aggregated": {k: {"mean": v["mean"], "std": v["std"]} for k, v in agg_results.items()}
            }
            
            agg_filename = f"results_{args.algo}_aggregated_{len(seeds)}seeds.json"
            with open(agg_filename, "w") as f:
                json.dump(agg_output, f, indent=2)
            print(f"Aggregated results saved to: {agg_filename}")
            
            # Log aggregated results to a summary wandb run
            summary_run = wandb.init(
                project="remdm-baselines",
                name=f"{args.algo.upper()}-MultiTask-SUMMARY",
                group=f"{args.algo.upper()}-MultiTask",
                config={"seeds": seeds, "n_seeds": len(seeds), **vars(args)},
                reinit=True
            )
            
            for key, stats in agg_results.items():
                wandb.log({
                    f"summary/{key}/mean": stats["mean"],
                    f"summary/{key}/std": stats["std"]
                })
            
            wandb.finish()
    
    print("\n" + "="*70)
    print(f" ALL {len(seeds)} SEED(S) COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()