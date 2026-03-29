"""
ReMDM Planner — Transformer-based Masked Discrete Diffusion for MiniHack.
"""

import torch
import torch.nn as nn

from config import HYPERPARAMS


class LocalDiffusionPlannerWithGlobal(nn.Module):
    """Local 9×9 crop + gated global 21×79 map. Dual-stream transformer for action-sequence diffusion.

    Local:
    Global: 21×79 map → 8 tokens via CNN + adaptive pool. Learnable gate (init ~0.05).
    Aux goal head: predicts normalized staircase (for training aux loss).
    """

    def __init__(self, action_dim, n_embd=256, n_head=4, n_layer=4, n_global_tokens=8):
        super().__init__()
        self.n_global_tokens = n_global_tokens

        # Local stream: 9×9 crop → 1 token
        self.embedding = nn.Embedding(6000, 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 9 * 9, n_embd)
        )
        self.action_emb = nn.Embedding(action_dim + 2, n_embd)
        self.timestep_emb = nn.Embedding(100, n_embd)
        self.pos_emb = nn.Embedding(HYPERPARAMS["seq_len"], n_embd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, batch_first=True,
            activation="gelu", norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.head = nn.Linear(n_embd, action_dim)

        # Global stream: 21×79 map → 8 tokens
        self.global_embedding = nn.Embedding(6000, 32)
        self.global_cnn = nn.Sequential(
            nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((2, 4))
        self.global_proj = nn.Linear(64, n_embd)
        self.global_gate = nn.Parameter(torch.tensor(-3.0))
        self.goal_head = nn.Sequential(
            nn.Linear(n_embd, 128), nn.GELU(), nn.Linear(128, 2),
        )

    def forward(self, local_obs, global_obs, actions, t_steps):
        x_local = self.embedding(local_obs).permute(0, 3, 1, 2)
        local_token = self.cnn(x_local).unsqueeze(1)

        x_global = self.global_embedding(global_obs).permute(0, 3, 1, 2)
        gf = self.global_cnn(x_global)
        gf = self.global_pool(gf)
        B = gf.shape[0]
        global_tokens = self.global_proj(
            gf.permute(0, 2, 3, 1).reshape(B, self.n_global_tokens, -1)
        )
        goal_pred = self.goal_head(global_tokens.mean(dim=1))  # before gate — aux loss has direct gradient to CNN
        gate = torch.sigmoid(self.global_gate)
        global_tokens = global_tokens * gate

        B, Seq = actions.shape
        positions = torch.arange(Seq, device=local_obs.device).unsqueeze(0).expand(B, -1)
        seq_emb = self.action_emb(actions) + self.timestep_emb(t_steps).unsqueeze(1) + self.pos_emb(positions)
        x = torch.cat([local_token, global_tokens, seq_emb], dim=1)
        n_prefix = 1 + self.n_global_tokens
        action_preds = self.head(self.transformer(x)[:, n_prefix:, :])
        return {"actions": action_preds, "goal_pred": goal_pred}

class LocalDiffusionPlanner(nn.Module):
    def __init__(self, action_dim, n_embd=256, n_head=4, n_layer=4):
        super().__init__()
        self.embedding = nn.Embedding(6000, 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, n_embd)
        )
        self.action_emb = nn.Embedding(action_dim + 2, n_embd)
        self.timestep_emb = nn.Embedding(100, n_embd)
        self.pos_emb = nn.Embedding(HYPERPARAMS["seq_len"], n_embd)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.head = nn.Linear(n_embd, action_dim)

    def forward(self, local_obs, global_obs, actions, t_steps):
        # We deliberately IGNORE global_obs here
        x_state = self.embedding(local_obs).permute(0, 3, 1, 2)
        state_emb = self.cnn(x_state).unsqueeze(1)
        B, Seq = actions.shape
        positions = torch.arange(Seq, device=local_obs.device).unsqueeze(0).expand(B, -1)
        seq_emb = self.action_emb(actions) + self.timestep_emb(t_steps).unsqueeze(1) + self.pos_emb(positions)
        x = torch.cat([state_emb, seq_emb], dim=1)
        out = self.transformer(x)
        return self.head(out[:, 1:, :])

class DualStreamPlanner(nn.Module):
    def __init__(self, action_dim, n_embd=256, n_head=4, n_layer=4):
        super().__init__()

        # Local Stream (Crop)
        self.local_embedding = nn.Embedding(6000, 32)
        self.local_cnn = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, n_embd // 2)
        )

        # Global Stream (Full Map)
        self.global_embedding = nn.Embedding(6000, 32)
        self.global_cnn = nn.Sequential(
            nn.Conv2d(32, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Flatten(),
            nn.Linear(3840, n_embd // 2)
        )

        self.action_emb = nn.Embedding(action_dim + 2, n_embd)
        self.timestep_emb = nn.Embedding(100, n_embd)
        self.pos_emb = nn.Embedding(HYPERPARAMS["seq_len"], n_embd)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.head = nn.Linear(n_embd, action_dim)

    def forward(self, local_obs, global_obs, actions, t_steps):
        # 1. Local
        x_local = self.local_embedding(local_obs).permute(0, 3, 1, 2)
        local_feat = self.local_cnn(x_local)

        # 2. Global
        x_global = self.global_embedding(global_obs).permute(0, 3, 1, 2)
        global_feat = self.global_cnn(x_global)

        # 3. Fuse
        state_emb = torch.cat([local_feat, global_feat], dim=1).unsqueeze(1)

        # 4. Transformer
        B, Seq = actions.shape
        positions = torch.arange(Seq, device=local_obs.device).unsqueeze(0).expand(B, -1)
        seq_emb = self.action_emb(actions) + self.timestep_emb(t_steps).unsqueeze(1) + self.pos_emb(positions)
        x = torch.cat([state_emb, seq_emb], dim=1)
        out = self.transformer(x)
        return self.head(out[:, 1:, :])