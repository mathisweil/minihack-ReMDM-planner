"""Dual-stream denoising transformer for MiniHack.

Architecture follows the
Craftax denoiser conventions (forward return format, obs-encoder pattern)
while using the MiniHack dual-stream design (local CNN + gated global
CNN + auxiliary goal head).
"""

from __future__ import annotations

import copy
import logging
import shutil
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class LocalDiffusionPlannerWithGlobal(nn.Module):
    """Dual-stream transformer for masked diffusion action planning.

    Combines a local 9x9 glyph crop with a gated global 21x79 map
    context. Produces action logits and an auxiliary staircase-coordinate
    prediction.

    Architecture:
        Local stream:  Embedding(6000,64) -> CNN(64->32->64) -> Linear -> 1 token
        Global stream: Embedding(6000,32) -> CNN(32->32->64) -> Pool(2,4)
                       -> Linear -> 8 tokens, gated by sigmoid(learnable scalar)
        Goal head:     mean(global_tokens) -> MLP -> [B,2] (before gate)
        Action stream: Embedding(14, n_embd) + timestep + position
        Transformer:   concat all -> TransformerEncoder -> last 64 tokens -> head

    Args:
        cfg: Config namespace with ``action_dim``, ``n_embd``, ``n_head``,
            ``n_layer``, ``n_global_tokens``, ``seq_len``,
            ``global_gate_init``, ``num_diffusion_steps``.
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        action_dim = cfg.action_dim
        n_embd = cfg.n_embd
        n_head = cfg.n_head
        n_layer = cfg.n_layer
        n_global_tokens = cfg.n_global_tokens
        seq_len = cfg.seq_len

        assert n_embd % n_head == 0, (
            f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        )

        self.n_global_tokens = n_global_tokens

        self.embedding = nn.Embedding(6000, 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, n_embd),
        )

        self.action_emb = nn.Embedding(action_dim + 2, n_embd)
        self.timestep_emb = nn.Embedding(
            cfg.num_diffusion_steps,
            n_embd,
        )
        self.pos_emb = nn.Embedding(seq_len, n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            dropout=getattr(cfg, "dropout", 0.0),
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(n_embd, action_dim)

        self.global_embedding = nn.Embedding(6000, 32)
        self.global_cnn = nn.Sequential(
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((2, 4))
        self.global_proj = nn.Linear(64, n_embd)
        self.global_gate = nn.Parameter(torch.tensor(cfg.global_gate_init))

        self.goal_head = nn.Sequential(
            nn.Linear(n_embd, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        local_obs: Tensor,
        global_obs: Tensor,
        action_seq: Tensor,
        t_discrete: int | Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass producing action logits and goal prediction.

        Args:
            local_obs: Local glyph crop. Shape ``[B, 9, 9]``, int.
            global_obs: Full glyph map. Shape ``[B, 21, 79]``, int.
            action_seq: Noisy action sequence. Shape ``[B, seq_len]``, int.
            t_discrete: Discrete timestep index (scalar int or ``[B]``).

        Returns:
            Dict with keys:
            - ``"actions"``: ``[B, seq_len, action_dim]`` logits.
            - ``"goal_pred"``: ``[B, 2]`` normalised staircase coords.
        """
        B, Seq = action_seq.shape
        device = local_obs.device

        # Local stream -> [B, 1, n_embd]
        x_local = self.embedding(local_obs)  # [B, 9, 9, 64]
        x_local = x_local.permute(0, 3, 1, 2)  # [B, 64, 9, 9]
        local_token = self.cnn(x_local).unsqueeze(1)  # [B, 1, n_embd]

        # Global stream -> [B, 8, n_embd]
        x_global = self.global_embedding(global_obs)  # [B, 21, 79, 32]
        x_global = x_global.permute(0, 3, 1, 2)  # [B, 32, 21, 79]
        gf = self.global_cnn(x_global)  # [B, 64, H', W']
        gf = self.global_pool(gf)  # [B, 64, 2, 4]
        global_tokens = gf.permute(0, 2, 3, 1)  # [B, 2, 4, 64]
        global_tokens = global_tokens.reshape(B, self.n_global_tokens, -1)  # [B, 8, 64]
        global_tokens = self.global_proj(global_tokens)  # [B, 8, n_embd]

        # Aux goal head (before gate for direct gradient to CNN)
        goal_pred = self.goal_head(global_tokens.mean(dim=1))  # [B, 2]

        # Apply gate
        gate = torch.sigmoid(self.global_gate)
        global_tokens = global_tokens * gate  # [B, 8, n_embd]

        # Action stream -> [B, seq_len, n_embd]
        positions = (
            torch.arange(
                Seq,
                device=device,
            )
            .unsqueeze(0)
            .expand(B, -1)
        )  # [B, seq_len]

        if isinstance(t_discrete, int):
            t_tensor = torch.full(
                (B,),
                t_discrete,
                dtype=torch.long,
                device=device,
            )
        else:
            t_tensor = t_discrete.long().to(device)

        seq_emb = (
            self.action_emb(action_seq)
            + self.timestep_emb(t_tensor).unsqueeze(1)
            + self.pos_emb(positions)
        )  # [B, seq_len, n_embd]

        # Concatenate: [local(1), global(8), actions(seq_len)]
        x = torch.cat(
            [local_token, global_tokens, seq_emb],
            dim=1,
        )  # [B, 1+8+seq_len, n_embd]

        # Transformer
        out = self.transformer(x)  # [B, 1+8+seq_len, n_embd]

        # Take last seq_len tokens for action predictions
        n_prefix = 1 + self.n_global_tokens
        action_logits = self.head(out[:, n_prefix:, :])  # [B, seq_len, action_dim]

        return {"actions": action_logits, "goal_pred": goal_pred}


class LocalDiffusionPlanner(nn.Module):
    """Local-only ablation model (no global stream, no goal head).

    Args:
        cfg: Config namespace.
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        action_dim = cfg.action_dim
        n_embd = cfg.n_embd
        seq_len = cfg.seq_len

        self.embedding = nn.Embedding(6000, 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, n_embd),
        )
        self.action_emb = nn.Embedding(action_dim + 2, n_embd)
        self.timestep_emb = nn.Embedding(cfg.num_diffusion_steps, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=cfg.n_head,
            dim_feedforward=n_embd * 4,
            dropout=getattr(cfg, "dropout", 0.0),
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layer,
        )
        self.head = nn.Linear(n_embd, action_dim)

    def forward(
        self,
        local_obs: Tensor,
        global_obs: Tensor,
        action_seq: Tensor,
        t_discrete: int | Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass (ignores global_obs).

        Args:
            local_obs: ``[B, 9, 9]`` int.
            global_obs: ``[B, 21, 79]`` int (ignored).
            action_seq: ``[B, seq_len]`` int.
            t_discrete: Timestep index.

        Returns:
            Dict with ``"actions"`` key only (no goal_pred).
        """
        B, Seq = action_seq.shape
        device = local_obs.device

        x_state = self.embedding(local_obs).permute(0, 3, 1, 2)
        state_emb = self.cnn(x_state).unsqueeze(1)  # [B, 1, n_embd]

        positions = (
            torch.arange(
                Seq,
                device=device,
            )
            .unsqueeze(0)
            .expand(B, -1)
        )

        if isinstance(t_discrete, int):
            t_tensor = torch.full(
                (B,),
                t_discrete,
                dtype=torch.long,
                device=device,
            )
        else:
            t_tensor = t_discrete.long().to(device)

        seq_emb = (
            self.action_emb(action_seq)
            + self.timestep_emb(t_tensor).unsqueeze(1)
            + self.pos_emb(positions)
        )
        x = torch.cat([state_emb, seq_emb], dim=1)
        out = self.transformer(x)
        return {"actions": self.head(out[:, 1:, :])}


def make_model(cfg: SimpleNamespace) -> nn.Module:
    """Instantiate the MiniHack denoising model.

    Args:
        cfg: Config namespace.

    Returns:
        ``LocalDiffusionPlannerWithGlobal``, or ``LocalDiffusionPlanner``
        when ``cfg.use_global_stream`` is False (local-only ablation).
    """
    if not getattr(cfg, "use_global_stream", True):
        return LocalDiffusionPlanner(cfg)
    return LocalDiffusionPlannerWithGlobal(cfg)


def _has_c_compiler() -> bool:
    """Check whether a C compiler is reachable by Triton.

    Checks the ``CC`` env var (set by conda activation scripts),
    then falls back to ``cc`` and ``gcc`` on ``PATH``.
    """
    import os

    cc_env = os.environ.get("CC")
    if cc_env and shutil.which(cc_env):
        return True
    return shutil.which("cc") is not None or shutil.which("gcc") is not None


def try_compile(model: nn.Module, cfg: SimpleNamespace) -> nn.Module:
    """Wrap *model* with ``torch.compile`` if enabled and a C compiler exists.

    Falls back to the uncompiled model when ``torch.compile`` is
    unavailable or Triton cannot find a C compiler (common on managed
    GPU nodes that lack ``gcc``/``cc``).

    Args:
        model: The raw (uncompiled) model.
        cfg: Config namespace; reads ``torch_compile`` bool.

    Returns:
        Compiled model, or *model* unchanged on fallback.
    """
    if not getattr(cfg, "torch_compile", False):
        return model
    if not hasattr(torch, "compile"):
        return model
    if not _has_c_compiler():
        logger.warning(
            "torch.compile requested but no C compiler found "
            "(CC env var, cc, gcc); falling back to eager mode"
        )
        return model
    logger.info("Compiling model with torch.compile")
    return torch.compile(model, mode="default")  # type: ignore[return-value]


class ModelEMA:
    """Exponential moving average of model parameters.

    Maintains a shadow copy of parameters updated as
    ``theta_ema <- decay * theta_ema + (1 - decay) * theta``.

    A parameter that never moves still moves its shadow. The update then
    computes ``decay * x + (1 - decay) * x``, which is not exactly ``x`` in
    float32. Measured on the shipped ablation architecture with the model
    held static for 500 updates at decay 0.999: 45 of 72 shadow tensors
    drift, the largest by 4.6e-05 on ``embedding.weight`` and most by around
    1e-07.

    This matters only for the group-C ablations, which are the only runs that
    freeze parameters. Their frozen tensors are exactly frozen -- every one
    measures a parameter delta of exactly 0.0 -- but the suite evaluates the
    EMA weights (``ablations/training.py`` applies them before each eval), so
    "frozen" is exact for the trained weights and approximate to the above
    magnitude for the evaluated ones. Left as it is: the drift is orders of
    magnitude below the resolution of any reported score, and excluding
    non-trainable tensors from the update would change the evaluated weights
    of every group-C run, which is a result-affecting change to take
    deliberately rather than a cleanup.

    Args:
        model: Source model.
        decay: EMA decay factor (default 0.999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self._decay = decay
        self._shadow: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            self._shadow[name] = param.data.clone()
        # Cached operand lists for the fused update, rebuilt only
        # when `update` is called with a different model object.
        self._fused_src: nn.Module | None = None
        self._fused_shadow: list[Tensor] = []
        self._fused_params: list[Tensor] = []

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters from *model*.

        Args:
            model: Source model whose parameters are blended in.
        """
        # One fused kernel per operation instead of a Python loop
        # over every parameter tensor. The loop issued two kernel launches
        # per parameter after every gradient step; `torch._foreach_*` issues
        # two in total. The arithmetic per element is unchanged.
        if self._fused_src is not model:
            shadow, params = [], []
            for name, param in model.named_parameters():
                shadow.append(self._shadow[name])
                params.append(param.data)
            self._fused_shadow = shadow
            self._fused_params = params
            self._fused_src = model
        torch._foreach_mul_(self._fused_shadow, self._decay)
        torch._foreach_add_(
            self._fused_shadow,
            self._fused_params,
            alpha=1.0 - self._decay,
        )

    def apply_to(self, model: nn.Module) -> None:
        """Copy shadow parameters into *model* (for inference).

        Args:
            model: Target model to overwrite.
        """
        for name, param in model.named_parameters():
            param.data.copy_(self._shadow[name])

    def state_dict(self) -> dict[str, Tensor]:
        """Return shadow parameter dict for serialisation.

        Returns:
            Dict mapping parameter names to EMA tensors.
        """
        return {k: v.clone() for k, v in self._shadow.items()}

    def load_state_dict(self, sd: dict[str, Tensor]) -> None:
        """Restore shadow parameters from *sd*.

        Strict, like ``nn.Module.load_state_dict``: keys *sd* omits and keys
        the shadow does not have are both errors. The shadow is what
        evaluation runs on (``apply_to``), so a truncated or foreign dict
        would otherwise leave freshly-initialised EMA weights in place,
        score them, and report the numbers as a result.

        Args:
            sd: State dict from a prior ``state_dict()`` call.

        Raises:
            RuntimeError: If *sd* omits a shadow key or carries an extra one.
        """
        missing = sorted(set(self._shadow) - set(sd))
        unexpected = sorted(set(sd) - set(self._shadow))
        if missing or unexpected:
            detail = ""
            if missing:
                detail += f"\n\tMissing key(s) in state_dict: {', '.join(missing)}."
            if unexpected:
                detail += (
                    f"\n\tUnexpected key(s) in state_dict: {', '.join(unexpected)}."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for ModelEMA:{detail}")
        for k, v in sd.items():
            self._shadow[k].copy_(v)

    def parameters(self):
        """Iterate over shadow parameter tensors.

        Yields:
            EMA parameter tensors.
        """
        yield from self._shadow.values()

    def make_eval_model(self, model: nn.Module) -> nn.Module:
        """Return a deep copy of *model* with EMA weights applied.

        Args:
            model: Template model (architecture).

        Returns:
            New model with shadow parameters.
        """
        eval_model = copy.deepcopy(model)
        self.apply_to(eval_model)
        eval_model.eval()
        return eval_model
