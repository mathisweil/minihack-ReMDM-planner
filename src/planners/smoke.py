import logging

import torch

from src.buffer import ReplayBuffer
from src.curriculum import DynamicCurriculum
from src.envs.minihack_env import collect_oracle_trajectory
from src.models.denoiser import ModelEMA, make_model
from src.planners.collect import DataCollector
from src.planners.inference import Evaluator, format_eval_results
from src.planners.logging import Logger
from src.planners.online import Trainer

logger = logging.getLogger(__name__)


def run_smoke(cfg) -> None:
    """Smoke test: collect oracle data, train briefly, eval."""

    device = cfg.device
    logger.info(f"Smoke test on {device}")

    # Collect a few oracle trajectories into the buffer
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seq_len, cfg.pad_token)
    for i, env_id in enumerate(cfg.id_envs):
        traj = collect_oracle_trajectory(env_id, seed=i, cfg=cfg)
        if traj is not None:
            buffer.add(traj)
    logger.info(f"Buffer seeded with {len(buffer)} windows")

    model = make_model(cfg).to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.dagger_lr)
    curriculum = DynamicCurriculum(cfg.id_envs, cfg.curriculum_queue_size)

    eval_model = ema.make_eval_model(model)
    collector = DataCollector(eval_model, buffer, curriculum, cfg, device)
    evaluator = Evaluator()
    log = Logger(cfg)

    trainer = Trainer(
        model, ema, optimizer, None, buffer, collector,
        evaluator, log, cfg, device,
    )
    trainer.train(start_iter=0)

    # Final eval
    eval_model = ema.make_eval_model(model)
    results = evaluator.evaluate(
        cfg.id_envs, eval_model, cfg.eval_episodes_per_env, cfg, device,
    )
    print(format_eval_results(results, label="Smoke"))
    log.log_eval(results, step=0, prefix="smoke_eval")
    mean_wr = float(sum(s["win_rate"] for s in results.values()) / len(results)) if results else 0.0
    log.log({"smoke_eval/mean_win_rate": mean_wr}, step=0)
    log.finish()