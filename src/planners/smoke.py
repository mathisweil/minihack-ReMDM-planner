import logging
import shutil
import tempfile

import torch

from src.buffer import ReplayBuffer
from src.curriculum import DynamicCurriculum
from src.envs.minihack_env import collect_oracle_trajectory
from src.models.denoiser import ModelEMA, make_model, try_compile
from src.planners.collect import DataCollector
from src.planners.inference import Evaluator, format_eval_results
from src.planners.logging import Logger
from src.planners.online import Trainer

logger = logging.getLogger(__name__)


def run_smoke(cfg) -> None:
    """Smoke test: collect oracle data, train briefly, eval.

    Smoke runs are throwaway, so checkpoints, config snapshots and eval
    JSONs go to a temporary directory instead of the repository tree
    (step-7 finding N6 / PARITY "Smoke-mode side effects"), **and that
    directory is removed when the run ends**.

    It was not. Every smoke run since the directory was introduced left one
    behind: 183 of them, 9.3 GB, had accumulated by 2026-08-19 -- enough to
    exhaust the 100 GB project quota and fail the test suite on
    `OSError: [Errno 122] Disk quota exceeded`, and 42 more (3.8 GB)
    reappeared within a day of the first clear-out. craftax's smoke path
    has always removed its own temporary expert directory the same way; this
    is that pattern, applied to the artefact directory.

    `ignore_errors=True` matches craftax and keeps a cleanup failure from
    masking the run's own result.
    """
    cfg.checkpoint_dir = tempfile.mkdtemp(prefix="remdm-smoke-")
    logger.info(f"Smoke artefacts -> {cfg.checkpoint_dir}")
    try:
        _run_smoke(cfg)
    finally:
        shutil.rmtree(cfg.checkpoint_dir, ignore_errors=True)


def _run_smoke(cfg) -> None:
    """The smoke run itself.

    Split out so `run_smoke` owns the temporary directory's whole lifetime;
    `cfg.checkpoint_dir` already points at it on entry.
    """

    device = cfg.device
    logger.info(f"Smoke test on {device}")

    # Collect a few oracle trajectories into the buffer
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.seq_len, cfg.pad_token)
    for i, env_id in enumerate(cfg.id_envs):
        traj = collect_oracle_trajectory(env_id, seed=i, cfg=cfg)
        if traj is not None:
            buffer.add(traj)
    logger.info(f"Buffer seeded with {len(buffer)} windows")

    raw_model = make_model(cfg).to(device)

    model = try_compile(raw_model, cfg)

    ema = ModelEMA(raw_model, decay=cfg.ema_decay)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg.dagger_lr,
        weight_decay=cfg.weight_decay,
    )
    curriculum = DynamicCurriculum(
        cfg.id_envs,
        cfg.curriculum_queue_size,
        cfg.curriculum_preseed,
    )

    collector = DataCollector(ema, raw_model, buffer, curriculum, cfg, device)
    evaluator = Evaluator()
    log = Logger(cfg)

    trainer = Trainer(
        model,
        ema,
        optimizer,
        None,
        buffer,
        collector,
        evaluator,
        log,
        cfg,
        device,
        raw_model=raw_model,
    )
    trainer.train(start_iter=0)

    # Final eval
    eval_model = ema.make_eval_model(raw_model)
    results = evaluator.evaluate(
        cfg.id_envs,
        eval_model,
        cfg.eval_episodes_per_env,
        cfg,
        device,
    )
    print(format_eval_results(results, label="Smoke"))
    log.log_eval(results, step=0, prefix="smoke_eval")
    mean_wr = (
        float(sum(s["win_rate"] for s in results.values()) / len(results))
        if results
        else 0.0
    )
    log.log({"smoke_eval/mean_win_rate": mean_wr}, step=0)
    log.finish()
