"""Hugging Face Hub upload utility.

Thin, decoupled upload script for checkpoints and eval JSONs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_to_hub(
    local_path: str,
    hub_path: str,
    repo_id: str,
    token: str,
) -> None:
    """Upload a single file to HF Hub.

    Args:
        local_path: Local file path.
        hub_path: Destination path in the repo.
        repo_id: HF Hub repo ID (e.g. ``"user/repo"``).
        token: HF API token.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hub_path,
            repo_id=repo_id,
        )
        logger.info(f"Uploaded {local_path} -> {repo_id}/{hub_path}")
    except Exception:
        logger.error(
            f"HF upload failed: {local_path}", exc_info=True,
        )


def upload_run(
    run_dir: str,
    repo_id: str,
    run_id: str,
    token: str,
) -> None:
    """Upload all ``.pth`` and ``.json`` files from a run directory.

    Files are placed under ``generalization/{run_id}/`` in the repo.

    Args:
        run_dir: Local directory containing checkpoint files.
        repo_id: HF Hub repo ID.
        run_id: Unique run identifier.
        token: HF API token.
    """
    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        logger.error(f"Run directory not found: {run_dir}")
        return

    for fpath in run_path.iterdir():
        if fpath.suffix in (".pth", ".json"):
            hub_path = f"generalization/{run_id}/{fpath.name}"
            upload_to_hub(str(fpath), hub_path, repo_id, token)


def maybe_upload_checkpoint(
    checkpoint_dir: str,
    run_id: str | None,
    repo_id: str | None = None,
) -> None:
    """Upload checkpoint if HF_TOKEN and run_id are set.

    Called from ``Trainer.save_checkpoint``. No-op if credentials
    are missing.

    Args:
        checkpoint_dir: Local checkpoint directory.
        run_id: Optional run identifier.
        repo_id: Optional HF repo ID.
    """
    token = os.environ.get("HF_TOKEN")
    if not token or not run_id:
        return
    if repo_id is None:
        repo_id = "remdm-minihack/checkpoints"
    upload_run(checkpoint_dir, repo_id, run_id, token)
