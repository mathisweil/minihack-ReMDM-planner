"""One-shot uploader for the COMP0258 demo HuggingFace repo.

Stages everything `demo.ipynb` needs (source code, stripped checkpoint,
ablation assets) into a public HF repo so that the marker only has to
upload the `.ipynb` file to Colab.

Usage
-----
    export HF_TOKEN=hf_xxx           # write-token from huggingface.co/settings/tokens
    .venv/bin/python scripts/hf_upload_demo.py \
        --repo-id <user>/remdm-minihack-demo \
        --staging tmp/hf_staging

If `tmp/hf_staging` does not exist, run the staging step first (see the
`stage` subcommand below).

Files staged
------------
- src/                       # full project source
- configs/                   # YAML configs (defaults.yaml in particular)
- environments/              # custom .des scenario files (if any)
- main.py, pyproject.toml, README.md
- checkpoint_inference.pth   # ema_state_dict only (~21 MB)
- assets/{score_comparison,group_comparison,per_env_delta,
          grad_alignment,score_delta,gradient_conflict_map,
          repr_drift,diagnosis_decision_tree}.png
- assets/{main_results,hypothesis_verdicts}.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS_TO_STAGE = (
    "score_comparison.png",
    "group_comparison.png",
    "per_env_delta.png",
    "grad_alignment.png",
    "score_delta.png",
    "gradient_conflict_map.png",
    "repr_drift.png",
    "diagnosis_decision_tree.png",
    "main_results.csv",
    "hypothesis_verdicts.csv",
)

IGNORE_PATTERNS = (
    "__pycache__",
    "*.pyc",
    ".DS_Store",
    "*.egg-info",
)


def _copytree(src: Path, dst: Path) -> None:
    """Copy *src* tree into *dst*, skipping IGNORE_PATTERNS."""
    shutil.copytree(
        src, dst,
        ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
        dirs_exist_ok=True,
    )


def stage(staging_dir: Path) -> None:
    """Build the upload payload at *staging_dir*.

    Args:
        staging_dir: Where to assemble the upload tree.
    """
    if staging_dir.exists():
        logger.info(f"clearing existing staging dir: {staging_dir}")
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    (staging_dir / "assets").mkdir()

    # Source trees
    for sub in ("src", "configs", "environments"):
        src = PROJECT_ROOT / sub
        if src.exists():
            _copytree(src, staging_dir / sub)
            logger.info(f"  staged {sub}/")

    # Top-level files
    for fname in ("main.py", "pyproject.toml", "README.md"):
        src = PROJECT_ROOT / fname
        if src.exists():
            shutil.copy2(src, staging_dir / fname)
            logger.info(f"  staged {fname}")

    # Stripped checkpoint (must be created beforehand by strip_checkpoint())
    ckpt_src = PROJECT_ROOT / "checkpoint_inference.pth"
    if not ckpt_src.exists():
        logger.info("  checkpoint_inference.pth missing -- stripping now")
        strip_checkpoint(
            PROJECT_ROOT / "checkpoint_final" / "online" / "final.pth",
            ckpt_src,
        )
    shutil.copy2(ckpt_src, staging_dir / "checkpoint_inference.pth")
    logger.info(
        f"  staged checkpoint_inference.pth "
        f"({ckpt_src.stat().st_size / 1e6:.1f} MB)"
    )

    # Ablation assets
    asset_dir = (
        PROJECT_ROOT / "experiments" / "rl_finetuning"
        / "outputs" / "minihack_final"
    )
    for name in ASSETS_TO_STAGE:
        src = asset_dir / name
        if not src.exists():
            logger.warning(f"  asset missing, skipping: {name}")
            continue
        shutil.copy2(src, staging_dir / "assets" / name)
        logger.info(f"  staged assets/{name}")

    total_mb = sum(
        f.stat().st_size for f in staging_dir.rglob("*") if f.is_file()
    ) / 1e6
    logger.info(f"\nstaging complete: {staging_dir}  ({total_mb:.1f} MB total)")


def strip_checkpoint(src: Path, dst: Path) -> None:
    """Save only ``ema_state_dict`` from *src* to *dst*.

    Args:
        src: Full DAgger checkpoint path.
        dst: Output path for the stripped checkpoint.

    Raises:
        FileNotFoundError: If *src* does not exist.
        KeyError: If ``ema_state_dict`` is missing from *src*.
    """
    import torch

    if not src.exists():
        raise FileNotFoundError(f"checkpoint not found: {src}")
    full = torch.load(src, map_location="cpu", weights_only=False)
    if "ema_state_dict" not in full:
        raise KeyError(f"checkpoint {src} has no ema_state_dict key")
    torch.save({"ema_state_dict": full["ema_state_dict"]}, dst)
    logger.info(
        f"stripped {src.name} ({src.stat().st_size / 1e6:.1f} MB) -> "
        f"{dst.name} ({dst.stat().st_size / 1e6:.1f} MB)"
    )


def upload(repo_id: str, staging_dir: Path, token: str) -> None:
    """Push *staging_dir* to a public HF model repo.

    Args:
        repo_id: HuggingFace repo ID, e.g. ``"user/remdm-minihack-demo"``.
        staging_dir: Local directory to upload.
        token: HF API token with write permission.

    Raises:
        FileNotFoundError: If *staging_dir* does not exist.
    """
    from huggingface_hub import HfApi, create_repo

    if not staging_dir.exists():
        raise FileNotFoundError(
            f"staging dir not found: {staging_dir}. Run `stage` first."
        )

    api = HfApi(token=token)
    logger.info(f"creating/ensuring public repo: {repo_id}")
    create_repo(
        repo_id=repo_id, token=token, exist_ok=True, private=False,
        repo_type="model",
    )

    logger.info(f"uploading {staging_dir} -> {repo_id}")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=list(IGNORE_PATTERNS),
        commit_message="Demo notebook payload (source + checkpoint + assets)",
    )
    logger.info(f"\ndone -- repo URL: https://huggingface.co/{repo_id}")
    logger.info(
        f"now find-and-replace TODO_HF_REPO_ID with {repo_id!r} in demo.ipynb"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-id",
        help="HF repo ID, e.g. user/remdm-minihack-demo",
    )
    p.add_argument(
        "--staging",
        default=str(PROJECT_ROOT / "tmp" / "hf_staging"),
        help="Staging directory (default: tmp/hf_staging)",
    )
    p.add_argument(
        "--stage-only",
        action="store_true",
        help="Only build the staging dir, do not upload.",
    )
    p.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip staging, only upload existing dir.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for the demo upload script."""
    args = parse_args()
    staging_dir = Path(args.staging).resolve()

    if not args.upload_only:
        stage(staging_dir)

    if args.stage_only:
        return

    if not args.repo_id:
        logger.error("--repo-id is required for upload")
        sys.exit(2)

    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    if not token:
        logger.error(
            "HF_TOKEN env var not set. Get a write token from "
            "https://huggingface.co/settings/tokens and run:\n"
            "    export HF_TOKEN=hf_xxx"
        )
        sys.exit(2)

    upload(args.repo_id, staging_dir, token)


if __name__ == "__main__":
    main()
