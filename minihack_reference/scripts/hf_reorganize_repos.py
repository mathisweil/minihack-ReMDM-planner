#!/usr/bin/env python3
"""
Reorganize Hugging Face remdm-minihack into two repos:
  - remdm-minihack/generalization (empty) — for DAgger generalization run uploads
  - remdm-minihack/archive_1 (all files from source) — checkpoints, datasets, etc.

Usage:
  HF_TOKEN=hf_xxx python hf_reorganize_repos.py [source_repo_id]

Examples:
  HF_TOKEN=hf_xxx python hf_reorganize_repos.py
  HF_TOKEN=hf_xxx python hf_reorganize_repos.py remdm-minihack/ReMDM-MiniHack
  HF_TOKEN=hf_xxx python hf_reorganize_repos.py remdm-research/remdm-minihack

Requires: HF token with write access to remdm-minihack org.
"""
import os
import sys
import tempfile
from pathlib import Path

def main():
    # Source repo: pass as arg, or use common options
    # remdm-research/remdm-minihack if transferred there; remdm-minihack/ReMDM-MiniHack otherwise
    source_repo = sys.argv[1] if len(sys.argv) > 1 else "remdm-minihack/ReMDM-MiniHack"
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN env var (requires write access to remdm-minihack org)")
        sys.exit(1)

    from huggingface_hub import HfApi, list_repo_files, hf_hub_download

    api = HfApi(token=token)
    org = "remdm-minihack"

    # 1. Create empty generalization repo
    gen_repo = f"{org}/generalization"
    try:
        api.create_repo(repo_id=gen_repo, private=True, exist_ok=True)
        print(f"Created/verified: {gen_repo} (empty)")
    except Exception as e:
        print(f"generalization: {e}")
        if "401" in str(e) or "403" in str(e):
            print("  -> Ensure HF_TOKEN has write access to remdm-minihack org")
            sys.exit(1)

    # 2. Create archive_1 repo
    archive_repo = f"{org}/archive_1"
    try:
        api.create_repo(repo_id=archive_repo, private=True, exist_ok=True)
        print(f"Created/verified: {archive_repo}")
    except Exception as e:
        print(f"archive_1: {e}")
        sys.exit(1)

    # 3. List all files in source repo (exclude .git)
    try:
        files = list_repo_files(repo_id=source_repo, token=token)
        files = [f for f in files if not f.startswith(".git")]
    except Exception as e:
        print(f"Cannot list {source_repo}: {e}")
        print("  -> Check repo_id (e.g. remdm-minihack/ReMDM-MiniHack)")
        sys.exit(1)

    if not files:
        print(f"No files in {source_repo}, archive_1 stays empty.")
        return

    # 4. Download source to temp dir, then upload to archive_1
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for fn in files:
            try:
                local = hf_hub_download(repo_id=source_repo, filename=fn, token=token, local_dir=tmp_path)
            except Exception as e:
                print(f"  Skip {fn}: {e}")
                continue

        # Upload: use upload_folder if we have dir structure, else upload each file
        # snapshot_download gives flat structure; list_repo_files gives paths
        # hf_hub_download with local_dir creates the folder structure
        print(f"Uploading {len(files)} files to {archive_repo}...")
        for fn in files:
            local_path = tmp_path / fn
            if local_path.exists():
                try:
                    api.upload_file(
                        path_or_fileobj=str(local_path),
                        path_in_repo=fn,
                        repo_id=archive_repo,
                        token=token,
                    )
                    print(f"  Uploaded: {fn}")
                except Exception as e:
                    print(f"  Fail {fn}: {e}")

    print("Done.")
    print(f"  {gen_repo}  <- empty")
    print(f"  {archive_repo}  <- full copy of {source_repo}")

if __name__ == "__main__":
    main()
