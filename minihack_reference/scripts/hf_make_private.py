#!/usr/bin/env python3
"""
Make a Hugging Face repo private.
Run after manually creating org and transferring repo.

Usage:
  HF_TOKEN=hf_xxx python hf_make_private.py remdm-research/remdm-minihack
"""
import os
import sys

def main():
    repo_id = sys.argv[1] if len(sys.argv) > 1 else "remdm-research/remdm-minihack"
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN env var")
        sys.exit(1)

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.update_repo_visibility(repo_id=repo_id, private=True)
    print(f"Repository {repo_id} is now private.")

if __name__ == "__main__":
    main()
