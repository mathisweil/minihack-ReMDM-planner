"""Verify the working tree against the stored parity references.

Usage:
    uv run python parity/check.py [--fast]

--fast skips the evaluation and short-training fingerprints (minutes) and
checks only forward passes and checkpoint schemas (seconds).

Exit code 0 = all green. Never fix a red check by re-capturing references.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parity.capture import (  # noqa: E402
    CHECKPOINTS, fixed_batch, load_model_pair, run_eval, run_train,
    schema_text,
)
from parity.fingerprint_lib import (  # noqa: E402
    PROJECT_ROOT, REFERENCE_DIR, Report, compare_array, compare_scalar,
    load_json, sha256_file,
)


def check_forward(report: Report, name: str, ckpt: str, cfg_path: str, atol: float) -> None:
    ref = np.load(REFERENCE_DIR / f"forward_{name}.npz")
    meta = load_json(REFERENCE_DIR / f"forward_{name}.json")

    got_hash = sha256_file(PROJECT_ROOT / ckpt)
    if got_hash != meta["checkpoint_sha256"]:
        report.add("FAIL", f"forward_{name}/checkpoint-file",
                   "checkpoint file changed on disk")
        return

    model, ema_model, cfg, _ = load_model_pair(ckpt, cfg_path)
    batch = fixed_batch(cfg)
    for key in ("local", "global", "actions", "t"):
        compare_array(report, f"forward_{name}/input_{key}",
                      batch[key].numpy(), ref[key], atol=0)
    with torch.no_grad():
        for tag, m in [("raw", model), ("ema", ema_model)]:
            out = m(batch["local"], batch["global"], batch["actions"], batch["t"])
            for okey, val in out.items():
                compare_array(report, f"forward_{name}/{tag}_{okey}",
                              val.numpy(), ref[f"{tag}_{okey}"], atol)


def check_schema(report: Report, name: str, ckpt: str) -> None:
    want = (REFERENCE_DIR / f"schema_{name}.txt").read_text()
    got = schema_text(ckpt)
    if got == want:
        report.add("PASS", f"schema_{name}")
    else:
        report.add("FAIL", f"schema_{name}", "key structure changed")


def main() -> None:
    fast = "--fast" in sys.argv
    report = Report()
    tol = load_json(REFERENCE_DIR / "tolerances.json")
    torch.set_num_threads(4)

    for name, ckpt, cfg_path in CHECKPOINTS:
        check_forward(report, name, ckpt, cfg_path, tol["forward_atol"])
        check_schema(report, name, ckpt)

    if not fast:
        for name, ckpt, cfg_path in CHECKPOINTS:
            want = load_json(REFERENCE_DIR / f"eval_{name}.json")["metrics"]
            got = run_eval(ckpt, cfg_path)
            compare_scalar(report, f"eval_{name}/win_rate",
                           got["win_rate"], want["win_rate"],
                           tol["eval_win_rate_atol"])
            compare_scalar(report, f"eval_{name}/avg_reward",
                           got["avg_reward"], want["avg_reward"],
                           tol["eval_reward_atol"])
            compare_scalar(report, f"eval_{name}/avg_steps",
                           got["avg_steps"], want["avg_steps"],
                           tol["eval_steps_atol"])

        want = load_json(REFERENCE_DIR / "train_offline.json")
        got = run_train(REFERENCE_DIR / "tiny_dataset.pt")
        for i, (g, w) in enumerate(zip(got["loss_history"], want["loss_history"])):
            compare_scalar(report, f"train_offline/loss[{i}]", g, w,
                           tol["train_loss_atol"])
        if tol["train_bit_reproducible"]:
            status = "PASS" if got["param_checksum"] == want["param_checksum"] else "FAIL"
            report.add(status, "train_offline/param_checksum")
        else:
            compare_scalar(report, "train_offline/param_mean",
                           got["param_stats"]["mean"], want["param_stats"]["mean"],
                           tol["train_loss_atol"])

    sys.exit(report.summary())


if __name__ == "__main__":
    main()
