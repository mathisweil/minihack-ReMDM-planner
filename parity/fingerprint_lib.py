"""Shared helpers for the parity regression harness.

Golden-output fingerprints protect the trained checkpoints: any code change
that alters what the repo computes must show up as a diff against the stored
references. See parity/capture.py (writes references) and parity/check.py
(compares against them).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PARITY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PARITY_DIR.parent
REFERENCE_DIR = PARITY_DIR / "reference"


def git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    return out.stdout.strip() or "unknown"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_tree(named_arrays: dict[str, np.ndarray]) -> str:
    """Order-independent checksum over a flat dict of named arrays."""
    h = hashlib.sha256()
    for name in sorted(named_arrays):
        h.update(name.encode())
        h.update(sha256_array(named_arrays[name]).encode())
    return h.hexdigest()


def array_stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "shape": list(np.shape(arr)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def save_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_json(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


class Report:
    """Collects PASS/FAIL lines and renders a summary."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.lines.append((status, name, detail))
        print(f"[{status:4}] {name}" + (f"  {detail}" if detail else ""))

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.lines if s == "FAIL")

    def summary(self) -> int:
        n = len(self.lines)
        print(f"\n{n - self.failed}/{n} checks passed")
        return 1 if self.failed else 0


def compare_scalar(
    report: Report, name: str, got: float, want: float, atol: float
) -> None:
    delta = abs(float(got) - float(want))
    if delta <= atol:
        report.add("PASS", name, f"delta={delta:.3g} (atol={atol:.3g})")
    else:
        report.add(
            "FAIL", name,
            f"got={got!r} want={want!r} delta={delta:.3g} atol={atol:.3g}",
        )


def compare_array(
    report: Report, name: str, got: np.ndarray, want: np.ndarray, atol: float
) -> None:
    if got.shape != want.shape:
        report.add("FAIL", name, f"shape {got.shape} != {want.shape}")
        return
    delta = float(np.max(np.abs(np.asarray(got, np.float64) - np.asarray(want, np.float64)))) if got.size else 0.0
    if delta <= atol:
        report.add("PASS", name, f"max|d|={delta:.3g} (atol={atol:.3g})")
    else:
        report.add("FAIL", name, f"max|d|={delta:.3g} atol={atol:.3g}")


def run_main(args: list[str], env_extra: dict | None = None, timeout: int = 900):
    """Run this repo's main.py in a subprocess and return CompletedProcess."""
    import os
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "disabled")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        timeout=timeout, env=env,
    )
