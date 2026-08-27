"""Publish tracked files as git committed them, not as the working tree has them.

``hf download <repo> --local-dir .`` writes the Hub's copies over the working
tree. The Hub repo carries its own ``README.md`` (the generated model card),
its own ``LICENSE`` and its own ``.gitattributes``, so a pull silently replaces
the repository's versions of the first two and creates the third. Comparing
``git ls-files`` against the Hub's file listing, those two are the *only*
tracked files a pull overwrites -- everything else the Hub holds
(``checkpoints/``, ``experiments/rl_finetuning/outputs/``, ``results/``) is
gitignored.

That matters because the publishers copy some files straight off the working
tree, so a pull-then-publish round trip re-publishes whatever the pull left
behind, laundering it as current. It already happened once: the Hub's
``LICENSE`` carried a superseded paper title for a full release cycle, and was
caught only by hand. Neither ``--dry-run`` nor the staged-tree listing would
have shown it, because both would have printed a ``LICENSE`` that was merely
the wrong one.

:func:`copy_tracked_file` closes that path by preferring the committed bytes,
which a download cannot touch. When git cannot answer -- no git binary, not a
checkout, nothing committed at that path -- it falls back to the working tree
and says loudly that the file went out unverified, so publishing still works
from a tarball or a slim container.

This module is imported by ``hf_upload.py`` and ``hf_upload_demo.py``. It is
deliberately dependency-free so the sibling ``minihack-ReMDM-planner`` can
carry a byte-identical copy; a ``diff`` between the two should stay empty.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# git is consulted, never required. A publish from a slim container or an
# exported tarball degrades to a warning rather than an exception.
_GIT_TIMEOUT_S = 10


def committed_bytes(relative_path: str, root: Path) -> tuple[bytes | None, str]:
    """The exact bytes of *relative_path* as committed at ``HEAD``.

    Returns ``(blob, "")`` when git answers, and ``(None, reason)`` when it
    cannot. *root* is a parameter rather than this module's own location so a
    test can point it at a throwaway checkout.

    Uses ``git cat-file blob``, the plumbing command, rather than ``git show``.
    ``show`` consults ``.gitattributes`` for textconv and eol conversion -- and
    the file a clobbering pull creates *is* a ``.gitattributes``, full of
    ``diff=lfs`` rules. Reading through plumbing makes this check structurally
    immune to the very artefact the bug leaves behind.
    """
    if shutil.which("git") is None:
        return None, "git is not on PATH"

    try:
        # HEAD:<path> resolves against the *enclosing* repository, so a working
        # copy nested inside another checkout would otherwise verify against
        # the wrong repo's file. Confirm root is the toplevel before trusting.
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
        )
        if top.returncode != 0:
            return None, _first_line(top.stderr) or "not a git checkout"
        if Path(top.stdout.decode().strip()).resolve() != Path(root).resolve():
            return None, f"{root} is not the root of a git checkout"

        # capture_output without text: a licence must be byte-exact, and text
        # mode applies universal-newline translation and locale decoding, which
        # would silently rewrite CRLF and make "byte-exact" a lie.
        blob = subprocess.run(
            ["git", "cat-file", "blob", f"HEAD:{relative_path}"],
            cwd=root,
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Covers the binary vanishing between which() and run(), and timeouts.
        return None, f"git could not be run ({type(exc).__name__})"

    if blob.returncode != 0:
        # git's own wording distinguishes "not a checkout" from "not committed"
        # without this code having to branch on either.
        return None, _first_line(blob.stderr) or "git could not read the blob"
    if not blob.stdout:
        # Publishing an empty licence is worse than publishing an unverified one.
        return None, f"HEAD:{relative_path} is an empty blob"
    return blob.stdout, ""


def _first_line(stderr: bytes) -> str:
    for line in stderr.decode("utf-8", "replace").splitlines():
        if line.strip():
            return line.strip()
    return ""


def copy_tracked_file(relative_path: str, destination: Path, root: Path) -> None:
    """Stage a git-tracked file, preferring the bytes git has committed.

    Warns on stderr when the working tree diverges from ``HEAD`` (the file is
    still published from ``HEAD``) and when git could not be consulted at all
    (the working-tree copy is published unverified).
    """
    source = Path(root) / relative_path
    blob, reason = committed_bytes(relative_path, root)

    if blob is None:
        shutil.copy2(source, destination)
        print(
            f"Warning: could not check {relative_path} against git ({reason}); "
            f"published the working-tree copy UNVERIFIED.\n"
            f"         Confirm it is current before trusting the release -- a "
            f"Hub download overwrites {relative_path} in place, so a stale copy "
            f"would publish silently.",
            file=sys.stderr,
        )
        return

    if source.is_file() and source.read_bytes() == blob:
        # Identical, so keep copy2 and its metadata: byte-for-byte the
        # behaviour this call replaced, for the overwhelmingly common case.
        shutil.copy2(source, destination)
        return

    destination.write_bytes(blob)
    print(
        f"Warning: {relative_path} in the working tree differs from the one "
        f"committed at HEAD; published the COMMITTED bytes.\n"
        f"         `hf download --local-dir .` overwrites LICENSE and README.md "
        f"with the Hub's own copies, and the next publish pushes them straight "
        f"back up -- that is how a LICENSE naming a superseded paper title was "
        f"once re-published as current.\n"
        f"         If the working-tree edit is intended, commit it and re-run; "
        f"otherwise `git checkout -- {relative_path}`.",
        file=sys.stderr,
    )


def dirty_paths(relative_paths: list[str], root: Path) -> list[str]:
    """Which of *relative_paths* git reports as modified, for a warn-only report.

    Directory-scoped counterpart to :func:`diverged_from_head`. Whole trees are
    not staged from ``HEAD``: ``Craftax_Baselines`` is a submodule, so
    ``HEAD:Craftax_Baselines/...`` does not resolve from the superproject, and
    none of the published directories is a path a Hub download overwrites
    anyway. Reporting is the proportionate response.

    Returns an empty list when git cannot answer -- an unavailable git is
    already reported by the file-level checks, and repeating it per directory
    would bury the message that matters.
    """
    if shutil.which("git") is None:
        return []
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--", *relative_paths],
            cwd=root,
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    seen = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        # Porcelain v1: two status columns, a space, then the path.
        entry = line[3:].strip()
        for candidate in relative_paths:
            if entry.startswith(candidate) and candidate not in seen:
                seen.append(candidate)
    return seen
