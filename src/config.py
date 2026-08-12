"""Configuration loader for ReMDM-MiniHack.

Loads YAML configs with deep-merge and CLI override support,
following the Craftax config pattern.
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import yaml

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*).

    Args:
        base: Base dictionary to merge into.
        override: Dictionary whose values take precedence.

    Returns:
        The merged dictionary (same object as *base*).
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# Valid config keys that do not appear in defaults.yaml: `device` is
# auto-selected at load time and serialised into checkpoint config snapshots.
_RUN_KEYS = {"device"}

# Config-file-only key naming a base config to inherit from. Resolved and
# stripped before merging, so it never reaches the namespace. Not valid as a
# --override.
_EXTENDS_KEY = "extends"

# Keys used by earlier code versions that survive in released checkpoint
# config snapshots (e.g. config_iter600.yaml on the HF Hub). Accepted so the
# documented snapshot-evaluation workflow keeps working; nothing reads them.
_LEGACY_SNAPSHOT_KEYS = {
    "checkpoint_every",
    "id_eval_every",
    "ood_eval_every",
    "max_iterations",
    "offline_epochs",
    # CH-1 (ADJUDICATION B-3): nucleus sampling replaced top-k; snapshots
    # from earlier runs still carry the key. It is accepted and unread.
    "top_k",
    # FIX-1 (ADJUDICATION B-1): the NELBO weight is no longer optional;
    # snapshots from earlier runs still carry the flag. Accepted, unread.
    "use_importance_weighting",
}


def _load_config_chain(path: Path) -> list[tuple[Path, dict]]:
    """Load a config and its ``extends`` ancestors, base first.

    ``extends`` names a base config, resolved relative to the file that
    declares it (absolute paths are also accepted). Later entries in the
    returned list override earlier ones.

    Args:
        path: Absolute path to the config file passed via ``--config``.

    Returns:
        ``[(path, raw_dict), ...]`` ordered base first, child last.

    Raises:
        ValueError: If the chain contains a cycle.
        FileNotFoundError: If a referenced base config does not exist.
    """
    chain: list[tuple[Path, dict]] = []
    seen: set[Path] = set()
    current: Path | None = path.resolve()

    while current is not None:
        if current in seen:
            visited = " -> ".join(p.name for p, _ in chain)
            raise ValueError(
                f"Circular 'extends' chain in configs: {visited} -> {current.name}"
            )
        seen.add(current)
        with open(current) as fh:
            raw = yaml.safe_load(fh) or {}
        chain.append((current, raw))

        base = raw.get(_EXTENDS_KEY)
        if not base:
            break
        base_path = Path(base).expanduser()
        if not base_path.is_absolute():
            base_path = current.parent / base_path
        base_path = base_path.resolve()
        if not base_path.exists():
            raise FileNotFoundError(
                f"Base config '{base_path}' referenced by '{current}' does not exist"
            )
        current = base_path

    chain.reverse()
    return chain


def _validate_keys(keys, allowed: set[str], source: str) -> None:
    """Reject unknown config keys instead of silently ignoring them.

    Args:
        keys: Keys to check.
        allowed: The full set of valid config keys.
        source: Label for the error message (file path or 'override').

    Raises:
        KeyError: If any key is not a known config key.
    """
    unknown = sorted(set(keys) - allowed)
    if unknown:
        raise KeyError(
            f"Unknown config key(s) {unknown} in {source}. "
            "Valid keys are defined in configs/defaults.yaml."
        )


def _cast_override(key: str, raw: str, current) -> object:
    """Cast a CLI override string to the type of the current config value.

    Args:
        key: Config key being overridden.
        raw: Raw string from the command line.
        current: Current (default or config-file) value, used for typing.

    Returns:
        Parsed Python value.

    Raises:
        TypeError: If the value cannot be interpreted as the key's type.
    """
    if isinstance(current, str):
        return raw

    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw

    if current is None or value is None:
        return value

    # YAML 1.1 reads '1e-4' as a string; accept scientific notation for
    # numeric keys.
    if (
        isinstance(current, (int, float))
        and not isinstance(current, bool)
        and isinstance(value, str)
    ):
        with contextlib.suppress(ValueError):
            value = float(value)

    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise TypeError(f"'{key}' expects a boolean, got '{raw}'")
        return value
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"'{key}' expects an integer, got '{raw}'")
        if isinstance(value, float):
            if not value.is_integer():
                raise TypeError(f"'{key}' expects an integer, got '{raw}'")
            value = int(value)
        return value
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"'{key}' expects a number, got '{raw}'")
        return float(value)
    if isinstance(current, list):
        if not isinstance(value, list):
            raise TypeError(f"'{key}' expects a list, got '{raw}'")
        return value
    return value


def load_config(
    config_path: str | None = None,
    cli_overrides: dict | None = None,
) -> SimpleNamespace:
    """Load configuration from YAML with optional overrides.

    1. Load ``configs/defaults.yaml``.
    2. Deep-merge *config_path* on top, base first if it declares ``extends``
       (skipped if it is the defaults file itself).
    3. Apply *cli_overrides* key=value pairs.
    4. Auto-select device (``cuda`` if available, else ``cpu``; honour
       ``DEVICE`` env-var).
    5. Validate invariants.

    Args:
        config_path: Path to a YAML file merged on top of defaults.
            ``None`` uses defaults only.
        cli_overrides: ``{key: value}`` pairs applied last.

    Returns:
        A ``SimpleNamespace`` containing all hyperparameters.

    Raises:
        AssertionError: If ``mask_token != action_dim`` or
            ``pad_token != action_dim + 1``.
    """
    if cli_overrides is None:
        cli_overrides = {}

    defaults_path = _PROJECT_ROOT / "configs" / "defaults.yaml"
    with open(defaults_path) as fh:
        cfg = yaml.safe_load(fh)

    allowed = set(cfg) | _RUN_KEYS

    if config_path is not None:
        config_path_resolved = Path(config_path)
        if not config_path_resolved.is_absolute():
            config_path_resolved = _PROJECT_ROOT / config_path_resolved
        if config_path_resolved.resolve() != defaults_path.resolve():
            for source, overrides in _load_config_chain(config_path_resolved):
                _validate_keys(
                    overrides,
                    allowed | _LEGACY_SNAPSHOT_KEYS | {_EXTENDS_KEY},
                    str(source),
                )
                _deep_merge(
                    cfg,
                    {k: v for k, v in overrides.items() if k != _EXTENDS_KEY},
                )

    _validate_keys(cli_overrides, allowed, "--override")
    for key, value in cli_overrides.items():
        if isinstance(value, str):
            value = _cast_override(key, value, cfg.get(key))
        cfg[key] = value

    # Device selection
    env_device = os.environ.get("DEVICE")
    if env_device:
        cfg["device"] = env_device
    elif "device" not in cfg:
        try:
            import torch

            cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            cfg["device"] = "cpu"

    ns = SimpleNamespace(**cfg)

    # Validation
    assert ns.mask_token == ns.action_dim, (
        f"mask_token ({ns.mask_token}) must equal action_dim ({ns.action_dim})"
    )
    assert ns.pad_token == ns.action_dim + 1, (
        f"pad_token ({ns.pad_token}) must equal action_dim + 1 ({ns.action_dim + 1})"
    )

    return ns


def make_run_dir(cfg: SimpleNamespace, tag: str = "run") -> Path:
    """Create a unique run subdirectory under ``cfg.checkpoint_dir``.

    Generates a directory named ``{tag}_{YYYYMMDD}_{HHMMSS}_{hex4}``
    to prevent concurrent runs from overwriting each other's
    checkpoints. Updates ``cfg.checkpoint_dir`` in place.

    Args:
        cfg: Config namespace (``checkpoint_dir`` is mutated).
        tag: Prefix for the directory name (e.g. ``"dagger"``,
            ``"offline"``).

    Returns:
        The created directory path.
    """
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)
    run_dir = Path(cfg.checkpoint_dir).resolve() / f"{tag}_{ts}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir = str(run_dir)
    logger.info("Checkpoint directory: %s", run_dir)
    return run_dir
