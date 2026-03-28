"""Configuration loader for ReMDM-MiniHack.

Loads YAML configs with deep-merge and CLI override support,
following the Craftax config pattern.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import yaml


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
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _cast_value(value: str) -> int | float | bool | str:
    """Best-effort cast of a CLI string to a Python scalar.

    Args:
        value: Raw string from the command line.

    Returns:
        Parsed Python value (int, float, bool, or str).
    """
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(
    config_path: str | None = None,
    cli_overrides: dict | None = None,
) -> SimpleNamespace:
    """Load configuration from YAML with optional overrides.

    1. Load ``configs/defaults.yaml``.
    2. Deep-merge *config_path* on top (if provided and different from defaults).
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
    with open(defaults_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    if config_path is not None:
        config_path_resolved = Path(config_path)
        if not config_path_resolved.is_absolute():
            config_path_resolved = _PROJECT_ROOT / config_path_resolved
        if config_path_resolved.resolve() != defaults_path.resolve():
            with open(config_path_resolved, "r") as fh:
                overrides = yaml.safe_load(fh) or {}
            _deep_merge(cfg, overrides)

    for key, value in cli_overrides.items():
        if isinstance(value, str):
            value = _cast_value(value)
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
        f"pad_token ({ns.pad_token}) must equal action_dim + 1 "
        f"({ns.action_dim + 1})"
    )

    return ns
