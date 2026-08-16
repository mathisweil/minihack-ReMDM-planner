"""Config layering: two-layer merge, key validation, delta-only presets."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from src.config import load_config

_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS = _ROOT / "configs"
_ABL_CONFIGS = _ROOT / "experiments" / "rl_finetuning" / "configs"
_ABL_DEFAULT = _ABL_CONFIGS / "ablations_default.yaml"


def _load_run_ablations():
    spec = importlib.util.spec_from_file_location(
        "run_ablations_under_test",
        _ROOT / "experiments" / "rl_finetuning" / "run_ablations.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ra():
    return _load_run_ablations()


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body)
    return path


# --------------------------------------------------------------------------
# two-layer merge
# --------------------------------------------------------------------------


def test_none_path_returns_empty(ra):
    assert ra._load_ablation_config(None) == {}


def test_base_loads_alone(ra):
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    assert ra._load_ablation_config(str(_ABL_DEFAULT)) == base


def test_empty_config_inherits_full_base(ra, tmp_path):
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    cfg = _write(tmp_path, "empty.yaml", "")
    assert ra._load_ablation_config(str(cfg)) == base


def test_machine_config_overrides_base(ra, tmp_path):
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    cfg = _write(tmp_path, "machine.yaml", "batch_size: 7\n")
    merged = ra._load_ablation_config(str(cfg))
    assert merged["batch_size"] == 7
    assert set(merged) == set(base)


def test_preset_does_not_inherit_from_another_preset():
    """Presets are a single layer over defaults.yaml; no config chains."""
    for path in _CONFIGS.glob("*.yaml"):
        raw = yaml.safe_load(path.read_text()) or {}
        assert "extends" not in raw, f"{path.name} declares extends"
    for path in _ABL_CONFIGS.glob("*.yaml"):
        raw = yaml.safe_load(path.read_text()) or {}
        assert "extends" not in raw, f"{path.name} declares extends"


def test_extends_rejected_as_config_key():
    """The mechanism is gone, so the key must now be an error, not ignored."""
    cfg = _write(_CONFIGS, "_tmp_extends.yaml", "extends: final_qmul_gpu.yaml\n")
    try:
        with pytest.raises(KeyError, match="extends"):
            load_config("configs/_tmp_extends.yaml")
    finally:
        cfg.unlink()


# --------------------------------------------------------------------------
# key validation
# --------------------------------------------------------------------------


def test_unknown_key_in_ablation_config_raises(ra, tmp_path):
    cfg = _write(tmp_path, "typo.yaml", "batch_sze: 512\n")
    allowed = set(yaml.safe_load(_ABL_DEFAULT.read_text()))
    with pytest.raises(KeyError, match="batch_sze"):
        ra._load_ablation_config(str(cfg), allowed=allowed)


def test_known_key_in_ablation_config_passes(ra, tmp_path):
    cfg = _write(tmp_path, "fine.yaml", "batch_size: 512\n")
    allowed = set(yaml.safe_load(_ABL_DEFAULT.read_text()))
    assert ra._load_ablation_config(str(cfg), allowed=allowed)["batch_size"] == 512


def test_shipped_ablation_configs_validate(ra):
    allowed = set(yaml.safe_load((_CONFIGS / "defaults.yaml").read_text())) | set(
        yaml.safe_load(_ABL_DEFAULT.read_text())
    )
    for name in ("ablations_final_qmul.yaml", "ablations_final_ucl.yaml"):
        ra._load_ablation_config(str(_ABL_CONFIGS / name), allowed=allowed)


def test_extends_rejected_as_cli_override():
    with pytest.raises(KeyError, match="extends"):
        load_config("configs/final_ucl_gpu.yaml", {"extends": "x.yaml"})


# --------------------------------------------------------------------------
# delta-only invariant
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset",
    sorted(p.name for p in _CONFIGS.glob("*.yaml") if p.name != "defaults.yaml"),
)
def test_preset_restates_no_inherited_value(preset):
    defaults = yaml.safe_load((_CONFIGS / "defaults.yaml").read_text())
    raw = yaml.safe_load((_CONFIGS / preset).read_text()) or {}
    restated = {k: v for k, v in raw.items() if k in defaults and defaults[k] == v}
    assert not restated, f"{preset} restates inherited values: {restated}"


@pytest.mark.parametrize(
    "preset",
    sorted(
        p.name
        for p in _ABL_CONFIGS.glob("*.yaml")
        if p.name not in {"ablations_default.yaml", "ablations_fast.yaml"}
    ),
)
def test_ablation_config_restates_no_inherited_value(preset):
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    raw = yaml.safe_load((_ABL_CONFIGS / preset).read_text()) or {}
    restated = {k: v for k, v in raw.items() if k in base and base[k] == v}
    assert not restated, f"{preset} restates inherited values: {restated}"


# --------------------------------------------------------------------------
# cross-machine drift
#
# With inheritance gone, nothing structurally stops the QMUL and UCL configs
# drifting apart, so the shared recipe is asserted here instead.
# --------------------------------------------------------------------------

#: The only keys on which the two paper-run configs may differ.
_MACHINE_KEYS = frozenset(
    {"num_collection_workers", "collect_num_workers", "collect_output"}
)


def test_paper_configs_differ_only_in_machine_keys():
    qmul = vars(load_config("configs/final_qmul_gpu.yaml"))
    ucl = vars(load_config("configs/final_ucl_gpu.yaml"))
    diverged = {
        k
        for k in set(qmul) | set(ucl)
        if k != "device" and qmul.get(k, "<absent>") != ucl.get(k, "<absent>")
    }
    assert diverged <= set(_MACHINE_KEYS), (
        "final_qmul_gpu.yaml and final_ucl_gpu.yaml must train an identical "
        f"model. They diverge on non-machine key(s): "
        f"{sorted(diverged - set(_MACHINE_KEYS))}. Move the shared value into "
        "configs/defaults.yaml."
    )


@pytest.mark.parametrize("preset", ["final_qmul_gpu.yaml", "final_ucl_gpu.yaml"])
def test_paper_config_holds_only_machine_keys(preset):
    raw = yaml.safe_load((_CONFIGS / preset).read_text()) or {}
    stray = set(raw) - set(_MACHINE_KEYS)
    assert not stray, (
        f"{preset} should hold only machine values; found {sorted(stray)}. "
        "Anything shared with the other cluster belongs in configs/defaults.yaml."
    )


# --------------------------------------------------------------------------
# offline compute-match pins
#
# The four offline_* keys silently override an env-step-derived value when
# non-null. defaults.yaml now sets them as part of the paper recipe, so any
# preset wanting the derived behaviour must pin them back to null explicitly.
# --------------------------------------------------------------------------

_OFFLINE_PINS = (
    "offline_total_grad_steps",
    "offline_eval_every_grad_steps",
    "offline_checkpoint_every_grad_steps",
    "offline_buffer_capacity",
)

#: Presets that must derive their offline budget rather than inherit the pin.
_DERIVES_OFFLINE_BUDGET = [
    "smoke.yaml",
    "ablation_local_only.yaml",
    "ucl_gpu_bigger_model.yaml",
    "ucl_gpu_learning_behaviour.yaml",
]


@pytest.mark.parametrize("preset", _DERIVES_OFFLINE_BUDGET)
def test_preset_pins_offline_overrides_to_null(preset):
    raw = yaml.safe_load((_CONFIGS / preset).read_text()) or {}
    missing = [k for k in _OFFLINE_PINS if raw.get(k, "<absent>") is not None]
    assert not missing, (
        f"{preset} must pin {missing} to null. Omitting them inherits the "
        "paper recipe's pins, which override the env-step-derived budget: "
        "smoke would train 60000 offline grad steps instead of 19."
    )


# --------------------------------------------------------------------------
# cross-machine poolability of ablation runs
#
# `run_ablations.py --merge` averages seeds of the same ablation across
# results.json files, so pooling two machine configs is only sound when they
# agree on everything that changes the trained model or the measured score.
# All published MiniHack ablation results were produced on the UCL 3090 Ti,
# which is therefore the reference.
# --------------------------------------------------------------------------

_REFERENCE_CONFIG = "ablations_final_ucl.yaml"

#: Keys that change the trained model or the score it is measured with.
_RESULT_AFFECTING = frozenset(
    {
        "max_iter",
        "batch_size",
        "episodes_per_iter",
        "grad_steps_per_iter",
        "lr",
        "weight_decay",
        "max_grad_norm",
        "diffusion_steps_collect",
        "use_amp",
        "eval_episodes",
    }
)

#: Configs whose runs may be merged with the reference.
_POOLABLE = {_REFERENCE_CONFIG}

#: Configs that must NOT be merged with the reference, mapped to the
#: result-affecting keys on which they are known to diverge.
_NOT_POOLABLE = {
    "ablations_final_qmul.yaml": frozenset(
        {
            "batch_size",  # 512 vs 4608: ~9x per-update SNR
            "episodes_per_iter",  # 20 vs 30: 10k vs 15k total episodes
            "diffusion_steps_collect",  # 3 vs 5: different collection policy
            "eval_episodes",  # 10 vs 20: noisier score
        }
    ),
}


def _machine_configs() -> list[str]:
    return sorted(p.name for p in _ABL_CONFIGS.glob("ablations_final_*.yaml"))


def _divergence(ra, name: str) -> set[str]:
    reference = ra._load_ablation_config(str(_ABL_CONFIGS / _REFERENCE_CONFIG))
    candidate = ra._load_ablation_config(str(_ABL_CONFIGS / name))
    return {
        k
        for k in _RESULT_AFFECTING
        if candidate.get(k, "<absent>") != reference.get(k, "<absent>")
    }


def test_every_machine_config_is_classified():
    """A new machine config must be declared poolable or not, never silently."""
    unclassified = set(_machine_configs()) - _POOLABLE - set(_NOT_POOLABLE)
    assert not unclassified, (
        f"Unclassified ablation machine config(s): {sorted(unclassified)}. "
        "Add to _POOLABLE (and align result-affecting keys with "
        f"{_REFERENCE_CONFIG}) or to _NOT_POOLABLE with the diverging keys."
    )


@pytest.mark.parametrize("name", sorted(_POOLABLE))
def test_poolable_config_matches_reference(ra, name):
    diverged = _divergence(ra, name)
    assert not diverged, (
        f"{name} is declared poolable with {_REFERENCE_CONFIG} but diverges "
        f"on result-affecting key(s): {sorted(diverged)}. Align them or move "
        "it to _NOT_POOLABLE."
    )


@pytest.mark.parametrize("name", sorted(_NOT_POOLABLE))
def test_not_poolable_divergence_is_recorded(ra, name):
    """Catches drift in both directions: new divergence, and silent alignment."""
    expected = _NOT_POOLABLE[name]
    actual = _divergence(ra, name)
    assert actual == set(expected), (
        f"{name} divergence from {_REFERENCE_CONFIG} has changed. "
        f"Recorded: {sorted(expected)}; actual: {sorted(actual)}. "
        "Update _NOT_POOLABLE, or move it to _POOLABLE if now aligned."
    )
