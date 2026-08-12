"""Config inheritance: extends resolution, key validation, delta-only presets."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from src.config import load_config, resolve_config_chain

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
# extends resolution
# --------------------------------------------------------------------------


def test_none_path_returns_empty(ra):
    assert ra._load_ablation_config(None) == {}


def test_empty_extends_opts_out(ra, tmp_path):
    cfg = _write(tmp_path, "child.yaml", "extends:\nbatch_size: 7\n")
    assert ra._load_ablation_config(str(cfg)) == {"batch_size": 7}


def test_empty_yaml_inherits_full_base(ra, tmp_path):
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    cfg = _write(_ABL_CONFIGS, "_test_empty.yaml", "")
    try:
        assert ra._load_ablation_config(str(cfg)) == base
    finally:
        cfg.unlink()


def test_self_cycle_raises(ra, tmp_path):
    cfg = _write(tmp_path, "loop.yaml", "extends: loop.yaml\n")
    with pytest.raises(ValueError, match="loop.yaml"):
        ra._load_ablation_config(str(cfg))


def test_two_file_cycle_raises(ra, tmp_path):
    _write(tmp_path, "a.yaml", "extends: b.yaml\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError) as excinfo:
        ra._load_ablation_config(str(tmp_path / "a.yaml"))
    assert "a.yaml" in str(excinfo.value) and "b.yaml" in str(excinfo.value)


def test_missing_base_raises_naming_both_files(ra, tmp_path):
    cfg = _write(tmp_path, "orphan.yaml", "extends: nope.yaml\n")
    with pytest.raises(FileNotFoundError) as excinfo:
        ra._load_ablation_config(str(cfg))
    assert "nope.yaml" in str(excinfo.value)
    assert "orphan.yaml" in str(excinfo.value)


def test_three_deep_chain_child_wins(ra, tmp_path):
    _write(tmp_path, "g.yaml", "extends:\nbatch_size: 1\nlr: 9.0\nmax_iter: 3\n")
    _write(tmp_path, "m.yaml", "extends: g.yaml\nbatch_size: 2\nlr: 8.0\n")
    _write(tmp_path, "c.yaml", "extends: m.yaml\nbatch_size: 3\n")
    merged = ra._load_ablation_config(str(tmp_path / "c.yaml"))
    assert merged == {"batch_size": 3, "lr": 8.0, "max_iter": 3}


def test_absolute_extends(ra, tmp_path):
    cfg = _write(tmp_path, "abs.yaml", f"extends: {_ABL_DEFAULT}\nbatch_size: 42\n")
    merged = ra._load_ablation_config(str(cfg))
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    assert merged["batch_size"] == 42
    assert set(merged) == set(base)


def test_base_does_not_self_extend(ra):
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    assert ra._load_ablation_config(str(_ABL_DEFAULT)) == base


def test_bare_relative_extends_outside_configs_fails_loudly(ra, tmp_path):
    cfg = _write(tmp_path, "copy.yaml", "extends: ablations_default.yaml\n")
    with pytest.raises(FileNotFoundError):
        ra._load_ablation_config(str(cfg))


def test_chain_is_base_first(tmp_path):
    _write(tmp_path, "base.yaml", "extends:\nlr: 1.0\n")
    child = _write(tmp_path, "child.yaml", "extends: base.yaml\nlr: 2.0\n")
    chain = resolve_config_chain(child)
    assert [p.name for p, _ in chain] == ["base.yaml", "child.yaml"]


# --------------------------------------------------------------------------
# key validation
# --------------------------------------------------------------------------


def test_unknown_key_in_ablation_config_raises(ra, tmp_path):
    cfg = _write(tmp_path, "typo.yaml", "extends:\nbatch_sze: 512\n")
    allowed = set(yaml.safe_load(_ABL_DEFAULT.read_text()))
    with pytest.raises(KeyError, match="batch_sze"):
        ra._load_ablation_config(str(cfg), allowed=allowed)


def test_known_key_in_ablation_config_passes(ra, tmp_path):
    cfg = _write(tmp_path, "fine.yaml", "extends:\nbatch_size: 512\n")
    allowed = set(yaml.safe_load(_ABL_DEFAULT.read_text()))
    assert ra._load_ablation_config(str(cfg), allowed=allowed) == {"batch_size": 512}


def test_shipped_ablation_configs_validate(ra):
    allowed = set(yaml.safe_load((_CONFIGS / "defaults.yaml").read_text())) | set(
        yaml.safe_load(_ABL_DEFAULT.read_text())
    )
    for name in ("final_ablations_qmul.yaml", "final_ablations_ucl.yaml"):
        ra._load_ablation_config(str(_ABL_CONFIGS / name), allowed=allowed)


def test_fast_overlay_carries_no_extends():
    """It is applied raw, so an extends key would leak into the namespace."""
    raw = yaml.safe_load((_ABL_CONFIGS / "ablations_fast.yaml").read_text())
    assert "extends" not in raw


def test_extends_rejected_as_cli_override():
    with pytest.raises(KeyError, match="extends"):
        load_config("configs/final_ucl_gpu.yaml", {"extends": "x.yaml"})


# --------------------------------------------------------------------------
# delta-only invariant
# --------------------------------------------------------------------------


def _inherited_values(path: Path, defaults: dict) -> dict:
    """Values *path* would see from its ancestors, excluding itself."""
    inherited = dict(defaults)
    for source, raw in resolve_config_chain(path):
        if source.resolve() == path.resolve():
            continue
        inherited.update({k: v for k, v in raw.items() if k != "extends"})
    return inherited


@pytest.mark.parametrize(
    "preset",
    sorted(p.name for p in _CONFIGS.glob("*.yaml") if p.name != "defaults.yaml"),
)
def test_preset_restates_no_inherited_value(preset):
    defaults = yaml.safe_load((_CONFIGS / "defaults.yaml").read_text())
    path = _CONFIGS / preset
    raw = yaml.safe_load(path.read_text()) or {}
    inherited = _inherited_values(path, defaults)
    restated = {
        k: v
        for k, v in raw.items()
        if k != "extends" and k in inherited and inherited[k] == v
    }
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
    restated = {
        k: v for k, v in raw.items() if k != "extends" and k in base and base[k] == v
    }
    assert not restated, f"{preset} restates inherited values: {restated}"


# --------------------------------------------------------------------------
# cross-machine poolability
#
# `run_ablations.py --merge` averages seeds of the same ablation across
# results.json files, so pooling two machine configs is only sound when they
# agree on everything that changes the trained model or the measured score.
# All published MiniHack ablation results were produced on the UCL 3090 Ti,
# which is therefore the reference.
# --------------------------------------------------------------------------

_REFERENCE_CONFIG = "final_ablations_ucl.yaml"

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
    "final_ablations_qmul.yaml": frozenset(
        {
            "batch_size",  # 512 vs 4608: ~9x per-update SNR
            "episodes_per_iter",  # 20 vs 30: 10k vs 15k total episodes
            "diffusion_steps_collect",  # 3 vs 5: different collection policy
            "eval_episodes",  # 10 vs 20: noisier score
        }
    ),
}


def _machine_configs() -> list[str]:
    return sorted(p.name for p in _ABL_CONFIGS.glob("final_ablations_*.yaml"))


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
