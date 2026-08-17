"""Config layering: two-layer merge, key validation, delta-only presets,
ablation poolability, W&B naming and publish-time checkpoint selection."""

import ast
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


# ---------------------------------------------------------------------------
# W&B naming (spec-config §6.5: the config keys govern; the "remdm-*"
# literals were dead fallbacks)
# ---------------------------------------------------------------------------


def test_wandb_names_come_from_the_config_not_a_literal():
    """Training and the ablation suite both take project and entity from
    the config, and the shipped names are the canonical ones."""
    cfg = load_config("configs/defaults.yaml")
    assert cfg.wandb_project == "minihack-ReMDM-planner"

    abl = yaml.safe_load((_ABL_CONFIGS / "ablations_default.yaml").read_text())
    assert abl["wandb_project"] == "minihack-ReMDM-planner-ablations"
    assert abl["wandb_entity"] == cfg.wandb_entity

    # No "remdm-*" literal survives as a runtime default in either path.
    for src in (
        _ROOT / "src" / "planners" / "logging.py",
        _ROOT / "experiments" / "rl_finetuning" / "run_ablations.py",
    ):
        assert "remdm-minihack" not in src.read_text(), src


def test_suite_wandb_init_takes_project_and_entity_from_cli_or_config():
    """The suite's wandb.init receives both names, CLI overriding config.

    The entity was a documented ablation-config key that nothing read
    (step-11 finding U2); this pins the wiring. Source-anchored because
    the call sits inside main() behind checkpoint loading.
    """
    import inspect

    ra = _load_run_ablations()
    src = inspect.getsource(ra.main)

    assert "project=(args.wandb_project or cfg.wandb_project)" in src
    assert 'entity=(args.wandb_entity or getattr(cfg, "wandb_entity", None))' in src


# ---------------------------------------------------------------------------
# Publish-time best-of-N selection (spec-config §6.3/§6.4; the
# checkpoint-selection mechanism PARITY records for this repo)
# ---------------------------------------------------------------------------


def _hf_upload():
    spec = importlib.util.spec_from_file_location(
        "hf_upload", _ROOT / "scripts" / "hf_upload.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shipped_defaults() -> dict:
    """The shipped `configs/defaults.yaml` as a published snapshot would be.

    `selection()` is called on a checkpoint's own config snapshot, which is a
    scrub of a config of exactly this shape. Feeding the shipped file rather
    than a hand-built dict is the point: the previous version of these tests
    supplied the keys the function happened to look for, so both were green
    while every released DAgger `selection.json` recorded nulls.
    """
    return yaml.safe_load((_CONFIGS / "defaults.yaml").read_text())


def test_selection_records_the_dagger_candidate_set():
    """A DAgger checkpoint publishes as best-of-N over periodic
    checkpoints, naming the metric, the cadence and the eval protocol.

    The cadence and budget are env-step denominated because that is the
    cadence DAgger checkpoints on; the selected point is its own iteration
    counter.
    """
    hf = _hf_upload()
    cfg = _shipped_defaults()
    stats = {"counter": "dagger_iteration", "value": 600}

    sel = hf.selection(cfg, stats, "id_winrate")

    assert sel["policy"] == "best-of-N over periodic checkpoints"
    assert sel["selected"] == {"dagger_iteration": 600}
    assert sel["selection_metric"] == "id_winrate"
    assert sel["candidates"] == {
        "unit": "env_steps",
        "every": cfg["checkpoint_every_timesteps"],
        "configured_max": cfg["total_timesteps"],
    }
    assert sel["candidates"]["every"] is not None
    assert sel["candidates"]["configured_max"] is not None
    assert sel["eval_protocol"] == {
        "episodes_per_env": cfg["checkpoint_eval_episodes"],
        "weights": "ema",
        "id_envs": cfg["id_envs"],
        "ood_envs": cfg["ood_envs"],
    }


def test_selection_switches_units_for_an_offline_checkpoint():
    """An offline checkpoint is selected over gradient steps, with the
    offline pins as the candidate set - not the env-step cadence."""
    hf = _hf_upload()
    cfg = _shipped_defaults()
    stats = {"counter": "gradient_step", "value": 40_000}

    sel = hf.selection(cfg, stats, None)

    assert sel["selected"] == {"gradient_step": 40_000}
    assert sel["selection_metric"] is None
    assert sel["candidates"] == {
        "unit": "gradient_steps",
        "every": cfg["offline_checkpoint_every_grad_steps"],
        "configured_budget": cfg["offline_total_grad_steps"],
    }
    assert sel["eval_protocol"]["weights"] == "ema"


@pytest.mark.parametrize(
    "missing",
    [
        "checkpoint_every_timesteps",
        "total_timesteps",
        "checkpoint_eval_episodes",
        "id_envs",
        "ood_envs",
    ],
)
def test_selection_rejects_a_snapshot_missing_a_key_it_records(missing):
    """An absent key fails the upload instead of publishing a null.

    This is the guard the released artefacts lacked: the two renamed keys
    read as `None` and shipped as `"every": null, "configured_max": null`.
    """
    hf = _hf_upload()
    cfg = _shipped_defaults()
    del cfg[missing]

    with pytest.raises(KeyError, match=missing):
        hf.selection(cfg, {"counter": "dagger_iteration", "value": 600}, None)


_SHIPPED_CONFIGS = (_CONFIGS / "defaults.yaml", _ABL_DEFAULT)
_SOURCE_DIRS = ("src", "experiments", "scripts")


def _production_sources() -> list[Path]:
    """Every production module: the entry point and the source packages."""
    found = [_ROOT / "main.py"]
    for name in _SOURCE_DIRS:
        found += sorted((_ROOT / name).rglob("*.py"))
    return found


def _normalise(key: str) -> str:
    """Config keys are lower case in both the YAML and the code here."""
    return key


# Every key read from a config object that no shipped YAML declares, with the
# reason it is not declared. Anything not listed here fails the test.
_NOT_FROM_A_CONFIG_FILE = frozenset(
    {
        # Set by run_ablations.main from --device, else auto-detected.
        "device",
        # Read from a published checkpoint's own config snapshot in
        # hf_upload.py, behind an explicit `or ENV_NAME` fallback.
        "env_name",
    }
)


# ---------------------------------------------------------------------------
# Config-key reachability (the class F-1 belongs to)
# ---------------------------------------------------------------------------
# F-1: `hf_upload.py::selection` read `checkpoint_every` and `max_iterations`
# long after both were renamed out of the config. `.get()` returned None, so
# every published DAgger selection.json recorded a null and nothing failed.
# The class is "a key the code reads that no shipped config declares", and the
# only way to keep it closed is to re-derive the set on every run.
#
# `_CONFIG_NAMES` are the identifiers a merged config is bound to across the
# production sources. Attributes and subscripts on those names are config
# reads; leading-underscore names are not -- they are the convention for a
# value the code stamps onto the namespace at run time.
_CONFIG_NAMES = frozenset({"config", "cfg", "merged", "conf"})
_NOT_CONFIG_ATTRS = frozenset(
    {"get", "items", "keys", "values", "update", "copy", "pop", "setdefault"}
)


class _ConfigKeyScanner(ast.NodeVisitor):
    """Collect every config key a module reads, with its call site."""

    def __init__(self) -> None:
        self.keys: dict[str, set[str]] = {}
        self.path = "?"

    def _record(self, key: str, node: ast.AST) -> None:
        if not key.startswith("_"):
            self.keys.setdefault(key, set()).add(f"{self.path}:{node.lineno}")

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in _CONFIG_NAMES
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self._record(node.args[0].value, node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in _CONFIG_NAMES
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self._record(node.args[1].value, node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in _CONFIG_NAMES
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            self._record(node.slice.value, node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in _CONFIG_NAMES
            and node.attr not in _NOT_CONFIG_ATTRS
        ):
            self._record(node.attr, node)
        self.generic_visit(node)


def _config_keys_read() -> dict[str, set[str]]:
    scanner = _ConfigKeyScanner()
    for source in _production_sources():
        scanner.path = str(source.relative_to(_ROOT))
        scanner.visit(ast.parse(source.read_text()))
    return scanner.keys


def _declared_keys() -> set[str]:
    declared: set[str] = set()
    for path in _SHIPPED_CONFIGS:
        declared |= set(yaml.safe_load(path.read_text()) or {})
    return declared


def test_every_config_key_the_code_reads_is_declared_in_a_shipped_config():
    """No production source may read a key no shipped config declares.

    Every unresolved key must be named in `_NOT_FROM_A_CONFIG_FILE` with the
    reason it is not declared -- derived at load, injected by the CLI, or read
    from a foreign config. That list is the point of the test: an entry is a
    visible decision, whereas F-1 was a silent one.
    """
    read = _config_keys_read()
    declared = {_normalise(key) for key in _declared_keys()}

    unresolved = {
        key: sorted(sites)
        for key, sites in read.items()
        if _normalise(key) not in declared and key not in _NOT_FROM_A_CONFIG_FILE
    }

    assert not unresolved, (
        "config keys read by production code that no shipped config declares "
        f"and no exemption covers: {unresolved}"
    )


def test_the_reachability_scan_reaches_the_code_it_claims_to_cover():
    """Guard the scanner itself.

    A scan that silently matched nothing would pass the test above forever.
    These floors pin that the production sources are found, that a healthy
    number of keys is recovered, and that the exemption list has not grown
    into an allowlist for everything.
    """
    read = _config_keys_read()

    assert len(_production_sources()) >= 20
    assert len(read) >= 80
    assert len(_declared_keys()) >= 80
    assert len(_NOT_FROM_A_CONFIG_FILE) <= len(read) // 4
    stale = sorted(_NOT_FROM_A_CONFIG_FILE - set(read))
    assert not stale, f"exemptions for keys nothing reads any more: {stale}"
