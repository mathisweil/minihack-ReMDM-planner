"""Config layering: two-layer merge, key validation, delta-only presets,
ablation poolability, W&B naming and publish-time checkpoint selection."""

import ast
import importlib.util
import json
import math
import os
import shutil
import subprocess
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
    cfg = _write(_CONFIGS, "_tmp_extends.yaml", "extends: final_minihack_gpu_h200.yaml\n")
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
    for name in ("ablations_final_minihack_gpu_h200.yaml", "ablations_final_minihack_gpu_24gb.yaml"):
        ra._load_ablation_config(str(_ABL_CONFIGS / name), allowed=allowed)


def test_extends_rejected_as_cli_override():
    with pytest.raises(KeyError, match="extends"):
        load_config("configs/final_minihack_gpu_24gb.yaml", {"extends": "x.yaml"})


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
# With inheritance gone, nothing structurally stops the GPU-H200 and GPU-24GB configs
# drifting apart, so the shared recipe is asserted here instead.
# --------------------------------------------------------------------------

#: The only keys on which the two paper-run configs may differ.
_MACHINE_KEYS = frozenset(
    {"num_collection_workers", "collect_num_workers", "collect_output"}
)


def test_paper_configs_differ_only_in_machine_keys():
    gpu_h200 = vars(load_config("configs/final_minihack_gpu_h200.yaml"))
    gpu_24gb = vars(load_config("configs/final_minihack_gpu_24gb.yaml"))
    diverged = {
        k
        for k in set(gpu_h200) | set(gpu_24gb)
        if k != "device" and gpu_h200.get(k, "<absent>") != gpu_24gb.get(k, "<absent>")
    }
    assert diverged <= set(_MACHINE_KEYS), (
        "final_minihack_gpu_h200.yaml and final_minihack_gpu_24gb.yaml must train an identical "
        f"model. They diverge on non-machine key(s): "
        f"{sorted(diverged - set(_MACHINE_KEYS))}. Move the shared value into "
        "configs/defaults.yaml."
    )


@pytest.mark.parametrize("preset", ["final_minihack_gpu_h200.yaml", "final_minihack_gpu_24gb.yaml"])
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
    "gpu_24gb_bigger_model.yaml",
    "gpu_24gb_learning_behaviour.yaml",
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
# All published MiniHack ablation results were produced on the RTX 3090 Ti,
# which is therefore the reference.
# --------------------------------------------------------------------------

_REFERENCE_CONFIG = "ablations_final_minihack_gpu_24gb.yaml"

# The result-affecting key set is declared once, in production, as
# ``run_ablations._RESULT_AFFECTING``: the same set that classifies these
# configs is the one ``--merge`` enforces on the configs a run recorded.

#: Configs whose runs may be merged with the reference.
_POOLABLE = {_REFERENCE_CONFIG}

#: Configs that must NOT be merged with the reference, mapped to the
#: result-affecting keys on which they are known to diverge.
_NOT_POOLABLE = {
    "ablations_final_minihack_gpu_h200.yaml": frozenset(
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
        for k in ra._RESULT_AFFECTING
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
# --merge enforces the policy above on the configs a run actually recorded.
# The classification tests read the shipped YAML; these read results.json,
# which is what an operator merges.
# ---------------------------------------------------------------------------


def _results_file(tmp_path: Path, name: str, config: dict, scores: list[float]):
    """Write a minimal results.json recording *config*.

    Args:
        tmp_path: Directory to write into.
        name:     File name.
        config:   The config to record, exactly as given.
        scores:   Per-seed scores for the single ablation in the file.

    Returns:
        The path, as a string.
    """
    payload = {
        "pretrained_score": 0.5,
        "config": config,
        "ablations": {
            "baseline_rl": {
                "score": sum(scores) / len(scores),
                "score_std": 0.0,
                "all_scores": scores,
                "history": {},
            }
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


@pytest.mark.parametrize("name", sorted(_NOT_POOLABLE))
def test_merge_refuses_a_not_poolable_pair(ra, tmp_path, name):
    """The refusal names every recorded diverging key, with both values."""
    paths = [
        _results_file(
            tmp_path,
            "ref.json",
            ra._load_ablation_config(str(_ABL_CONFIGS / _REFERENCE_CONFIG)),
            [1.0],
        ),
        _results_file(
            tmp_path,
            "cand.json",
            ra._load_ablation_config(str(_ABL_CONFIGS / name)),
            [2.0],
        ),
    ]
    with pytest.raises(ValueError) as excinfo:
        ra._merge_result_files(paths)

    message = str(excinfo.value)
    assert "not poolable" in message
    for key in _NOT_POOLABLE[name]:
        assert key in message, f"{key} is a recorded divergence but unnamed"


def test_merge_refuses_a_file_that_records_no_config(ra, tmp_path):
    """A distinct refusal: absent is not equal, so it is never merged on trust."""
    paths = [
        _results_file(
            tmp_path,
            "ref.json",
            ra._load_ablation_config(str(_ABL_CONFIGS / _REFERENCE_CONFIG)),
            [1.0],
        ),
        _results_file(tmp_path, "bare.json", {}, [2.0]),
    ]
    with pytest.raises(ValueError, match="records no config"):
        ra._merge_result_files(paths)


def test_a_merged_config_is_one_input_file_and_the_merge_is_recorded(ra, tmp_path):
    """The config recorded beside merged results is one input file's, whole,
    and which one is written next to it (spec-ablations §1.3).

    Merging the configs key by key produced a config that matched no input
    file. The poolability guard forbids the result-affecting keys from
    diverging, so the chimera can only form out of the rest -- worker
    counts, output paths, the W&B run name -- but those are what tell a
    reader which machine produced the numbers, and a per-key blend names a
    run that never happened.

    Derivation: two poolable files differing only in the W&B project, a key
    the guard does not police. The blend takes the second file's value while
    every other key comes from the first; the fix takes the first file's
    config entire, and records that it did.
    """
    config_a = ra._load_ablation_config(str(_ABL_CONFIGS / _REFERENCE_CONFIG))
    config_a["wandb_project"] = "run-on-gpu-24gb"
    config_b = dict(config_a)
    config_b["wandb_project"] = "run-on-gpu-h200"
    paths = [
        _results_file(tmp_path, "a.json", config_a, [1.0, 2.0]),
        _results_file(tmp_path, "b.json", config_b, [3.0]),
    ]
    _, _, merged_config = ra._merge_result_files(paths)

    # The whole of the first file's config, not a key-by-key blend.
    assert merged_config == config_a
    assert merged_config["wandb_project"] == "run-on-gpu-24gb"
    assert ra._merge_result_files(paths[::-1])[2] == config_b

    # And the merge itself is on the record.
    provenance = {"inputs": paths, "config_from": paths[0]}
    payload = json.loads(
        ra._results_to_json({}, 0.5, merged_config, provenance).decode()
    )
    assert payload["merge_provenance"] == provenance
    # A single run carries no provenance block, so its presence marks a merge.
    assert "merge_provenance" not in json.loads(
        ra._results_to_json({}, 0.5, merged_config).decode()
    )


def test_merge_still_pools_two_runs_of_the_same_config(ra, tmp_path):
    """The guard is a refusal on wrong input only: a poolable pair merges to
    the same values it did before the guard existed."""
    config = ra._load_ablation_config(str(_ABL_CONFIGS / _REFERENCE_CONFIG))
    paths = [
        _results_file(tmp_path, "a.json", config, [1.0, 2.0]),
        _results_file(tmp_path, "b.json", config, [3.0]),
    ]
    merged, pretrained, merged_config = ra._merge_result_files(paths)

    assert merged["baseline_rl"]["all_scores"] == [1.0, 2.0, 3.0]
    assert merged["baseline_rl"]["score"] == pytest.approx(2.0)
    assert merged["baseline_rl"]["score_std"] == pytest.approx(math.sqrt(2 / 3))
    assert pretrained == pytest.approx(0.5)
    assert merged_config == config


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


def _plant(directory: Path) -> None:
    """Write the file minihack discovery looks for."""
    (directory / "iter1.pth").write_text("")


def test_publish_discovery_is_at_the_released_layout_and_skips_downloads(tmp_path):
    """Checkpoint discovery finds the released layout and never the Hub
    download copies under `checkpoints/hf/` (both repos' README, Publishing).

    The two repos failed this in opposite directions. craftax used a
    fixed-depth glob, which happened to exclude `checkpoints/hf/` because a
    download sits one level deeper than a real checkpoint -- true by
    arithmetic, not by intent, and silent if the layout ever changed depth.
    minihack used a recursive `rglob("*.pth")`, which on its live tree found
    ten directories, **none of them at the released layout**: seven raw
    `dagger_<timestamp>/` run directories, the repository root with two loose
    files, and two artefacts already published, which a publish would have
    pushed back up into a nested `checkpoints/hf/checkpoints/...` tree.

    Both now discover at the released layout and both skip `checkpoints/hf/`
    explicitly, so the exclusion holds at any depth.
    """
    hf = _hf_upload()
    monkey_ckpts = tmp_path / "checkpoints"
    released = monkey_ckpts / "online" / "Some-Model-100M"
    download = monkey_ckpts / "hf" / "checkpoints" / "online" / "Some-Model-100M"
    stray = monkey_ckpts / "dagger_20260101_000000_abcd"
    for d in (released, download, stray):
        d.mkdir(parents=True)
        _plant(d)
    # A download parked at the released depth is still a download.
    shallow_download = monkey_ckpts / "hf" / "Some-Model-100M"
    shallow_download.mkdir(parents=True)
    _plant(shallow_download)

    hf.CKPTS = monkey_ckpts
    found = hf.discover_checkpoints()

    assert set(found) == {released}, sorted(str(p) for p in found)
    assert not any(hf._is_download_copy(p) for p in found)
    assert hf._is_download_copy(download)
    assert hf._is_download_copy(shallow_download)



def test_the_published_config_drops_the_same_environment_keys_in_both_repos():
    """The published config keeps the recipe and drops provenance, by one
    declaration that is byte-identical across the repos (both READMEs).

    Neither repo dropped `use_wandb`: it starts with neither `wandb_` nor
    `hub_`, so minihack's prefix rule missed it, and craftax removed only the
    nested `_wandb` blob, leaving `USE_WANDB`, `WANDB_ENTITY` and
    `WANDB_PROJECT` in the released `config.yaml`. No credential was exposed
    either way -- those live in `_wandb` and `wandb-metadata.json`, both
    already removed -- but the published surface advertised an account and a
    project that are nothing to do with the recipe, and the two repos
    advertised different ones.

    Keys are compared lower-cased because craftax records them UPPERCASE and
    minihack lower-case.
    """
    hf = _hf_upload()

    for key in ("_wandb", "use_wandb", "USE_WANDB", "wandb_project",
                "WANDB_ENTITY", "hub_repo_id", "HUB_TOKEN"):
        assert hf.is_environment_key(key), key

    for key in ("lr", "LR", "batch_size", "NUM_ENVS", "noise_schedule",
                "use_amp", "USE_AMP", "hubris", "wandbish"):
        assert not hf.is_environment_key(key), key


# ---------------------------------------------------------------------------
# Publishing: the scrub is APPLIED at every depth, not merely correct
# ---------------------------------------------------------------------------
# The suite already asserts is_environment_key() CLASSIFIES correctly. Nothing
# asserted it was APPLIED below the top level, and in the sibling repo it was
# not: scrub_abs_paths() recursed with shorten_paths but filtered environment
# keys only at the top of the document, so WANDB_PROJECT and WANDB_ENTITY
# survived one level down inside a nested config_snapshot and were published,
# while the uploader printed a successful scrub. The predicate was proven and
# its use was sampled. These pin the use.


def test_environment_keys_are_dropped_below_the_top_level():
    """The regression test for the sibling's leak: a nested config snapshot."""
    hf = _hf_upload()
    doc = {
        "lr": 3e-4,
        "resume_metadata": {
            "config_snapshot": {
                "use_wandb": True,
                "wandb_project": "minihack-ReMDM-planner",
                "wandb_entity": "myopic-planner",
                "batch_size": 4608,
            },
        },
    }
    out = hf.scrub(doc)

    snap = out["resume_metadata"]["config_snapshot"]
    assert snap == {"batch_size": 4608}, snap
    assert out["lr"] == 3e-4
    # The whole document, flattened, must mention no environment key at all.
    assert hf.environment_key_paths(out) == []


def test_environment_keys_are_dropped_inside_lists():
    """Nesting through a list is the other way past a dict-only filter."""
    hf = _hf_upload()
    out = hf.scrub({"runs": [{"wandb_run_id": "a07wlxl7", "seed": 0}]})
    assert out == {"runs": [{"seed": 0}]}


def test_environment_key_paths_names_where_the_leak_is():
    """A scrub that reports 'done' without naming what it removed is how the
    sibling's leak stayed invisible."""
    hf = _hf_upload()
    paths = hf.environment_key_paths(
        {"a": {"b": {"wandb_project": "p"}}, "use_wandb": True}
    )
    assert sorted(paths) == ["a.b.wandb_project", "use_wandb"]


def test_the_scrub_still_shortens_paths_at_depth():
    """Both passes recurse; fixing the filter must not lose the shortener."""
    hf = _hf_upload()
    out = hf.scrub({"outer": {"dataset_path": "/very/long/cluster/path/data.npz"}})
    assert out["outer"]["dataset_path"] == "path/data.npz"


def test_dropping_environment_keys_preserves_mapping_type():
    """A state dict must stay an OrderedDict; the published checkpoint keeps
    its structure, and only the environment keys go."""
    from collections import OrderedDict

    hf = _hf_upload()
    out = hf.drop_environment_keys(
        OrderedDict([("wandb_run_id", "x"), ("model_state_dict",
                     OrderedDict([("w", 1), ("b", 2)]))])
    )
    assert isinstance(out, OrderedDict)
    assert isinstance(out["model_state_dict"], OrderedDict)
    assert list(out["model_state_dict"]) == ["w", "b"]
    assert "wandb_run_id" not in out


def test_a_clean_document_is_returned_unchanged():
    """No environment key anywhere means nothing is rewritten -- the property
    scrub_checkpoint relies on to copy bytes rather than re-save them."""
    hf = _hf_upload()
    doc = {"lr": 1e-4, "nested": {"batch_size": 8, "envs": ["a", "b"]}}
    assert hf.scrub(doc) == doc
    assert hf.environment_key_paths(doc) == []


# ---------------------------------------------------------------------------
# Publishing: the scrub REACHES every staged file, not just the ones named
# ---------------------------------------------------------------------------
# Depth was one half. Reach was the other: stage_runs copied results.json with
# shutil.copy2 and copied tables/, figures/ and gdelta/ with copytree, so
# nothing in a run directory was ever scrubbed. The released results.json
# carried wandb_project, wandb_entity, baselines_wandb_project and the
# absolute path of the machine it ran on. scrub() was correct and simply never
# called. Naming the files to scrub would repeat the mistake one level up, so
# the tree is walked instead.


def test_a_substring_rule_catches_wandb_keys_that_are_not_prefixed():
    """`baselines_wandb_project` was live in both released config.yaml files;
    the sibling's `RESUME_WANDB_RUN_ID` is the same shape."""
    hf = _hf_upload()
    for key in ("baselines_wandb_project", "RESUME_WANDB_RUN_ID",
                "eval_wandb_entity"):
        assert hf.is_environment_key(key), key
    # The underscore keeps it narrow: these must still survive.
    for key in ("wandbish", "hubris", "bandwidth"):
        assert not hf.is_environment_key(key), key


def test_the_staged_tree_is_scrubbed_including_copied_directories(tmp_path):
    """The regression test for the reach bug, in the real staged shape."""
    hf = _hf_upload()
    run = tmp_path / "experiments" / "rl_finetuning" / "outputs" / "abl"
    (run / "gdelta").mkdir(parents=True)
    (run / "results.json").write_text(json.dumps({
        "config": {"wandb_entity": "myopic-planner", "batch_size": 4608,
                   "baselines_wandb_project": "x"},
        "config_from": "/cs/student/project_msc/2025/dsml/someone/tmp/cfg.yaml",
    }))
    (run / "gdelta" / "gdelta_aggregate.json").write_text(json.dumps(
        {"sources": ["/cs/student/project_msc/2025/dsml/someone/out/s0.json"]}
    ))
    (run / "diagnosis.md").write_text(
        "Run at /cs/student/project_msc/2025/dsml/someone/out/results.json.\n"
    )

    changed = hf.scrub_staged_tree(tmp_path)

    res = json.loads((run / "results.json").read_text())
    assert res["config"] == {"batch_size": 4608}
    assert res["config_from"] == "tmp/cfg.yaml"
    agg = json.loads((run / "gdelta" / "gdelta_aggregate.json").read_text())
    assert agg["sources"] == ["out/s0.json"]
    assert "/cs/student" not in (run / "diagnosis.md").read_text()
    assert len(changed) == 3, changed


def test_shortening_leaves_relative_paths_alone(tmp_path):
    """The path regex must anchor to an absolute path.

    Unanchored it matched from the middle of a relative one, so the generated
    LaTeX comment `experiments/rl_finetuning/analysis/tables.py` became
    `experimentsanalysis/tables.py` -- a scrub corrupting a file it had no
    business touching.
    """
    hf = _hf_upload()
    keep = "Generated by experiments/rl_finetuning/analysis/tables.py."
    assert hf.shorten_text_paths(keep) == keep
    assert hf.shorten_text_paths(
        "at /cs/student/project_msc/2025/dsml/someone/out/results.json now"
    ) == "at out/results.json now"

    # A narrow character class does not merely under-match here: the
    # lookbehind refuses to restart mid-path, so an absolute path containing
    # `@` or `+` would fail to match at all and ship UNSHORTENED. `@` is where
    # an email-like component appears in a home directory path.
    assert hf.shorten_text_paths("/home/user@dom/proj/run/file.txt") == "run/file.txt"
    assert hf.shorten_text_paths("/cs/a+b/c/d/e.txt") == "d/e.txt"

    tex = tmp_path / "tables"
    tex.mkdir()
    (tex / "results.tex").write_text(f"% {keep}\n\\newcommand{{\\mhX}}{{1}}\n")
    before = (tex / "results.tex").read_bytes()
    assert hf.scrub_staged_tree(tmp_path) == []
    assert (tex / "results.tex").read_bytes() == before


def test_a_clean_staged_tree_keeps_its_bytes(tmp_path):
    """Nothing to scrub means nothing rewritten, so an untouched release does
    not churn on the Hub."""
    hf = _hf_upload()
    f = tmp_path / "tables"
    f.mkdir()
    (f / "main_results.csv").write_text("Method,Score\nbaseline_rl,0.43\n")
    before = (f / "main_results.csv").read_bytes()
    assert hf.scrub_staged_tree(tmp_path) == []
    assert (f / "main_results.csv").read_bytes() == before


def test_inference_payloads_are_scrubbed_not_merely_shortened(tmp_path):
    """stage_inference shortened paths and filtered no keys."""
    hf = _hf_upload()
    src = tmp_path / "eval.json"
    src.write_text(json.dumps({"wandb_project": "p", "mean_win_rate": 0.44}))
    hf.INFERENCE = tmp_path / "results" / "inference"
    hf.ROOT = tmp_path
    staging = tmp_path / "staging"
    hf.stage_inference(staging, [src])
    out = json.loads((staging / "results" / "inference" / "eval.json").read_text())
    assert out == {"mean_win_rate": 0.44}


# ---------------------------------------------------------------------------
# Publishing: the licence is the one git committed, not the one on disk
# ---------------------------------------------------------------------------
# `hf download --local-dir .` writes the Hub's copies over the working tree.
# The Hub repo carries its own README.md (the model card) and its own LICENSE,
# and comparing `git ls-files` against the Hub listing those two are the only
# tracked files a pull overwrites. Publishing read LICENSE straight off the
# tree, so a pull-then-publish round trip re-published whatever the pull left
# behind. That shipped a LICENSE naming a superseded paper title, caught only
# by hand -- neither `--dry-run` nor the staged-tree listing would show it,
# because both print a LICENSE that is merely the wrong one.

_STALE_LICENSE = (
    'Copyright (c) 2026 The authors of "The Double Intractability of '
    'Reinforcement Learning for Discrete Diffusion Planners"\n'
)
_CURRENT_LICENSE = (
    'Copyright (c) 2026 The authors of "Return-Weighted ELBO Fine-Tuning '
    'Degrades Masked Diffusion Planners"\n'
)

requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is not on PATH"
)


def _checkout(root: Path, license_text: str) -> None:
    """A throwaway checkout with LICENSE committed.

    Isolated from the developer's git config: hooks, commit templates and
    commit.gpgsign would otherwise leak in and make this machine-dependent.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "LICENSE").write_text(license_text)
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], cwd=root, env=env, capture_output=True, check=True
    )
    run("init", "-q", "--template=")
    run("add", "LICENSE")
    run(
        "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
        "commit", "-qm", "licence",
    )


def _staged_license(hf, root: Path, staging: Path) -> bytes:
    """Run the real `stage()` against *root* and return the staged LICENSE.

    The module constants are bound from ROOT at import (hf_upload.py, top), so
    rebinding ROOT alone would leave `stage_inference`'s
    `INFERENCE.relative_to(ROOT)` pointing at the real repository and raising.
    """
    hf.ROOT = root
    hf.CKPTS = root / "checkpoints"
    hf.RUNS = root / "experiments" / "rl_finetuning" / "outputs"
    hf.INFERENCE = root / "results" / "inference"
    hf.PAPER_FIGURES = root / "results" / "paper_figures"
    staging.mkdir(parents=True, exist_ok=True)
    # This repo's stage() carries a trailing `metric` the sibling's does not.
    hf.stage(staging, {}, [], [], [], None)
    return (staging / "LICENSE").read_bytes()


@requires_git
def test_the_published_licence_is_the_one_git_committed(tmp_path):
    """A clean checkout publishes the committed bytes."""
    repo = tmp_path / "repo"
    _checkout(repo, _CURRENT_LICENSE)
    staged = _staged_license(_hf_upload(), repo, tmp_path / "staging")
    assert staged == _CURRENT_LICENSE.encode()


@requires_git
def test_a_clobbered_licence_publishes_gits_bytes_not_the_working_trees(
    tmp_path, capsys
):
    """The regression test for the incident: a LICENSE overwritten by a Hub
    download must not reach the Hub, and the operator must be told."""
    repo = tmp_path / "repo"
    _checkout(repo, _CURRENT_LICENSE)
    # Exactly what `hf download --local-dir .` did.
    (repo / "LICENSE").write_text(_STALE_LICENSE)

    staged = _staged_license(_hf_upload(), repo, tmp_path / "staging")

    assert staged == _CURRENT_LICENSE.encode()
    assert staged != _STALE_LICENSE.encode()
    assert b"Double Intractability" not in staged
    assert "differs from the one committed at HEAD" in capsys.readouterr().err


@requires_git
def test_the_published_licence_is_byte_exact(tmp_path):
    """CRLF and a missing trailing newline survive.

    Fails the moment `capture_output=True` is 'tidied' into `text=True`, or
    `git cat-file blob` is swapped back for `git show`.
    """
    repo = tmp_path / "repo"
    awkward = 'Copyright (c) 2026\r\nNo trailing newline here.'
    repo.mkdir()
    (repo / "LICENSE").write_bytes(awkward.encode())
    _checkout(repo, awkward)

    staged = _staged_license(_hf_upload(), repo, tmp_path / "staging")
    assert staged == awkward.encode()


def test_publishing_from_a_non_git_checkout_still_works_and_says_so(
    tmp_path, capsys
):
    """A tarball or slim container must still publish -- with a warning."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "LICENSE").write_text(_CURRENT_LICENSE)

    staged = _staged_license(_hf_upload(), repo, tmp_path / "staging")

    assert staged == _CURRENT_LICENSE.encode()
    assert "UNVERIFIED" in capsys.readouterr().err


@requires_git
def test_an_uncommitted_licence_falls_back_to_the_working_tree(tmp_path, capsys):
    """`git init` with nothing committed is a distinct branch from 'no repo'."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "LICENSE").write_text(_CURRENT_LICENSE)
    subprocess.run(["git", "init", "-q", "--template="], cwd=repo, check=True)

    staged = _staged_license(_hf_upload(), repo, tmp_path / "staging")

    assert staged == _CURRENT_LICENSE.encode()
    assert "UNVERIFIED" in capsys.readouterr().err


def test_git_missing_falls_back_rather_than_failing_the_publish(
    tmp_path, capsys, monkeypatch
):
    """Decision: git is consulted, never required."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "LICENSE").write_text(_CURRENT_LICENSE)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    staged = _staged_license(_hf_upload(), repo, tmp_path / "staging")

    assert staged == _CURRENT_LICENSE.encode()
    assert "git is not on PATH" in capsys.readouterr().err


def test_the_model_card_no_longer_recommends_a_clobbering_download():
    """The card told users to run the command that causes the bug.

    `snapshot_download(repo_id=..., local_dir=".")` with no narrowing writes
    the Hub's own README.md, LICENSE and .gitattributes over a working copy's.
    """
    hf = _hf_upload()
    row = {
        "path": "checkpoints/online/Minihack-Online-Diffusion-DAgger-100M",
        "role": "Diffusion planner (online DAgger)",
        "env": "MiniHack-Room-Random-5x5-v0",
        "arch": "4L, d_model 256, 4 heads, horizon 64, 5M params",
        "step": "563",
        "detail": "5,657,661 env steps",
        "size": "100 MB",
        "restores": "`model_state_dict`",
        "file": "iter563.pth",
        "config": "config_563.yaml",
    }
    # This repo's model_card() takes fig_rows *and* a trailing metric.
    card = hf.model_card("owner/repo", [row], [], [], [], 1.0, None)

    assert 'snapshot_download(repo_id="owner/repo", local_dir=".")' not in card
    assert "ignore_patterns" in card
    for name in ("README.md", "LICENSE", ".gitattributes"):
        assert name in card, name


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


_HF_ONLINE = (
    _ROOT
    / "checkpoints"
    / "hf"
    / "checkpoints"
    / "online"
    / "Minihack-Online-Diffusion-DAgger-100M"
)
_needs_artefact = pytest.mark.skipif(
    not (_HF_ONLINE / "selection.json").exists(),
    reason="released HF checkpoints not downloaded to checkpoints/hf/",
)


@_needs_artefact
def test_the_released_dagger_selection_is_the_documented_historical_exception():
    """The released DAgger `selection.json` keeps its null candidate set.

    It was written by a `selection()` that read `checkpoint_every` and
    `max_iterations` after both had been renamed out of the config, so
    `.get()` returned None. Author decision 2026-08-17: the artefact stays
    as published and is historical/noncanonical -- the checkpoint's own
    config snapshot carries the real cadence and budget, so nothing is
    lost, and rewriting a released file is not worth the churn.

    This fails if the artefact is silently re-published, or if the README's
    historical note stops explaining it. A re-publish is fine -- it just has
    to retire the note and this test together, exactly as the craftax N3
    rename did.
    """
    released = json.loads((_HF_ONLINE / "selection.json").read_text())

    assert released["candidates"] == {
        "unit": "dagger_iterations",
        "every": None,
        "configured_max": None,
    }

    note = (_ROOT / "README.md").read_text()
    assert "Historical note" in note
    assert '"every": null' in note and "940000" in note


@_needs_artefact
def test_a_publish_from_the_current_code_records_the_real_candidate_set():
    """The same snapshot, run through the current `selection()`.

    The published nulls were never a data loss: the released
    `config.yaml` declares both canonical keys, so the corrected reader
    recovers the cadence and budget from the artefact itself. This pins
    that a re-publish would fix the record rather than repeat it.
    """
    hf = _hf_upload()
    cfg = hf.scrub(yaml.safe_load((_HF_ONLINE / "config.yaml").read_text()))

    sel = hf.selection(cfg, {"counter": "dagger_iteration", "value": 563}, None)

    assert sel["candidates"] == {
        "unit": "env_steps",
        "every": cfg["checkpoint_every_timesteps"],
        "configured_max": cfg["total_timesteps"],
    }
    assert sel["candidates"]["every"] is not None
    assert sel["candidates"]["configured_max"] is not None


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


def _is_config_ref(node: ast.AST) -> bool:
    """Is *node* a reference to a merged config?

    Two shapes reach one and both are production: a bare name (``cfg.KEY``)
    and an attribute chain ending in a config name (``self.cfg.KEY``,
    ``ctx.cfg.KEY``). Matching only the first made the scan blind to a fifth
    of the ablation recipe's readers (sweep S0-3, gate F-7).
    """
    if isinstance(node, ast.Name):
        return node.id in _CONFIG_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _CONFIG_NAMES
    return False


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
            and _is_config_ref(node.func.value)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self._record(node.args[0].value, node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and _is_config_ref(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self._record(node.args[1].value, node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            _is_config_ref(node.value)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            self._record(node.slice.value, node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            _is_config_ref(node.value)
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


# ---------------------------------------------------------------------------
# Ablation descriptions (sweep S0-2)
#
# `AblationSpec.description` is what `--list` prints and what every run logs,
# and nothing read it: reverting either the F-2 or the F-3 description fix
# left the whole suite green. The table below is the shared canon, character
# for character with the sibling repo -- which is also what caught two
# typographic divergences between them (`baseline_rl`'s dash and
# `kl_penalty`'s "vs"), aligned to ASCII in the same commit.
#
# It is deliberately NOT compared with the `experiments/README.md` table:
# those cells are short labels in a different register ("Soft KL constraint
# vs pretrained") and the registry carries the mechanism sentence. 24 of 25
# differ by design; only `mixed_replay` coincides, because F-2 fixed it by
# copying the README wording.
# ---------------------------------------------------------------------------

_EXPECTED_DESCRIPTIONS = {
    "action_diversity": (
        "Baseline ELBO with degenerate (all-same-action) plans discarded"
    ),
    "advantage_clip": (
        "Baseline ELBO with PPO-style advantage clipping to [1-eps, 1+eps]"
    ),
    "attention_only": (
        "Baseline ELBO updating only attention weights (Q/K/V/O); FFN frozen"
    ),
    "baseline_rl": (
        "Return-weighted ELBO -- no modifications"
    ),
    "bc_all": (
        "Uniform ELBO on all rollout windows (no advantage weighting)"
    ),
    "bc_wins": (
        "Uniform ELBO on win windows only (no advantage weighting)"
    ),
    "entropy_bonus": (
        "Baseline ELBO minus entropy bonus (encourages action diversity)"
    ),
    "ewc": (
        "ELBO + Elastic Weight Consolidation (Fisher diagonal regularisation)"
    ),
    "ffn_only": (
        "Baseline ELBO updating only FFN layers; attention frozen"
    ),
    "frozen_backbone": (
        "Baseline ELBO training the action head and token embeddings (backbone frozen)"
    ),
    "gradient_surgery": (
        "PCGrad: RL gradient projected to remove conflict with BC gradient"
    ),
    "head_only": (
        "Baseline ELBO updating only the final linear projection"
    ),
    "kl_penalty": (
        "Return-weighted ELBO + soft KL penalty vs pretrained"
    ),
    "layer_ablation_top1": (
        "Baseline ELBO updating only the top-1 transformer block + head"
    ),
    "layer_ablation_top2": (
        "Baseline ELBO updating only the top-2 transformer blocks + head"
    ),
    "layer_ablation_top3": (
        "Baseline ELBO updating only the top-3 transformer blocks + head"
    ),
    "llrd": (
        "Baseline ELBO with Layer-wise Learning Rate Decay"
    ),
    "lora": (
        "Baseline ELBO with LoRA adaptation (rank-r attention projections only)"
    ),
    "low_t": (
        "Return-weighted ELBO restricted to low-t (fine-detail) regime"
    ),
    "mixed_replay": (
        "Self-replay: the run's own past online windows resampled into each batch"
    ),
    "normalized_adv": (
        "Baseline ELBO with (A - mean) / (std + eps) advantage normalisation"
    ),
    "reward_filtering": (
        "Baseline ELBO trained only on top-75th-percentile return windows"
    ),
    "reward_model": (
        "Baseline ELBO with advantages re-weighted by a learned MLP reward model"
    ),
    "running_stats": (
        "Baseline ELBO with EMA running mean/std for advantage normalisation"
    ),
    "t_curriculum": (
        "ELBO with t range annealed from high-t to low-t over training"
    ),
    "trust_region_kl": (
        "Baseline ELBO + hard KL trust region via quadratic barrier"
    ),
}


def test_every_registered_ablation_has_its_pinned_description():
    """Character for character, and the same table in the sibling repo."""
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    assert set(REGISTRY) == set(_EXPECTED_DESCRIPTIONS), (
        f"registry/table mismatch: only in registry "
        f"{sorted(set(REGISTRY) - set(_EXPECTED_DESCRIPTIONS))}, only in "
        f"table {sorted(set(_EXPECTED_DESCRIPTIONS) - set(REGISTRY))}"
    )
    wrong = {
        name: (spec.description, _EXPECTED_DESCRIPTIONS[name])
        for name, spec in REGISTRY.items()
        if spec.description != _EXPECTED_DESCRIPTIONS[name]
    }
    assert not wrong, f"description drift: {wrong}"


def test_every_ablation_names_a_hypothesis_and_is_listed_in_the_readme():
    """The two other strings a run surfaces. The README check is by name
    only, for the register reason in the comment above."""
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    readme = (_ROOT / "experiments" / "README.md").read_text()
    for name, spec in sorted(REGISTRY.items()):
        assert spec.hypothesis.strip(), f"{name} has no hypothesis"
        assert f"`{name}`" in readme, f"{name} is in no README table"


def test_the_reachability_scan_sees_config_reads_through_an_attribute():
    """`self.cfg.KEY` and `ctx.cfg.KEY` are production shapes; the scanner
    matched only a bare `cfg.KEY` and was blind to both (sweep S0-3, gate
    F-7). All four access forms are covered, and an unrelated attribute
    chain must still be ignored."""
    scanner = _ConfigKeyScanner()
    scanner.path = "<synthetic>"
    scanner.visit(
        ast.parse(
            "cfg.plain_name\n"
            "self.cfg.via_self\n"
            "ctx.cfg.via_ctx\n"
            "self.cfg.get('via_self_get', 0)\n"
            "self.cfg['via_self_subscript']\n"
            "getattr(self.cfg, 'via_self_getattr', 0)\n"
            "unrelated.attr.not_a_config\n"
        )
    )
    assert set(scanner.keys) == {
        "plain_name",
        "via_self",
        "via_ctx",
        "via_self_get",
        "via_self_subscript",
        "via_self_getattr",
    }
