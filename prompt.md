# Flatten config inheritance to two layers

The sibling repo `../craftax-ReMDM-planner/` has just had this done; read its
`configs/`, `main.py`, `experiments/rl_finetuning/run_ablations.py` and both
READMEs for the target shape.

Goal: `defaults.yaml` holds the shared final recipe, and each GPU config holds
only that machine's values. No config may inherit from another config.

**Investigate before changing anything.** The two repos differ; do not assume
the craftax findings transfer. Report what you find, then apply.

## Scope

- `configs/final_ucl_gpu.yaml` currently declares `extends: final_qmul_gpu.yaml`.
  Remove that link. Move everything the pair shares into `defaults.yaml`, so
  each of `final_qmul_gpu.yaml` and `final_ucl_gpu.yaml` keeps only its own
  machine values. Establish what those genuinely are: for craftax it was just
  `num_envs` and `seed`, here it looked like the hardware perf knobs and output
  paths, so check rather than copy.
- `experiments/rl_finetuning/configs/`: the machine configs should change only
  what they are meant to change and inherit the rest from
  `ablations_default.yaml`. Keys identical across *all* machine configs and
  different from the base belong in the base.
- Once nothing uses it, strip the `extends` machinery from `src/config.py` and
  `run_ablations.py`, and its tests in `tests/test_config.py`. Keep
  `run_ablations.py`'s implicit `ablations_default.yaml` base: that is a
  separate mechanism and is still in use.

## Investigate first

1. **Blast radius.** Moving values into `defaults.yaml` changes what *every*
   other preset resolves to. `smoke.yaml`, `ablation_local_only.yaml`,
   `ucl_gpu_bigger_model.yaml` and `ucl_gpu_learning_behaviour.yaml` all sit on
   those defaults. For each, list which moving keys it currently inherits, and
   pin the old value in that file so its resolved config does not move. These
   presets are experiments; silently redefining them invalidates their results.
2. **Primary/legacy pairs.** In craftax, five keys had an env-frame PRIMARY form
   that silently overrides a LEGACY update-count form when non-null. Moving the
   PRIMARY keys into the defaults therefore overrode the LEGACY value that three
   ablation presets were built around. Check whether this repo has the same
   pattern (`*_frames`, `*_cycles`, `*_final` against `*_steps`, `*_max`,
   `*_decay`, or similar). Where it does, the pin must be an explicit `null`.
3. **Bare-run behaviour.** After the change, running with no `--config` uses the
   full recipe rather than a small baseline. Confirm that is wanted, and check
   nothing (CI, smoke, docs) depended on the cheap default.
4. **Duplication left behind.** Anything shared by a subset of presets but not
   all of them now has to be restated in each. Say where that happens and how
   many keys it is, so the drift risk is on the record.

## Verify

Snapshot every preset's fully merged config before the change, apply it,
re-resolve, and diff. **Expect zero values moved**, for `configs/` and the
ablation configs alike. Any key that moves is a missing pin: report it, do not
absorb it. Then `pytest`, `--mode smoke`, a `--fast` ablation run, and
`ruff check` / `ruff format --check` against a pre-change baseline.

Note that with inheritance gone, nothing stops the QMUL and UCL configs
drifting apart. Add or keep a test asserting they differ only in the intended
machine keys.

## Docs

Update `README.md` and `experiments/README.md`: the precedence chain (no
`extends` layer), what `defaults.yaml` now is, the preset table, and any
primary/legacy hazard found in step 2. Only claim what a file actually does.
