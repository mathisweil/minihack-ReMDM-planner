# Task: make the RL fine-tuning ablation suite affordable, on the 4070 Ti

**This is prompt 2 of 2.** `PERF_OFFLINE_PROMPT_MINIHACK.md` ran first and owns the shared
files. Start from its HEAD and read its report before doing anything; if it touched
`src/buffer.py`, `src/planners/inference.py`, `src/diffusion/sampling.py` or
`src/models/denoiser.py`, those changes are yours to build on, not to rediscover.

Also read `PERF_GPU_RESULTS_MINIHACK.md` and `PERF_MEASURE_3090_RESULTS_MINIHACK.md`. They
contain measured constants for this exact box that you should not re-derive.

## Where you are

`minihack-ReMDM-planner` on the 4070 Ti box (`outback.cs.ucl.ac.uk`, RTX 4070 Ti SUPER,
16 GB, i7-14700K, 20 physical cores / 28 threads). A uv project. Repo conventions are in
`CLAUDE.md` one level up: UK English, no em dashes, evidence for every claim.

Confirm the box and record the output before anything else.

```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run --no-sync python -c "import torch;p=torch.cuda.get_device_properties(0);print(p)"
lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)'
python -c "import tempfile;print('TMPDIR:',tempfile.gettempdir())"
```

The two traps from prompt 1 apply unchanged: `uv` needs `--extra cuda12` on this box and
flipping resolutions corrupts the shared NFS venv (recover with
`uv sync --extra cuda12 --reinstall`, then `uv run --no-sync`); and `$TMPDIR` defaults to an
NFS home while local xfs is at `/tmp`. Export `TMPDIR` to local disk and say that you did.
`TMPDIR` matters more here than anywhere else in the repo, for the reason in Phase 1.

## What this pass is, and is not

This is a speed pass. `ABLATION_SMOKE_PROMPT_MINIHACK.md` is a separate task covering whether
the 25 ablations are structurally correct. If you hit an ablation that crashes or produces
nonsense, record it, skip it, and hand it there. Do not repair science in this pass.

Equally, do not change what any ablation computes. The suite's output is a comparison
*between* ablations, so a change that speeds up one arm and not another is worse than no
change at all.

## The scale, and where the time goes

`final_ablations_ucl.yaml`: `max_iter: 500`, `episodes_per_iter: 30`, `grad_steps_per_iter: 1`,
`eval_every: 25`, `eval_episodes: 20`, over 25 registered ablations against 4 ID envs.

Per ablation-seed that is **15,000 model episodes collected and 500 gradient steps**. The
suite is a collection workload with a training step attached, which is the opposite of the
DAgger arm's balance, and it means the levers are different ones.

Projected from constants measured on this box in `PERF_GPU_RESULTS_MINIHACK.md`, and clearly
a projection until you measure it:

| | per ablation-seed | x25 ablations |
|---|---|---|
| env construction on local xfs, at 25.6 ms/env | 384 s | 2.7 h |
| env construction on NFS `$TMPDIR`, at 86.3 ms/env | 1,295 s | 9.0 h |
| with the env pool, at 0.00 ms per recycled acquire | ~0 s | ~0 h |

That is construction alone, before a single env step. Multiply by `num_seeds`.

---

## Preflight

```
export TMPDIR=/tmp/remdm-$USER && mkdir -p "$TMPDIR"
git status --short && git log --oneline -8
uv sync --extra cuda12
uv run --no-sync python -m pytest tests -q
uv run --no-sync ruff check src tests
uv run --no-sync python experiments/rl_finetuning/run_ablations.py --list
ls -la checkpoints* ../checkpoints* 2>&1 | head -20
```

Expect the documented pre-existing lint errors; do not fix them. Note that `experiments/` is
not in the lint scope, so treat any ruff output there as pre-existing unless you introduced it.

**Which checkpoint and which main config is a `needs-human` question, not a guess.**
`run_ablations.py:488-489` falls back to `configs/defaults.yaml`, and the ablation config is
merged on top of it (`run_ablations.py:508`), so the *main* config decides the architecture
and therefore the VRAM. `defaults.yaml:19-21` is `n_embd: 256`, `n_head: 4`, `n_layer: 4`,
matching the DAgger recipe. But `final_ablations_ucl.yaml`'s header says the checkpoint came
from `configs/qmul_gpu.yaml`, and the only reference run the 3090 Ti pass found on disk was a
wider model (`ucl_gpu_bigger_model.yaml:19-20`, `n_embd: 384`, `n_head: 6`), which would move
every memory number below. A checkpoint whose dimensions do not match the config will fail at
`load_state_dict` (`training.py:741`), so this resolves itself loudly rather than silently,
but resolve it before you measure. Record which pair you used and carry it through every
measurement so the numbers are internally consistent.

---

## Phase 1: measure one ablation end to end, before changing anything

Run `baseline_rl` for a reduced `--max-iter` at the real `final_ablations_ucl.yaml` settings
otherwise, long enough to cross an `eval_every: 25` boundary and a `cka_every: 50` boundary.
The suite already logs most of what you need at `training.py:1104-1120`.

Record per iteration:

| Quantity | Where from |
|---|---|
| iteration wall time | `speed/iter_time_sec` |
| collect / train split | `speed/collect_time_sec`, `speed/train_step_time_sec` |
| of collect: env construction, env stepping, GPU sampling | instrument `_collect_training_data_gpu` |
| effective batch | `train/effective_batch_size`, already logged |
| peak VRAM | `speed/gpu_memory_mb`, already logged |
| cost of each diagnostic | time the blocks at `training.py:1144-1247` individually |
| cost of one eval | the block at `training.py:1250` |

Then a projection to the full suite, with `num_seeds` stated, and labelled a projection.

**Two gates before you write any code.**

1. **VRAM.** `batch_size: 4608` in `final_ablations_ucl.yaml` was written for a 24 GB card.
   Scaling the certified 6,678 MB at batch 2048 gives a projection of about 15.0 GB against
   this card's 15,974 MB total, and that projection assumes the 256-dim model; the 384-dim
   alternative above would be worse. That is not obviously survivable, especially with the
   diagnostic passes at `training.py:1171-1184` and `1144-1154` running a full extra forward
   and backward at the same batch. But **check `train/effective_batch_size` before assuming it
   binds**: the batch is data-limited by what 30 episodes yield, `local_b = local_obs[:batch_size]`
   at `training.py:994-997` takes whatever is there, and 30 episodes may well produce fewer
   than 4,608 windows. Measure the distribution over iterations, then decide whether there is
   a VRAM problem at all.
   If it does OOM: **do not lower `batch_size`.** Gradient accumulation over microbatches
   reproduces the same gradient exactly, provided the loss reduction is a plain mean over the
   batch. Check that it is, in `losses.py`, before claiming exactness, and pin it with a test
   that the accumulated gradient matches the single-batch gradient to within float tolerance.
   If the reduction turns out to be weighted in a way accumulation cannot reproduce, stop and
   report rather than approximating.
2. **Where the time actually is.** If collection is not the majority of iteration time, the
   change list below is wrong and you should follow your profile instead. Say so in the report.
   C0 was found that way on the DAgger pass and beat every change on that prompt's list.

---

## Phase 2: the changes

One commit each, suite and lint green at every commit.

**X1. Use the env pool. This is the whole pass.** `training.py:519` calls
`make_env(env_id, None, cfg)` directly and `training.py:614-616` closes every env in a
`finally`, so the suite constructs and destroys 30 environments per iteration, 15,000 per
ablation-seed. The pool that fixed exactly this for DAgger already exists in
`src/envs/minihack_env.py` and is used by `collect.py` and `inference.py`: `acquire_env`,
`release_env` on success, `discard_env` on the exception path. A recycled acquire measured
0.00 ms on this box against 25.6 ms for a construction on local disk.

Two things to get right. `shaped_reward` must stay at its default `True` here: the episode
return drives the advantage weighting at `training.py:952-962` and the win-rate tracking
against `win_threshold`, so turning shaping off would change what every ablation optimises.
And `discard_env`, not `release_env`, on the failure path, so a broken env is never recycled.

`REMDM_MAX_IDLE_ENVS` bounds the idle pool; the default is 64 and 30 concurrent envs sits
under it. Watch RSS anyway, per Phase 4.

**X2. Apply C1 to the three places on this path that never got it.** Certified worth 3.1% of
the gradient step on the 3090 Ti and 0.9% here; the transfers below are larger.

- `training.py:552-565`: the replan H2D does `.long()` on the host before `.to(device)`, so
  int16 glyph maps cross PCIe as int64. Widen on the device, as `online.py` and `offline.py`
  now do.
- `training.py:354-356`: `_extract_windows` widens a whole episode's observations to int64 on
  the host, and the result is concatenated and transferred at `training.py:649-652`. At the
  projected several thousand windows per iteration this is the largest single transfer in the
  loop. Keep the buffer dtype across the transfer and widen on the device.
- `inference.py:187-200`: the same pattern in the shared evaluator. If prompt 1 already fixed
  it, skip and say so.

int16 to int64 widening is lossless for glyph IDs. Pin it with a test rather than asserting it.

**X3. Stop rebuilding the eval model every iteration.** `training.py:899` calls
`ema.make_eval_model(raw_model)` inside the iteration loop, which is
`copy.deepcopy(model)` plus an EMA apply (`denoiser.py:428-440`), so it allocates and copies
a full model per iteration, 500 times per ablation-seed, plus `training.py:1251` and
`training.py:1292`. Keep one persistent eval model and refresh its weights in place. The
weights it is used with must be identical to what the current code produces; verify that
directly rather than by inspection.

**X4. Remove the `.item()` storm in the health metrics.** `training.py:1129-1141` computes
`param_norm` and `param_drift_from_init` with one `.item()` per parameter tensor, twice, so
roughly 144 device syncs every 10 iterations. `torch._foreach_norm` over the parameter list
with a single sync at the end gives the same value. This is the same class of defect as C0
and C2, both of which were worth more than they looked.

**X5. Stop zeroing 50 MB per iteration.** `training.py:526-536` pre-allocates
`(n, max_steps + 1, map_h, map_w)` int16 history, which is 30 x 501 x 21 x 79 x 2 bytes, about
49.9 MB, allocated and zeroed every iteration and mostly unused because episodes end well
before 500 steps. Size it from actual episode lengths, or reuse one buffer across iterations.
Measure it first: it may be cheap enough to leave.

**X6. Settle `torch_compile: false`.** Both final ablation configs disable it with the
comment "compilation overhead not worth it for 500 iters". At `grad_steps_per_iter: 1` that is
500 gradient steps per ablation-seed and the model is rebuilt for every ablation and every
seed, so compilation would be paid 25 x `num_seeds` times. The comment is probably right, but
it has never been measured and the diagnostics add forward and backward passes it did not
account for. Measure both ways on one ablation and either confirm the comment with numbers or
change the flag. `try_compile` falls back silently to eager when no C compiler is visible
(`denoiser.py:317`), so grep the log rather than trusting the flag; `/usr/bin/gcc` was present
on this box during the DAgger pass.

---

## Phase 3: what must not change

Every ablation must stay comparable with every other ablation and with the published tables.
Pinned:

`max_iter`, `episodes_per_iter`, `grad_steps_per_iter`, `batch_size` (see the gate above),
`lr`, `weight_decay`, `max_grad_norm`, `diffusion_steps_collect`, `eval_every`,
`eval_episodes`, every diagnostic cadence (`grad_align_every`, `repr_drift_every`,
`t_analysis_every`, `cka_every`, `per_layer_every`, `t_analysis_n_bins`, `cka_batch_size`),
the advantage knobs (`win_threshold`, `return_weight_floor`, `return_weight_cap`), and every
per-group hyperparameter in Groups A, B and D.

Also pinned: which environments are sampled and how. `_collect_training_data_gpu` draws
`random.choice(cfg.id_envs)` per episode at `training.py:512`, seeded per ablation at
`training.py:734-736`. Any change that alters the sequence of env IDs or seeds changes the
data, and the seeds are what make the multi-seed tables mean anything.

Reducing the diagnostics is not a speed-up available to you. They are the experiment.

**One inconsistency to report, not fix.** `final_ablations_ucl.yaml` sets no `num_seeds` and
ends on a dangling `# -- Output / Logging` header, so it inherits `num_seeds: 1` from
`ablations_default.yaml:121`, while `final_ablations_qmul.yaml:67` sets 3 with the note
"matching the published 3-run tables". The UCL config also inherits rather than sets the wandb
keys. Whether the UCL runs were meant to be 1 seed or 3 changes the compute budget by 3x and
is the author's call. Put it in the report.

---

## Phase 4: verify

The bar here is that the ablations still compute the same thing, and it is a higher bar than
the DAgger pass because there is no single loss curve to check.

1. **Same-seed comparison on at least three ablations from different groups.** Suggested:
   `baseline_rl` (A), `gradient_surgery` (B, exercises the PCGrad path and the AMP unscale
   interaction at `training.py:1007-1014`), and `mixed_replay` (A, exercises
   `MixedReplayBuffer`). Same seed, same `--max-iter`, before and after. Compare the training
   loss series, `train/effective_batch_size`, and the eval scores. Establish a noise floor
   first by running identical code twice: AMP and kernel selection are non-deterministic, and
   the collection path is stochastic, so the control band here will be much wider than the
   DAgger pass's 2.8e-4 and you need to know how wide before you can read anything.
2. **Env-step counts and episode counts** should be identical before and after for a fixed
   seed. This is the sharpest available check that X1 and X2 changed nothing behavioural, and
   it is the check that caught the DAgger changes being clean.
3. **`tests/test_env_reuse.py`** green, since X1 makes the suite depend on it.
4. **RSS, open file descriptors and `$TMPDIR` entries** across a long single-ablation run. The
   pool holds instances alive and its failure mode is gradual, so a short run will not show
   it. The 3090 Ti pass saw a flat 2.8 GB plateau over 112 DAgger iterations with clean release
   after each eval; you want the same shape here over 500.
5. **One full ablation end to end**, and its final score against the pre-change tree.
6. **Suite and lint**, plus tests for whatever you changed.

---

## Deliverable

`PERF_EXPERIMENTS_RESULTS_MINIHACK.md`, in the repo's evidence style:

- Box identification, `TMPDIR`, and which checkpoint and main config you ran against.
- The Phase 1 breakdown: where iteration time actually goes, with the collect split broken
  down far enough to see construction separately from stepping.
- The VRAM answer: measured effective batch distribution, measured peak, and whether 4608
  ever binds on this card.
- Per change: what it was worth, measured, or why the profile closed it.
- Phase 4 results, including the noise floor you established and the env-step identity check.
- A run-plan table: minutes per ablation-seed before and after, hours for 25 ablations at
  `num_seeds: 1` and at 3, and peak VRAM with headroom. Derive the iteration cost from
  measurement and say which parts are projections.
- A go / no-go for launching the full suite on this box, with conditions.
- What you did not do and why, including anything you handed to
  `ABLATION_SMOKE_PROMPT_MINIHACK.md`.

Commit only when the suite and lint are green. Ask before committing; do not push.
