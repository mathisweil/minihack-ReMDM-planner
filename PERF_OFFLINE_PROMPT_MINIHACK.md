# Task: make the offline BC arm fast enough to run, on the 4070 Ti

**This is prompt 1 of 2.** `PERF_EXPERIMENTS_PROMPT_MINIHACK.md` follows and covers the
RL fine-tuning ablation suite. Both run on the same box. This one owns every shared file,
so run it first and let prompt 2 branch off its result.

The DAgger arm has already been through this treatment on two boxes. Read
`PERF_GPU_RESULTS_MINIHACK.md` and `PERF_MEASURE_3090_RESULTS_MINIHACK.md` before starting:
they contain measured constants for this exact hardware that you should not re-derive, and
two environment traps that cost hours to find.

## Where you are

`minihack-ReMDM-planner` on the 4070 Ti box (`outback.cs.ucl.ac.uk`, RTX 4070 Ti SUPER,
16 GB, i7-14700K, 20 physical cores / 28 threads). A uv project. Repo conventions are in
`CLAUDE.md` one level up: UK English, no em dashes, evidence for every claim (command and
output, file and line). Everything in this pass happens here; there is no second box.

Confirm the box first and record the output. If this is not the 4070 Ti, stop and say so.

```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run --no-sync python -c "import torch;p=torch.cuda.get_device_properties(0);print(p)"
lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)'
free -g
python -c "import tempfile;print('TMPDIR:',tempfile.gettempdir())"
df -h "$(python -c 'import tempfile;print(tempfile.gettempdir())')" /tmp
```

`free -g` is not decoration. This pass is the first time the full offline dataset has ever
been built, and host RAM is the thing most likely to stop it.

### Two traps, already paid for

1. **`uv sync` and `uv run` must carry `--extra cuda12` on this box.** The default
   resolution is a CUDA 13 torch build and the driver here is 560.35.03. Worse, flipping
   between the two resolutions corrupts the venv, because the `nvidia-*-cu12` and
   `nvidia-*-cu13` wheels share `nvidia/<lib>/` namespace directories. Symptom:
   `ImportError: libcudnn.so.9` while `uv sync` reports nothing to do. Recovery:
   `uv sync --extra cuda12 --reinstall`. Sync once, then use `uv run --no-sync` throughout.
   The `.venv` is shared over NFS with the 3090 Ti box, so a bare `uv sync` breaks that box too.
2. **`$TMPDIR` is on NFS**, with local xfs at `/tmp`. Measured here: `make_env` 86.3 ms/env
   on NFS against 25.6 ms local. Export `TMPDIR` to local disk before every run in this pass
   and say in your report that you did.

## What already landed, and what it means here

Five changes to the shared gradient step (C0 vectorised staircase lookup, C1 int16 over
PCIe, C2 device-side metrics, C3 fused EMA, C4 fused AdamW) are certified on both boxes.
The offline path already inherits all of them: `offline.py:92-98` is C4, `offline.py:254-258`
is C1, `offline.py:219-224` is C2, and `auxiliary_goal_loss` carries C0.

Measured on the 3090 Ti, offline at `offline_batch_size: 2048`: **161.3 → 108.8 ms per
gradient step, -32.6%**, matching DAgger's -31.0% because it is the same step. On this box
DAgger measured 166.5 → 121.9 ms.

So the gradient step is done. What has never been measured is everything around it, and the
reason is simple: **`data/` does not exist and the full dataset has never been built.** The
3090 Ti pass ran offline on a 400-trajectory stand-in. The real recipe is
`collect_episodes_per_env: 5000` (`final_ucl_gpu.yaml:157`), which is 50x larger.

---

## Preflight

```
export TMPDIR=/tmp/remdm-$USER && mkdir -p "$TMPDIR"
git status --short && git log --oneline -8
uv sync --extra cuda12
uv run --no-sync python -m pytest tests -q
uv run --no-sync ruff check src tests
ls -la data/ checkpoints* 2>&1 | head
```

Expect 150 tests passing and 4 pre-existing ruff errors in `tests/test_failure_behaviour.py`
and `tests/test_method_spec.py`, plus 8 in `scripts/profile_dagger.py`. Do not fix those.

---

## Phase 1: build the dataset, and measure what it costs

This phase is the point of the pass. Do it before touching any code.

```
uv run --no-sync python main.py --mode collect --config configs/final_ucl_gpu.yaml
```

Record, with commands and outputs:

| Quantity | How |
|---|---|
| wall clock, and whether `collect_num_workers: 8` saturates the box | time it, watch `uptime` |
| trajectories written, steps per trajectory, per-env counts | the collector logs all three |
| `data/oracle_bc_ucl.pt` size on disk | `ls -l` |
| peak RSS during collection | sample `/proc/<pid>/status` |

Then load it, which is the part at risk:

| Quantity | How |
|---|---|
| `torch.load` wall clock and peak RSS | `run_offline` calls it at `offline.py:709` |
| windows after `load_offline_data` | `len(buffer)`, logged at `offline.py:723` |
| `_ensure_cache` wall clock and peak RSS | first `buffer.sample` call triggers it |
| steady RSS once training starts | `/proc` sampling |

**Three things to check explicitly, because each is a defect if it fires.**

1. **Silent truncation.** `offline_buffer_capacity: 1500000` (`final_ucl_gpu.yaml:145`).
   Scaling the 3090 Ti's measured 400 trajectories / 41,780 steps gives a projection of
   about 2.09M windows for 20,000 trajectories, which is over the cap. `buffer.py:74-75`
   then keeps `self._offline[:capacity]` and drops the rest with no warning. The order it
   truncates in is `as_completed` order (`collect_oracle.py:120`), so the dropped tail is
   whatever finished last, which skews towards longer and harder episodes, and it is not
   reproducible across runs or across a change in `collect_num_workers`. Measure the real
   window count. If it exceeds the cap, **report it and stop short of choosing a fix**: raising
   the cap, sampling uniformly before truncating, and reducing `collect_episodes_per_env` are
   three different experiments, and which one the paper wants is not your call. Say what the
   per-env window counts are before and after truncation so the size of the bias is on record.
2. **Triple storage.** `_slice_trajectory` (`buffer.py:184-195`) emits one window per
   timestep, each holding `loc.copy()` and `glob.copy()`. `_ensure_cache` (`buffer.py:82-107`)
   then builds a second, stacked copy in a Python loop over every window, and `self._offline`
   is never released afterwards. At a projected ~4 KB per window that is two multi-GB copies
   plus the loaded `.pt`. Measure all three peaks. If it fits comfortably, say so and leave
   the code alone.
3. **The cache-build loop.** `buffer.py:103` is a Python-level loop over every window. Time
   it. At the projected scale this is a one-off startup cost, not a per-step cost, so it only
   matters if it is minutes rather than seconds.

Then measure the step itself, at the real config, over at least 200 gradient steps after a
warm-up, with `torch.compile` given its compile step and those steps discarded:

| Quantity | Where from |
|---|---|
| ms per gradient step, samples/s | wrap the loop |
| `buffer.sample` time as a share of the step | time it separately |
| the three H2D copies at `offline.py:256-258` | time them separately |
| peak VRAM allocated and reserved | `torch.cuda.max_memory_allocated/reserved` |
| projected hours for 60,000 steps | derived, and label it a projection |

**Decision gate.** If `buffer.sample` plus the three H2D copies together are under 2% of the
step, **do not build a prefetcher**. Write down the number, say the data path is not worth
touching, and go straight to Phase 3. The step is GPU-bound on this box after C0; assume
nothing about where the remaining time is.

VRAM is not expected to be a problem: 2048 was certified at 6,678 MB allocated on both boxes,
which leaves about 9 GB spare here. Confirm rather than assume, and if it OOMs, stop and
report. **Do not lower `offline_batch_size`**: it is pinned to the compute-matched comparison.

---

## Phase 2: the changes, if Phase 1 justifies them

One commit each, tests and lint green at every commit, so any one can be dropped.

**O1. Prefetch the next batch.** Only if the gate opened. `offline.py:249` samples on the
main thread and 256-258 copy synchronously from unpinned memory, so the gather and the
transfer are both serialised against 60,000 consecutive gradient steps with nothing to hide
behind. This is the case DAgger did not have, because DAgger has collection between training
blocks. Prompt 1 of the DAgger pair deliberately deferred pinned staging buffers for exactly
this reason: it declined to add a `non_blocking=True` that documents a guarantee unpinned
memory does not give. Do it properly here: a single-slot background prefetch of
`buffer.sample`, pinned staging buffers, `non_blocking=True`, one CUDA event to order the
consume against the copy. The sampled indices must come from the same RNG stream in the same
order, so a fixed seed still gives the same batches. Pin that in a test rather than arguing it.

**O2. One `make_eval_model` per eval point.** `offline.py:392` and `offline.py:420` each call
it, and both cadences come from `offline_eval_every_grad_steps: 5000`
(`offline.py:154-164`), so they fire on the same step with identical weights.
`make_eval_model` is `copy.deepcopy(model)` plus an EMA apply (`denoiser.py:428-440`), so
this is a whole redundant model copy per eval point. Hoist it. Twelve evals over the run, so
this is small; take it because it is three lines, not because it is worth much.

**O3. Anything Phase 1 actually found.** The list above is what reading the code suggests.
Phase 1 measures. If the profile puts the time somewhere else, follow the profile, and say in
the report that you did. C0 was found this way on the DAgger pass and was worth more than
every change on the original list combined.

**Not on the list, deliberately.** `_loss_track` at `offline.py:224` allocates one device
float per gradient step, which is 240 KB at 60,000 steps: leave it. The
`speed/train_step_time_sec` window that contains an eval reads high (recorded on the 3090 Ti
pass): it is a reporting quirk in a metric nothing depends on, leave it.

---

## Phase 3: what must not change

Offline BC is the compute-matched baseline for the paper's headline comparison. No speed-up
is worth invalidating it. Pinned:

`offline_total_grad_steps: 60000`, `offline_batch_size: 2048`, `offline_lr`,
`offline_grad_clip`, the cosine schedule and its `eta_min`, `weight_decay`, `aux_loss_weight`,
`ema_decay`, `label_smoothing`, `loss_weight_clip`, `seq_len`, `n_embd`, `n_head`, `n_layer`,
`n_global_tokens`, `num_diffusion_steps`, `diffusion_steps_eval`, `offline_eval_every_grad_steps`,
`offline_checkpoint_every_grad_steps`, `eval_episodes_per_env`, `checkpoint_eval_episodes`.

`60000 x 2048` is the compute budget DAgger is matched against, not a tuning knob. The
comment at `final_ucl_gpu.yaml:117-128` explains why the batch is 2048 and not larger even
though the VRAM would allow it; that reasoning holds harder on a 16 GB card.

`collect_episodes_per_env: 5000` is pinned as a fairness parameter, not a performance one
(`final_ucl_gpu.yaml:152-156`). Raising `collect_num_workers` is fine on its own terms, the
per-task seeds at `collect_oracle.py:100-105` are deterministic, but note that it changes
completion order and therefore changes which windows survive truncation. That interaction is
another reason truncation needs reporting rather than working around.

The model architecture is out of scope.

---

## Phase 4: verify

Bitwise equality is unavailable under AMP. Verify what is verifiable.

1. **Loss trajectory.** 200 gradient steps at a fixed seed from the same buffer, at the
   merge-base and at HEAD. Establish the noise floor first by running identical code twice;
   on the DAgger pass that floor was 2.8e-4 max absolute delta, and every shipped change sat
   inside it. A systematic one-signed offset means a change altered the maths.
2. **Batch identity for O1.** If you built the prefetcher, assert that the sequence of
   sampled index arrays at a fixed seed is identical to the sequential path, for at least
   50 steps. This is the claim the change rests on.
3. **A real run to the first checkpoint.** `offline_checkpoint_every_grad_steps: 10000`, so
   this is 10,000 steps. Compare the loss curve and the first eval JSON against the
   pre-change tree at the same seed. Win rates at `eval_episodes_per_env: 50` over 4 ID envs
   carry real standard error, so this catches gross divergence, not fine structure. Say that.
4. **RSS over that run.** This is the arm where host memory is the risk. Sample `/proc`
   throughout and report start, plateau, and peak.
5. **Suite and lint**, and the new tests for whatever you changed.

---

## Deliverable

`PERF_OFFLINE_RESULTS_MINIHACK.md`, in the repo's evidence style, structured like
`PERF_GPU_RESULTS_MINIHACK.md`:

- Box identification and the `TMPDIR` you used, first, so every number is anchored.
- The Phase 1 dataset table: collection cost, file size, window count, load and cache time,
  and the three RSS peaks. This is the part nobody has ever measured, and it is the most
  valuable thing in the report even if you change no code.
- Whether truncation fires, by how much, and the per-env window counts either side of it.
- The before/after step table, and the projection to 60,000 steps with its assumptions stated.
- Per change: what it was worth, measured, or the sentence explaining why the gate closed it.
- Phase 4 results.
- Hours per seed for `--mode offline` on this box, and how that compares with the 3090 Ti's
  certified 1.94 h. Label it as this box's number, not a revision of the target's.
- What you did not do and why.

Commit only when the suite and lint are green. Ask before committing; do not push.

**Hand-off to prompt 2.** Say explicitly which shared files you touched. `src/buffer.py`,
`src/planners/inference.py`, `src/diffusion/sampling.py` and `src/models/denoiser.py` are all
reachable from the ablation suite, and prompt 2 needs to start from your HEAD rather than
rediscover the same ground.
