# Offline BC results: MiniHack, on the 4070 Ti

Input: `PERF_OFFLINE_PROMPT_MINIHACK.md` (prompt 1 of 2). Successor: `PERF_EXPERIMENTS_PROMPT_MINIHACK.md`.

**Every number here was measured on the box below, with `TMPDIR` on local disk.** Projections
say so in the line itself.

## This prompt ran third, not first

The prompt is prompt 1 of 2, says it owns every shared file, and asks prompt 2 to branch off
its HEAD. It was in fact run **after** both `PERF_EXPERIMENTS_PROMPT_MINIHACK.md` and
`ABLATION_SMOKE_PROMPT_MINIHACK.md`, so the dependency runs the other way. What that changed:

- It started from `273ba47`, not from the state prompt 2 assumed.
- **`src/planners/inference.py` was already taken by prompt 2** (commit `e6b4bbf`, PERF-X2),
  which applied the int16-over-PCIe reorder at `inference.py:187-200` that this pass would
  otherwise have owned. Nothing here conflicts with it.
- The preflight expectation of "150 tests passing" is now 164, because prompt 2 and the smoke
  task added 14 between them.

**Shared files touched by this pass: none.** The only source change is
`src/planners/offline.py`, which the ablation suite does not import. `src/buffer.py`,
`src/planners/inference.py`, `src/diffusion/sampling.py` and `src/models/denoiser.py` are
**unchanged by this pass**, so nothing here invalidates prompt 2's results.

`CLAUDE.md` does not exist in the repo or one level up. Conventions were taken from the
sibling `PERF_*_RESULTS_MINIHACK.md` files.

## The box

```
$ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
name, memory.total [MiB], driver_version
NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB, 560.35.03

$ lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)'
CPU(s):                               28
Model name:                           Intel(R) Core(TM) i7-14700K
Thread(s) per core:                   2
Core(s) per socket:                   20

$ free -g
               total        used        free      shared  buff/cache   available
Mem:              62           4          39           0          18          57
Swap:             31           0          31

$ python -c "import tempfile;print('TMPDIR:',tempfile.gettempdir())"
TMPDIR: /cs/student/project_msc/2025/dsml/mathweil/tmp

$ df -h <TMPDIR> /tmp
evs2:/cs/student/project_msc/2025   22T   19T  3.9T  83% /cs/student/project_msc/2025
/dev/mapper/vg_outback-lv_root     300G  122G  178G  41% /
```

This is the 4070 Ti SUPER, 16 GB, torch 2.13.0+cu126. **62 GB of host RAM**, which is the
number that decides whether this pass is feasible at all.

Both documented traps applied. The venv had already been repaired with
`uv sync --extra cuda12 --reinstall` during prompt 2's pass and every command since has used
`uv run --no-sync`. **`export TMPDIR=/tmp/remdm-$USER` was set for every command in this pass**,
confirmed by `tempfile.gettempdir()` returning `/tmp/remdm-mathweil`.

## Preflight

| Check | Result |
|---|---|
| `git status --short` | clean at `273ba47` (plus untracked `logs/` from the smoke task) |
| `pytest tests -q` | 164 passed |
| `ruff check src tests` | 4 errors, all pre-existing |
| `ls data/` | **did not exist** |
| `ls checkpoints*` | did not exist |

---

## Phase 1: the dataset, built for the first time

### Collection

```
$ time uv run --no-sync python main.py --mode collect --config configs/final_ucl_gpu.yaml
Collection complete in 36.3s
  Trajectories: 20000 (0 failures)
  Total steps: 1912931
  MiniHack-Room-Random-5x5-v0: 5000 eps, 16751 steps, avg 3.4 steps/ep
  MiniHack-Room-Random-15x15-v0: 5000 eps, 50600 steps, avg 10.1 steps/ep
  MiniHack-Corridor-R2-v0: 5000 eps, 1206145 steps, avg 241.2 steps/ep
  MiniHack-MazeWalk-9x9-v0: 5000 eps, 639435 steps, avg 127.9 steps/ep
Saved 20000 trajectories to data/oracle_bc_ucl.pt (6371.7 MB)

real 1m45.515s   user 4m47.775s   sys 0m17.781s
```

| Quantity | Measured |
|---|---|
| collection wall | **36.3 s** at 550.9 episodes/s |
| serialisation wall | **67 s** (`torch.save` of 6.4 GiB, single-threaded, onto NFS) |
| total wall | 1 m 45.5 s |
| trajectories | 20,000, **0 failures** |
| total steps (= windows) | **1,912,931** |
| file size | 6,681,186,345 bytes (6,371.7 MiB) |
| collector peak RSS | **33.30 GB** |

**`collect_num_workers: 8` does not saturate the box.** Eight workers on 20 physical cores.
The arithmetic confirms they were individually busy: 36.3 s of collection at 8 workers is
~290 CPU-seconds, against a measured `user` of 287.8 s for the whole command, so the workers
were saturated and the remaining 67 s of serialisation was single-threaded. Raising the worker
count would cut the 36 s but not the 67 s, and it would change `as_completed` order and
therefore which windows survive truncation, which is the interaction the next section is about.

The collector peak RSS of 33.30 GB is the parent process only (the 8 workers are its children
and were sampled separately at negligible size). The peak is reached during `torch.save`, which
serialises the whole 6.4 GiB trajectory list; the plateau before it is 20.22 GB.

### Load

```
torch.load                68.00 s   RSS  13.61 GB   HWM  20.15 GB
trajectories           20,000
load_offline_data          2.77 s   RSS  16.26 GB   HWM  20.15 GB
len(buffer)            1,500,000
after releasing .pt              RSS  16.26 GB   (freed 0.00 GB)
_ensure_cache (1st sample)   2.06 s   RSS  22.12 GB   HWM  22.12 GB
buffer.sample (warm)      0.772 ms per call
steady RSS             22.12 GB
peak RSS (HWM)         22.12 GB
```

| Stage | Wall | RSS after |
|---|---|---|
| `torch.load` (`offline.py:709`) | **68.00 s** | 13.61 GB, peak 20.15 GB during the read |
| `load_offline_data` slice into windows | 2.77 s | 16.26 GB |
| `_ensure_cache` (first `buffer.sample`) | **2.06 s** | 22.12 GB |
| steady state | | **22.12 GB of 62 GB** |

Startup is 73 s against a projected 2 h of training, **1.0%**, so none of it is worth touching.

### The three checks

**1. Silent truncation: it fires, and the bias is not spread evenly.**

`offline_buffer_capacity: 1500000` against 1,912,931 windows. `buffer.py:73-75` keeps
`self._offline[:capacity]` and drops the rest with no warning.

| env | windows before | windows after | kept | share before | share after |
|---|---|---|---|---|---|
| `MiniHack-Room-Random-5x5-v0` | 16,751 | 16,751 | 100.0% | 0.9% | 1.1% |
| `MiniHack-Room-Random-15x15-v0` | 50,600 | 50,600 | 100.0% | 2.6% | 3.4% |
| `MiniHack-Corridor-R2-v0` | 1,206,145 | 1,206,145 | 100.0% | 63.1% | 80.4% |
| **`MiniHack-MazeWalk-9x9-v0`** | **639,435** | **226,504** | **35.4%** | 33.4% | **15.1%** |
| total | 1,912,931 | 1,500,000 | 78.4% | | |

**412,931 windows are dropped, 21.6% of the dataset, and 100% of the loss falls on one
environment.** MazeWalk-9x9 keeps just over a third of its data; its share of the training set
drops from 33.4% to 15.1% while Corridor-R2's rises to 80.4%. The offline baseline would train
on a set that under-weights one of the four ID environments by more than half, then be
evaluated on all four equally.

The mechanism is submission order: tasks are built env by env (`collect_oracle.py:100-105`), and
for tasks this short `as_completed` order tracks submission order closely, so the tail that
falls off the end is entirely the last env in `id_envs`. That also means the identity of the
victim env is an artefact of list order, and the exact cut inside MazeWalk moves with
`collect_num_workers`.

**Reported, not fixed.** Raising the cap, sampling uniformly before truncating, and reducing
`collect_episodes_per_env` are three different experiments and the choice is the author's. The
per-env counts above are on record so the size of the bias is known either way.

**2. Triple storage: real, and it fits.** Measured peaks: 20.15 GB during `torch.load`,
16.26 GB after slicing, 22.12 GB with the stacked cache built. Against 62 GB total and 57 GB
available, that is comfortable, so **the code is left alone** as the prompt directs.

One observation for the record: `del data; gc.collect()` freed **0.00 GB** of RSS. Two reasons.
`_slice_trajectory` keeps `a = actions_arr[start:end]`, a numpy *view*, so every trajectory's
actions array stays reachable from the buffer; and glibc does not return the freed
`local`/`global` arenas to the OS. Neither matters at 62 GB, but it means the loaded `.pt`
cannot be assumed to be reclaimed.

**3. The cache-build loop: 2.06 s.** `buffer.py:103` is a Python-level loop over all 1.5M
windows. Two seconds, once, at startup. Not worth touching.

---

## The gradient step, and the decision gate

`baseline` measurement drives the real `run_offline` at `configs/final_ucl_gpu.yaml`
unmodified except for the step count. The loop at `offline.py:249` begins every step with
`buffer.sample`, so the interval between consecutive sample calls is exactly one gradient step.
250 steps, first 50 discarded to cover `torch.compile` and warm-up.

| Quantity | Measured |
|---|---|
| step, mean | **121.34 ms** (median 121.81, min 35.58, max 209.09) |
| throughput | **16,878 samples/s** at batch 2048 |
| `buffer.sample` | 0.963 ms, **0.79%** of the step |
| the three H2D copies (`offline.py:256-258`) | 0.438 ms, **0.36%** of the step |
| **gate: sample + H2D** | **1.401 ms, 1.15% of the step** |
| peak VRAM allocated | **6,351 MiB** |
| peak VRAM reserved | 6,406 MiB |
| projected 60,000 steps | **2.02 h** (projection) |

Per-copy, at batch 2048:

| Copy | Time | Bytes crossing PCIe |
|---|---|---|
| `local` | 0.029 ms | 0.33 MB int16 |
| `global` | 0.324 ms | 6.80 MB int16 |
| `actions` | 0.068 ms | 1.05 MB int64 |

121.34 ms against DAgger's certified 121.9 ms on this box: the same step, as expected, since
both inherit C0 to C4.

**Peak VRAM has to be sampled inside the loop.** `offline.py:375` calls
`reset_gpu_memory_stats()` every `offline_log_every` steps, so reading
`torch.cuda.max_memory_allocated()` after the run returns the peak since the last reset (170 MiB)
rather than the run's peak. The 6,351 MiB above is the maximum over per-step samples. Against
15,974 MiB that leaves 9.6 GB spare, and no run in this pass came near an OOM.

**The gate is closed at 1.15%, under the 2% bar. The data path is not worth touching, so O1
was not built.** The step is GPU-bound: 98.85% of it is the forward, backward and optimiser
work that C0 to C4 already certified.

---

## Phase 2: per change

### O1. Prefetch the next batch: not built

Closed by the gate above. `buffer.sample` plus the three H2D copies are 1.401 ms of a 121.34 ms
step. A perfect prefetcher, hiding all of it, would save 1.15%, and it would add a background
thread, pinned staging buffers, a CUDA event and a new class of ordering bug to the arm that is
the paper's compute-matched baseline. Not worth it.

### O2. One `make_eval_model` per eval point: taken

`offline.py:392` and `offline.py:420` each called `ema_model.make_eval_model(_ema_source)`, and
both cadences derive from `offline_eval_every_grad_steps` (`offline.py:157-161`), so they fire
on the same step against the same source and each built its own `copy.deepcopy(model)` plus EMA
apply. The eval model is now built at most once per step and shared, kept lazy so the two blocks
stay independent if the cadences ever diverge.

Worth very little, as the prompt says: twelve eval points over a 60,000-step run, and a deepcopy
of a 5.24 M-parameter model is ~3 ms, so this is about 36 ms of a 2 h run. It is in because it
is three lines and provably neutral.

`test_offline_builds_one_eval_model_per_eval_point` drives the real `train_offline` with a stub
evaluator and counts the copies. On the pre-change code it fails with `assert 6 == 3`, one
redundant copy per eval point; with the change it passes, and it additionally asserts that the
ID and OOD blocks are handed the same model object.

### O3. Anything the profile found: nothing

The profile put 98.85% of the step inside the certified gradient step and 1.0% of the run in
startup. There is no third thing on this path. Recorded as a null result rather than padded.

---

## Phase 4: verification

### 1. Loss trajectory and the noise floor

200 gradient steps, seed 0, same buffer, evals disabled. Identical code run twice on each tree
to establish the floor first.

| Comparison | max abs delta | mean abs delta | sign split |
|---|---|---|---|
| **noise floor**, pre vs pre | 3.263e-02 | 3.232e-03 | 82/200 positive |
| **noise floor**, post vs post | 4.038e-02 | 3.847e-03 | 166/200 positive |
| change, pre vs post | 4.977e-02 | 5.127e-03 | 43/200 positive |
| change, pre vs post (second pair) | 1.448e-02 | 2.140e-03 | 143/200 positive |

The pre-versus-post differences sit inside the identical-code band, and the sign split moves
between 43/200 and 143/200 across the two pairings, which is the opposite of the systematic
one-signed offset that would indicate altered maths.

The floor here is two orders of magnitude wider than the DAgger pass's 2.8e-4, and the reason
is worth stating rather than hiding:

| step | pre_a | pre_b | post_a | post_b |
|---|---|---|---|---|
| 0 | 2.5767419 | 2.5767419 | 2.5767419 | 2.5767419 |
| 1 | 2.6258802 | 2.6258802 | 2.6258802 | 2.6258802 |
| 2 | 1.8163356 | 1.8163358 | 1.8163354 | 1.8163353 |
| 199 | 0.0332420 | 0.0332112 | 0.0330636 | 0.0330004 |

**Steps 0 and 1 are bitwise identical across all four runs**, so the RNG stream, the sampled
batches and the `t` draws are exactly reproducible and the data path is deterministic.
Divergence begins at step 2 at 1.6e-5 and amplifies through training: max abs delta is 1.6e-5
over the first 10 steps and 1.2e-3 over the last 100. The large headline figures come from the
middle of the run, where the loss is falling steeply from 2.6 to 0.03 and small weight
differences move it visibly. The floor is wide because the run is short and steep, not because
the pipeline is loose.

### 2. Batch identity for O1

Not applicable. No prefetcher was built, so there is no claim to pin.

### 3. A real run to the first checkpoint

10,000 gradient steps, seed 0, `configs/final_ucl_gpu.yaml` unmodified, on each tree. This
crosses two eval points (5,000 and 10,000) and writes the first step checkpoint, so it is the
first run in which O2 can fire at all.

| | Pre-change | Post-change |
|---|---|---|
| final loss at step 10,000 | 0.002504332 | 0.002938163 |
| eval at step 10,000, ID mean | 0.7000 | 0.7150 |
| eval at step 10,000, OOD mean | 0.0733 | 0.0733 |
| Room-Random-5x5 | 1.00 | 1.00 |
| Room-Random-15x15 | 1.00 | 1.00 |
| Corridor-R2 | 0.54 | 0.60 |
| MazeWalk-9x9 | 0.26 | 0.26 |

Loss over all 10,000 steps: mean abs delta **1.011e-03**, and the sign split is
**5,161 of 10,000 positive**, which is as close to a coin flip as this test can produce. By
segment:

| steps | max abs delta | mean abs delta | mean loss, pre | mean loss, post |
|---|---|---|---|---|
| 0 to 1,000 | 1.157e-01 | 2.392e-03 | 0.07168 | 0.07363 |
| 4,000 to 5,000 | 1.200e-02 | 2.614e-04 | 0.00357 | 0.00359 |
| 9,000 to 10,000 | 9.078e-03 | 8.280e-04 | 0.00267 | 0.00266 |

The ID means differ by 0.015, which is three Corridor-R2 episodes out of 50. At
`eval_episodes_per_env: 50` the standard error on a win rate near 0.57 is about 0.070, so this
comparison catches gross divergence and nothing finer. It found none, and the OOD means are
identical.

**The wall clocks of these two runs are not comparable and are not quoted as a result.** The
pre-change run overlapped for roughly three minutes with a duplicate process that had not
exited when expected, so both were sharing the GPU. The step-level timing above, measured with
the box otherwise idle, is the figure to use.

### 4. RSS over that run

Sampled every 5 s inside the training process across both 10,000-step runs.

| | Pre-change | Post-change |
|---|---|---|
| start | 0.48 GB | 0.48 GB |
| plateau | 23.15 GB | 23.44 GB |
| peak | 23.60 GB | 23.59 GB |

Flat after startup, and the same on both trees. Against 62 GB total the arm has roughly 2.6x
headroom, so host memory is not the constraint the prompt feared it might be, **at this
dataset size**. It would become one if the truncation cap were raised much beyond 1.5M windows:
the cache alone is about 3.4 KB per window.

### 5. Suite and lint

| | Before | After |
|---|---|---|
| `pytest tests -q` | 164 passed | **165 passed** (1 new) |
| `ruff check src tests` | 4 pre-existing errors | the same 4, unchanged |

---

## Hours per seed

| | |
|---|---|
| build the dataset, once | **1 m 45 s** (36 s collect, 67 s save) |
| load and cache, per run | **73 s** |
| 60,000 gradient steps at 121.34 ms | **2.02 h** (projection from 200 timed steps) |
| 12 evals plus 6 checkpoints | a few minutes, not separately isolated |
| **total, `--mode offline`, per seed** | **about 2.1 h on this box** |

The 3090 Ti's certified figure is 1.94 h. **This is this box's number, not a revision of the
target's**: 2.02 h against 1.94 h for the same 60,000 steps is the expected small gap between a
4070 Ti SUPER and a 3090 Ti on an identical GPU-bound step, and it says nothing about what the
3090 Ti will do.

The 73 s load is per run and the 1 m 45 s build is once for all seeds.

## What was not done

- **O1, the prefetcher.** Closed by the measured gate at 1.15%. The reasoning is above.
- **Any fix for the truncation.** Reported with per-env counts, as instructed. Three candidate
  fixes are three different experiments.
- **Any change to the triple storage.** Measured at 22.12 GB steady against 62 GB and left
  alone, as the prompt directs when it fits.
- **`_loss_track` and the `speed/train_step_time_sec` eval-window quirk.** Both excluded by the
  prompt; neither was touched.
- **Raising `collect_num_workers`.** It would cut the 36 s collection phase but not the 67 s
  single-threaded save, and it changes `as_completed` order and therefore which windows survive
  truncation. Not a change to make while truncation is unresolved.

## Hand-off

**Shared files touched: none.** The only source change is `src/planners/offline.py`, which the
ablation suite does not import.

Because this prompt ran third rather than first, the hand-off is inverted: prompt 2 has already
landed `PERF-X1` to `PERF-X7` and the smoke task has landed one fix, all on `src/` and
`experiments/`. `src/planners/inference.py` carries prompt 2's `PERF-X2` change. Nothing in this
pass conflicts with any of it, and `src/buffer.py`, `src/diffusion/sampling.py` and
`src/models/denoiser.py` remain untouched by all three passes.

One item for whoever runs the suite next: **`data/oracle_bc_ucl.pt` now exists**, 6.68 GB, and
is gitignored. It cost 1 m 45 s to build and is deterministic given `seed: null` resolving to 0
at `collect_oracle.py:88`, so it can be rebuilt rather than copied.
