# Target-box certification: MiniHack DAgger speed-up (prompt 2 of 2)

Input: `PERF_MEASURE_3090_PROMPT_MINIHACK.md`. Predecessor: `PERF_GPU_PROMPT_MINIHACK.md`
and its report `PERF_GPU_RESULTS_MINIHACK.md`, measured on the 4070 Ti dev box.

Every figure here was measured on the target box. Prompt 1's figures are quoted only for
comparison and are labelled as the dev box's.

## The box

```
$ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
name, memory.total [MiB], driver_version
NVIDIA GeForce RTX 3090 Ti, 24564 MiB, 590.44.01

$ uv run python -c "import torch;p=torch.cuda.get_device_properties(0);print(p)"
_CudaDeviceProperties(name='NVIDIA GeForce RTX 3090 Ti', major=8, minor=6,
  total_memory=24114MB, multi_processor_count=84, L2_cache_size=6MB)

$ lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)'
CPU(s):                  24
Model name:              12th Gen Intel(R) Core(TM) i9-12900K
Thread(s) per core:      2
Core(s) per socket:      16

$ hostname
wigeon-l.cs.ucl.ac.uk
$ python -c "import tempfile;print('TMPDIR:',tempfile.gettempdir())"
TMPDIR: /cs/student/project_msc/2025/dsml/mathweil/tmp     # NFS, see below
```

RTX 3090 Ti, 24 GB, i9-12900K, 16 physical cores / 24 threads. This is the target box the
prompt describes. Prompt 1 ran on `outback.cs.ucl.ac.uk`; this is `wigeon-l`.

`os.cpu_count()` is **24** here against 28 on the dev box, so `collect_batch_gpu` sizes its
oracle thread pool at `min(30, 24) = 24` rather than 28 (`collect.py:455`). Immaterial at 30
episodes, recorded because the prompt asks.

### Three conditions that qualify every number below

**1. This box is shared, and was busy throughout.** Two other users held GPU memory and CPU
for the whole session:

```
$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
347502, 6858 MiB      # another user, present for 3h+
403677, 470 MiB       # another user
386426, 316 MiB       # another user
$ uptime
 20:07:54 up 20:07,  3 users,  load average: 4.52, 4.63, 4.30
```

That is 7,644 MiB of foreign GPU memory and a load average of 4.5 before my own work. It
contaminated the first measurements badly: a single-shot run at HEAD gave 251.5 ms per
gradient step, and the same code measured 110.1 ms twenty minutes later. **Every timing
below therefore comes from interleaved A/B rounds with the minimum taken per arm**, never
from a single run: contention only ever adds time, so the minimum is the best available
estimate of the uncontended cost, and interleaving stops drift in someone else's job from
being read as a difference in mine. Within-arm spread after this treatment is under 0.6 ms.

**2. `$TMPDIR` is on NFS, and it breaks the collection gate.** See Phase 1.

**3. The venv is shared between the two boxes over NFS.** The repo and `.venv` live on the
same `evs2:` mount visible from both hosts, and the driver here (590.44.01) supports CUDA 13
while the dev box's (560.35.03) does not. A bare `uv sync` would resolve to the CUDA 13
build and break the other box. The existing `cuda12` resolution (torch 2.13.0+cu126) runs on
both, so it was kept and every command used `uv run --no-sync`. That also means both boxes
ran the identical torch build, which is what makes the two passes comparable at all.

## Preflight

| Check | Result |
|---|---|
| `git status --short` | clean at `9af96ba` |
| `git log --oneline -12` | prompt 1's six PERF commits + doc on top of `6d1e37d` |
| `uv run python -m pytest tests` | **157 passed, 2 deselected** in 271.98s (150 after the C6 revert) |
| `uv run python -m pytest tests/test_env_reuse.py -v` | **6 passed** in 5.11s |
| `uv run ruff check src tests` | 4 errors, the documented pre-existing ones |
| `uv run ruff check scripts/profile_dagger.py` | 8 errors, as documented |
| `ls checkpoints/` | **does not exist** — see Phase 4 |

`test_env_reuse.py` is green on this box, so env reuse is safe in this NLE build and the
collection change stands. The suite takes 272s here against 30s on the dev box; that is the
CPU-bound MiniHack env tests on the slower cores plus NFS, not a code difference.

## `scripts/profile_dagger.py`

The prompt's named command runs and `--config` takes effect — the echoed header shows
`Batch size: 2048`, `AMP: True`, `Episodes/iteration: 30`, so it is profiling the real
config and not `defaults.yaml`. Two things about it:

**It crashed** at the memory audit before this pass, on every invocation at the real config:

```
File "scripts/profile_dagger.py", line 643, in run_profiling
    loss.backward()
RuntimeError: Error: accessing gradient tensor output of CUDAGraphs that has been
overwritten by a subsequent run.
```

The audit reuses `.grad` tensors that inductor allocated inside a CUDA graph capture during
the profiled loop. The audit does not accumulate gradients, so `model.zero_grad(set_to_none=True)`
before it is the fix, and is applied here. The script now completes.

**Its absolute numbers are not the run's.** Every component is wrapped in a timer that calls
`torch.cuda.synchronize()` on entry and exit, which serialises the pipeline it is measuring,
and its rollout is a sequential B=1 reconstruction rather than the real
`collect_batch_gpu`. At the real config it reports 29.2 s of model rollout per iteration
against 3.7 s measured on the real path, and 339 ms per gradient step against 110 ms. Its
memory audit also caps the batch at `min(dagger_batch_size, 1024)`, so it cannot answer the
VRAM question either. It is useful for the *relative* breakdown inside a step; the numbers
below come from driving the real `DataCollector.collect_batch_gpu` and `Trainer._train_step`.

## Phase 1: before and after, on this box

`configs/final_ucl_gpu.yaml` unmodified, `TMPDIR` on local disk, mean of warm iterations
(iteration 0 excluded: it pays `torch.compile` and a cold env pool).

| Quantity | merge-base `6d1e37d` | HEAD `9af96ba` | Change |
|---|---|---|---|
| collect wall time | 3.54 s | 3.73 s | flat (identical code) |
| &nbsp;&nbsp;env reset | 0.12 s | 0.16 s | |
| &nbsp;&nbsp;env step | 0.57 s | 0.57 s | |
| &nbsp;&nbsp;GPU inference | 0.22 s | 0.22 s | |
| &nbsp;&nbsp;oracle rollout | 2.62 s | 2.78 s | |
| train wall time (100 steps) | 16.22 s | 11.20 s | **-31.0%** |
| ms per grad step | 162.16 | 111.95 | **-31.0%** |
| samples/s | 12,629 | 18,293 | **+44.9%** |
| iteration wall time | 19.76 s | 14.93 s | **-24.5%** |
| collect / train split | 17.9% / 82.1% | 25.0% / 75.0% | |
| env steps per iteration | 10,800.3 | 10,800.3 | identical |
| peak VRAM allocated | 6,675.3 MB | 6,677.6 MB | flat |
| peak VRAM reserved | 7,006.6 MB | 7,065.3 MB | flat |

The dev box measured -26.5% on the gradient step; this box gets **-31.0%**. The changes are
worth more here, and every individual one is (Phase 3).

Collection is unchanged code and the identical env-step count confirms it: the small
difference is in `oracle_rollout_sec`, 30 threaded BFS rollouts on a shared box.

### Decision gate: failed on NFS, passes on local disk

On `$TMPDIR` as configured, collection is **6.64 s per iteration — over the 5 s gate**. The
prompt lists three candidate causes; it is the first of them. Measured directly:

| | NFS `$TMPDIR` | local `/tmp` |
|---|---|---|
| `make_env` cold, pool bypassed | 115.7 ms/env | **17.2 ms/env** |
| first pooled acquire | 161.7 ms | 35.4 ms |
| pooled recycle | 0.00 ms | 0.00 ms |
| `env.reset` on a recycled env | 8.44 ms | **1.24 ms** |

6.7x on construction and 6.8x on reset, and reset is paid per episode and is *not* amortised
by the pool. The dev box saw only 3.4x, so this box is hurt roughly twice as badly.

Same benchmark, same seeds, only `TMPDIR` changed:

| | NFS | local xfs |
|---|---|---|
| collect per iteration | 6.64 s | **3.73 s** |
| &nbsp;&nbsp;env reset | 0.97 s | 0.16 s |
| iteration wall time | 17.88 s | **14.93 s** |

**Setting `TMPDIR` to local disk is a requirement for the real runs, not an optimisation.**
It is a one-line environment change worth 2.9 s per iteration, and it is what moves
collection back under the gate. The env pool itself is working: `env_reset_sec` for 30 envs
is 7.4 s on a cold pool and 0.03-0.5 s once warm.

## Phase 2: VRAM certification

At `dagger_batch_size: 2048`, AMP on, `torch.compile` on, over 200 gradient steps:

| | Allocated | Reserved |
|---|---|---|
| training step, batch 2048 | **6,677.6 MB** | **7,065.3 MB** |
| checkpoint eval, `eval_episodes_per_env: 50` | 122.8 MB | 163.6 MB |
| torch-visible total | 24,114 MB | |
| headroom on an idle card | **17,436 MB (17.4 GB)** | |

`final_ucl_gpu.yaml:18-19` claimed "~6-8 GB peak" with "comfortable headroom". **The comment
is correct.** 6.68 GB allocated, 7.07 GB reserved, 17.4 GB spare. It has been sharpened with
the measured figures rather than corrected.

The prompt expected the checkpoint eval to be the high-water mark. **It is not, by 54x.**
Eval runs under `@torch.no_grad()` with B <= 50, so it peaks at 123 MB; training at batch
2048 dominates. Eval is a *host*-memory and file-descriptor event, not a VRAM one — see
Phase 4.

Because the box is shared, the practical headroom is smaller than the idle figure: with the
7,644 MiB other users held during this session, my 7.07 GB reserved brought the card to
about 14.6 GB of 24.5 GB. Still comfortable, but the 17.4 GB is not all reliably available.

### `num_collection_workers` is dead on the CUDA path

Confirmed: `online.py:147` computes `use_gpu_batch = str(self.device).startswith("cuda") and
n_eps > 1`, which is always true here (`episodes_per_iteration: 30`), so `collect_batch_parallel`
is unreachable and `num_collection_workers` feeds nothing. `DataCollector.__init__`
(`collect.py:258-265`) still builds one CPU deep copy of the model per worker whenever the
value is non-zero. The comment in `final_ucl_gpu.yaml` justified the value by this box's core
count; it now records that the knob has no effect on GPU and what the non-zero value costs.
The knob was not wired up: that would change collection behaviour, which is out of scope for
a certification pass.

## Phase 3: the `carry` set, settled

Round-robin over three rounds, all seven variants sampled in every round, 200 gradient steps
each after a 20-step warm-up, minimum per variant. Each `no_cX` row is HEAD with that one
change reverted, so its delta is what the change is *worth*.

| Variant | runs (ms/step) | min | change is worth | dev box said | Final |
|---|---|---|---|---|---|
| HEAD | 110.08, 110.27, 110.63 | 110.08 | — | — | — |
| no_C0 | 155.33, 155.46, 155.67 | 155.33 | **+41.1%** | +33.9% | **keep** |
| no_C1 | 113.44, 113.94, 114.21 | 113.44 | **+3.1%** | +0.9% | **keep** |
| no_C2 | 111.89, 111.68, 111.95 | 111.68 | **+1.5%** | +0.25% | **keep** |
| no_C3 | 110.86, 110.91, 110.85 | 110.85 | **+0.7%** | +0.33% | **keep** |
| no_C4 | 112.89, 112.85, 112.90 | 112.85 | **+2.5%** | +0.50% | **keep** |
| bf16 (C6 enabled) | 109.40, 109.69, 109.77 | 109.40 | -0.6% | -1.8% | **revert** |

**Every change in the `carry` set is worth more on the target than on the dev box, and all
four are now `keep`.** That is the finding that justifies having run both passes:

- **C3 and C4** are kernel-launch-overhead fixes, and the prompt predicted they would
  understate on the dev box's faster cores. They did: C4 is 5x more valuable here (2.5% vs
  0.50%), C3 2x (0.7% vs 0.33%). C2, also a sync fix, is 6x more valuable (1.5% vs 0.25%).
- **C1** is the bandwidth-sensitive one, and it is 3.4x more valuable here (3.1% vs 0.9%)
  despite this card's wider bus — the transfer is over PCIe, not on-card, and the host-side
  int64 cast this removes is CPU work on the slower CPU.
- **C6 (bf16)** moves the other way, as the prompt predicted for the opposite reason: 0.6%
  here against 1.8% on the dev box. Ampere GeForce halves FP16-with-FP32-accumulate tensor
  throughput relative to Ada, so there is less for bf16 to win. Combined with the trajectory
  evidence below, **C6 has been reverted** (`af1db9a`): it shipped as an off-by-default flag
  so this pass could re-measure bf16, that measurement is now done and negative on both
  boxes, and by the prompt's own rule a change that buys nothing on either box is code with
  no purchase. `use_amp: true` continues to mean fp16 with GradScaler, as every released
  checkpoint was trained. The suite drops from 157 to 150 tests with C6's seven gone.

Where the two boxes disagreed, in one line: **nothing reversed sign.** Every change that
helped on the dev box helps more here; the only change whose value shrank is the one that is
off by default and rejected on other grounds.

### C5: `torch.compile` is engaging

Checked, not assumed. `/usr/bin/gcc` and `/usr/bin/cc` are present, `_has_c_compiler()` is
`True`, `try_compile` returns an `OptimizedModule`, the log line is
`INFO src.models.denoiser: Compiling model with torch.compile`, and **zero of the 21 Phase 3
logs contain the "falling back to eager mode" warning**. Unlike the dev box, this card does
not emit inductor's `Not enough SMs to use max_autotune_gemm` message — it has 84 SMs against
66 — so the two boxes may not be running identical kernel selections. That is a difference in
what inductor chose, not in what the code does.

## Phase 4: validation against the recipe

### 1. Loss trajectories

200 gradient steps, fixed seed, same deterministic buffer. The right control is two runs of
*identical* code, since AMP and kernel selection are non-deterministic:

| Comparison vs HEAD | signed mean delta | mean abs delta |
|---|---|---|
| HEAD vs HEAD (control) | -1.05e-04 | 1.24e-04 |
| HEAD vs HEAD, 3rd run (control) | -1.63e-04 | 1.77e-04 |
| merge-base | -1.46e-04 | 1.55e-04 |
| no_C0 | +9.81e-06 | 1.96e-05 |
| no_C1 | -2.55e-04 | 2.68e-04 |
| no_C2 | -2.59e-04 | 2.78e-04 |
| no_C3 | -2.08e-04 | 2.38e-04 |
| no_C4 | -1.16e-04 | 1.27e-04 |
| **bf16** | **-6.24e-03** | **6.24e-03** |

The prompt's criterion is a *systematic* offset. The control band is -1.05e-04 to -1.63e-04:
identical code, run twice, differs by that much. merge-base and every reverted-change variant
sit inside or within 1.6x of that band. **bf16 sits 38 to 60x outside it**, and is the only
comparison where the offset is one-signed and large. Mean loss over 200 steps: 0.54931 for
bf16 against 0.55529-0.55561 for everything else.

C1 must be exact, and is: the widening is int16 to int64, lossless for glyph IDs, pinned by
`tests/test_gpu_step_perf.py`. C0 is bitwise-verified against the loop it replaced, and its
trajectory delta here (1.96e-05) is the smallest in the table — an order of magnitude inside
the control band.

### 2. A real run to the first checkpoint

The prompt asks to compare against `checkpoints/`. **That directory does not exist in this
checkout**, and the only reference run on disk
(`../checkpoints_ucl_bigger_model/dagger_20260403_024043_7c94/`, with evals at iters 250 /
500 / 750) was trained with a **different model**: `n_embd: 384`, `n_head: 6`,
`dagger_batch_size: 4608` against this config's 256 / 4 / 2048. Its win rates are not
comparable to a `final_ucl_gpu.yaml` run, so the literal comparison the prompt asks for is
unavailable. What was done instead:

A real run at the unmodified config, `--seed 0`, `total_timesteps` capped at 1,000,000 to
reach the first checkpoint (`checkpoint_every_timesteps: 940000`), `TMPDIR` on local disk.
It completed 112 iterations in 31.5 minutes, fired one periodic ID/OOD eval and two
checkpoint evals, and wrote both checkpoints and their eval JSONs.

| iter | env steps | iter s | collect s | train s | samples/s | oracle s | model s |
|---|---|---|---|---|---|---|---|
| 0 | 10,731 | 26.88 | 5.43 | 21.46 | 9,545 | 2.83 | 2.59 |
| 10 | 126,498 | 17.33 | 6.36 | 10.96 | 18,679 | 5.33 | 1.01 |
| 20 | 235,162 | 12.76 | 1.75 | 11.01 | 18,609 | 0.83 | 0.92 |
| 40 | 445,071 | 12.19 | 1.17 | 11.02 | 18,593 | 0.46 | 0.70 |
| 60 | 569,470 | 12.35 | 1.34 | 11.01 | 18,609 | 0.42 | 0.92 |
| 80 | 724,513 | 17.67 | 6.76 | 10.91 | 18,765 | 5.70 | 1.05 |
| 100 | 914,502 | 14.63 | 3.60 | 11.03 | 18,572 | 2.80 | 0.79 |
| 110 | 995,923 | 14.32 | 3.32 | 11.00 | 18,621 | 2.53 | 0.78 |

**Warm mean: 14.30 s per iteration — collect 3.30 s, train 10.99 s, 18,628 samples/s.**

The training half is flat to within 0.13 s across the whole run (10.91-11.04 s, i.e.
109-110 ms per gradient step), and matches the isolated benchmark exactly. All the variance
is in collection, and all of *that* is the oracle BFS: `oracle_rollout_sec` swings 0.42 to
5.70 s while `model_rollout_sec` stays at 0.70-1.09 s and `env_reset_sec` at 0.03 s.

Eval results, from the checkpoint JSONs:

| | iter 103 (941,772 steps) | iter 112 (1,005,510 steps) |
|---|---|---|
| ID mean win rate | 0.465 | 0.520 |
| &nbsp;&nbsp;Room-Random-5x5 | 0.96 | 0.96 |
| &nbsp;&nbsp;Room-Random-15x15 | 0.32 | 0.50 |
| &nbsp;&nbsp;Corridor-R2 | 0.42 | 0.40 |
| &nbsp;&nbsp;MazeWalk-9x9 | 0.16 | 0.22 |
| OOD mean win rate | 0.047 | 0.040 |

The recipe is learning as expected this early: the easiest ID env is nearly solved, the
harder three are climbing, OOD is still near zero at 1M of 5.65M env steps. Nothing here
suggests the changes altered behaviour, and the loss-trajectory table above is the sharper
test of that.

**An observability defect found on the way.** The periodic ID/OOD eval fired (the
iterations 40-50 window is 36 s longer than its neighbours, and 50 episodes x 7 envs is
about that) but produced **no output at all**: `Logger.log` only prints when
`step % 10 == 0` (`logging.py:161`), and the eval landed on an iteration that was not a
multiple of 10. With `use_wandb: false` those eval results are computed and then discarded.
The real runs have `use_wandb: true` so the numbers reach W&B, and checkpoint evals are
written to JSON regardless, but a wandb-less run silently loses its periodic evals.

### 3. The offline BC path

`offline.py` shares the batch-prep code C1 and C2 touch. No dataset existed
(`data/` is absent), so one was collected: `--mode collect` with
`collect_episodes_per_env=100` produced 400 trajectories and 41,780 steps in 1.3 s. Offline
training then ran at the real `offline_batch_size: 2048` through the first eval:

| | merge-base | HEAD |
|---|---|---|
| ms per grad step | 161.3 | **108.8** |
| samples/s | 12,700 | **18,821** |
| loss at the logged steps | 0.0212 → 0.0192 | 0.0214 → 0.0132 |

**-32.6%**, matching the DAgger figure of -31.0%, which is the expected result since it is
the same step. The first eval fired correctly and produced sane win rates (ID mean 0.125,
OOD mean 0.067 after 500 steps on a 400-trajectory dataset). One logging artefact noticed:
`speed/train_step_time_sec` includes the eval when a log window contains one, so that single
window reads 13.4 s against 5.4 s either side. It is a reporting quirk in a metric nothing
depends on; not fixed.

### 4. Env pool behaviour

Sampled once a minute from `/proc` across the whole 112-iteration run, including two
checkpoint evals at `eval_episodes_per_env: 50`:

| | start | steady state | during checkpoint eval | after |
|---|---|---|---|---|
| RSS | 2,573 MB | **2,809-2,817 MB** | 2,938 MB | 2,848 MB |
| open fds | 99 | 126-132 | **210** | 128 |
| `$TMPDIR` size | 76 MB | 82 MB | 92 MB | 82 MB |
| `$TMPDIR` entries | 80 | 100 | 141 | 100 |

**No leak.** RSS climbs for the first twelve minutes as the replay buffer fills to
`buffer_capacity: 10000`, then sits flat at 2,809-2,817 MB for the following fifteen
minutes. The eval spikes are transient and fully released: +129 MB RSS, +80 fds, +41 temp
directories while 50 live envs exist, all returned afterwards. `REMDM_MAX_IDLE_ENVS` did not
need turning down.

This is the change whose failure mode is gradual, so the plateau over 111 iterations
matters more than the peak. For the 5.65M-step runs, RSS should be watched once rather than
assumed, but nothing here suggests it will not hold.

## Phase 5: the run plan

All figures from this box, `TMPDIR` on local disk. "Before" is merge-base `6d1e37d` with the
same collection time, since collection is unchanged; "after" is HEAD `9af96ba`.

| Quantity | Before | After |
|---|---|---|
| s per iteration | 19.52 (collect 3.30 / train 16.22) | **14.30** (collect 3.30 / train 10.99) |
| ms per grad step | 162.16 | **109.9** |
| samples/s | 12,629 | **18,628** |
| iterations to `total_timesteps: 5650000` | 689 | 689 |
| hours per seed, `--mode online` | 3.94 | **2.95** |
| hours per seed, `--mode offline` (60,000 grad steps) | 2.82 | **1.94** |
| hours for 3 seeds, both modes | 20.3 | **14.7** |
| peak VRAM allocated / reserved | 6,675 / 7,007 MB | 6,678 / 7,065 MB |
| headroom against 24 GB (idle card) | 17.4 GB | **17.4 GB** |

**The speed-up saves 5.6 hours across the planned 3 seeds x 2 modes, a 28% reduction.**

### How the iteration count was derived

Not assumed. Measured across the 112-iteration real run: 8,972 env steps per iteration
averaged over the whole run, but **8,197 over its second half**, because episodes shorten as
the policy improves. The table uses the second-half rate (689 iterations); the whole-run
average would give 630. If the rate keeps falling beyond the 1M steps observed, the true
count rises above 689 and the hours rise proportionally — this is a projection from
early-training episode lengths, as the prompt requires, not a measurement of the full run.
The per-iteration wall time is far more solid: the training half was flat to within 0.13 s
across all 112 iterations.

The eval overhead (688 s per seed) assumes 12 periodic ID+OOD evals and 6 checkpoint evals at
the measured 22.1 s (ID) and 16.1 s (OOD) for 50 episodes per env.

### Go / no-go

**Go**, with three conditions.

1. **Set `TMPDIR` to local disk.** Not optional. On the NFS default, collection is 6.64 s per
   iteration against 3.73 s, which breaches the prompt's own gate and adds roughly 0.6 h per
   online seed. `export TMPDIR=/tmp/<something>` before launching.
2. **This box is shared.** Another user held 7,644 MiB of GPU memory and a load average of
   4.5 for the whole session. The run fits alongside that (7.1 GB reserved + 7.6 GB foreign
   = 14.6 GB of 24.5 GB), but the wall-clock projections assume the contention level seen
   here. A heavier neighbour will stretch them, and a single contended measurement misled by
   2.3x during this pass before interleaving caught it.
3. **Watch RSS once on the first long run.** The pool plateaued at 2.8 GB over 112
   iterations with clean release after each eval, which is the behaviour wanted, but 5.65M
   steps is 6x longer than what was observed.

Nothing found in this pass argues against launching. The recipe is untouched: every pinned
parameter is unchanged, `dagger_batch_size: 2048` fits with 17.4 GB spare on an idle card,
the loss trajectory sits inside the same-code control band, and the one change that moved it
(bf16) has been reverted out of the tree.

### Commits from this pass

| Commit | What |
|---|---|
| `af1db9a` | Revert C6: bf16 rejected on both boxes, flag removed |
| `e5eb2dd` | Fix the CUDAGraphs crash in `profile_dagger.py`'s memory audit |
| `c5acce5` | Certified VRAM, the dead worker knob, the TMPDIR requirement in the config header |
| this one | This report |

Final state of prompt 1's six changes after certification: **C0, C1, C2, C3, C4 keep;
C6 revert.**

## What was not done, and why

- **A merge-base twin of the real run.** The before/after behavioural comparison rests on the
  loss-trajectory table, the identical env-step counts, and prompt 1's seed-matched
  before/after run on the dev box. A second 30-minute run here would have added little: the
  sharper evidence is the 200-step trajectory against a measured control band.
- **Wiring up `num_collection_workers`.** It is dead on CUDA and its only cost is 167.7 MB of
  RAM and 0.06 s of startup for 8 unused CPU model copies (5,241,935 parameters, 21.0 MB
  each). Making it live would change collection behaviour; the comment now records the truth
  instead.
- **The oracle BFS, which is now the whole collection cost.** `oracle_rollout_sec` is 0.42 to
  5.70 s per iteration while everything else in collection is under 1.1 s. At 23% of
  iteration time it does not justify the process-pool rework the prompt names as the next
  lever, but it is where the remaining collection time is, and it is the thing to attack if
  collection ever matters again.
- **The model architecture**, per the prompt. For the record, the global stream's 435 MB fp32
  activation is why a 5.2M-parameter model needs 6.7 GB at batch 2048.
