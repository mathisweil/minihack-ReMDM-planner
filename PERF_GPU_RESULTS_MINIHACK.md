# GPU-half results: MiniHack DAgger speed-up (prompt 1 of 2)

Input: `PERF_GPU_PROMPT_MINIHACK.md`. Successor: `PERF_MEASURE_3090_PROMPT_MINIHACK.md`.

**Every number here was measured on the 4070 Ti dev box.** None of it certifies the
target. Where the two boxes differ in kind, that is called out per change.

## The box this ran on

```
$ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
name, memory.total [MiB], driver_version
NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB, 560.35.03

$ uv run python -c "import torch;p=torch.cuda.get_device_properties(0);print(p)"
_CudaDeviceProperties(name='NVIDIA GeForce RTX 4070 Ti SUPER', major=8, minor=9,
  total_memory=15974MB, multi_processor_count=66, L2_cache_size=48MB)

$ lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)'
CPU(s):                  28
Model name:              Intel(R) Core(TM) i7-14700K
Thread(s) per core:      2
Core(s) per socket:      20

$ hostname
outback.cs.ucl.ac.uk
```

torch 2.13.0+cu126, Python 3.12.13. This is the 4070 Ti dev box the prompt anticipated
(the SUPER variant, 16 GB), not the 3090 Ti.

### Two environment facts that bite

**1. `uv sync` / `uv run` must carry `--extra cuda12` on this box.** The driver is
560.35.03 (CUDA ceiling 12.6), and the default no-extra resolution is PyPI torch 2.13.0
built for CUDA 13.0, which fails with `RuntimeError: The NVIDIA driver on your system is
too old (found version 12060)`. Worse, flipping between the two resolutions corrupts the
venv: the `nvidia-*-cu12` and `nvidia-*-cu13` wheels share `nvidia/<lib>/` namespace
directories, so uninstalling one side deletes the other side's files while leaving it
recorded as installed. The symptom is `ImportError: libcudnn.so.9` or
`cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` with `uv sync` reporting nothing to do. The
fix is `uv sync --extra cuda12 --reinstall`. Sync once, then use `uv run --no-sync`.

**2. `$TMPDIR` is on NFS.** `tempfile.gettempdir()` is
`/cs/student/project_msc/2025/dsml/mathweil/tmp`, mounted `evs2:/cs/student/project_msc/2025`
over NFS, while local xfs is available at `/tmp` (178 GB free). Env construction is
filesystem-bound, so this is the condition the prompt warned about. Measured cost below.

## Preflight

| Check | Result |
|---|---|
| `git status --short` | clean; `git log --oneline -3` = `6d1e37d`, `70f508e`, `3fe4444` |
| `uv run python -m pytest tests -q` | 136 passed, 2 deselected |
| `uv run python -m pytest tests/test_env_reuse.py -v` | 6 passed in 1.39s |
| `uv run ruff check src tests` | 4 errors, all the documented pre-existing ones |
| CUDA | available, RTX 4070 Ti SUPER, bf16 supported |

`test_recycled_env_matches_fresh_env` is green here, so env reuse is safe in this NLE
build and the collection change stands.

## Phase 1: the split before any GPU change

Measured with a harness that drives the real `DataCollector.collect_batch_gpu` and the
real `Trainer._train_step`, at `configs/final_ucl_gpu.yaml` unmodified
(`episodes_per_iteration: 30`, `grad_steps_per_iteration: 100`,
`dagger_batch_size: 2048`, `use_amp: true`, `torch_compile: true`). `scripts/profile_dagger.py`
was not used for the headline numbers: it reconstructs the loop rather than calling it,
and as the prompt notes it does not isolate `make_env`. Timing a real `collect_batch_gpu`
call covers that gap directly. Figures are the mean of iterations 1-2 (iteration 0 pays
`torch.compile` and a cold env pool).

| Quantity | Before | After | Change |
|---|---|---|---|
| collect wall time | 3.79 s | 4.15 s | oracle-thread noise, see below |
| &nbsp;&nbsp;env reset (30 envs) | 0.341 s | 0.325 s | unchanged |
| &nbsp;&nbsp;env step | 0.575 s | 0.553 s | unchanged |
| &nbsp;&nbsp;GPU inference | 0.180 s | 0.176 s | unchanged |
| &nbsp;&nbsp;oracle rollout (threaded BFS) | 2.683 s | 3.076 s | unchanged code; thread-timing noise |
| train wall time (100 steps) | 16.71 s | 12.28 s | **-26.5%** |
| ms per grad step | 167.10 | 122.79 | **-26.5%** |
| samples/s | 12,256 | 16,678 | **+36.1%** |
| iteration wall time | 20.50 s | 16.43 s | **-19.9%** |
| env steps per iteration | 10,735.5 | 10,735.5 | identical (same seeds) |
| peak VRAM | 6,677.6 MB | 6,676.5 MB | unchanged |

Nothing in the collection path was touched, and the identical env-step count confirms it:
the collect-time difference is entirely in `oracle_rollout_sec`, which is 30 BFS rollouts
across a thread pool and varies run to run.

**Decision gate: passed.** Collection is 3.8 s per iteration against the 5 s threshold,
and 18% of iteration time. In the short real run below it settles lower still (1.4 s at
iteration 10) once the policy improves and episodes shorten. The collection fixes landed:
`env_reset_sec` for 30 envs is 2.46 s on a cold pool (82 ms/env) and 0.13-0.55 s once warm
(4-18 ms/env).

**VRAM at `dagger_batch_size: 2048`: 6,677 MB peak allocated** (`torch.cuda.max_memory_allocated()`),
7,111 MB reserved before the changes and 6,730 MB after. No OOM, and 9 GB of headroom on
this 16 GB card. The
`final_ucl_gpu.yaml` header's "~6-8 GB peak" claim is correct — it can stay. Prompt 2
still needs to certify it on the 3090 Ti, but there is no memory risk to engineer around.

## Phase 2: the changes

Measured on a 200-step train-only benchmark at the real config, fixed seed and a
deterministic buffer, so the only variation between runs is GPU non-determinism. The
noise floor from two identical baseline runs is **0.06% on time and 2.8e-4 on any single
loss value**, which is what "within noise" means below.

| # | Change | ms/step | delta vs previous | Verdict | Commit |
|---|---|---|---|---|---|
| — | baseline (two runs) | 166.46 / 166.56 | — | — | `6d1e37d` |
| C1 | int16 over PCIe, widen on GPU | 164.99 | -0.9% | `keep` | `b300627` |
| **C0** | **vectorise the staircase lookup** | **123.19** | **-25.3%** | **`keep`** | `46ced6c` |
| C2 | device-side step metrics | 122.88 | -0.25% | `carry` | `d05b64e` |
| C3 | fused EMA update | 122.47 | -0.33% | `carry` | `0150862` |
| C4 | `fused=True` AdamW | 121.86 | -0.50% | `carry` | `33fc9b3` |
| C5 | confirm `torch.compile` engages | — | no change needed | — | — |
| C6 | bf16 instead of fp16 + scaler | 119.67 | -1.8%, **loss moves** | `carry`, off by default | `0c637e2` |

Committed in dependency order (`46ced6c` C0 first, then `b300627` C1), while the ms/step
column above is in the order they were measured, which is why C1 appears above C0 there.
Each change is a separate commit so one can be dropped without unpicking the others; the
tests that pin each change ship in its own commit, and the suite and lint are green at
every one.

Cumulative: **166.51 → 121.86 ms/step, a 1.37x speed-up on the gradient step.**

### C0 (not in the original list, and the one that mattered)

Profiling the step first, rather than starting from the list, found that
`find_staircase_from_glyphs` (`src/diffusion/loss.py`) looped over the batch calling
`is_stair[b].nonzero()`. `nonzero` must return its output size to the host, so every
sample forced a device sync: **2,048 syncs per gradient step**, with
`cudaStreamSynchronize` accounting for 71% of CPU time. The step was CPU-bound, not
GPU-bound.

`nonzero` returns row-major-ordered indices, so `positions[0]` is the lowest flat index
that is set; taking the minimum masked flat index reproduces it exactly with no host
round-trip.

Profiler, 10 gradient steps, before → after:

| Metric | Before | After |
|---|---|---|
| `aten::nonzero` calls | 20,510 | 0 |
| `cudaStreamSynchronize` calls | 20,620 | 100 |
| `cudaLaunchKernel` calls | 68,755 | 2,530 |
| Self CPU total | 1.692 s | 1.217 s |
| Self CUDA total | 1.297 s | 1.204 s |

CPU total now matches CUDA total, i.e. the step is GPU-bound and further launch-overhead
work has little left to win *on this box*. That is the main reason C2/C3/C4 read as small
here and may not on the target.

Exactness was verified directly rather than inferred: the new implementation is bitwise
equal to the loop across 22 cases on both CPU and CUDA (random maps, no staircase, all
four glyph variants, every-cell staircase, 2-D input, and a flat-index sweep over
positions 0, 1, 78, 79, 80, 1658).

### C1

`local`, `global` and `actions` now transfer at their buffer dtype and widen on the
device, in both `online.py` and `offline.py`. The buffer holds glyph maps as int16
(`minihack_env.py:720-721`), so `global` crossed as int64 at 27.2 MB per step; it now
crosses at 6.8 MB.

The `.int()` variant the prompt suggested measuring was tried and is **not** worth it:
122.06 ms/step against 121.86 for `.long()` (i.e. no gain, inside noise) for 15 MB less
VRAM, in exchange for a non-standard index dtype flowing through the model. Reverted to
`.long()`.

`non_blocking=True` was left off deliberately: `torch.from_numpy` gives unpinned memory,
where the copy is synchronous regardless, so the flag would document a guarantee the code
does not have. Pinned staging buffers would be a real change with real bookkeeping, and
with the step now GPU-bound there is nothing here to win on this box. Flagged for prompt 2
rather than done blind.

### C2, C3, C4

All three are launch-overhead and sync fixes, all sound in principle, all near the noise
floor on this box. Per the prompt's inverted rule, none is reverted: this box has the
faster CPU of the two and the step is now GPU-bound here, so each understates.

- **C2**: `Trainer._train_step_device` returns detached 0-dim tensors and the loop sums
  them on device, with one `.tolist()` where the metrics are logged. `_train_step` remains
  as a float-returning wrapper so existing callers and tests are unaffected. `offline.py`
  records per-step losses into a device buffer materialised once after the loop.
- **C3**: `ModelEMA.update` uses `torch._foreach_mul_`/`torch._foreach_add_` over cached
  operand lists: 144 kernel launches per step become 2. Verified bitwise identical to the
  loop across 50 updates on all 72 parameter tensors.
- **C4**: `fused=True` on both AdamW constructions, gated on CUDA.

### C5: `torch.compile` is engaging

Not assumed — checked. `/usr/bin/gcc` and `/usr/bin/cc` are both present, `_has_c_compiler()`
returns `True`, the log line is `INFO src.models.denoiser: Compiling model with torch.compile`
with no fallback warning, `try_compile` returns an `OptimizedModule`, and the profiler shows
`Torch-Compiled Region: 0/0`, `CompiledFunctionBackward` and named Triton kernels. Inductor
does emit `Not enough SMs to use max_autotune_gemm mode` on this 66-SM card; the 3090 Ti has
84 SMs, so prompt 2 may see a different autotune path.

### C6: bf16 wins on time and loses on the loss

Implemented as an opt-in `amp_dtype: fp16|bf16` config key, defaulting to `fp16`, so the
default behaviour is byte-for-byte the released recipe and prompt 2 can measure the
alternative with one CLI override. The scaler is enabled only for fp16.

bf16 is 1.8% faster here (119.67 vs 121.86 ms/step) but **moves the loss trajectory by two
orders of magnitude more than the noise floor**: max |Δloss| 3.67e-2 and mean 4.19e-3,
against a 2.83e-4 / 5.20e-5 noise floor. Mean loss over 200 steps is 0.54934 against
0.55332. That is the mantissa difference, not scheduling. The prompt's rule is to drop it
if the loss curve moves at all, and it does; it stays off. Left in the tree as a measurable
flag because the fp16-vs-bf16 ratio is one of the things that differs most between these
two cards, so the trade may look different on the target — but the loss trajectory will
not, and that is the reason not to use it.

## Loss trajectories

200 steps, same seed, same deterministic buffer, compared against baseline run A.

| Variant | ms/step | max abs Δloss | mean abs Δloss | mean loss | last-20 mean |
|---|---|---|---|---|---|
| baseline A (reference) | 166.46 | — | — | 0.55332 | 0.14200 |
| baseline B (**noise floor**) | 166.56 | 2.83e-04 | 5.20e-05 | 0.55337 | 0.14202 |
| C1 | 164.99 | 3.05e-04 | 5.93e-05 | 0.55337 | 0.14203 |
| C0+C1 | 123.19 | 1.90e-04 | 4.64e-05 | 0.55327 | 0.14195 |
| +C2 | 122.88 | 3.49e-04 | 6.28e-05 | 0.55338 | 0.14201 |
| +C3 | 122.47 | 3.90e-04 | 7.79e-05 | 0.55339 | 0.14206 |
| +C4 (final, fp16) | 121.86 | 1.68e-04 | 3.11e-05 | 0.55334 | 0.14197 |
| C1 `.int()` variant (rejected) | 122.06 | 3.61e-04 | 6.91e-05 | 0.55338 | 0.14204 |
| +C6 bf16 (**off by default**) | 119.67 | **3.67e-02** | **4.19e-03** | 0.54934 | 0.14266 |

Every shipped change sits inside the run-to-run noise band. Only bf16 leaves it, by ~100x.

C1 in particular cannot change values, and this is checked rather than argued: int16 to
int64 widening is exact for glyph IDs, and `tests/test_gpu_step_perf.py` pins that the
device-side widening equals the host-side cast it replaced.

## Verification

1. **Suite and lint**: `157 passed, 2 deselected`; `ruff check src tests` back to the same
   4 pre-existing errors (`tests/test_failure_behaviour.py`, `tests/test_method_spec.py`).
   None of them were touched. One new C420 introduced during this work was fixed, not
   suppressed.
2. **New tests**: `tests/test_gpu_step_perf.py`, 21 tests (suite total 136 → 157), pinning the invariants each
   change relies on — the staircase lookup against the loop it replaced (including the
   row-major tie-break and the pad sentinel), the fused EMA against the per-parameter
   loop, the device/float metric contract and the empty-buffer no-op, exact int16 widening,
   and the `amp_dtype` mapping plus the fp16-only scaler rule. CPU-only apart from two
   CUDA-gated scaler cases, which run (not skip) on this box.
3. **Loss trajectory**: table above.
4. **Short real run**: see below.

### Short real run, before and after

One DAgger run to the first checkpoint through `main.py`, same seed (7), same overrides
(`total_timesteps=160000`, eval and checkpoint at 150000, 10 episodes per env), on the
pre-change tree and the post-change tree.

| | Before | After |
|---|---|---|
| wall clock, whole run | 419 s (19:00:02 → 19:07:01) | **324 s** (18:53:43 → 18:59:07) |
| iteration 0 (pays `torch.compile`) | 25.83 s | 26.15 s |
| iteration 10 | 18.12 s | **13.51 s** |
| &nbsp;&nbsp;of which train | 16.998 s | **12.121 s** |
| &nbsp;&nbsp;of which collect | 1.126 s | 1.386 s |
| &nbsp;&nbsp;samples/s | 12,048 | **16,896** |
| mean loss, iteration 0 | 1.0375 | 1.0375 |
| mean loss, iteration 10 | 0.0152 | 0.0152 |
| env steps at checkpoint 1 / 2 | 157,468 / 169,149 | 157,468 / 169,149 |
| ID win rate, `eval_iter18.json` | 0.25 | 0.25 |
| OOD win rate, `eval_iter18.json` | 0.00 | 0.00 |
| ID win rate, `eval_iter20.json` | 0.15 | 0.15 |
| OOD win rate, `eval_iter20.json` | 0.00 | 0.00 |

1.29x on the whole run including evals and checkpointing. The behavioural evidence is
stronger than the prompt asked for: the env-step counts are identical at both checkpoints,
the logged mean losses agree to four decimal places at both logged iterations, and the
aggregate win rates match exactly at both checkpoints. Per-env win rates at `iter18` are
identical (0.3 / 0.4 / 0.2 / 0.1); at `iter20` two envs swap a single episode each
(15x15 0.2 → 0.1, MazeWalk 0.1 → 0.2) with the aggregate unchanged, which is GPU
non-determinism in sampling, not a behavioural change. At 10 episodes per env the standard
error on a per-env win rate is about 0.13, so this check catches gross divergence rather
than fine structure.

### Env pool behaviour over the run (Phase 4 item 4)

Sampled from `/proc` during the post-change run: RSS 2,623 → 2,696 MB and open fds 98 →
127 over ~4 minutes and 21 iterations, thread count flat at 53. Across the benchmark
iterations RSS was 2,436 MB mean with fds at 79-81. No unbounded growth in either, and no
fd leak. The pool's failure mode is gradual, so this is worth re-checking on a run long
enough to reach `eval_episodes_per_env: 50`, where up to 50 live plus 64 idle envs can
coexist — that configuration was not exercised here (the short run used 10).

### `$TMPDIR` on NFS

The prompt asked to confirm `$TMPDIR` is on local disk. It is not: it is on an NFS mount,
with local xfs available at `/tmp`. Measured, same box, same process, `MiniHack-Room-Random-5x5-v0`:

| | NFS (`$TMPDIR` as set) | local xfs (`/tmp`) |
|---|---|---|
| `make_env` cold, pool bypassed | 86.3 ms/env | **25.6 ms/env** |
| first pooled acquire | 86.7 ms | 28.4 ms |
| pooled recycle | 0.00 ms | 0.00 ms |
| `env.reset` on a recycled env | 2.94 ms | 2.43 ms |

NFS makes construction 3.4x more expensive. The env pool is what keeps this from
mattering: a recycled acquire is free on both, so steady-state collection is unaffected,
which is why collection still passes the gate at 3.8 s. What it still costs is every
construction the pool cannot avoid — filling the pool at startup (30 envs, ~1.8 s extra)
and eval, where `eval_episodes_per_env: 50` can want 50 live envs of an env-id at once
(~3 s extra per env-id, per cold fill).

Pointing `TMPDIR` at local disk is free and removes the penalty outright. Recommended for
the real runs, and worth checking on the target box, which may share the same NFS home.


## What is not done, and why

- **`num_collection_workers: 8` is still dead on the CUDA path.** `DataCollector.__init__`
  allocates 8 CPU model copies that only `collect_batch_parallel` uses, and CUDA never
  takes that branch (`online.py:157-167`). Wasted memory and startup time, not wasted
  wall-clock per iteration. Not touched: it is a collection-path change and the
  measurement does not demand it.
- **Sequential env stepping in the model rollout** (`collect.py:618-647`) is 0.55 s of the
  3.8 s collect, with the oracle BFS at 2.68 s the larger half. Moving env stepping into
  worker processes is the prompt's named next lever; the measurement does not justify
  starting it.
- **The global stream** is untouched, per Phase 3. For the record, it is where the memory
  goes: `nn.Embedding(6000, 32)` over `[2048, 21, 79]` materialises a 435 MB fp32
  activation per forward, and it is the largest single tensor in the step.
- **Pinned staging buffers** for the H2D copies: see C1.

## For prompt 2

- Re-run the same comparison on the 3090 Ti and decide C2, C3, C4. They are launch-overhead
  fixes measured on the faster CPU of the two boxes, against a step that is now GPU-bound
  here; the target's i9-12900K should show more.
- Certify peak VRAM at `dagger_batch_size: 2048` on 24 GB. Expect ~6.7 GB.
- Decide bf16. The recommendation from here is **no**, on trajectory grounds, whatever the
  throughput says.
- Hours per seed is deliberately not projected here. This box's iteration time is 16.43 s
  with a cold model; the short real run shows env steps per iteration falling from ~7,900
  to ~4,500 as the policy improves, so iteration count rises above the naive
  `total_timesteps / env_steps_per_iteration` estimate as the run progresses. Prompt 2
  supersedes any projection made from this box.
