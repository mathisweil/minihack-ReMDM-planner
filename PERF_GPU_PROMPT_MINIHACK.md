# Task: implement the GPU half of the MiniHack DAgger speed-up

**This is prompt 1 of 2.** It implements and correctness-checks the changes on a development GPU. `PERF_MEASURE_3090_PROMPT_MINIHACK.md` then certifies them on the target box and produces the numbers the real runs are planned from. Do not do prompt 2's job here, and do not skip it: the decisions it makes cannot be made from a development box.

## Where you are

`minihack-ReMDM-planner` on a CUDA box. A uv project: `uv sync`, run with `uv run`. Repo conventions are in `CLAUDE.md` one level up; UK English, no em dashes, evidence for every claim (command and output, file and line).

The collection half of this work is **already done and in the tree you are pulling**. Your job is the GPU half, plus confirming on real hardware that the collection half landed. Read the next section before touching anything: it tells you what changed and what the measured baseline was, so you can tell a regression from a difference in hardware.

**Measure before you change anything, and measure after each change.** Every number in this document was measured on an Apple laptop CPU. The ratios should carry; the absolute values will not. Do not carry my figures into a report as if you had observed them.

### Which box are you on, and what it can decide

The intended target for the real runs is the **3090 Ti box** (24 GB VRAM, i9-12900K, 16 physical cores / 24 threads). This prompt is expected to run on the **4070 Ti dev box** (16 GB VRAM, i7-14700K, 20 physical cores / 28 threads). Record what you are actually on before anything else, and do not infer it from this paragraph:

```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run python -c "import torch;p=torch.cuda.get_device_properties(0);print(p)"
lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)'
```

| Question | Dev box (4070 Ti) can answer | Target box (3090 Ti) only |
|---|---|---|
| Does the change run, and is it correct? | **Yes.** This is the point of this pass. | |
| Does the loss trajectory stay put? | **Yes.** Hardware independent. | |
| Do the collection fixes hold under CUDA and Linux? | **Yes.** First real run of `collect_batch_gpu`. | |
| Rough ranking of which GPU changes help | Provisional only | Final |
| Peak VRAM at `dagger_batch_size: 2048` | Indicative, and only if it fits | **Certifying** |
| ms per grad step, samples/s, hours per seed | No | **Yes** |
| Whether to revert a change that looked marginal | **No.** Carry it forward. | **Yes** |

Two specific reasons the numbers do not transfer, both worth keeping in mind while you read your own measurements:

- **Memory bandwidth.** The 3090 Ti runs a 384-bit GDDR6X bus against the 4070 Ti's 192-bit, and Ada carries far more L2. The model's global stream embeds 2048 x 1659 = 3.4M glyph lookups per forward into a 435 MB fp32 activation, so it is bandwidth-sensitive, and anything touching it will land differently on the two cards.
- **Single-core speed.** The i7-14700K here is faster per core than the target's i9-12900K. C3 and C4 below are kernel-launch-overhead fixes, which are CPU-bound. They will look **less** valuable on this box than they will on the target. A change that reads as noise here may be real there.

If you are in fact on the 3090 Ti, run this prompt anyway and then run prompt 2 in the same session: the two passes are still distinct, and prompt 2 covers the certification work this one deliberately omits.

---

## What has already been changed

Three defects in the collection path, all fixed, all trajectory-preserving. Verified byte-identical (same actions, same SHA-256 over every observation, same rewards) across three seeds on all four ID envs before and after.

| # | Change | Files | Measured effect |
|---|---|---|---|
| 1 | `obs_keys` no longer requests `pixel` | `src/envs/minihack_env.py:191` | `env.step` 3.56 ms to 0.22 ms |
| 2 | Envs recycled through a bounded pool instead of rebuilt per episode | `minihack_env.py` (`_EnvPool`, `acquire_env`, `release_env`, `borrow_env`, `close_env_pool`), `collect.py`, `inference.py` | construction 183 ms to 3.0 ms per episode |
| 3 | Reward shaping skipped where the reward is discarded | `minihack_env.py:step`, collection call sites | a further ~2x on step cost |

Why they were so expensive:

- **`pixel`** made NLE render the full tiled RGB screen every step. Nothing in the repo ever read it. Grep confirms: the wrapper only touches `glyphs` and `chars`.
- **Env construction** looked cheap and was not. `MiniHack.__init__` unconditionally calls `_patch_nhdat` (`minihack/base.py:305`), which copies the NetHack data directory into a fresh temp dir and forks `lev_comp` then `dlb` to rebuild the data archive. At `episodes_per_iteration: 30` the loop paid that 60 times per iteration, roughly 11 s of pure fork and file copying.
- **Shaping** ran a full BFS plus two `np.argwhere` scans per step to build a reward that DAgger collection discards. It is still on for eval, which reports `avg_reward`.

End-to-end A/B on the real `collect_oracle_trajectory`, 10 episodes per env, episode lengths asserted identical:

```
env                                    legacy    current   speedup  steps
MiniHack-Room-Random-5x5-v0            222.6ms       2.9ms     76.0x     2
MiniHack-Room-Random-15x15-v0          231.5ms       6.3ms     36.9x     8
MiniHack-Corridor-R2-v0                926.8ms      36.4ms     25.5x   203
MiniHack-MazeWalk-9x9-v0               788.8ms      21.2ms     37.2x   161
TOTAL per oracle episode               542.4ms      16.7ms     32.5x
```

`tests/test_env_reuse.py` pins all of it: pixel stays out of `obs_keys`, a recycled env is trajectory-identical to a fresh one, the pool actually recycles and stays bounded, disabling shaping leaves transitions untouched, and the hot oracle path returns its env rather than leaking it. 136 tests pass, ruff is clean on the changed files.

### Two things about that work you must check rather than trust

1. **`collect_batch_gpu` never runs on CPU**, so the test suite does not cover it (`online.py:137` only takes that branch on CUDA). It was exercised by hand on CPU and works, but this box is the first place it runs for real. Watch iteration 1 closely.
2. **The env pool holds instances alive.** The cap is idle instances only; peak live still tracks caller concurrency, so eval at `eval_episodes_per_env: 50` can have 50 live plus up to 64 idle. Each holds a NetHack vardir of a few MB. If RSS or `$TMPDIR` becomes a problem, lower `REMDM_MAX_IDLE_ENVS` (default 64, read at import in `minihack_env.py`). Also confirm `$TMPDIR` is on local disk, not NFS: env construction is filesystem-bound and a network mount would make even the reduced cost hurt.

---

## Preflight

All of it, before you start. Report and stop on any failure rather than working around it.

```
git status --short && git log --oneline -3
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import os,tempfile; print('TMPDIR:', tempfile.gettempdir())" && df -h "$(python -c 'import tempfile;print(tempfile.gettempdir())')"
uv run python -m pytest tests -q
uv run python -m pytest tests/test_env_reuse.py -v
uv run ruff check src tests
```

`tests/test_env_reuse.py` must be green here, not just on the laptop. If `test_recycled_env_matches_fresh_env` fails on this box, stop: env reuse is not safe in this NLE build and the whole collection change has to come out.

---

## Phase 1: measure the current split, change nothing

`scripts/profile_dagger.py` already exists and reports per-component timings for rollout and gradient steps. Use it first:

```
uv run python scripts/profile_dagger.py --config configs/final_ucl_gpu.yaml
```

That `--config` flag was added as part of this work. Before it, the script always loaded `configs/defaults.yaml` no matter what you passed, which profiles `dagger_batch_size: 3584` with **`use_amp` and `torch_compile` off**: a materially different machine from the one the run uses. If you see those values in the output, the flag did not take effect and the numbers are worthless.

Two further caveats, both worth reading the script for:

- It forces `buffer_capacity: 500` for a fast start, against 10,000 in the real run. That changes `_ensure_cache` rebuild cost and buffer sampling, not the GPU step.
- It times `env_reset` and `env_step` separately but does **not** isolate `make_env`, which was the dominant cost before change 2. Add that timing, or time a full `collect_batch_gpu` call directly, otherwise you cannot confirm change 2 landed.

What to record, per iteration, at the real config (`configs/final_ucl_gpu.yaml`, `episodes_per_iteration: 30`, `grad_steps_per_iteration: 100`, `dagger_batch_size: 2048`):

| Quantity | Where from |
|---|---|
| collect wall time, split into env reset / env step / GPU inference | `DataCollector._last_profile` |
| train wall time and ms per grad step | wrap the 100-step loop |
| samples/s | `100 * 2048 / train_time` |
| peak VRAM | `torch.cuda.max_memory_allocated()` |
| env steps per iteration | `sum(model_steps) + sum(oracle_steps)` |

Then project: `total_timesteps / env_steps_per_iteration` gives the iteration count (expect roughly 700 at 5.65M), times the per-iteration wall time gives hours per seed, times 3 seeds.

**Decision gate.** After the collection fixes, collection should be a small minority of iteration time (projection: 1 to 2 s against 25 to 30 s before). If collection is still above 5 s per iteration on this box, stop and report before doing any GPU work. Likely causes, in order: `$TMPDIR` on a network mount, the pool being bypassed on some path, or model episodes running far longer here than the 152-step random-policy mean the projection assumed. Diagnosing that is worth more than any change below.

The 5 s threshold is a smell test, not a target. It is deliberately loose enough to survive the difference between these boxes: collection is now mostly single-threaded env stepping, so it should if anything be *faster* on the 14700K than on the target's 12900K. Collection above 5 s means something is wrong in kind, not in degree.

**VRAM at this point.** Record `torch.cuda.max_memory_allocated()` at the real `dagger_batch_size: 2048`. The `final_ucl_gpu.yaml` header claims roughly 6 to 8 GB peak, which is a comment nobody has verified. Three outcomes:

- It fits with room to spare: good, note the figure, and prompt 2 will certify it on 24 GB.
- It fits but is close to 16 GB: note it prominently. The target has 24 GB so the run is still viable, but the config comment is wrong and should be corrected.
- It OOMs: **stop and report. Do not lower the batch size.** `dagger_batch_size: 2048` is pinned to the released checkpoint recipe and to the compute-matched offline baseline (Phase 3). An OOM at 16 GB says nothing about 24 GB, so this is a finding to hand to prompt 2, not a problem to engineer around here. If you need to keep working, do the remaining GPU changes at a reduced batch **for timing ratios only**, and label every number produced that way.

---

## Phase 2: the GPU-side changes

Ranked by expected value. Do them one at a time, measure after each, and record the delta.

**Do not revert anything on this box's evidence alone.** That rule is inverted from the usual one, deliberately. C3 and C4 are kernel-launch-overhead fixes and this box has the faster CPU of the two, so they will understate here. C1 and C6 touch bandwidth and this box has roughly half the target's. Classify each change instead:

| Verdict | Meaning | Who decides |
|---|---|---|
| `keep` | Clear win here, and no reason it would not hold | you, confirmed by prompt 2 |
| `carry` | Neutral or noise here, but sound in principle | **prompt 2** |
| `revert` | Actively slower, or changes the loss trajectory | you, now |

Only the third is yours to act on unilaterally, and a loss-trajectory change is a correctness failure rather than a performance one. Everything else ships to prompt 2 with its measurement attached, so the target box can re-run the same comparison and decide.

Each change below is small and independent. Commit them as separate commits so prompt 2 can bisect or drop one without unpicking the others.

**C1. Stop sending 4x the bytes over PCIe.** `online.py:361-363` and `offline.py:245-247` cast to `.long()` on the CPU before transfer, so `global` crosses as int64: 27.2 MB per step, 2.72 GB per iteration, against 6.8 MB and 0.68 GB if you transfer int16 and widen on the GPU.

```python
# from
global_t = torch.from_numpy(global_np).long().to(self.device)
# to
global_t = torch.from_numpy(global_np).to(self.device, non_blocking=True).long()
```

`nn.Embedding` accepts int32 as well as int64, so `.int()` is an option worth measuring against `.long()`. Pinned staging buffers help `non_blocking` do anything at all; without pinning the copy is still synchronous. Applies to `local`, `global` and `actions` in both files.

**C2. Remove the per-step GPU syncs.** `online.py:418-421` calls `.item()` four times per grad step, 400 forced syncs per iteration, each stalling the CPU until the GPU drains. Accumulate the loss tensors on device and call `.item()` once per iteration when the metrics are actually logged. `offline.py:302,329-345` has the same pattern, worse: it is inside the logging block but `loss.item()` at 302 runs every step.

**C3. Fuse the EMA update.** `denoiser.py:369-373` is a Python loop over 72 parameter tensors, so 144 kernel launches, run after every grad step (`online.py:193`). `torch._foreach_mul_` and `torch._foreach_add_` make it two. Cache the shadow and parameter lists once rather than rebuilding them per call.

**C4. `fused=True` on AdamW.** `online.py:687` and `offline.py:92`. One line, 72 param tensors, safe with AMP on CUDA.

**C5. Confirm `torch.compile` is actually engaging.** `try_compile` (`denoiser.py:317`) silently returns the uncompiled model when it cannot find `cc` or `gcc`, which is common on managed GPU nodes. It logs a warning; check the log rather than assuming. `torch_compile: true` in the config means nothing if that fallback fired. If it did, either make a compiler visible or stop paying attention to the flag. Note that only the training model is compiled; collection and eval deliberately use the raw model, which is correct (deep-copying a compiled module breaks FX tracing).

**C6. Measure bf16 against fp16 plus GradScaler.** Both Ampere and Ada support bf16, so this is testable here. It would remove the scaler's `unscale_` pass and its inf checks. This one is a genuine experiment, not a known win: measure it, and if the loss curve moves at all, drop it. Default to `carry` rather than `keep` even if it wins here, because the fp16-vs-bf16 throughput ratio is one of the things that differs most between these two cards.

If Phase 1 says collection still dominates after all this, the next lever is that the model rollout steps its 30 envs sequentially in one Python loop (`collect.py:615-617`) while 15 cores idle, and `num_collection_workers: 8` is dead on the CUDA path (it only feeds `collect_batch_parallel`, which CUDA never takes, while `DataCollector.__init__` still allocates 8 unused CPU model copies). Moving env stepping into persistent worker processes is the fix. Do not start it unless the measurement demands it.

---

## Phase 3: what must not change

These are pinned to the released `iter600` checkpoint recipe and to the compute-matched offline BC baseline. Changing any of them invalidates the paper's comparison, and no speed-up is worth that:

`total_timesteps`, `grad_steps_per_iteration`, `dagger_batch_size`, `episodes_per_iteration`, `dagger_lr`, `weight_decay`, `ema_decay`, `buffer_capacity`, `efficiency_multiplier`, `seq_len`, `n_embd`, `n_head`, `n_layer`, `n_global_tokens`, `num_diffusion_steps`, `diffusion_steps_eval`, `diffusion_steps_collect`, `replan_every`, `offline_total_grad_steps`, `offline_batch_size`.

Also out of scope: the model architecture. The global stream embeds 2048 x 1659 = 3.4M glyph lookups per forward and is the memory-heavy part of the model, but shrinking it changes the checkpoint and the recipe. Leave it. Note it in your report if you want, do not act on it.

`100 grad steps x 2048 batch` is the compute budget the offline baseline is matched against. It is not a tuning knob.

---

## Phase 4: verification

Bitwise equality is not available: AMP and cuDNN kernel selection are non-deterministic on GPU. So verify what is verifiable.

1. **Full suite plus lint**: `uv run python -m pytest tests -q` and `uv run ruff check src tests`. The four ruff errors in `tests/test_failure_behaviour.py` and `tests/test_method_spec.py` are pre-existing, as are the eight in `scripts/profile_dagger.py`, which has never been in the lint scope. Do not fix any of them as part of this work.
2. **Loss trajectory**: run ~200 grad steps with a fixed seed before and after your changes, from the same seeded buffer, and compare the loss curves. They should track within noise. A systematic offset means a change altered the maths, not just its scheduling. C1 in particular must not change values: int16 to int64 and int16 to int32 widening are both exact for glyph IDs.
3. **A short real run**: one DAgger run to the first checkpoint, then compare `eval_iter*.json` win rates against `checkpoints/` from an earlier run at the same point. Different by more than seed noise means stop and investigate.
4. **Watch iteration 1 for env pool behaviour**: RSS, open file descriptors, `$TMPDIR` usage. The pool is the one change whose failure mode is gradual rather than immediate.

---

## Deliverable

You are handing work to prompt 2, so the deliverable is a branch plus a table, not a conclusion.

Report back, in the repo's evidence style (command and output, file and line):

- **The box you were actually on**, from the commands in the first section. Every number below is meaningless without it.
- A before/after table of the Phase 1 quantities, each labelled with that hardware.
- Per change: `keep` / `carry` / `revert`, the measurement behind it, and the commit sha.
- Peak VRAM at `dagger_batch_size: 2048`, or the OOM and at what batch you fell back to for ratios.
- The loss-trajectory comparison for every change that touches the training step, especially C1.
- Whether the Phase 1 decision gate passed, and if not, what it turned out to be.
- Anything you found that is not on this list, especially in the collection path, which is newly changed and least proven.

Do **not** report hours-per-seed as a plan. If you project it, label it as this box's figure and note that prompt 2 supersedes it.

Commit only when the suite and lint are green, one commit per change. Ask before committing; do not push.
