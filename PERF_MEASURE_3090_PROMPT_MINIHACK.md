# Task: certify the MiniHack DAgger speed-up on the 3090 Ti and produce the run plan

**This is prompt 2 of 2.** `PERF_GPU_PROMPT_MINIHACK.md` implemented and correctness-checked the changes on a development GPU (a 4070 Ti, 16 GB, i7-14700K). It deliberately did not decide anything that depends on hardware. This pass runs on the box the real training will use and settles those questions.

You are producing the numbers the paper's runs get planned from. Treat every figure that arrives from prompt 1 as provisional and re-measure it here.

## Where you are

`minihack-ReMDM-planner` on the target box: RTX 3090 Ti, 24 GB VRAM, i9-12900K (16 physical cores, 24 threads). A uv project: `uv sync`, run with `uv run`. Repo conventions are in `CLAUDE.md` one level up; UK English, no em dashes, evidence for every claim (command and output, file and line).

Confirm the box before anything else and record the output. If this is not a 3090 Ti with 24 GB, stop: this prompt has no meaning on other hardware, and prompt 1 is the one to run instead.

```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run python -c "import torch;p=torch.cuda.get_device_properties(0);print(p)"
lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)'
python -c "import tempfile;print('TMPDIR:',tempfile.gettempdir())" && df -h "$(python -c 'import tempfile;print(tempfile.gettempdir())')"
```

## What arrives from prompt 1

A branch with the collection fixes (already in the tree before prompt 1 started) plus up to six GPU-side changes as separate commits, each classified:

| Verdict | What it means for you |
|---|---|
| `keep` | Won on the dev box. Confirm it also wins here. |
| `carry` | Neutral or noise there, sound in principle. **You decide.** |
| `revert` | Already removed. Nothing to do. |

The `carry` set is the substance of this pass. Two classes of change were expected to understate on the dev box, for opposite reasons, and both should be re-measured here rather than trusted:

- **C3 (foreach EMA) and C4 (fused AdamW)** are kernel-launch-overhead fixes, so they are CPU-bound. The dev box has the faster per-core CPU, so these will have looked smaller there than they are here.
- **C1 (int16 transfers) and C6 (bf16)** are bandwidth-sensitive. This card has a 384-bit GDDR6X bus against the dev box's 192-bit, and less L2, so the balance shifts the other way.

Read prompt 1's report before starting. If a change was reverted for changing the loss trajectory, do not resurrect it.

---

## Preflight

```
git status --short && git log --oneline -12
uv sync
uv run python -m pytest tests -q
uv run python -m pytest tests/test_env_reuse.py -v
uv run ruff check src tests
ls checkpoints/
```

`tests/test_env_reuse.py` must be green on this box specifically. It pins that a recycled environment is trajectory-identical to a fresh one, which is an NLE-build-level claim, not a portable one. If it fails here, the collection change comes out regardless of what happened on the dev box.

Pre-existing lint: four errors in `tests/test_failure_behaviour.py` and `tests/test_method_spec.py`, eight in `scripts/profile_dagger.py`. Do not fix them.

---

## Phase 1: baseline, on this box, with the real config

Re-run prompt 1's Phase 1 protocol so the two boxes are directly comparable. Same script, same config, same seeds.

```
uv run python scripts/profile_dagger.py --config configs/final_ucl_gpu.yaml
```

`--config` was added as part of this work. Without it the script silently profiles `configs/defaults.yaml`, which is `use_amp: false`, `torch_compile: false`, `dagger_batch_size: 3584`: a different machine from the one the run uses. Check the echoed config before trusting any output. Note also that it forces `buffer_capacity: 500` against 10,000 in the real run, and that it does not isolate `make_env`.

Record, per iteration at `configs/final_ucl_gpu.yaml` (`episodes_per_iteration: 30`, `grad_steps_per_iteration: 100`, `dagger_batch_size: 2048`):

| Quantity | Where from |
|---|---|
| collect wall time, split env reset / env step / GPU inference | `DataCollector._last_profile` |
| train wall time, ms per grad step | wrap the 100-step loop |
| samples/s | `100 * 2048 / train_time` |
| peak VRAM | `torch.cuda.max_memory_allocated()` |
| env steps per iteration | `sum(model_steps) + sum(oracle_steps)` |
| collect / train split as a percentage | derived |

Do this twice: once at the merge-base commit that predates the GPU changes, and once at branch HEAD. That gives a clean before/after on identical hardware, which is the table this whole exercise exists to produce.

**Decision gate.** Collection should now be a small minority of iteration time. If it is above 5 s per iteration here, stop and diagnose before anything else: `$TMPDIR` on a network mount, the env pool being bypassed on some path, or model episodes far longer than the 152-step random-policy mean the projection assumed. Note that collection is mostly single-threaded env stepping, so it may be modestly *slower* here than on the dev box's faster cores. That is expected; a multiple is not.

---

## Phase 2: VRAM certification

This is the question only this box can answer, and one of the two reasons prompt 2 exists.

1. Peak allocated and peak reserved at `dagger_batch_size: 2048`, AMP on, `torch.compile` on, over at least 200 grad steps so allocator behaviour settles.
2. The same during a checkpoint eval, which runs `eval_episodes_per_env: 50` episodes in lockstep, holds 50 live envs, and is the memory high-water mark of the loop in practice.
3. Headroom against 24 GB, stated plainly.

`configs/final_ucl_gpu.yaml:18-19` claims "~6-8 GB peak" with "comfortable headroom". Nobody has verified that comment. Confirm or correct it, and if it is wrong, fix the comment in the same commit as your report.

While you are in that file, `final_ucl_gpu.yaml:20-21` justifies `num_collection_workers: 8` by this box's core count. That knob is dead on the CUDA path: `online.py:137` always takes the `collect_batch_gpu` branch, and `num_collection_workers` only feeds `collect_batch_parallel`, which CUDA never reaches. `DataCollector.__init__` still allocates 8 unused CPU deep copies of the model because of it. Correct the comment, and either wire the knob up or record in it that the value has no effect on GPU.

If prompt 1 reported an OOM at 16 GB, this is where it gets resolved. **Still do not lower `dagger_batch_size`.** It is pinned to the released `iter600` checkpoint recipe and to the compute-matched offline BC baseline; changing it invalidates the paper's comparison. If it will not fit in 24 GB either, that is a finding to report, not to engineer around.

---

## Phase 3: settle the `carry` set

For each `carry` change, and for each `keep` change you want to confirm:

1. Measure at branch HEAD.
2. Revert just that commit, measure again, restore.
3. Record the delta in ms per grad step and samples/s, over at least 200 steps after a warm-up (`torch.compile` pays its compile cost once; discard those steps).

Then decide `keep` or `revert` on this box's evidence, and say which. A change that is neutral here and neutral there gets reverted: it is code with no purchase.

Two things to check while you are in here, both cheap and both invisible on the dev box:

- **`torch.compile` fallback.** `try_compile` (`denoiser.py:317`) silently returns the uncompiled model when Triton cannot find `cc` or `gcc`, which is common on managed nodes. It logs a warning. Grep the log rather than assuming; `torch_compile: true` in the config means nothing if that fallback fired. If it did, either make a compiler visible or stop treating the flag as meaningful.
- **`os.cpu_count()` sizing.** `collect_batch_gpu` sizes its oracle thread pool as `min(n_episodes, os.cpu_count() or 4)`, which is 24 here against 28 on the dev box. Immaterial at 30 episodes, but note the actual value so the collection numbers are interpretable.

---

## Phase 4: validate against the recipe, not just the clock

Bitwise equality is unavailable: AMP and cuDNN kernel selection are non-deterministic on GPU. Verify what is verifiable.

1. **Loss trajectory.** ~200 grad steps at a fixed seed from the same seeded buffer, at the merge-base and at HEAD. They should track within noise. A systematic offset means a change altered the maths rather than its scheduling. C1 in particular must be exact: int16 to int32 and int16 to int64 widening are both lossless for glyph IDs.
2. **A real run to the first checkpoint.** `uv run python main.py --mode online --config configs/final_ucl_gpu.yaml --seed 0 --override checkpoint_dir=<scratch>`. Compare the resulting `eval_iter*.json` win rates against the existing `checkpoints/` evals at the same point. Different by more than seed noise means stop and investigate rather than proceeding to the full runs.
3. **The offline path too.** `offline.py` shares the batch-prep code C1 and C2 touch, and the BC baseline is 60,000 grad steps at batch 2048. Run at least to the first eval and confirm the loss curve and step time. It is half the paper's comparison and it is easy to forget because the speed complaint was about DAgger.
4. **Env pool behaviour over a long run.** RSS, open file descriptors, `$TMPDIR` usage across several iterations and at least one checkpoint eval. The pool holds instances alive; the cap bounds idle ones only, so peak live still tracks caller concurrency (30 in collection, 50 in eval). `REMDM_MAX_IDLE_ENVS` turns it down. This is the one change whose failure mode is gradual rather than immediate, which is exactly why a short run will not show it.

---

## Phase 5: the run plan

This is the deliverable the rest of the project is waiting on.

| Quantity | Before | After |
|---|---|---|
| s per iteration, collect / train split | | |
| iterations to `total_timesteps: 5650000` | | |
| hours per seed, `--mode online` | | |
| hours per seed, `--mode offline` (60,000 grad steps) | | |
| hours for 3 seeds, both modes | | |
| peak VRAM, headroom against 24 GB | | |

Derive the iteration count from measured env steps per iteration rather than assuming: it depends on episode length, which changes as the policy improves, so state it as a projection from early-training episode lengths and say so.

Then a go / no-go for launching the real runs, with whatever caveats you attach.

---

## What must not change

Pinned to the released `iter600` checkpoint recipe and to the compute-matched offline BC baseline. No speed-up is worth invalidating the comparison:

`total_timesteps`, `grad_steps_per_iteration`, `dagger_batch_size`, `episodes_per_iteration`, `dagger_lr`, `weight_decay`, `ema_decay`, `buffer_capacity`, `efficiency_multiplier`, `seq_len`, `n_embd`, `n_head`, `n_layer`, `n_global_tokens`, `num_diffusion_steps`, `diffusion_steps_eval`, `diffusion_steps_collect`, `replan_every`, `offline_total_grad_steps`, `offline_batch_size`.

`100 grad steps x 2048 batch` is the compute budget the offline baseline is matched against, not a tuning knob.

The model architecture is out of scope. The global stream embeds 2048 x 1659 = 3.4M glyph lookups per forward into a 435 MB fp32 activation and is the memory-heavy part of the model, but shrinking it changes the checkpoint and the recipe. Note it if you like; do not act on it.

---

## Deliverable

In the repo's evidence style (command and output, file and line):

- Box identification output, first, so every number below is anchored.
- The Phase 1 before/after table, measured at merge-base and at HEAD on this box.
- Phase 2 VRAM certification, and whether `final_ucl_gpu.yaml:18-19` was right.
- Per change: final `keep` or `revert`, this box's measurement, and how it compared with the dev box's. Call out explicitly any change where the two boxes disagreed, since that is the finding that justifies having run both.
- Phase 4 results: loss trajectories, the checkpoint-eval comparison against existing `checkpoints/`, the offline path, and the long-run pool behaviour.
- The Phase 5 run plan table and the go / no-go.

Commit only when the suite and lint are green. Ask before committing; do not push.
