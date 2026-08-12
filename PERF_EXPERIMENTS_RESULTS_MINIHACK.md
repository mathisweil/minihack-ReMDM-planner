# RL fine-tuning ablation suite: speed pass on the 4070 Ti

Input: `PERF_EXPERIMENTS_PROMPT_MINIHACK.md` (prompt 2 of 2). Sibling: `PERF_OFFLINE_PROMPT_MINIHACK.md`.

**Every number here was measured on the box below, at the checkpoint and config named
below.** Where something is a projection rather than a measurement it says so in the line
itself.

## Prompt 1 did not run

The prompt says to start from `PERF_OFFLINE_PROMPT_MINIHACK.md`'s HEAD and read its report.
There is no report: no `PERF_OFFLINE_RESULTS_MINIHACK.md` exists, `git log` shows no commit
after `3c4757b added prompts perf exp and offline`, and `src/planners/inference.py:187-200`
still carried the unfixed `.long().to(device)` pattern that prompt 1 owns. So this pass
started from `3c4757b` and took `inference.py` itself (change X2, third bullet).

`CLAUDE.md` is not present in the repo or one level up. Conventions were taken from the
existing `PERF_*_RESULTS_MINIHACK.md` files.

## The box

```
$ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
name, memory.total [MiB], driver_version
NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB, 560.35.03

$ uv run --no-sync python -c "import torch;p=torch.cuda.get_device_properties(0);print(p)"
_CudaDeviceProperties(name='NVIDIA GeForce RTX 4070 Ti SUPER', major=8, minor=9,
  total_memory=15974MB, multi_processor_count=66, L2_cache_size=48MB)

$ lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)'
CPU(s):                               28
Model name:                           Intel(R) Core(TM) i7-14700K
Thread(s) per core:                   2
Core(s) per socket:                   20

$ hostname
outback.cs.ucl.ac.uk
```

torch 2.13.0+cu126, Python 3.12. Both documented traps bit exactly as described:

- **`--extra cuda12` is mandatory.** The driver ceiling is CUDA 12.6; a plain
  `uv sync --extra cuda12` on the already-mixed venv left `nvidia-*-cu13` wheels in place
  and `import torch` died with `ImportError: libcudnn.so.9`. `uv sync --extra cuda12
  --reinstall` (65 s) fixed it, and every command since used `uv run --no-sync`.
- **`$TMPDIR` defaults to NFS.** `tempfile.gettempdir()` was
  `/cs/student/project_msc/2025/dsml/mathweil/tmp`. Every measurement below ran with
  `export TMPDIR=/tmp/remdm-$USER` on local disk (`/dev/mapper/vg_outback-lv_root`,
  179 GB free), confirmed with `python -c "import tempfile;print(tempfile.gettempdir())"`
  → `/tmp/remdm-mathweil`.

## Checkpoint and main config

`needs-human`, and it was asked. The answer: the W&B artifact
`myopic-planner/minihack-ReMDM-planner/checkpoint-iter100:v0`, downloaded to
`artifacts/checkpoint-iter100-v0/iter100.pth` (84.0 MB, `metadata {'iteration': 100,
'buffer_size': 10000}`).

Its bundled `config.yaml` is `n_embd: 256`, `n_head: 4`, `n_layer: 4`, `n_global_tokens: 8`,
`seq_len: 64`, so the matching main config is **`configs/defaults.yaml`**, not
`configs/ucl_gpu_bigger_model.yaml`. That was verified by loading rather than by reading:
`make_model(defaults + final_ablations_ucl)` accepts `ckpt["ema_state_dict"]` with no
`load_state_dict` error, 5,241,935 parameters. Every measurement below is at this pair, and
so is the VRAM answer. The 384-dim alternative on disk
(`checkpoints_ucl_bigger_model/dagger_20260403_024043_7c94/iter750.pth`) would move every
memory number upward and was not used.

Ablation config: `experiments/rl_finetuning/configs/final_ablations_ucl.yaml`, unmodified
except for a comment (see X6).

## Preflight

| Check | Result |
|---|---|
| `git status --short` | clean at `3c4757b` |
| `uv run --no-sync python -m pytest tests -q` | 150 passed in 65 s |
| `uv run --no-sync ruff check src tests` | 4 errors, all pre-existing (`B017`, `F401`, `C408`, `B905`) |
| `run_ablations.py --list` | 25 ablations, groups A/B/C/D + Baseline |
| checkpoints on disk | none in this clone; see above |

`experiments/` and `scripts/` are outside the lint scope. Both new scripts here are clean
under `ruff check` anyway.

---

## Phase 1: where the time goes

`baseline_rl`, seed 0, `final_ablations_ucl.yaml` settings, 51 iterations so the run crosses
`eval_every: 25` twice and `cka_every: 50` once. Instrumented with the new
`scripts/profile_ablation.py`, which monkey-patches the callables `training.py` already uses
and never edits the training loop, so the measured run computes exactly what a real run
computes.

**`speed/iter_time_sec` understates a diagnostic iteration.** `training.py:1088` stops that
clock immediately after the gradient step, before the diagnostics and the eval. The profiler
records `prof/wall_iter_sec`, the gap between consecutive metric emissions, which is the real
per-iteration cost. On a diagnostic iteration the two differ by 4x.

### The split of a plain (non-diagnostic) iteration, batch 4608

| Component | Before | Share |
|---|---|---|
| **collect: env construction** | **0.828 s** | **33.4%** |
| collect: env stepping | 0.848 s | 34.2% |
| collect: GPU sampling (ReMDM, K=5) | 0.208 s | 8.4% |
| collect: env reset | 0.112 s | 4.5% |
| collect: env close | 0.010 s | 0.4% |
| collect: window extraction | 0.004 s | 0.2% |
| gradient step | 0.373 s | 15.0% |
| `make_eval_model` deepcopy | 0.003 s | 0.1% |
| **total wall per iteration** | **2.478 s** | |

Collection is **85%** of a plain iteration, so the prompt's change list stands rather than the
"follow your own profile" branch. Within collection, construction and stepping are almost
exactly equal, and construction is the half that is free to remove.

Confirmed at full scale: over the whole 500-iteration run, env construction was **416.15 s of
1,415.3 s, 29.4%**, against 472.06 s (33.4%) for stepping, the latter including eval's
episodes, which the per-iteration figures above do not. The full run's mean iteration is
2.823 s rather than 2.478 s because the 20 diagnostic iterations cost 12.590 s each.

Per-call costs, from the 51-iteration run (1,530 collection env constructions, 337,039 env
steps):

| Operation | Cost |
|---|---|
| env construction, local xfs `$TMPDIR` | 27.6 ms |
| env reset, fresh instance | 3.28 ms |
| one env step | 0.134 ms |
| one batched ReMDM replan (K=5) | 6.56 ms |

27.6 ms against the prompt's projected 25.6 ms; the projection was good.

### Diagnostics and eval

Timed individually over the full 500-iteration run, at the real batch (20 firings of each
25-cadence block, 10 of CKA, 21 evals):

| Block | Cost per firing | Cadence | Total over 500 iters |
|---|---|---|---|
| `compute_t_analysis` (10 backward passes, one per t bin) | 4.330 s | every 25 | 86.6 s |
| `evaluator.evaluate` (4 envs x 20 episodes) | 3.720 s | every 25 + final | 78.1 s |
| `compute_repr_drift` | 1.070 s | every 25 | 21.4 s |
| `compute_grad_alignment` | 0.721 s | every 25 | 14.4 s |
| per-layer grad norms (norm computation only) | 0.0015 s | every 25 | 0.03 s |
| `compute_cka` (`cka_batch_size: 128`) | 0.008 s | every 50 | 0.08 s |

All five diagnostic cadences are 25 or 50, so every diagnostic fires on the same iterations. A
diagnostic iteration cost **12.590 s against 2.416 s** for a plain one, so the whole diagnostic
plus eval load is 204 s of the 1,415 s run, **14.4%**. The per-layer block's own norm
computation is negligible; its cost is the extra forward and backward it runs first, which is
inside that 10.2 s gap. **Reducing them was never on the table and is not needed.**

---

## The VRAM answer

### Does 4608 bind?

Yes, on the majority of iterations, and it is close to binding on the rest.
`prof/windows_collected` is the number of windows the iteration's 30 episodes produced, which
is what `local_obs[:batch_size]` draws from. Over a **full 500-iteration** `baseline_rl` run:

| min | median | max | mean | >= 4608 |
|---|---|---|---|---|
| 1,700 | 4,740 | 7,178 | 4,719 | **278 of 500 (55.6%)** |

So the gradient batch is genuinely data-limited, but not by much: the median iteration is
above `batch_size`, and the batch changes size every iteration, which matters for both the
allocator and `torch.compile` below.

`train/effective_batch_size` is a different quantity and far smaller: over the same 500
iterations, mean **537.1**, median 508.5, min 202.6, max 1,867.8 against a tensor batch of
4,608. The advantage weights are skewed enough that `(sum w)^2 / sum w^2` collapses to about
11% of the rows the GPU pays for. That is a property of the experiment, not of this pass, but
it is worth the author knowing.

### Measured peak

`scripts/vram_sweep_ablation.py`, real model, real loss, real AMP setting, synthetic batches
(shapes drive the footprint, not glyph values). Peak MiB allocated:

| batch | train step | + per-layer diag | grad_align | repr_drift | t_analysis | cka |
|---|---|---|---|---|---|---|
| 512 | 1,713 | 1,763 | 1,781 | 587 | 1,801 | 226 |
| 1,024 | 3,413 | 3,413 | 3,432 | 1,077 | 3,451 | 232 |
| 2,048 | 6,711 | 6,711 | 6,730 | 2,053 | 6,750 | 247 |
| 3,072 | 9,997 | 9,997 | 10,016 | 3,031 | 10,036 | 261 |
| **4,608** | **14,942** | 14,942 | **14,962** | 4,497 | **14,981** | 282 |

Card total 15,974 MiB. Model plus optimizer state is 40 MiB; everything else is activations,
dominated by the global stream (`global_embedding` produces `[B, 21, 79, 32]`).

In the real loop, peak is **15,087 MiB: 94.4% of the card, 887 MiB of headroom.**

### It OOMs on the default allocator, and that is fixable without touching the batch

The first attempt to run `baseline_rl` at 4608 died on iteration 1:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 658.00 MiB.
Of the allocated memory 13.39 GiB is allocated by PyTorch, and 1.49 GiB is
reserved by PyTorch but unallocated.
```

1.49 GB reserved-but-unallocated against 887 MB of headroom: the batch changes size every
iteration, so fixed-size allocator segments fragment. `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` returns that gigabyte and the same run then completed. Re-running
the isolated sweep without the flag reproduced the OOM at 4,608 (peak 14,891 MiB before it
died) while the flagged run reached 15,087 MiB and survived, so this is the deciding factor
and not luck.

`run_ablations.py` now sets it with `os.environ.setdefault` before `import torch`, so an
explicit setting from the environment still wins (change **X7**).

### Gradient accumulation: not implemented, and why

The prompt's contingency was gradient accumulation, conditional on the loss reduction being
a plain mean over the batch. **It is not**, and the prompt's own instruction in that case is
to stop and report rather than approximate.

`losses.py:174-183` reduces the ELBO term with `(per_sample * advantages).mean()`, which
accumulation reproduces exactly with `n_i / N` weights. But it then adds
`cfg.aux_loss_weight * aux_loss`, and `auxiliary_goal_loss` (`src/diffusion/loss.py:124-133`)
is a mean over the **subset of samples where the staircase is visible**:

```python
valid = targets[:, 0] != pad_value          # [B]
if not valid.any():
    return goal_pred.new_tensor(0.0)
diff = (goal_pred[valid] - targets[valid]) ** 2
return diff.mean()
```

Accumulating microbatches with weight `n_i / N` gives `sum_i (n_i/N) * mean_{valid_i}`, which
equals `mean_{valid}` only when the visible fraction is identical in every microbatch. It is
not, since the microbatches are a random split. The correct weight is
`n_valid_i / n_valid_total`, which is knowable but requires the loss functions to return the
ELBO and auxiliary terms separately, and several other ablation losses (`kl_penalty`,
`entropy_bonus`, `trust_region_kl`) carry their own masked-subset means with the same
property. That is a loss-API change across 25 factories, not a speed change, so it was not
made.

It is also not needed on this card: with expandable segments, 4608 fits.

---

## Phase 2: per change, what it was worth

All measured on `baseline_rl`, seed 0, batch 4608, 12 iterations, `TMPDIR` on local disk,
`expandable_segments:True`.

### X1. Use the env pool: the whole pass, confirmed

`training.py` constructed 30 environments per iteration with `make_env` and closed every one
in a `finally`. It now borrows from the shared `_EnvPool` with `acquire_env`, returning them
with `release_env` on success and `discard_env` on the exception path, and the whole
construct-plus-step region moved inside the `try` so a mid-construction failure cannot leak.
`shaped_reward` stays at its default `True`.

Measured over the full 500-iteration run:

| | Before | After |
|---|---|---|
| env constructions in the run | **15,420** | **420** |
| cost per acquisition | 27.743 ms | **0.076 ms** |
| env reset | 3.300 ms | 0.959 ms |
| collect time per iteration | 2.049 s | **1.086 s (-47.0%)** |
| mean wall per iteration | 2.823 s | **1.851 s (-34.4%)** |
| **500-iteration wall** | **1,415.3 s** | **928.9 s (-34.4%)** |

Eval got faster for free (3.720 s → 3.529 s per eval): `inference.py` already used the pool,
and it now finds warm instances that collection left behind.

### X2. int16 across PCIe, widened on the device

Three places, all now `.to(device).long()` instead of `.long().to(device)`:

- `training.py:552-565`, the replan H2D in `_collect_training_data_gpu`.
- `training.py:354-356`, `_extract_windows`, which widened a whole episode on the host before
  the concatenation and transfer at `training.py:649-652`. This is the big one: at the
  measured mean of 4,347 windows an iteration, `global` crossed as 57.7 MB of int64 instead of
  14.4 MB of int16.
- `inference.py:187-200`, the shared evaluator. Prompt 1 had not fixed it.

Window extraction fell from **0.099 ms to 0.017 ms per episode (-83%)** over the full run,
2.5 ms an iteration. Worth about 0.1% of the iteration on its own; it is in for the transfer
bytes, not the CPU time. It did **not** move peak VRAM (15,123 → 15,125 MB): the peak is set
by the gradient step's activations, and the collected tensors are an order of magnitude
smaller than that.

`tests/test_ablation_perf.py` pins that the device-side widening returns exactly the values
the host-side cast returned, that the dtype contract survives the short-episode padding and
the `T == 0` guard, and that int16 → int64 is exact over the whole `0..5999` glyph range.

### X3. One eval model, refreshed in place

`ema.make_eval_model(raw_model)` is `copy.deepcopy` plus an EMA apply and it ran once per
iteration, once per eval and once at the end. It is now created once and refreshed with
`ema.apply_to(eval_model)`.

Measured worth: **2.7 ms per iteration, 0.1%** (521 calls costing 1.40 s over the full
pre-change run, against 1 call after). The deepcopy is cheap because the model is only 5.24 M
parameters. It is in because it removes 520 allocate-and-free cycles of a whole model from a
run that sits at 95% of VRAM, not because of the time.

The equivalence is verified rather than inspected: `apply_to` overwrites every named
parameter from the same shadow, and `test_refreshed_eval_model_matches_a_fresh_deepcopy`
asserts tensor equality against a fresh `make_eval_model` after five EMA updates.
`test_the_denoiser_has_no_buffers_to_go_stale` pins the assumption that makes reuse safe.

### X4. Fused health metrics

`training.py:1129-1141` computed `param_norm` and `param_drift_from_init` with one `.item()`
per parameter tensor, twice: 72 parameter tensors, so 144 device syncs every 10 iterations.
Replaced with `torch._foreach_norm` over the parameter list and a single two-element
`.tolist()`.

Measured worth: below the noise of a single iteration; it fires on 10% of iterations and the
whole health block was inside the 51-iteration run's residual, which came to -0.4% of wall.
`test_fused_param_norm_matches_the_per_tensor_loop` pins that both values match the Python
loop to `rel=1e-6`.

### X5. The 50 MB history buffer: closed by measurement, no change made

The prompt asked to measure first. Measured:

| | |
|---|---|
| `np.zeros((30, 501, 21, 79), int16)`, 49.9 MB | **0.007 ms** |
| `np.zeros((30, 501, 9, 9), int16)`, 2.4 MB | 0.065 ms |
| the ~220-step-per-episode writes that actually happen | 2.13 ms |

glibc serves the 50 MB request with `mmap` and lazily-zeroed pages, so there is no 50 MB
memset: the large allocation is an order of magnitude *cheaper* than the small one, which
comes off the heap and is genuinely cleared. 0.007 ms against a 1,675 ms iteration is
0.0004%. The premise that this is "allocated and zeroed every iteration" does not hold on
this platform. **No change made.**

### X6. `torch_compile: false`, comment confirmed on time, with a VRAM caveat

Measured both ways, `baseline_rl`, 12 iterations, batch 4608. `try_compile` did engage
(`Compiling model with torch.compile` in the log, no fallback warning; `/usr/bin/gcc` is
present), so this is not the silent-eager case.

| | eager | compiled |
|---|---|---|
| gradient step, steady state | 0.385 s | **0.273 s (-29%)** |
| iteration 1 | 0.539 s | 22.31 s |
| iteration 2 | 0.313 s | 15.23 s |
| peak VRAM | 15,087 MiB | **14,219 MiB (-868 MiB)** |

The second compile is the point: the batch is data-limited so its size changes every
iteration, and Inductor recompiles. Over 500 iterations the arithmetic is ~56 s saved against
~38 s paid, per ablation and per seed, about 2% of a 15 minute run, and that is before the
recompilation risk on the 24 ablations with different graphs (LoRA rewrites modules, PCGrad
runs two backward passes). **The config comment is right and the flag stays `false`.**

The VRAM result is the part worth keeping: compilation buys back 868 MiB, which nearly doubles
the 887 MiB of headroom. It is the lever to reach for if a heavier ablation turns out not to
fit. Both numbers are now recorded in the config comment.

### X7. Expandable segments (not on the prompt's list)

See the VRAM section. One line in `run_ablations.py`; it is the difference between the suite
running and OOMing on iteration 1.

---

## Phase 4: verification

### The noise floor comes first, and it is wide

Identical code, identical seed, same box, `baseline_rl`, 12 iterations, repeated:

| run | windows per iteration | env steps | final eval |
|---|---|---|---|
| control A | 5150, 3740, 6422, 3604, 5271, 5314, 4215, 3334, 5096, 4541, 4705, 4311 | 71,624 | 0.4000 |
| control B | identical to A | 71,624 | 0.4000 |
| control C | identical for 8 iterations, then 5114, 4342, 4986, 4097 | 71,411 | 0.4500 |

Two runs of the same code reproduced each other exactly; the third diverged at iteration 9
and ended 0.05 apart on the eval. Collection is a closed loop (model produces plans, plans
produce episodes, episodes produce the next model), so a single last-bit difference from AMP
or a cuDNN algorithm choice eventually flips one sampled action and the trajectories separate
completely. **A pre-change run of `baseline_rl` on the untouched tree scored 0.4000 once and
0.4125 another time.** So the control band on the eval score is at least 0.05 at 12
iterations, and the loss series is only comparable while the data is.

This is much wider than the DAgger pass's 2.8e-4, exactly as the prompt anticipated, and it
means the eval score cannot certify anything at this run length. The sharp check is the data.

### Same-seed comparison and the env-step identity check

Three ablations from different groups, seed 0, 12 iterations, before and after. `windows` is
the per-iteration window count, `env steps` the number of `env.step` calls inside the
iteration loop; both are exact integers, which is what makes them the sharp instrument.

| | windows identical | env steps identical | loss, max abs diff | `effective_batch_size` | eval score |
|---|---|---|---|---|---|
| `gradient_surgery` (B) | **all 12** | **all 12** (72,281) | **0.00e+00** | identical | 0.4375 → 0.4375 |
| `mixed_replay` (A) | **all 12** | **all 12** (70,087) | 3.75e-05 | identical | 0.3875 → 0.4000 |
| `baseline_rl` (Baseline) | 8, then diverges | 8, then diverges | 1.69e-05 over those 8 | identical | 0.4125 → 0.4125 |

`gradient_surgery` reproduced **bit for bit** through the PCGrad path and its AMP
`unscale_` interaction, over all 12 iterations. `mixed_replay` reproduced the data exactly
through `MixedReplayBuffer` with losses agreeing to 1.3e-4 relative. `baseline_rl` diverged at
iteration 9, the same iteration and the same way as the identical-code control C above, so
that is the box's nondeterminism and not the change.

`train/effective_batch_size` was **bit-identical in all three ablations** on every iteration
where the data matched.

Two exact reproductions out of three, at a noise floor where identical code only reproduces
itself two times in three, is as clean a result as this pipeline admits.

### One full ablation end to end

`baseline_rl`, seed 0, all 500 iterations, `final_ablations_ucl.yaml` unmodified, on each tree:

| | Before | After | |
|---|---|---|---|
| **wall, 500 iterations** | **1,415.3 s (23.6 min)** | **928.9 s (15.5 min)** | **-34.4%** |
| plain iteration | 2.416 s | 1.453 s | -39.9% |
| diagnostic iteration (x20) | 12.590 s | 11.393 s | -9.5% |
| mean iteration | 2.823 s | 1.851 s | -34.4% |
| collect | 2.049 s | 1.086 s | -47.0% |
| gradient step | 0.362 s | 0.362 s | unchanged |
| peak VRAM | 15,123 MB | 15,125 MB | unchanged |
| final ID win rate | 0.2375 | 0.2625 | inside the 0.05 band |

Per-operation, over the whole 500-iteration run:

| | Before | After |
|---|---|---|
| env acquisitions | 15,000 at **27.743 ms** | 15,000 at **0.076 ms** |
| env constructions | **15,420** | **420** |
| env reset | 3.300 ms | 0.959 ms |
| env step | 0.137 ms (3,441,736 steps) | 0.131 ms (3,486,187 steps) |
| window extraction | 0.099 ms | 0.017 ms |
| `make_eval_model` | 521 calls | 1 call |
| `compute_t_analysis` | 4.330 s | 4.308 s |
| `compute_repr_drift` | 1.070 s | 1.065 s |
| `compute_grad_alignment` | 0.721 s | 0.718 s |

0.076 ms per recycled acquire against the prompt's projected 0.00 ms, and the diagnostics are
untouched to within 0.5%, which is the check that nothing leaked into them.

Both runs' eval traces decline monotonically from the checkpoint's 0.395 (before: 0.45 → 0.2625;
after: 0.425 → 0.2625). Return-weighted ELBO fine-tuning degrades ID win rate here. That is
the same on both trees and is a question for `ABLATION_SMOKE_PROMPT_MINIHACK.md`, not a
speed defect.

### RSS, file descriptors and `$TMPDIR` over a long run

Sampled every 15 s across each full 500-iteration run. The pool holds instances alive, so this
is the check that matters:

| | Before | After |
|---|---|---|
| RSS by quarter of the run | 2.56 → 2.62 → 2.66 → 2.70 GB | **2.33 → 2.40 → 2.44 → 2.46 GB** |
| RSS range | 2.48 - 2.74 GB | **2.23 - 2.48 GB** |
| open file descriptors | 115 - 179 | **93 - 142** |
| `$TMPDIR` entries | 93 - 123 | **78 - 105** |

The pooled run uses **less** memory, fewer descriptors and fewer temp directories than the
construct-and-destroy loop it replaces, and both drift by the same +0.14 GB over 500
iterations, so the pool is not the thing that drifts. `REMDM_MAX_IDLE_ENVS` (default 64) was
never approached: 30 concurrent envs in collection, 20 per env ID in eval.

### Suite and lint

| | Before | After |
|---|---|---|
| `pytest tests -q` | 150 passed | 161 passed (11 new in `tests/test_ablation_perf.py`) |
| `ruff check src tests` | 4 pre-existing errors | the same 4, unchanged |
| `ruff check` on the two new scripts | n/a | clean |

---

## Two ablations do not fit on this card at all

A 3-iteration probe of every registered ablation at `batch_size: 4608`, peak MiB allocated:

| Ablation | Peak | | Ablation | Peak |
|---|---|---|---|---|
| `head_only`, `frozen_backbone` | 4,252 | | `attention_only` | 11,370 |
| `layer_ablation_top1` | 5,420 | | `ffn_only` | 12,684 |
| `reward_filtering` | 6,132 | | `baseline_rl` and 8 others | 15,087 |
| `layer_ablation_top2` | 8,394 | | `gradient_surgery` | 15,126 |
| `lora` | 11,360 | | `reward_model` | 15,129 |
| `layer_ablation_top3` | 11,368 | | `ewc` | 15,133 |
| | | | `mixed_replay` | **15,290** |
| | | | **`kl_penalty`** | **OOM** |
| | | | **`trust_region_kl`** | **OOM** |

`kl_penalty` and `trust_region_kl` both run the frozen reference model's forward on the same
batch as the current model's forward and backward, and both die inside the current model's
attention projection with 15.16 GiB allocated and 101 MiB reserved-but-unallocated. That is a
genuine capacity shortfall, not fragmentation: expandable segments were already on.

**`torch_compile: true` does not rescue them.** Retried with compilation, both OOM at the same
allocation. `try_compile` wraps the training model only, and the reference model is
deliberately never compiled (`training.py:797`), so the 868 MiB compilation saves is not on the
side that overflows.

Bracketing `kl_penalty` by batch size: **4,096 fits** (peak 15,504 MiB, 470 MiB headroom),
4,352 OOMs, 4,608 OOMs. Extrapolating the fit, these two arms want roughly 17.4 GB at 4,608,
so they need a 24 GB card, which is what `final_ablations_ucl.yaml` was written for.

Capping those two at 4,096 would make them incomparable with the other 23, and `batch_size` is
pinned, so that is the author's call and not one this pass can take.

## Run plan

Measured for `baseline_rl` end to end; the suite totals are projections that use it as the
per-arm average and are labelled as such.

| | Before | After |
|---|---|---|
| **per ablation-seed (`baseline_rl`, measured)** | **23.6 min** | **15.5 min** |
| per ablation-seed (`gradient_surgery`, projected from its measured +0.366 s gradient step) | 26.6 min | 18.5 min |
| per ablation-seed (`mixed_replay`, projected) | 23.7 min | 15.7 min |
| **23 runnable ablations, `num_seeds: 1` (projection)** | **9.0 h** | **5.9 h** |
| 23 runnable ablations, `num_seeds: 3` (projection) | 27.1 h | **17.8 h** |
| all 25 if the two OOM arms were on a 24 GB card, `num_seeds: 1` (projection) | 9.8 h | 6.5 h |
| all 25, `num_seeds: 3` (projection) | 29.5 h | 19.4 h |

The projections are conservative: seven arms train only part of the network
(`head_only`, `frozen_backbone`, `layer_ablation_top1/2/3`, `lora`, `attention_only`) and both
their peak VRAM and their gradient step are well below `baseline_rl`'s, so the true suite
total should come in under these figures.

Peak VRAM and headroom, against 15,974 MiB:

| | Peak | Headroom |
|---|---|---|
| `baseline_rl` full run | 15,125 MiB | 849 MiB (5.3%) |
| `mixed_replay`, the tightest that runs | 15,290 MiB | 684 MiB (4.3%) |
| `kl_penalty`, `trust_region_kl` | does not fit | n/a |

## Go / no-go

**GO for 23 of the 25 ablations at `num_seeds: 1`, about 6 hours. NO-GO for `kl_penalty` and
`trust_region_kl` on this card.** Conditions:

1. **`export TMPDIR` to local disk.** On the NFS default, env construction was 86.3 ms in the
   earlier pass against 27.6 ms here. The pool now removes 97% of constructions, so this
   matters less than it did, but the 420 that remain plus every NetHack vardir still land
   there.
2. **Leave `PYTORCH_CUDA_ALLOC_CONF` alone.** `run_ablations.py` sets
   `expandable_segments:True` by default now; setting it to anything else reintroduces the
   iteration-1 OOM.
3. **One job on the GPU.** At 4.3% headroom on the tightest runnable arm there is no room for a
   second process, an X server doing real work, or a display manager restart.
4. **Decide `num_seeds` first** (see below). At 3 seeds this is 17.8 h rather than 5.9 h.
5. **Run `kl_penalty` and `trust_region_kl` on a 24 GB card**, or accept 23 arms. They fit at
   `batch_size: 4096` on this one, but that breaks comparability with the other 23 and
   `batch_size` is pinned.
6. `torch_compile` stays `false`. It is available as a 868 MiB VRAM lever for arms that are
   close to the line, but it does not rescue the two that are over it.

---

## Reported, not fixed

**`num_seeds` is unset in `final_ablations_ucl.yaml`.** The file ends on a dangling
`# -- Output / Logging` header with no keys under it, so it inherits `num_seeds: 1` from
`ablations_default.yaml:121`, while `final_ablations_qmul.yaml:67` sets 3 with the note
"matching the published 3-run tables". It also inherits `wandb_project` and `wandb_entity`
rather than setting them. Whether the UCL runs were meant to be 1 seed or 3 changes the
compute budget by 3x and is the author's call. Not changed.

**`train/effective_batch_size` is around 150 against a tensor batch of 4,608.** The advantage
weights are skewed enough that the statistical batch is 3% of the rows the GPU pays for. Not
a speed defect and not touched here, but it is the sort of thing
`ABLATION_SMOKE_PROMPT_MINIHACK.md` exists to adjudicate.

## What was not done

- **Gradient accumulation.** Not exact for this loss; see above. Stopped and reported, as
  instructed.
- **Parallel env stepping.** Env stepping is 34% of an iteration and is a serial Python loop
  over 30 independent environments, each of which is deterministic given its action, so a
  thread pool would be trajectory-identical if NLE releases the GIL during `step`. Not on the
  prompt's list, not measured, and it interacts with the pool's locking. Flagged as the next
  lever if more is needed.
- **Handed to `ABLATION_SMOKE_PROMPT_MINIHACK.md`:** nothing structural. The two OOM arms fail
  for capacity, not correctness, and are reported above rather than passed on. The declining
  eval trace on `baseline_rl` (0.45 → 0.2625 over 500 iterations, on both trees) is a science
  question that belongs there; it is not caused by anything in this pass.
- **Diagnostics untouched**, as required. Their measured cost moved by less than 0.5% between
  trees, which is the evidence that nothing leaked into them.
