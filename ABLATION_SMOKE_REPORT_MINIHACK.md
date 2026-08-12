# MiniHack ablation suite: preventive smoke test

Input: `ABLATION_SMOKE_PROMPT_MINIHACK.md`. Started from HEAD `813e6ae`, tree clean.

> **Memory at `batch_size: 4608` on a 24 GB card is unverified and stays open.** This box is a
> 16 GB RTX 4070 Ti SUPER. Phase 3 probed every ablation at `--batch-size 512` on the CLI,
> which measures this card and says nothing about the 3090 Ti that runs the suite. That check
> belongs on the 24 GB card before the relaunch. See "What this does not clear" for a
> measurement at 4,608 on *this* card that is relevant for a different reason.
>
> **The final combined `results.json` holds 25 of 25 ablations.** Asserted, not inferred from
> the plots.

## Result in one line

One defect found, in `mixed_replay`, fixed in `27740ab`. Everything else was green first time.

---

## Preflight

| Check | Result |
|---|---|
| HEAD at start | `813e6ae`, tree clean |
| path contains whitespace | no (`/cs/student/project_msc/2025/dsml/mathweil/minihack_temp/minihack-ReMDM-planner`) |
| `$TMPDIR` | `/tmp/remdm-mathweil`, local disk, no whitespace |
| torch / CUDA | 2.13.0+cu126, available, RTX 4070 Ti SUPER |
| card | 16,376 MiB total (15,974 MiB usable per torch) |
| `pytest tests -q` | 161 passed |
| `ruff check experiments` | clean |
| registry | 25 ablations |

**Environments are correctly built, not the silently degraded default level.** The path check
alone is not enough, so a freshly constructed env of each ID env was inspected for a goal
staircase:

| Env | Goal glyphs in global map | Distinct glyphs |
|---|---|---|
| `MiniHack-Room-Random-5x5-v0` | 1 | 4 |
| `MiniHack-Room-Random-15x15-v0` | 1 | 4 |
| `MiniHack-Corridor-R2-v0` | 1 | 11 |
| `MiniHack-MazeWalk-9x9-v0` | 1 | 4 |

Each has exactly one staircase, and the glyph variety differs by env (Corridor 11 against 4),
so `mh_patch_nhdat.sh` produced distinct real levels rather than one default.

### Three prompt assumptions that do not hold in this clone

| Assumed | Actual |
|---|---|
| `/workspace/minihack-ReMDM-planner` | repo is at `/cs/student/project_msc/.../minihack_temp/minihack-ReMDM-planner`; runs wrote to a scratchpad, never to `experiments/rl_finetuning/outputs/` |
| `uv sync --extra cuda` | the extra is `cuda12` on this box; `cuda` does not exist |
| `RETRAIN_LOG.md`, `scripts/select_best_checkpoint.py`, `checkpoints/` | none of the three exist here |

**Checkpoint.** Since `select_best_checkpoint.py` is absent, the checkpoint was taken as the
W&B artifact `myopic-planner/minihack-ReMDM-planner/checkpoint-iter100:v0`, which is the
seed-0 retrained DAgger run (`retrain_fix1/online_s0/dagger_20260811_210823_ee39/iter100.pth`,
the only checkpoint in that run directory, so `<best>` is trivially `iter100`). It satisfies
the FIX-B4 constraint: `raw_model.load_state_dict(ckpt["ema_state_dict"])` succeeds strictly,
5,241,935 parameters against `configs/defaults.yaml`. No tolerant key loading was added.

---

## The analysis path, exercised first

`generate_summary_tables`, `generate_all_plots` and `generate_diagnosis_report` are called at
`run_ablations.py:702-704`, inside the per-ablation loop and outside the `try`, so a defect
there aborts the whole suite. It was exercised before the sweep, on the first ablation.

**Clean.** All plots, all CSV/TeX tables and `diagnosis.md` generated, including
`gradient_conflict_map.png`, `score_delta.png`, `per_env_delta.png` and
`diagnosis_decision_tree.png`. No matplotlib 3.11 removal bit: `plots.py:488` already uses
`tick_labels`, and nothing else in the analysis path uses a removed keyword.

One correction to the prompt's expectation. It predicts a skipped ablation "shows up as an
empty `ablations` dict". For a single-ablation run it produces **no `results.json` at all**:
the file is written after the `try` block, and the `continue` at `run_ablations.py:641` skips
both the write and the analysis. The count check is still the right check; the artefact is
absent rather than empty.

---

## Phase 1 and Phase 4: the 25, isolated

Each ablation in its own process, `--fast --num-seeds 1`, against
`final_ablations_ucl.yaml` (the hyperparameter-complete config), checkpoint as above. Phase 1
is the sweep before the fix, Phase 4 the re-run after it. Peak MiB is Phase 3, at
`--batch-size 512`, and is **reserved** as reported by `nvidia-smi`, not allocated: the
caching allocator holds a pool, so this is a ceiling and not a live footprint.

| Ablation | Phase 1 | Skipped? | nan/inf? | Phase 3 (batch 512) | Peak MiB reserved | Phase 1 s | Phase 4 s | Failure | Fix |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_rl` | pass | no | no | pass | 2344 | 23 | 23 | | |
| `kl_penalty` | pass | no | no | pass | 2584 | 22 | 22 | | |
| `ewc` | pass | no | no | pass | 2428 | 24 | 24 | | |
| `llrd` | pass | no | no | pass | 2338 | 23 | 22 | | |
| `lora` | pass | no | no | pass | 1878 | 24 | 25 | | |
| `mixed_replay` | **FAIL** | **yes** | no | pass | 2484 | 9 | 23 | ring-buffer overflow | `27740ab` |
| `trust_region_kl` | pass | no | no | pass | 2584 | 22 | 23 | | |
| `t_curriculum` | pass | no | no | pass | 2344 | 23 | 23 | | |
| `entropy_bonus` | pass | no | no | pass | 2344 | 23 | 23 | | |
| `gradient_surgery` | pass | no | no | pass | 2414 | 22 | 22 | | |
| `advantage_clip` | pass | no | no | pass | 2342 | 23 | 24 | | |
| `normalized_adv` | pass | no | no | pass | 2344 | 23 | 22 | | |
| `bc_wins` | pass | no | no | pass | 2344 | 22 | 22 | | |
| `low_t` | pass | no | no | pass | 2344 | 22 | 22 | | |
| `frozen_backbone` | pass | no | no | pass | 1110 | 21 | 21 | | |
| `head_only` | pass | no | no | pass | 1110 | 21 | 21 | | |
| `attention_only` | pass | no | no | pass | 1886 | 22 | 22 | | |
| `ffn_only` | pass | no | no | pass | 2038 | 22 | 21 | | |
| `layer_ablation_top1` | pass | no | no | pass | 1220 | 21 | 22 | | |
| `layer_ablation_top2` | pass | no | no | pass | 1550 | 22 | 22 | | |
| `layer_ablation_top3` | pass | no | no | pass | 1900 | 23 | 22 | | |
| `reward_filtering` | pass | no | no | pass | 2284 | 22 | 22 | | |
| `running_stats` | pass | no | no | pass | 2344 | 22 | 22 | | |
| `action_diversity` | pass | no | no | pass | 2344 | 24 | 23 | | |
| `reward_model` | pass | no | no | pass | 2404 | 23 | 23 | | |

Phase 1 total 548 s, Phase 4 total 561 s. The 9 s against 23 s on `mixed_replay` in Phase 1
is the tell: a swallowed failure is faster than a success, and it exited 0.

**`nan`/`inf` did not materialise anywhere.** The prompt flagged AMP overflow in an added
penalty term as the quiet failure mode, since it would complete and report a number. Every
Group A ablation (`ewc`, `kl_penalty`, `trust_region_kl`, `llrd`, `lora`) produced 50 finite
losses and 5 finite eval scores, checked numerically in `results.json` rather than by grepping
the log.

### The count check

The only check that proves the skip path never fired, since single-ablation runs each have
their own `results.json`:

```
$ uv run --no-sync python -c "import json; d=json.load(open('.../smoke_all/results.json'))['ablations']; \
    print(len(d), 'of 25'); assert len(d) == 25, sorted(set(ABL) - set(d))"
25 of 25
ASSERTION PASSED: no ablation was skipped
```

One combined `--all --fast --num-seeds 1` sweep, 491 s, exit 0, zero "FAILED - skipping" lines
and zero non-finite values in the log or the results.

---

## The defect

**`mixed_replay`, `MixedReplayBuffer.push`, `training.py:204-222`.**

```
File "experiments/rl_finetuning/ablations/training.py", line 217, in push
    self._local[:rest] = local_obs[first:]
RuntimeError: The expanded size of the tensor (500) must match the existing
size (561) at non-singleton dimension 0. Target sizes: [500, 9, 9].
Tensor sizes: [561, 9, 9].
```

**Root cause.** The wrap-around branch splits the incoming batch at the ring boundary: it
writes `first = capacity - start` rows at the end of the buffer and `rest = n - first` rows at
the front. That is correct only while `n <= capacity`. When one push carries more windows than
the buffer holds, `rest` exceeds `capacity`, so the destination slice `self._local[:rest]` is
clamped to `capacity` rows while the source still has `rest`, and the assignment raises.

Concretely: `ablations_fast.yaml:27` sets `mixed_replay_buffer_size: 500`, the first iteration
collected 1,061 windows, so `start = 0`, `first = 500`, `rest = 561` into a 500-row
destination.

**Fix.** An oversized push now keeps the most recent `capacity` windows, which is exactly what
the ring discipline would have left behind had the rows been written one at a time. No
hyperparameter, loss, seed or diagnostic changes.

**Exposure at the real config.** The new branch does not fire at
`final_ablations_ucl.yaml`: the buffer is 10,000 windows and a 500-iteration run measured
between 1,700 and 7,178 windows per iteration. It is reachable there in principle, since 30
episodes of up to 500 steps bound the count at 15,000, so this is a **latent** defect at the
real size and a **certain** one at the smoke size. It would have cost the `mixed_replay` arm
of any `--fast` validation run silently.

Three tests were added beside the existing round-trip, which only ever exercised a push that
fits: one that straddles the boundary, one larger than the whole buffer, and the same after a
partial fill so the overflow path runs with a non-zero write index.

### The Craftax defects, checked not ported

| Craftax defect (`2d41229`) | Here |
|---|---|
| per-layer grad-norm `lax.cond` leaf mismatch under LoRA | Not applicable. Eager PyTorch, `compute_per_layer_grad_norms` returns a name-keyed dict and skips `param.grad is None`. `lora` passed Phase 1 and Phase 3. Not investigated further, per the prompt. |
| LoRA merge dropped a `.reshape` | Analogue checked. `lora` runs clean and its per-ablation `torch.save(trained_model.state_dict(), ...)` writes a LoRA-shaped state dict, which is expected: `training.py` re-initialises the EMA after `apply_lora_to_model` for that reason, and `remove_lora_from_model` runs after the final eval. Nothing downstream reads those per-ablation checkpoints. |
| `ax.boxplot(labels=)` removed in matplotlib 3.11 | Already correct at `plots.py:488` (`tick_labels`). The rest of the analysis path was exercised end to end and generated every artefact. |

---

## Phase 3: memory at batch 512

Every ablation allocates and runs its buffers at `--batch-size 512 --max-iter 3 --eval-every 1`,
so the eval path's allocation is included. All 25 exit 0.

**Does deliver:** every ablation's buffers allocate and run at batch 512, and a peak reserved
VRAM ranking across the 25.

**Does not deliver:** any statement about `batch_size 4608` on 24 GB. That check stays open and
belongs on the 3090 Ti before the relaunch.

Ranking, highest first: `trust_region_kl` and `kl_penalty` 2,584 MiB, `mixed_replay` 2,484,
`ewc` 2,428, `gradient_surgery` 2,414, `reward_model` 2,404, the Group B and D arms 2,338-2,344,
`reward_filtering` 2,284, `ffn_only` 2,038, `layer_ablation_top3` 1,900, `attention_only` 1,886,
`lora` 1,878, `layer_ablation_top2` 1,550, `layer_ablation_top1` 1,220, `head_only` and
`frozen_backbone` 1,110.

The ordering is the informative part: the two arms that run the frozen reference model on the
same batch sit at the top, and the partial-training arms scale with how much of the network
they touch.

### What this does not clear

Separately from this task, `PERF_EXPERIMENTS_RESULTS_MINIHACK.md` measured the suite at its own
`batch_size: 4608` **on this 16 GB card**, and found `kl_penalty` and `trust_region_kl` OOM
there while the other 23 fit at up to 15,290 MiB. That is a figure from this card and does not
constrain the 24 GB box, where all 25 are expected to fit. It is recorded here only because it
identifies the same two arms this phase ranks highest, which is a consistent story rather than
a coincidence.

---

## Do not decide, report

### 1. `final_ablations_qmul.yaml` is not hyperparameter-complete

Verified exhaustively rather than spot-checked. **24 keys** are present in
`final_ablations_ucl.yaml` and absent from both `final_ablations_qmul.yaml` and
`configs/defaults.yaml`, so each resolves through a `getattr(cfg, key, default)` fallback in
code. Every one of the 24 currently agrees with the UCL value, so there is **no numerical
difference today**:

| Key | UCL value | Code default | Agree |
|---|---|---|---|
| `adv_clip_eps` | 0.2 | `0.2` | yes |
| `entropy_coef` | 0.01 | `0.01` | yes |
| `ewc_fisher_batches` | 20 | `20` | yes |
| `ewc_lambda` | 100.0 | `100.0` | yes |
| `kl_coef` | 0.1 | `0.1` | yes |
| `llrd_decay` | 0.9 | `0.9` | yes |
| `lora_alpha` | 16.0 | `16.0` | yes |
| `lora_rank` | 8 | `8` | yes |
| `mixed_replay_buffer_size` | 10000 | `10000` | yes |
| `mixed_replay_ratio` | 0.25 | `0.25` | yes |
| `return_weight_cap` | 5.0 | `5.0` | yes |
| `return_weight_floor` | 0.1 | `0.1` | yes |
| `reward_filter_percentile` | 75 | `75` | yes |
| `reward_model_depth` | 2 | `2` | yes |
| `reward_model_lr` | 0.001 | `1e-3` | yes |
| `reward_model_train_steps` | 50 | `50` | yes |
| `reward_model_width` | 64 | `64` | yes |
| `running_stats_ema_decay` | 0.99 | `0.99` | yes |
| `t_curriculum_end` | 0.2 | `0.2` | yes |
| `t_curriculum_start` | 0.8 | `0.8` | yes |
| `t_curriculum_steps` | 200 | `200` | yes |
| `t_max_low` | 0.2 | `0.2` | yes |
| `trust_region_kl` | 0.05 | `0.05` | yes |
| `win_threshold` | 0.5 | `0.5` | yes |

It is one default change away from silently diverging, and this matters because
`RETRAIN_LOG.md` (absent from this clone) is said to make the QMUL config the primary MiniHack
ablation run. **Changed nothing.**

`num_seeds` is the same class of gap in the other direction: QMUL sets 3, the UCL config sets
nothing and inherits 1 from `ablations_default.yaml:121`.

### 2. Two stale config comments

| File | Comment | Actual |
|---|---|---|
| `final_ablations_ucl.yaml:9` | "Full default batch_size (3072)" | `batch_size: 4608` at line 18, and the default is 3072, so the comment is wrong twice |
| `final_ablations_qmul.yaml:7` | "Larger batch (1024 vs 3072)" | `batch_size: 512` at line 16, which is smaller, and neither number appears |

**Flagged, changed neither.**

### 3. A question, not a decision

This prompt assumes the 4070 Ti is the test box and the real suite runs on the 24 GB card.
`PERF_EXPERIMENTS_PROMPT_MINIHACK.md`, run immediately before this, assumes the opposite: that
the full 25-ablation suite is to be launched **on this box**. Those two cannot both be true.
If the real suite is meant to run here, the UCL batch size does not fit for two arms and any
batch that does fit changes what every ablation measures. **Not begun. State which box the
suite launches on.**

---

## Diff

One commit, `27740ab`, on top of `813e6ae`:

```
 experiments/rl_finetuning/ablations/training.py |  15 ++++++++++++-
 tests/test_smoke_experiments.py                 |  68 ++++++++++++++++++++++++
```

`experiments/rl_finetuning/ablations/training.py`, in `MixedReplayBuffer.push`:

```python
        n = local_obs.shape[0]
        if n == 0:
            return
+       if n >= self.capacity:
+           # One iteration can collect more windows than the whole
+           # buffer holds. The wrap-around branch below splits the batch
+           # at the buffer boundary and would then try to write n - first
+           # rows into a destination of at most `capacity`. Keep the most
+           # recent `capacity` windows instead, which is exactly what the
+           # ring discipline would have left behind had they been written
+           # one at a time.
+           local_obs = local_obs[-self.capacity :]
+           global_obs = global_obs[-self.capacity :]
+           x0 = x0[-self.capacity :]
+           returns = returns[-self.capacity :]
+           n = self.capacity
        start = self._write_idx % self.capacity
```

Plus `test_mixed_replay_buffer_wraps_without_losing_rows`,
`test_mixed_replay_buffer_survives_a_push_larger_than_itself` and
`test_mixed_replay_buffer_handles_an_oversized_push_after_a_partial_fill`.

| Gate | Result |
|---|---|
| `pytest tests -q` | 164 passed |
| `pytest tests/test_failure_behaviour.py -q` | 7 passed, unweakened |
| `ruff check experiments` | clean |
| `ruff check src tests` | the 4 documented pre-existing errors, unchanged |

The `try/except` at `run_ablations.py:640` was not touched or weakened, and none was added
elsewhere.

---

## Wall clock and hardware

| | |
|---|---|
| Phase 1 sweep, 25 isolated | 548 s |
| Phase 3 memory probe, 25 | ~600 s |
| Phase 4 re-run, 25 isolated | 561 s |
| Phase 4 combined `--all` | 491 s |
| Hardware | RTX 4070 Ti SUPER 16 GB, i7-14700K, 20 cores / 28 threads, `outback.cs.ucl.ac.uk` |
| torch | 2.13.0+cu126 |
| HEAD at start | `813e6ae` |

A `--fast` ablation is 22 s here rather than the minutes the prompt anticipated, because
`PERF-X1` (commit `d7bf59e`, landed immediately before this task) replaced 30 environment
constructions per iteration with pool acquisitions, and MiniHack collection is exactly the
CPU-bound path that change addresses.

## Unresolved

- **Memory at `batch_size: 4608` on 24 GB.** Not testable here. Belongs on the 3090 Ti.
- **Which box the suite launches on.** See question 3 above.
- **`num_seeds` for the UCL config.** 1 by inheritance against 3 in QMUL.
- Nothing was left red. All 25 ablations pass all three checks, and the combined
  `results.json` holds 25 of 25.
