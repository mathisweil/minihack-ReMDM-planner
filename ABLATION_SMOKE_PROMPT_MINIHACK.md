# Task: smoke-test and repair every ablation in the MiniHack suite

## Where you are

`/workspace/minihack-ReMDM-planner` on the GPU box, a clone of `mathisweil/minihack-ReMDM-planner`. A uv project: `uv sync`, run with `uv run`.

**This box is an RTX 4070 Ti, 16 GB VRAM. The suite it is testing is configured for 24 GB** (`final_ablations_ucl.yaml`, `batch_size: 4608`, AMP on). You are here to find code defects, which are hardware independent. You cannot certify memory behaviour for the real run from this card, and you must not re-size anything to make it fit. Phase 3 says what is and is not in scope.

**This is preventive, not a post-mortem.** Nothing has failed here yet. The sibling Craftax suite died at ablation 5 of 25 and cost a night of GPU, and three defects were fixed there in `craftax-ReMDM-planner` commit `2d41229`. This task checks the MiniHack suite before it is launched, not after.

Record the HEAD sha before you start and confirm the tree is clean. There is no fix to pull. If the tree is dirty, stop and report it rather than discarding anything.

---

## The failure mode here is different, and quieter

`run_ablations.py:613-645` wraps the per-ablation seed loop in `try/except Exception`, logs `Ablation '%s' FAILED — skipping to next.`, and continues. A broken ablation does not stop the suite.

So the MiniHack failure is not a dead run. It is a suite that **finishes with fewer than 25 entries while looking complete**: `results.json` holds only what survived, and the analysis stage builds tables, plots and the diagnosis report from that. A missing arm does not announce itself in a figure.

Two consequences that shape this whole task:

1. **The check that matters most is a count.** After any run, assert `results.json` holds 25 ablations. Never conclude from the plots that the suite is whole.
2. **The analysis stage is the one thing that can still kill the run.** `generate_summary_tables`, `generate_all_plots` and `generate_diagnosis_report` are called at `run_ablations.py:697-699`, inside the per-ablation loop and **outside** the `try`. An analysis defect therefore aborts the entire suite at ablation 1. In Craftax the equivalent defect, `ax.boxplot(labels=)` removed in matplotlib 3.11, only surfaced after the whole suite had run. Here it would surface immediately and destroy everything after it. Exercise the analysis path first, before the sweep, with `--analyze-only` against any existing `results.json`.

---

## Do not port the Craftax bug hunt blindly

| Craftax defect (`2d41229`) | Transfers here? |
|---|---|
| per-layer grad-norm `lax.cond` branches disagreed under LoRA, 161 leaves against 113 | **No.** This repo is eager PyTorch. `compute_per_layer_grad_norms` (`diagnostics/gradient.py:109`) returns a dict keyed by parameter name and skips `param.grad is None`. There is no fixed-size array and no branch to mismatch. Do not go looking for it. |
| LoRA merge dropped the `.reshape` its sibling had | **Check the analogue.** LoRA here changes the model's `state_dict` keys, and `training.py:787` re-initialises the EMA for exactly that reason. Anything downstream that assumes base keys after LoRA is the same class of defect: the per-ablation `torch.save(trained_model.state_dict(), ...)` writes a LoRA-shaped state dict for that one ablation. |
| `ax.boxplot(labels=)` | **Already correct** at `analysis/plots.py:488`, which uses `tick_labels`. Sweep the rest of the analysis path for other matplotlib 3.11 removals, given it runs after every ablation. |

What is genuinely MiniHack-shaped, and where to look instead:

- **`requires_grad = False` ablations.** `frozen_backbone`, `head_only`, `attention_only`, `ffn_only`, `layer_ablation_top1/2/3` freeze parameters. Anything assuming a populated `.grad`, a non-empty optimiser param group, or a fixed parameter count is the defect shape here.
- **AMP is on** (`use_amp: true`). An fp16 overflow in an added penalty term (EWC, trust region, KL) produces a `nan` loss, not an exception. The `try/except` will not catch it, the ablation will complete, and it will report a number. Check the logged loss for `nan`/`inf` in every Group A ablation rather than trusting exit code 0.
- **Silent env degradation.** See the path caveat in preflight. It does not raise.

---

## Preflight

All of it, before you start. Report and stop on any failure rather than working around it.

```
cd /workspace/minihack-ReMDM-planner && git status --short && git log --oneline -1
uv sync && uv sync --extra cuda
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv
pwd | grep -q ' ' && echo "PATH HAS SPACES - STOP" || echo "path ok"
ls checkpoints/
```

**The path check is not a formality.** MiniHack's `mh_patch_nhdat.sh` interpolates paths unquoted and fails silently on whitespace, leaving every environment as the same default level with no goal staircase. `src/envs/minihack_env.py` detects this and substitutes a Python implementation, but a run on a whitespace path that slipped through would produce plausible numbers from the wrong levels. Confirm both that the path is clean and that a freshly built environment actually has a goal.

**The checkpoint is a hard requirement and a real constraint.** `training.py:740-741` does `torch.load(...)` then `raw_model.load_state_dict(ckpt["ema_state_dict"])`, a strict load of that exact key. FIX-B4 dropped every legacy-compatibility read, so a pre-FIX-1 checkpoint is **expected to fail loudly, by design**. Per `RETRAIN_LOG.md` section 3 the suite consumes the seed-0 retrained DAgger checkpoint, `<RUN_DIR_online_s0>/iter<best>.pth`, with `<best>` computed mechanically by `uv run python scripts/select_best_checkpoint.py <RUN_DIR>`. If only pre-FIX-1 checkpoints are on this box, report that and stop. Do not restore tolerant key loading to make a smoke test run: that would delete a deliberate safety property to test a suite that is not ready.

---

# Phase 1: structural sweep, all 25, isolated

Each ablation runs in its own process. That is not for isolation from crashes, which the `try/except` already provides, but so that one log file maps to one ablation and a failure cannot be lost in 25 ablations of output.

`--fast` (`run_ablations.py:494-499`) loads `configs/ablations_fast.yaml` as an override layer: `max_iter 50, batch_size 128, episodes_per_iter 2, grad_steps_per_iter 1, eval_every 10, eval_episodes 5`, diagnostics every 10 (`cka` 25), `ewc_fisher_batches 5`, `reward_model_train_steps 10`, `mixed_replay_buffer_size 500`. Every conditional diagnostic therefore fires within the 50 iterations, which is what makes this a structural test and not a shortcut. Merge order is main config, `--ablations-config`, fast, CLI (`run_ablations.py:508`), so CLI overrides survive `--fast`.

Pass `final_ablations_ucl.yaml`, not the QMUL config: it is the hyperparameter-complete one, and `--fast` overrides only sizes on top of it. See the last section for why that distinction matters.

**Wall clock will not resemble Craftax.** MiniHack collection is CPU-bound through NLE, not GPU-bound. Each ablation collects 50 x 2 episodes and evaluates 5 episodes across 7 environments (4 ID, 3 OOD) every 10 iterations. Time the first ablation and extrapolate before assuming the sweep fits in your session.

Verify the names against `--list` and every flag against `--help` before starting. `--use-wandb` here is `store_true` with default `False`, so W&B is disabled by omitting it. There is no `--no-use-wandb` and no `--ppo-checkpoint`; both exist in the Craftax repo and neither exists here.

```
cd /workspace/minihack-ReMDM-planner
mkdir -p logs/smoke

CKPT=<RUN_DIR_online_s0>/iter<best>.pth
CFG=experiments/rl_finetuning/configs/final_ablations_ucl.yaml

ABL="baseline_rl kl_penalty ewc llrd lora mixed_replay trust_region_kl t_curriculum \
entropy_bonus gradient_surgery advantage_clip normalized_adv bc_wins low_t \
frozen_backbone head_only attention_only ffn_only layer_ablation_top1 \
layer_ablation_top2 layer_ablation_top3 reward_filtering running_stats \
action_diversity reward_model"

for a in $ABL; do
  SECONDS=0
  uv run python experiments/rl_finetuning/run_ablations.py \
    --checkpoint "$CKPT" --ablations-config "$CFG" \
    --ablations "$a" --fast --num-seeds 1 \
    --output-dir /workspace/smoke/"$a" 2>&1 | tee logs/smoke/"$a".log
  echo "$a exit=${PIPESTATUS[0]} secs=$SECONDS" | tee -a logs/smoke/status.txt
done
```

Run these loops under `bash`. `${PIPESTATUS[0]}` is not optional and is bash syntax: `$?` after a pipe reports `tee`, and zsh spells the array differently.

**Exit code 0 is not a pass here.** The `try/except` means a failed ablation still exits 0. Grep every log for the skip message and for non-finite losses before calling anything green:

```
grep -l "FAILED — skipping" logs/smoke/*.log
grep -liE "\bnan\b|\binf\b" logs/smoke/*.log
for a in $ABL; do
  uv run python -c "import json; print('$a', list(json.load(open('/workspace/smoke/$a/results.json'))['ablations']))"
done
```

`results.json` is `{"pretrained_score", "config", "ablations": {name: ...}}` (`run_ablations.py:227-234`), so a skipped ablation shows up as an empty `ablations` dict rather than an error.

Run the whole sweep before fixing anything. The full failure set up front is worth more than the first failure early.

---

# Phase 2: repair, one ablation at a time

For each failure, in registry order:

1. Read the traceback in `logs/smoke/<name>.log`. Identify the defect at file and line. The traceback will be there even though the process exited 0, because the handler uses `logger.exception`.
2. State the root cause before editing. For a `requires_grad` or `state_dict` key problem, give the parameter names on both sides.
3. Apply the smallest fix that is correct. Prefer consolidating a duplicated implementation over patching one copy.
4. Re-run that one ablation with the Phase 1 command. Do not proceed while it is still red, and re-apply the three checks above rather than reading the exit code.
5. The full test suite and `uv run ruff check experiments/` must both pass. **`tests/test_failure_behaviour.py` matters especially**: it exists to prove FIX-B1 to FIX-B4 fail loudly rather than silently degrading. A fix that swallows an error will break it, and breaking it is the signal that the fix is wrong, not that the test is.
6. Commit that fix alone, with a message stating the failure and the fix. Do not push. The author reviews before anything reaches `main`.

---

# Phase 3: memory behaviour, at the largest size this card can honestly test

`--fast` runs at batch 128, so it cannot exercise the buffers that `ewc` (Fisher diagonal), `lora` (adapters plus optimiser state) and `mixed_replay` (transition buffer) carry at scale.

**Do not run the UCL config at its own batch size on this card.** It is `batch_size: 4608` with AMP, sized for a 24 GB 3090 Ti. An OOM here would measure this 4070 Ti and say nothing about the machine that runs the suite.

**Do not switch to `final_ablations_qmul.yaml` to get a smaller batch.** Unlike the Craftax pair, these two configs are not the same config at two sizes: the QMUL one omits the entire Group A, B and D hyperparameter block. Swapping it in would quietly change what several ablations do. Instead keep the UCL config and override the single size knob on the CLI, which leaves every ablation hyperparameter identical:

```
for a in $ABL; do
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > /tmp/vram_$a.txt &
  SAMPLER=$!
  uv run python experiments/rl_finetuning/run_ablations.py \
    --checkpoint "$CKPT" --ablations-config "$CFG" \
    --ablations "$a" --batch-size 512 --max-iter 3 --eval-every 1 --num-seeds 1 \
    --output-dir /workspace/mem/"$a" 2>&1 | tee logs/smoke/mem_"$a".log
  ST=${PIPESTATUS[0]}; kill $SAMPLER
  echo "$a exit=$ST peak_mib=$(sort -n /tmp/vram_$a.txt | tail -1)" | tee -a logs/smoke/mem_status.txt
done
```

`--batch-size 512` is the value the repo's own 8 GB QMUL config uses, so it fits 16 GB with headroom. `--max-iter 3 --eval-every 1` forces an eval inside three iterations so the eval path's allocation is included.

Report the `nvidia-smi` figure as **reserved**, not allocated: PyTorch's caching allocator holds a pool, so this over-reports live tensors and is a ceiling rather than a footprint. It is still the right number for "does this fit". There is no JAX-style preallocation setting to disable here, so nothing needs an environment variable.

What this phase delivers, stated in exactly these terms in the report:

- **Does deliver:** every ablation's buffers allocate and run at batch 512, plus a measured peak reserved-VRAM ranking across the 25.
- **Does not deliver:** any statement about `batch_size 4608` on 24 GB. That check stays open and belongs on the 3090 Ti before the relaunch. Say so explicitly rather than letting a green Phase 3 read as a clearance.

---

# Phase 4: confirm and report

Re-run Phase 1 end to end. Every ablation must be green by all three checks, not by exit code. Then run one combined sweep of all 25 into a single output directory, with `--all --fast --num-seeds 1`, and assert the count:

```
uv run python -c "import json; d=json.load(open('/workspace/smoke_all/results.json'))['ablations']; \
print(len(d), 'of 25'); assert len(d) == 25, sorted(set('$ABL'.split()) - set(d))"
```

That is the only check that proves the skip path never fired. Running them individually cannot prove it, because each single-ablation run has its own `results.json`.

Write `/workspace/ABLATION_SMOKE_REPORT_MINIHACK.md`:

| Ablation | Phase 1 | Skipped? | nan/inf? | Phase 3 (batch 512) | Peak MiB reserved | Seconds | Failure | Fix (commit) |

followed by the full diff, the total wall clock, the hardware this ran on, the HEAD sha it started from, and any item you did not resolve. Head the report with two lines: that memory at `batch_size 4608` is unverified and why, and the count of ablations present in the final `results.json` out of 25.

---

# Constraints

1. **Never change what an ablation measures.** Fixes make the code run correctly. A change to a loss, a hyperparameter, a seed, a diagnostic's definition or a config value changes the experiment. If the only way to make an ablation run is such a change, stop, report it and leave it red.
2. **Never re-size a config to fit this card.** `batch_size` belongs to the experiment. Phase 3 overrides it on the CLI for a probe and writes nothing; the config files are not edited by this task.
3. **Do not remove or weaken the `try/except` at `run_ablations.py:613`,** and do not add one anywhere else. It is deliberate. Your job is to report every ablation it swallows, not to change the policy.
4. **Evidence for every claim.** File and line for code, command and output for anything executed. Never state a number you have not measured. A peak VRAM figure from this card is a figure from this card, and must be labelled as such.
5. **Do not touch any existing directory under `experiments/rl_finetuning/outputs/`.** Every run in this task writes to `/workspace/smoke/` or `/workspace/mem/`.
6. **Do not push, do not start the real suite.** The launch is the author's call.
7. UK English, no em dashes, short structured entries, tables over prose.

# Do not decide these, report them

- **`final_ablations_qmul.yaml` is not hyperparameter-complete.** It omits the whole Group A, B and D block that `final_ablations_ucl.yaml` carries, and `configs/defaults.yaml` does not supply those keys either, so each resolves through a `getattr(cfg, key, default)` fallback in code. Spot-checked, those defaults currently equal the UCL values (`ewc_fisher_batches 20`, `llrd_decay 0.9`, `lora_alpha 16.0`, `lora_rank 8`, `mixed_replay_buffer_size 10000`, `mixed_replay_ratio 0.25`, `return_weight_cap 5.0`, `reward_filter_percentile 75`, `win_threshold 0.5`), so there is no numerical difference today. It is one default change away from silently diverging, and `RETRAIN_LOG.md` already patches the same gap for `num_seeds` with an explicit `--num-seeds 3`. This matters because `RETRAIN_LOG.md` section 3 makes the QMUL config the primary MiniHack ablation run. Verify the complete key set, tabulate config value against code default for each, and report. Change nothing.
- **Two stale config comments.** `final_ablations_ucl.yaml` says "Full default batch_size (3072)" above `batch_size: 4608`. `final_ablations_qmul.yaml` says "Larger batch (1024 vs 3072)" above `batch_size: 512`. Flag both, change neither.
- **This prompt assumes the 4070 Ti is the test box** and the real suite runs on the 24 GB card. If the author intends to run the real suite here, that is a different task: the UCL batch size does not fit, and any batch that does changes what every ablation measures. Do not begin that. State it as a question.
