# Task: the runs the paper review says are missing, MiniHack side

The sibling `craftax-ReMDM-planner/RUNS.md` covers Craftax Classic only. This file is its
MiniHack counterpart, and it exists because one of the review's findings lands **harder here
than it does on Craftax**: the $\bar{A}$ step-size confound was refuted on Craftax at a ratio
of 0.931 and declined there, but the MiniHack ratio is 1.738, where the confound is real and
nothing tracks it.

> **STATUS: DONE — 2026-08-27. Do not re-run any of this without asking.**
> Jobs A and B were already complete; Runs 1, 2 and 3 (reduced: 1e-4 and 1e-5) were executed
> on canada-l and the GPU was released. Results, exact numbers, artefact paths and what is
> still outstanding are in **[REPORT — executed 2026-08-27](#report--executed-2026-08-27)** at
> the bottom of this file. Read that before acting on anything below it — the instructions that
> follow are the original task spec, kept for reference, and two of their pointers are wrong
> (the Craftax `gdelta_verification/` path, and the claim that `results.tex` had never been
> emitted). Headline: **Run 2 contradicts the manuscript's central inference.**

**Use the `/ucl-gpu` skill to find and reserve a free GPU on the UCL CS duck cluster before
starting anything under "Runs", and release it when done.** Read `CLAUDE.md` and
`experiments/README.md` first. Two of the items below need no accelerator at all — do those
first, because the GPU runs depend on their output.

**Sizing, from the paper's own accounting:** the MiniHack suite ran all 25 conditions and all
three seeds on one GPU in 20.2 GPU-hours (`16-appendix-hyperparameters.tex:6`), so **one
condition x 3 seeds is about 0.8 GPU-hours**.

**The duck cluster is the reference hardware.** Every host is a single RTX 3090 Ti
(24,564 MiB), which is exactly the machine behind
`experiments/rl_finetuning/configs/ablations_final_minihack_gpu_24gb.yaml` — the reference
config every published MiniHack number was produced under. So a reserved duck host needs no
config change of any kind.

---

## Hard constraints — read before touching any config

- **Never edit `configs/final_*` or `experiments/rl_finetuning/configs/ablations_final_*`** to
  fit the machine. `CLAUDE.md` forbids it and `tests/test_config.py` enforces the sibling
  rules. Use `ablations_final_minihack_gpu_24gb.yaml` as-is.
- **`ablations_final_minihack_gpu_h200.yaml` is not poolable** with it — it diverges on
  `batch_size` (512 vs 4608), `episodes_per_iter` (20 vs 30), `diffusion_steps_collect`
  (3 vs 5) and `eval_episodes` (10 vs 20), all of which change the result. If you end up on
  anything other than a 24 GB card, stop and say so rather than running the H200 config and
  comparing across.
- Presets are delta-only. Restating a defaults value silently pins it, and `test_config.py`
  will fail. The ablation suite has **no `--override`**; `--lr`, `--seed`, `--num-seeds`,
  `--max-iter`, `--batch-size` and `--eval-every` are the CLI flags that exist.
- **Every run must use the same pretrained checkpoint the 25 published conditions start from:**
  `checkpoints/online/Minihack-Online-Diffusion-DAgger-100M/iter563.pth`, with **its own
  `config.yaml` snapshot**. The model is built from the config, not the checkpoint, so a
  mismatch raises at load. Both are on the Hub:

  ```sh
  uv run hf download mathisweil/remdm-minihack-checkpoints \
      --include "checkpoints/online/Minihack-*/**" --local-dir .
  ```

- **Release the `/ucl-gpu` hold immediately before launching the job on that host**
  (`gpufleet.sh release <host>`), then start at once. The hold allocates ~9.8 GiB of the card's
  24 GiB and a real job competes with it. Once the job is resident the card scans as busy
  anyway.
- **The install path must not contain spaces.** MiniHack's `mh_patch_nhdat.sh` fails silently
  and falls back to the Python path in `src/envs/minihack_env.py`.
- `experiments/rl_finetuning/outputs/` is **not** gitignored here, unlike the sibling repo, so
  run outputs appear as untracked files. Leave them untracked; do not add a `.gitignore` line
  without asking.
- Record the wandb run id for each run. Every compute figure in the paper is sourced to one.
- **Do not edit anything under the manuscript's `src/`.** These are findings for the authors.

---

## Before you reserve — two jobs that need no accelerator

### A. Regenerate the tables and the `results.tex` macros  (~20 minutes, no compute)

**`results.tex` has never been emitted, on either side.** The emitter exists and is at sibling
parity — `analysis/tables.py:749` sets `_MACRO_PREFIX = "mh"` against the sibling's `rw`, and
`_MACRO_SCALE = 100.0` converts the fractional win rate to the percentage points the manuscript
prints. Nobody has run it, so the `mh` macro namespace the manuscript is meant to `\input` does
not exist as a file.

**The published `results.json` is not in this repo and not in the `Publication/` folder.** Both
local copies there are the **H200 run** — `pretrained_score` 0.70, `batch_size` 512,
`baseline_rl` 0.5833 — which is not the published suite and not poolable with it. The published
run is on the Hub:

```sh
uv run python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("mathisweil/remdm-minihack-checkpoints",
                      "experiments/rl_finetuning/outputs/minihack_ablations/results.json"))
PY
```

It records `pretrained_score` 0.475, `batch_size` 4608, `episodes_per_iter` 30, 25 conditions,
3 seeds. Then:

```sh
uv run python experiments/rl_finetuning/run_ablations.py \
    --analyze-only --emit-tex-macros \
    --results-path <downloaded>/results.json \
    --output-dir results/minihack_published
```

Read the **real macro names** off the regenerated `tables/results.tex` and record them. The
workspace notes are stale on this point: `state.md` says the MiniHack emitter produces
`\mhBaselineRlScore` and no `\mhDeltaPretrained*`, but `tables.py:923-936` emits
`\mhScore<Tag>`, `\mhScoreSd<Tag>` and `\mhDeltaPretrained<Tag>` — the sibling scheme. Commit
`1132d97` closed that gap and nobody re-emitted to confirm.

Two import defects to record, **not** to fix here: the `.tex` export carries
`\label{tab:main-results}`, identical to the Craftax export's, so the two collide; and the CSV
Score column is fractions while the macros are already scaled x100.

Do the same on the Craftax side, so both namespaces exist as files.

### B. Measure $g_\delta$ at publication settings  (CPU, no reservation)

The measurement has been run **once, at 3 rollout seeds x 3 draws**. The paper's table is 8
draws, and the result was never persisted — it exists only in a session transcript, so there is
no artefact behind any MiniHack number. NLE 1.3.0 imports cleanly from this repo's `.venv`,
which also means `09-appendix-derivation.tex:68` ("we could not build the NetHack Learning
Environment on the machine used for this measurement") no longer describes the situation.

**Verify the checkpoint first.** The open caveat is that `iter563.pth` was never confirmed as
the checkpoint behind the 47.50 pretrained score. The downloaded `results.json` records
`pretrained_score` 0.475; confirm its config snapshot and checkpoint reference agree before
treating anything from this run as publishable. **If they disagree, stop and report** — do not
substitute a different checkpoint.

```sh
uv run python experiments/rl_finetuning/run_ablations.py \
    --measure-gdelta --gdelta-seeds 0 1 2 --gdelta-draws 8 \
    --checkpoint checkpoints/online/Minihack-Online-Diffusion-DAgger-100M/iter563.pth \
    --results-path <downloaded>/results.json \
    --output-dir results/minihack_published
```

`--results-path` matters: it makes the weight transforms measured the ones the published run
actually trained under. Writes `gdelta/gdelta_seed{0,1,2}.json` and `gdelta/gdelta_aggregate.json`;
`--emit-tex-macros` then picks the aggregate up as `\mhGdelta*`. The same pass produces the
**shuffled-$\delta$ null**, which has never been persisted on this side either.

Cross-check against the sibling artefact
(`craftax-ReMDM-planner/experiments/rl_finetuning/outputs/gdelta_verification/gdelta_aggregate.json`,
3 seeds x 8 draws, `eq4_residual_max` 4.44e-07). The 3-draw MiniHack run reported a max relative
residual of 2.14e-05, two orders looser; if that does not tighten at 8 draws, say so.

**Read $\bar{A}_{\text{base}} / \bar{A}_{\text{clip}}$ off this run, not off the transcript.**
It sizes Run 1. For orientation only, the 3-draw run gave baseline $\bar{A}$ 0.488 and
`advantage_clip` 0.840, a ratio of 1.738.

---

## Run 1 — advantage clipping at matched effective step  (~0.8 GPU-hours, 3 seeds)

**Highest value of the three. This is the run where MiniHack is worse off than Craftax.**

Eq. 4 makes $\bar{A}$ the effective learning rate. On Craftax the two arms of the paper's
central control are matched to within 7 per cent (ratio 0.931), which is why the sibling
`RUNS.md` Run 2 was declined. Here the ratio is **1.738**: the clipped arm takes a step roughly
1.7 times larger than the baseline's, in exactly the direction the drift-vs-score ordering says
determines the outcome. The paper does not publish MiniHack $g_\delta$ and scopes its claim to
Craftax Classic, so this is not currently a hole in the PDF — but it is the thing that has to be
true before any MiniHack decomposition number can be published.

Rerun `advantage_clip`, 3 seeds, from the same checkpoint, with the learning rate scaled by
$\bar{A}_{\text{base}} / \bar{A}_{\text{clip}}$ from job B. On the 3-draw ratio that is
$3\times10^{-4} / 1.738 \approx 1.7\times10^{-4}$; use the measured value. `--lr` is a CLI flag,
so **no config file is edited**.

```sh
uv run python experiments/rl_finetuning/run_ablations.py \
    --checkpoint checkpoints/online/Minihack-Online-Diffusion-DAgger-100M/iter563.pth \
    --ablations-config experiments/rl_finetuning/configs/ablations_final_minihack_gpu_24gb.yaml \
    --ablations advantage_clip --num-seeds 3 --lr <matched> \
    --output-dir results/minihack_matched_step --use-wandb
```

**Do not `--merge` this into the published results.** `lr` is in
`run_ablations._RESULT_AFFECTING`, so the merge will refuse by design. Report it alongside the
published numbers instead: `advantage_clip` 33.75 (+/- 5.10), `baseline_rl` 43.75 (+/- 6.12),
pretrained 47.50.

**What the outcome means — state it plainly, do not spin it:**
- If the 10-point gap between clipping and the baseline survives matching, the MiniHack control
  is clean and the decomposition numbers become publishable on this side.
- If it collapses, the MiniHack comparison was measuring step size, and the paper's decision to
  rest its claim on Craftax Classic becomes load-bearing rather than incidental. Say so.

---

## Run 2 — the unweighted-ELBO-on-all-rollouts arm  (~0.8 GPU-hours, 3 seeds)

The MiniHack counterpart of the sibling's Run 1. The manuscript's headline inference is that the
degradation comes from fine-tuning on self-generated rollouts rather than from the return
weighting, and it concedes there is no arm separating the two. **The arm has not been added on
either side.**

Fine-tune from the same checkpoint on the same on-policy rollouts with **uniform weights** —
plain behavioural cloning on self-generated data, no return weighting. Everything else identical
to `baseline_rl`: same rollout collection, same iteration count, same learning rate, same 3
seeds.

Add it as an `AblationSpec` in `experiments/rl_finetuning/ablations/registry.py` following the
existing pattern, not as a one-off script. The loss already exists: `_core_loss` takes
`advantages: Tensor | None` and takes the uniform branch on `None`
(`experiments/rl_finetuning/ablations/losses.py:126-162`), so this is a spec entry and a factory
that passes `None`, not new numerics. Run `uv run pytest` afterwards — `test_config.py` and
`test_spec_ablations.py` both guard the registry.

**What the outcome means:**
- If it degrades about as much as `baseline_rl` (43.75), the self-generated-data account is
  supported and the paper's central inference becomes a measurement.
- If it degrades **less**, the return weighting does carry blame and the paper's conclusion is
  wrong.
- If it degrades **more**, the weighting was protective — the reading the paper currently keeps
  open as the inverse of its own.

Do not adjust anything to reach a preferred outcome. Note the paper rests its claim on Craftax,
so the Craftax version of this arm is the one that settles the manuscript; the MiniHack version
tells you whether the answer is environment-specific.

---

## Run 3 — learning-rate sweep  (~3 GPU-hours; ask before starting)

**Larger than the other two. Check with the author before running it.**

The review's version of this is "fine-tuning uses the pretraining learning rate". On MiniHack
the arithmetic is sharper than that: the suite fine-tunes at `lr: 3.0e-4`
(`experiments/rl_finetuning/configs/ablations_default.yaml:25`) while the checkpoint it starts
from was produced by online DAgger at `dagger_lr: 0.00003` (`configs/defaults.yaml:148`) —
**ten times the learning rate that trained it**. No condition in the suite varies it.

Check that against what the manuscript actually claims before relying on the framing; the
"identical to the pretraining LR" line is a Craftax statement and may not transfer.

Full version: `baseline_rl` at 3e-4, 1e-4, 3e-5, 1e-5, three seeds each — 12 runs, ~3.2
GPU-hours. Reduced version: 1e-4 and 1e-5 only, 6 runs, ~1.6 GPU-hours, which still answers
"does *any* learning rate recover 47.50?".

---

## Not a run, but tracked — MiniHack has no external anchor

The sibling has `scripts/eval_ppo_expert.py` (never run, its Run 3). **MiniHack has no
expert-eval script at all.** Its DAgger teacher is the built-in BFS oracle and is never scored,
so a reader cannot tell whether the pretrained planner at 47.50 is close to its teacher or far
below it — and if it is far below, the negative result may be about a weak checkpoint rather
than about the objective.

Writing that harness is a code task, not a run. It is listed here so it is tracked rather than
done silently. Ask before starting it.

---

## Report back

One table: run, condition, learning rate, per-seed scores, mean, seed sd, wandb run id,
wall-clock, GPU-hours.

Then answer, in one line each:
1. Is `iter563.pth` confirmed as the checkpoint behind the 47.50 pretrained score?
2. What is $\bar{A}_{\text{base}} / \bar{A}_{\text{clip}}$ at 3 seeds x 8 draws, and how much of
   the norm ratio survives the shuffled-$\delta$ null?
3. Does the clipping gap survive matching on effective step?
4. Does the unweighted-rollout arm degrade more, less, or the same as `baseline_rl`?
5. (If run) Does any learning rate recover 47.50?

Release the GPU reservation when finished. **Do not edit anything under the manuscript's
`src/`** — these are findings for the authors. If a run contradicts the paper, say so plainly.

---

# REPORT — executed 2026-08-27

**Status: jobs A and B were already complete on arrival; Runs 1, 2 and 3 (reduced) were
executed and are done. All four GPU runs finished; the reservation is released.** Nothing
under any manuscript `src/` was touched, and no `final_*` or ablation machine config was
edited.

Everything ran on **canada-l** (duck cluster, RTX 3090 Ti, 24,564 MiB) under
`experiments/rl_finetuning/configs/ablations_final_minihack_gpu_24gb.yaml` **unmodified**,
from `checkpoints/online/Minihack-Online-Diffusion-DAgger-100M/iter563.pth`.
Only one host was ever free — the fleet was rescanned every 7 minutes for 3.5 hours and
never showed a second free card (5 hosts down, 17 held by other users), so the four runs
ran back-to-back rather than in parallel.

## The table

| Run | Condition | lr | per-seed | mean | seed sd | wandb | wall-clock | GPU-h |
|---|---|---|---|---|---|---|---|---|
| 1 | `advantage_clip` matched step | 1.636e-4 | 38.75 / 31.25 / 41.25 | **37.08** | 4.25 | `xy1s1ic7` | 49m56s | 0.83 |
| 2 | `bc_all` (new arm) | 3e-4 | 36.25 / 35.00 / 30.00 | **33.75** | 2.70 | `6vpbqtp4` | 50m23s | 0.84 |
| 3a | `baseline_rl` | 1e-4 | 42.50 / 43.75 / 46.25 | **44.17** | 1.56 | `5fzphilq` | 51m01s | 0.85 |
| 3b | `baseline_rl` | 1e-5 | 47.50 / 46.25 / 46.25 | **46.67** | 0.59 | `zadg8rtj` | 48m02s | 0.80 |

Total **3.32 GPU-hours**, 16:03–19:26 BST. Scores are ID win rate in percentage points.
W&B project `myopic-planner/minihack-ReMDM-planner-ablations`.

Published reference these sit against (`experiments/rl_finetuning/outputs/minihack_ablations/results.json`):
pretrained **47.50**, `baseline_rl` **43.75 ± 6.12**, `advantage_clip` **33.75 ± 5.10**,
`bc_wins` **45.83 ± 3.58**. RUNS.md's quoted values were checked against that file and match.

## The five answers

1. **Yes.** Runs 1 and 2 each re-evaluated `iter563.pth` under the reference config and logged
   `Pretrained baseline ID win rate: 0.4750` — 47.50 exactly, twice, independently.
2. **$\bar{A}_{\text{base}}/\bar{A}_{\text{clip}} = 0.545$** (i.e. $\bar{A}_{\text{clip}}/\bar{A}_{\text{base}} = 1.834 \pm 0.195$
   at 3 seeds x 8 draws, up from 1.738 at 3 draws). The shuffled-$\delta$ null leaves **16.1%**
   of the baseline norm ratio (1.956 -> 0.316), so ~84% is genuine return signal; the cosine
   collapses 0.803 -> -0.032. The null retains 15–17% across all four transforms.
3. **Only partly — about a third of the gap is step size.** Matching moves clipping
   33.75 -> 37.08, closing 3.33 of the 10.00-point gap; 6.67 points remain. Neither the
   residual (t = 1.55) nor the original gap (t = 2.17) is separable from zero at 3 seeds.
   **The paper's decision to rest its claim on Craftax Classic is therefore load-bearing, not
   incidental.**
4. **More — by 10.00 points (t = 2.59).** `bc_all` 33.75 vs `baseline_rl` 43.75; it ranks 24th
   of 26 conditions. **This contradicts the manuscript's headline inference:** the return
   weighting is *protective*, the reading the paper keeps open as the inverse of its own.
   The three-way contrast localises it — `baseline_rl` (return-weighted, all windows) 43.75;
   `bc_all` (uniform, all windows) 33.75; `bc_wins` (uniform, wins only) 45.83. Dropping the
   weighting while keeping every rollout window costs 10 points; filtering to wins recovers it.
   **The sibling replicates this on Craftax** (`bc_all` 4.73 vs `baseline_rl` 8.34, pretrained
   11.98), so the answer is not environment-specific.
5. **Effectively yes, at 1e-5.** 3e-4 -> 43.75, 1e-4 -> 44.17, 1e-5 -> **46.67 ± 0.59**, which is
   0.83 short of 47.50 — under one eval episode (4 envs x 20 episodes ⇒ 1.25-point granularity),
   and one seed hit 47.50 exactly. Seed sd tightens monotonically as lr falls: 6.12 -> 1.56 -> 0.59.

## Where the artefacts are

> **Warning: `results/` is gitignored (`.gitignore:274`), so all four new run directories are
> untracked and local-only.** RUNS.md warned only that `experiments/rl_finetuning/outputs/` is
> *not* ignored, but its own Run 1 command writes to `results/`. Copy anything the paper cites
> out of `results/` before relying on it. No `.gitignore` line was added — that needs asking.

| What | Path |
|---|---|
| Run 1 — clipping at matched step | `results/minihack_matched_step/` |
| Run 2 — `bc_all` | `results/minihack_bc_all/` |
| Run 3a — `baseline_rl` @1e-4 | `results/minihack_lr_1e-4/` |
| Run 3b — `baseline_rl` @1e-5 | `results/minihack_lr_1e-5/` |
| Published 25 + `bc_all` pooled (26) | `results/minihack_published_plus_bcall/` |
| Published suite (25 conditions) | `experiments/rl_finetuning/outputs/minihack_ablations/results.json` |
| `mh` macro namespace | `experiments/rl_finetuning/outputs/minihack_ablations/tables/results.tex` |
| $g_\delta$ + shuffled null | `experiments/rl_finetuning/outputs/minihack_ablations/gdelta/gdelta_{seed0,seed1,seed2,aggregate}.json` |
| `rw` macro namespace (Craftax) | `craftax-ReMDM-planner/experiments/rl_finetuning/outputs/craftax_classic_ablations/tables/results.tex` |
| Craftax $g_\delta$ aggregate | `craftax-ReMDM-planner/experiments/rl_finetuning/outputs/craftax_classic_ablations/gdelta/gdelta_aggregate.json` |

Each run directory carries `results.json`, `diagnosis.md`, `checkpoint_<name>.pth`, `figures/`
and `tables/` in the standard layout of experiments/README.md §Output structure.

**Craftax review runs already exist** (from an earlier session), useful for cross-environment
comparison: `craftax-ReMDM-planner/experiments/rl_finetuning/outputs/review_anchor_baseline_rl`
(8.34), `review_run1_bc_all` (4.73), `review_run2_advclip_lr_matched` (4.92),
`review_run4_baseline_lr1e-4` (10.15), `review_run4_baseline_lr1e-5` (10.81); pretrained 11.98.

## Job A — the macro namespaces (was already done, verified)

Both namespaces exist as files: `mh` (338 macros) and `rw` (238). The emitter produces the
**sibling scheme**, confirming `1132d97` closed the gap and that `state.md` is stale — the real
families are:

```
\mhPretrainedScore  \mhBatchSize  \mhPooledSeedSd
\mhScore<Tag>  \mhScoreSd<Tag>  \mhDeltaPretrained<Tag>  \mhDeltaBaseline<Tag>
\mhEss<Tag>  \mhCvA<Tag>  \mhEnv<Tag><Layout>
\mhGroup{Best,Worst,Mean,Sd,N,Delta}{Baseline,A,B,C,D}
\mhGdelta{Abar,AbarRatio,CvA,Ess,Ratio,RatioSd,Cos,CosSd,RatioShuf,RatioShufSd,CosShuf,CosShufSd}<Tag>
\mhGdelta{NSeeds,NParams,Batch,RandomCosSd,BcSelfCos,BcSelfCosSd,EqFourResidual}
```

`<Tag>` is the CamelCase condition name **with digits spelled out** — `LayerAblationTopOne`,
not `LayerAblationTop1`. Layout tags likewise: `RoomRandomFivexFive`,
`RoomRandomOneFivexOneFive`, `CorridorRTwo`, `MazeWalkNinexNine`.

**Both import defects reproduce and were left unfixed, as instructed:** the `.tex` export
carries `\label{tab:main-results}`, identical to the Craftax export's, so the two collide; and
the CSV `Score` column is fractional while the macros are already scaled x100.

**Provenance of the published `results.json`, now settled:** the local copy at
`experiments/rl_finetuning/outputs/minihack_ablations/results.json` is *not* byte-identical to
the Hub copy, but differs **only** in three scrubbed wandb name fields
(`wandb_project`, `wandb_entity`, `baselines_wandb_project`). All 25 scores and every history
value are identical. The local file is the published run pre-anonymisation; the Hub copy is the
de-identified export. A sha mismatch here is the publish transform, not a different run.

## Job B — $g_\delta$ at publication settings (was already done, verified)

Artefacts are 3 seeds x 8 draws, stamped `checkpoint_step: 563`, `batch: 4608`, measured against
the published `results.json`.

| transform | $\bar{A}$ | $\bar{A}$ ratio | $\|g_\delta\|/\|\nabla L_{BC}\|$ | shuffled | % surviving | cos | cos shuffled |
|---|---|---|---|---|---|---|---|
| `baseline_rl` | 0.462 | 1.000 | 1.956 | 0.316 | 16.1% | 0.803 | -0.032 |
| `advantage_clip` | 0.838 | 1.834 | 0.114 | 0.018 | 15.5% | 0.838 | -0.196 |
| `normalized_adv` | -0.000 | 0.000 | 0.768 | 0.133 | 17.3% | 0.803 | -0.007 |
| `bc_wins` | 1.000 | 2.192 | 2.422 | 0.412 | 17.0% | 0.771 | -0.162 |

**The Eq. 4 residual did NOT tighten at 8 draws — it got slightly worse:** 2.14e-05 (3 draws)
-> **4.84e-05** (8 draws), against Craftax's 4.8e-07. Two orders looser. Worth resolving before
any MiniHack decomposition number is published.

**RUNS.md's cross-check path above is wrong:** there is no
`craftax-ReMDM-planner/experiments/rl_finetuning/outputs/gdelta_verification/`. The Craftax
aggregate lives at `.../outputs/craftax_classic_ablations/gdelta/gdelta_aggregate.json` and
reports `eq4_residual_max` **4.8e-07**, not 4.44e-07.

## Code changes made (Run 2's arm)

`bc_all` added as a 26th `AblationSpec`, group B — a spec entry plus a factory passing `None`,
no new numerics, exactly as this file specified:

- `experiments/rl_finetuning/ablations/losses.py` — `make_loss_bc_all`, `del advantages` then
  `_core_loss(..., None, ...)`.
- `experiments/rl_finetuning/ablations/registry.py` — import + `AblationSpec`.
- `tests/test_smoke_experiments.py` — registry count 25 -> 26.
- `tests/test_config.py` — pinned description added.
- `tests/test_spec_ablations.py` — new guard
  `test_bc_all_ignores_the_advantages_and_averages_over_the_whole_batch`.
- `experiments/README.md` — 25 -> 26 ablations, 15 -> 16 loss factories, table row.

`uv run pytest`: **414 passed, 6 skipped, 11 deselected.**

**Sibling parity:** the Craftax repo had already added `bc_all` with the same name and the same
`del advantages` mechanism; the description, hypothesis and docstring here were aligned to it
character-for-character, so `test_config.py`'s pinned table matches across the two repos. One
deliberate divergence: the MiniHack side has the `test_spec_ablations.py` guard above and the
sibling does not — **consider adding it there** (sibling has 39 spec tests, MiniHack now 40).

**`bc_all` is poolable with the published suite** — `--merge` accepted it (machine confirmation
that it agrees on every key in `run_ablations._RESULT_AFFECTING`), because it ran at the default
`lr: 3.0e-4` under the reference config. The pooled 26-condition set was written to
`results/minihack_published_plus_bcall/`; **the published `results.json` was left untouched.**
Run 1 was deliberately *not* merged, as this file requires — its `lr` differs.

## Full ranking, published 25 + `bc_all` (pretrained 47.50)

```
 1 head_only            49.58 ± 3.86     14 trust_region_kl      42.50 ± 1.02
 2 gradient_surgery     48.75 ± 3.54     15 low_t                42.08 ± 5.24
 3 layer_ablation_top1  48.75 ± 2.04     16 llrd                 42.08 ± 1.18
 4 frozen_backbone      46.67 ± 0.59     17 mixed_replay         41.25 ± 4.68
 5 bc_wins              45.83 ± 3.58     18 layer_ablation_top2  40.42 ± 1.56
 6 entropy_bonus        44.58 ± 3.28     19 attention_only       39.58 ± 2.36
 7 kl_penalty           44.17 ± 2.12     20 layer_ablation_top3  39.17 ± 2.57
 8 reward_filtering     44.17 ± 5.03     21 running_stats        38.33 ± 3.12
 9 baseline_rl          43.75 ± 6.12     22 action_diversity     37.50 ± 4.08
10 ffn_only             43.75 ± 1.02     23 advantage_clip       33.75 ± 5.10
11 ewc                  43.33 ± 1.18     24 bc_all               33.75 ± 2.70   <- new
12 lora                 43.33 ± 1.56     25 reward_model         32.08 ± 1.18
13 t_curriculum         42.92 ± 2.57     26 normalized_adv       12.08 ± 4.12
```

## Still outstanding

- **Run 3, full version.** The reduced sweep (1e-4, 1e-5) was chosen and run. The `3e-5` arm was
  not run, and `3e-4` was not re-run as a reproducibility check — ~1.6 GPU-hours to complete.
- **The MiniHack expert-eval harness** (this file's "Not a run, but tracked" section). Not
  started; it needs asking first. Still true that the BFS oracle teacher is never scored, so
  nobody can tell whether 47.50 is near its teacher or far below it.
- **The manuscript LR claim was not verified.** This file asks to check the "identical to the
  pretraining LR" line against the manuscript; the `.tex` sources are not in this workspace, so
  it could not be checked. The config arithmetic *is* confirmed: suite `lr: 3.0e-4`
  (`ablations_default.yaml:25`) vs `dagger_lr: 3.0e-5` (`configs/defaults.yaml:148`) — 10x.
- **The two `.tex` import defects** remain unfixed, as instructed.
