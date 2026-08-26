# Task: MiniHack-side defects from the paper review

An external review of the manuscript surfaced four items that can only be settled in this
repository. The sibling `craftax-ReMDM-planner` has its own prompt files for the Craftax half;
the paper-figure generator lives there and reads **this** repo's
`results/experiments/rl_finetuning/outputs/minihack_ablations/results.json`, so item 3 below is
a cross-repo dependency.

Read `CLAUDE.md` first — the sibling-parity rule governs items 1 and 4. Run `uv run pytest`
when done.

---

## 1. Two MiniHack numbers in the paper have no table behind them

The manuscript states, for MiniHack baseline RL, a fall from 47.5% to 43.8% **± 6.1**, and
separately that **three conditions finish nominally above the checkpoint**.

Neither is checkable by a reader:

- the ± 6.1 appears in no table — the group-summary table prints `--` in the StdDev column for
  the Baseline RL row;
- there is **no per-condition post-loop MiniHack table anywhere in the paper**, so the
  "three conditions" claim cannot be verified at all.

Craftax has a full per-condition results table. MiniHack does not, and the parity rule says it
should.

**Do:**
- Recover the baseline's per-seed final win rates from `results.json` and report the seed
  standard deviation. Confirm whether it is 6.1.
- Emit a per-condition MiniHack table — condition, final win rate, seed sd, Δ vs pretrained,
  Δ vs baseline — mirroring the Craftax table's columns and LaTeX export path in
  `experiments/rl_finetuning/analysis/tables.py`.
- From that table, state exactly **which** conditions finish above the pretrained checkpoint
  and how many. If it is not three, say so.

Report the numbers. Do not edit the manuscript.

---

## 2. Verify the ESS / CV_A counter caveat

The paper's hyperparameter appendix claims, for **both** suites:

> Both suites log an effective sample size $\mathrm{ESS} = (\sum_i A_i)^2 / \sum_i A_i^2$ over
> the applied weight batch at every iteration [...] The counter is evaluated on the weights
> produced by the shared weighting step, **before** any condition-specific transform applied
> inside the loss function. Conditions that reshape weights inside the loss, namely
> `advantage_clip` and `normalized_adv`, therefore log a value that does not reflect that
> transform. [...] Conditions that change the weighting step itself, namely `bc_wins`,
> `running_stats`, `reward_model` and `reward_filtering`, log a value that does reflect their
> change.

This is a claim about where in the code the counter is read. **Verify it against this repo's
training loop and loss functions**, and confirm the stated batch size `B = 4,608`.

If the counter is actually read at a different point, that invalidates a published figure's
caption and its appendix caveat — report it precisely and do not adjust the code to fit the
paper.

---

## 3. Confirm `repr_drift_kl` is populated for `baseline_rl`

The Craftax repo's `scripts/paper_figures.py` builds the score-vs-KL figure by iterating
conditions and skipping any whose `repr_drift_kl` series is empty:

```python
kl = series(entry, "repr_drift_kl")
if not kl.size:
    continue
```

The review found the figure's legend advertises a "Baseline RL" marker that does not appear in
either panel — including the MiniHack panel. Check whether `repr_drift_kl` is present and
non-empty for `baseline_rl` in this repo's emitted `results.json`. If it is missing, that is
the cause on the MiniHack side; recover it.

---

## 4. Decide the `measure_gdelta.py` parity question, and record the decision

The sibling repo has `experiments/rl_finetuning/measure_gdelta.py`, which measures the return
term $g_\delta$ of the return-weighted ELBO decomposition. **This repo has no equivalent.**

The paper states the reason as a hard limit:

> It is Craftax Classic only, since the MiniHack planner needs the NetHack Learning
> Environment, which we could not build on the machine used for this measurement.

`CLAUDE.md`'s parity rule exempts environment- and framework-forced divergences, so this is
plausibly a legitimate exemption rather than a gap. But it is currently recorded **only in the
manuscript**, nowhere in the code.

**Do one of the two, and say which:**

- **Port it.** This is a PyTorch repo, so the Orbax sharding workaround the JAX version needs
  does not apply, and the measurement itself is three gradient evaluations on one batch with
  three different weight vectors — it runs on CPU in minutes. If NLE builds here, port it,
  match the sibling's CLI and output-JSON schema exactly, and note that this **closes a stated
  limitation of the paper**.
- **Document the exemption.** If NLE genuinely will not build, record that in
  `experiments/README.md` next to where the diagnostics are described, so the absence is a
  recorded decision rather than an apparent oversight.

Either way, also mirror whatever the sibling adds to its version: per-variant `Abar` reporting
and a shuffled-$\delta$ null control. See `../craftax-ReMDM-planner/GDELTA_VERIFICATION.md`.

---

## 5. Parity: `--emit-tex-macros`

The sibling is gaining a `--emit-tex-macros` flag on `run_ablations.py` that writes
`tables/results.tex`, one `\newcommand` per headline number, so the manuscript can cite
generated macros instead of hand-typed literals. There is already uncommitted work in this
repo's `run_ablations.py` and `analysis/tables.py`.

Bring this repo to parity: same flag name, same emitted schema, same macro-naming rule, same
tests. Macro names must be valid TeX control sequences (letters only), so the mangling rule for
names like `advantage_clip` must match the sibling's exactly — otherwise the two `results.tex`
files collide when the manuscript inputs both.

---

## 6. Report back

State: the baseline seed sd and whether it is 6.1; which conditions finish above the
checkpoint; whether the ESS-counter caveat is accurate as written; whether `repr_drift_kl`
exists for `baseline_rl`; and which way the `measure_gdelta.py` parity decision went.

Do not edit anything under the manuscript's `src/`.

---

## 7. Compute

**Items 1–3 need no GPU.** They are analysis of the `results.json` this repo has already
emitted, plus a table-export change. Do them first, locally.

Items 4 and 5 need a GPU only if you decide to port `measure_gdelta.py` **and** NLE builds —
and even then the measurement is three gradient evaluations on a single batch, which runs on
CPU in minutes.

If anything here does turn out to need a GPU, use the **`/ucl-gpu`** skill to find and reserve
one, and release it when done. Do not start a long run without checking with the author: the
decisive runs for this paper are the Craftax ones in
`../craftax-ReMDM-planner/RUNS.md`. MiniHack parity versions of those arms are optional and
should wait until the Craftax results are in — the paper's central claim rests on Craftax
Classic, and the manuscript already concedes MiniHack is not a second confirmation.
