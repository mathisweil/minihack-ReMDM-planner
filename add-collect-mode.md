# Task: Add `--mode collect` to the MiniHack ReMDM project

## Goal

Add a new `--mode collect` that runs the BFS oracle across the 4 in-distribution MiniHack environments, collects oracle demonstration trajectories, and saves them as a `.pt` dataset file that `--mode offline` can consume directly.

This is the missing first stage of the pipeline. Currently the README documents:

```
[Stage 1]  Offline BC on oracle demos     main.py --mode offline --data dataset.pt
```

…but there is no documented way to produce `dataset.pt`. The smoke test collects a handful of oracle trajectories internally, and the DAgger loop generates oracle data on the fly, but neither produces a reusable standalone dataset. This task fills that gap.

---

## Analysis phase — read these files first

Before writing any code, read every file listed below and take notes. You need to understand the existing conventions so the collect mode integrates cleanly.

### 1. Understand the oracle

Read `src/envs/minihack_env.py`:
- How is the `AdvancedObservationEnv` constructed? What observations does it return (local crop, global map, staircase coords)?
- How does the BFS oracle work? What is its interface — does it take an env and return actions step-by-step, or does it run a full episode?
- What action space does it use? (The model uses 12 actions + MASK=12 + PAD=13, so valid oracle actions should be in `[0, 11]`.)

### 2. Understand the data format the offline trainer expects

Read `src/planners/offline.py`:
- What format does `--data` expect? A `.pt` file containing what — a dict of tensors? A list?
- What keys/fields does it need per sample or per window? Likely: `local_obs`, `global_obs`, `actions`, `goal_coords`, possibly `dones`.
- How does it slice episodes into `seq_len=192` windows? Does it do the windowing itself, or does it expect pre-windowed data?
- Is there episode-boundary masking (skipping windows that span a `done`)?

### 3. Understand how DAgger and smoke already collect oracle data

Read `src/planners/online.py` and `src/planners/smoke.py`:
- How does DAgger call the oracle? Trace the code path from "oracle rollout on the same seed" to "add trajectory to buffer".
- How does the smoke test collect its few oracle trajectories?
- What does the data look like when it enters the `ReplayBuffer`?

Read `src/planners/collect.py`:
- What do `run_model_episode` and `DataCollector` currently do?
- Can any of this be reused or adapted for oracle-only collection?
- If `DataCollector` is coupled to the model (needs a diffusion model to run), you'll need a separate oracle collection function.

### 4. Understand the replay buffer

Read `src/buffer.py`:
- What is the `ReplayBuffer` sample format? Each sample is a dict with which keys?
- The buffer "pins offline data at the front" — does the offline trainer use this buffer, or does it use a separate `Dataset`/`DataLoader`?
- What shape are the tensors? This is critical for format compatibility.

### 5. Understand the entry point and config

Read `main.py`:
- How are modes dispatched? What pattern do `offline`, `dagger`, `inference`, `smoke` follow?
- What CLI args exist? How are `--mode`, `--data`, `--config`, `key=value` overrides parsed?

Read `configs/defaults.yaml`:
- Are there any collection-related config keys already? If not, you'll add new ones.

Read `src/config.py`:
- How does config loading work? How do new keys get registered?

---

## Design specification

After reading the code, implement `--mode collect` following these requirements:

### Behaviour

1. Loop over the 4 ID environments (the same list used in DAgger training):
   - `MiniHack-Room-Random-5x5-v0`
   - `MiniHack-Room-Random-15x15-v0`
   - `MiniHack-Corridor-R2-v0`
   - `MiniHack-MazeWalk-9x9-v0`

2. For each environment, run the BFS oracle for `collect_episodes_per_env` episodes (default: 5000). Each episode:
   - Create the env with `AdvancedObservationEnv` using the same wrapper stack as the rest of the project.
   - Step the oracle until the episode ends (win or max steps).
   - Record at every step: `local_obs`, `global_obs`, `action`, `done`, and `goal_coords` (staircase position from the observation, for the auxiliary loss).
   - Store the full episode trajectory.

3. After collecting all episodes, slice them into `seq_len`-length windows:
   - A window is valid only if no `done=True` occurs within positions `[0, seq_len-1)` (the last position can be a done — that's the episode ending).
   - Episodes shorter than `seq_len` should be **right-padded** with `PAD` tokens (action=13) and the observations at the last valid step repeated. Mark these padded positions so the loss can ignore them.
   - Episodes longer than `seq_len` produce multiple windows via a sliding stride (default: `seq_len // 2 = 96`, configurable as `collect_stride`).

4. Save the dataset as a `.pt` file (a dict of stacked tensors):
   ```python
   {
       "local_obs":   torch.Tensor,   # [N, seq_len, 9, 9]  or whatever the obs shape is
       "global_obs":  torch.Tensor,   # [N, seq_len, 21, 79] or similar
       "actions":     torch.LongTensor,  # [N, seq_len] values in [0..11] + PAD=13
       "goal_coords": torch.Tensor,   # [N, 2] staircase coords (for aux loss)
       "valid_mask":  torch.BoolTensor,  # [N, seq_len] True for real steps, False for padding
       "env_id":      torch.LongTensor,  # [N] which environment this window came from
   }
   ```
   **Important:** Match the exact field names, dtypes, and shapes that `src/planners/offline.py` expects. If the offline trainer expects a different format, match that format — not the one above. The schema above is a starting guess; the code is the source of truth.

5. Print progress: episodes collected, windows extracted, dataset size, per-env stats, wall-clock time.

### Performance

- NLE environments are single-threaded. Use `multiprocessing` (or `concurrent.futures.ProcessPoolExecutor`) to run multiple environment instances in parallel. Default: `collect_num_workers` = `min(cpu_count(), 8)`.
- Batch the window slicing — don't do it one episode at a time in Python loops if you can vectorise with torch/numpy.
- Print an ETA based on episodes completed so far.

### Config keys to add (in `configs/defaults.yaml`)

```yaml
# Data collection
collect_episodes_per_env: 5000      # Oracle episodes per ID environment
collect_stride: 96                  # Sliding window stride (seq_len // 2)
collect_num_workers: 8              # Parallel environment workers
collect_output: "data/dataset.pt"   # Default output path
```

### CLI interface

```bash
# Default: 5000 episodes per env, output to data/dataset.pt
python main.py --mode collect

# Custom episode count and output
python main.py --mode collect collect_episodes_per_env=2000 collect_output=data/small_dataset.pt

# Override via config
python main.py --mode collect --config configs/collect_large.yaml
```

The `--output` flag from `--mode inference` should NOT be reused — use `collect_output` as a config key to keep the config system uniform.

---

## Implementation plan — write this before coding

After reading all the files, write a numbered plan with:
1. What new files you'll create (if any) vs what existing files you'll modify.
2. For each file, what functions/classes you'll add or change.
3. The exact data format you'll produce, confirmed against what `offline.py` actually loads.
4. How you'll reuse existing oracle/env code vs what's new.
5. How you'll wire it into `main.py` and `configs/defaults.yaml`.

Get the plan right before writing code. If you discover that the offline trainer expects a different format than what's described above, adapt to what the code actually expects.

---

## Implementation order

1. **Add config keys** to `configs/defaults.yaml` and confirm `src/config.py` picks them up.
2. **Write the collection logic.** Either add a `run_collect` function in `src/planners/collect.py` (if the existing file is suitable) or create a new file. Prefer extending the existing file if it doesn't break DAgger's use of `DataCollector`.
3. **Wire into `main.py`** — add `collect` to the mode dispatch.
4. **Test format compatibility.** Write a quick sanity check:
   ```python
   dataset = torch.load("data/dataset.pt")
   # Assert all expected keys exist
   # Assert shapes are correct
   # Assert action values are in valid range [0..11] + PAD=13
   # Assert no done=True in the middle of a window (except right-padded ones)
   ```
5. **Update README.md** — add a "Stage 0 — Collect oracle demonstrations" section to the pipeline and the usage docs. Include the commands, config keys, and recommended episode counts. Update the pipeline diagram.

---

## Verification checklist

After implementation, verify:

- [ ] `python main.py --mode collect collect_episodes_per_env=10 collect_output=data/test.pt` runs without error
- [ ] The output `.pt` file loads and has the correct keys, shapes, and dtypes
- [ ] Action values are all in `[0, 11]` for real steps and `13` (PAD) for padded positions
- [ ] `valid_mask` is `True` for real steps, `False` for padding
- [ ] No window contains a `done=True` before the last valid position (episode-boundary correctness)
- [ ] `python main.py --mode offline --data data/test.pt` can load the collected dataset and start training (even if you immediately Ctrl+C — just confirm it loads)
- [ ] The collect mode prints progress with per-env episode counts and an ETA
- [ ] `configs/defaults.yaml` has the new keys and they work as overrides from the CLI
- [ ] `README.md` pipeline diagram is updated and the collect command is documented
- [ ] No existing modes (`offline`, `dagger`, `inference`, `smoke`) are broken — test `python main.py --mode smoke` if possible

---

## Things to watch out for

- **Observation format:** The model takes `local_obs` (9×9 glyphs) and `global_obs` (21×79 glyphs). Make sure you capture exactly the observations that the env wrapper returns, in the same format (numpy arrays? torch tensors? integer glyph IDs?). Check what `AdvancedObservationEnv` returns.
- **Goal coordinates:** The model has an auxiliary goal prediction head that predicts staircase `[row, col]`. Check how DAgger extracts this from the observation — it might be in the info dict or derived from the global map. The collect mode must capture the same thing.
- **Window observation:** The offline trainer might only need the **first** observation of each window (the conditioning obs for the plan), not all 192 observations. Check this — it changes the dataset shape significantly. If it only needs `obs[0]`, store `[N, 9, 9]` not `[N, 192, 9, 9]`.
- **NLE env cleanup:** NLE envs must be properly closed. Use try/finally or context managers. Leaked env processes will accumulate.
- **Reproducibility:** Set and save the random seed. The dataset should be reproducible given the same seed and episode count.
