# Performance Optimisation — minihack-ReMDM-planner

You are optimising a discrete diffusion action planner for MiniHack. The codebase is functional and audited — this is about wall-clock speed, not correctness. The DAgger training loop is the bottleneck: each iteration runs 10 sequential episodes (model rollout + oracle rollout + efficiency filter), then 100 gradient steps. On CPU, collection dominates. On GPU, the repeated single-sample forward passes during collection are poorly utilised.

**Your job:** Profile the current codebase, identify the top bottlenecks with real timing data, then implement the highest-impact optimisations. Do not guess — measure first, then act.

---

## Phase 1 — Profile before touching anything

Before writing any optimisation code, **measure the current baseline**. Run a targeted profiling session — not a full training run, just enough to get timing breakdowns.

### 1a. Time the DAgger iteration components

Write a standalone profiling script (`scripts/profile_dagger.py`) that:

1. Loads the model and a small buffer (smoke-test size is fine)
2. Runs **3 full DAgger iterations** and times each component separately:
   - `curriculum.sample_next_env()` — should be negligible
   - **Model rollout** (single episode): env.reset + loop of (select_action or greedy_sample + env.step) × max_steps
   - **Oracle rollout** (single episode): env.reset + loop of (get_oracle_action + env.step) × max_steps
   - **Efficiency filter** — should be negligible
   - **Buffer add** — should be negligible
   - **Single gradient step**: buffer.sample + forward + loss + backward + optimizer.step + EMA update
3. Reports a table like:
   ```
   Component                  | Total (s) | Per-call (ms) | % of iteration
   Model rollout (10 eps)     |     X.XX  |       XXX     |      XX%
   Oracle rollout (10 eps)    |     X.XX  |       XXX     |      XX%
   Gradient steps (100)       |     X.XX  |       XXX     |      XX%
   Buffer sampling (100)      |     X.XX  |       XXX     |      XX%
   Other                      |     X.XX  |       XXX     |      XX%
   TOTAL per iteration        |     X.XX  |               |     100%
   ```
4. Within the model rollout, break down further:
   - Time spent in `model.forward()` (the denoising passes)
   - Time spent in `env.step()` 
   - Time spent in `env.reset()`
   - Overhead (tensor creation, numpy↔torch conversion, etc.)
5. Report device info: GPU name (if available), CUDA version, CPU count

**Do not proceed to Phase 2 until you have this table.** The numbers determine which optimisations matter.

### 1b. Quick memory audit

While profiling, also measure:
- Peak GPU memory during a gradient step (with current batch_size)
- Peak GPU memory during model forward (single sample, as used in collection)
- How much headroom exists for batch_size increases

---

## Phase 2 — Plan optimisations based on profile data

After profiling, create a prioritised optimisation plan. Here are the candidates — **rank them by expected impact based on your actual profiling numbers**, not by theoretical appeal.

### Candidate A: Parallel episode collection (multiprocessing)

The 10 DAgger episodes per iteration are fully independent. Each one runs (model_rollout → oracle_rollout → efficiency_filter) on a different env/seed. Use Python `multiprocessing` to run them in parallel.

**Why multiprocessing, not vectorised envs:**
- Each episode has paired model+oracle rollouts on the same seed — this pairing must be preserved per-worker
- MiniHack/NLE uses process-level state — `AsyncVectorEnv` works but adds complexity for marginal benefit over raw multiprocessing
- The model rollout within each episode is sequential (replan every 16 steps, execute plan) — vectorising that requires managing N plan states
- Multiprocessing is much less invasive: wrap `collect_episode()` in a worker function, spawn N workers, gather results

**Implementation sketch:**
```python
from multiprocessing import Pool

def _collect_worker(args):
    """Worker: runs one (model_rollout, oracle_rollout, filter) episode."""
    env_id, seed, model_state_dict, cfg = args
    # Load model in worker (each process gets its own copy)
    model = build_model(cfg)
    model.load_state_dict(model_state_dict)
    model.eval()
    # Run paired rollouts
    return collect_episode(model, env_id, seed, cfg)

def collect_parallel(model, episodes, cfg, num_workers=None):
    num_workers = num_workers or min(episodes, os.cpu_count() or 4)
    # Prepare args: curriculum samples envs, generates seeds
    args = [(env_id, seed, model.state_dict(), cfg) for env_id, seed in tasks]
    with Pool(num_workers) as pool:
        results = pool.map(_collect_worker, args)
    return results
```

**Caveats to handle:**
- Model weights must be serialised to each worker (state_dict). For a ~2M param model this is fast.
- Each worker creates its own MiniHack env — NLE handles this fine in subprocesses.
- GPU inference in workers: if using CUDA, either (a) keep model on CPU in workers (simpler, oracle is CPU-bound anyway), or (b) use `torch.cuda` with `spawn` start method.
- Curriculum updates happen AFTER all episodes return — this is a slight deviation from the reference where updates are interleaved, but the curriculum adapts slowly (100-episode queue) so this is negligible.

**Config:**
```yaml
num_collection_workers: 4   # 0 = sequential (reference behaviour)
```

### Candidate B: Batched model inference during collection

Currently, the model processes observations one at a time during collection: `[1, 9, 9]` local, `[1, 21, 79]` global. If using GPU, this drastically underutilises it.

Two approaches depending on whether you implement Candidate A:

**Without parallel collection (sequential episodes):** Not much to batch — you're processing one env at a time. The denoising loop runs `diffusion_steps` forward passes per replan, each on a single sample. You could batch the K denoising steps into one pass with `[K, seq_len]` but the steps are sequential (each depends on the previous mask), so this doesn't work.

**With parallel collection (or vectorised envs):** If N workers collect simultaneously and you centralise model inference, you can batch observations from N environments into `[N, 9, 9]` and run a single forward pass. This requires an inference server pattern:
- Workers send observations to a central queue
- Main process batches them and runs the model
- Results are sent back to workers

This is higher complexity. **Only pursue if profiling shows model.forward() is the bottleneck during collection AND you're on GPU.** If oracle BFS or env.step() dominates, batched inference won't help.

### Candidate C: Optimise the oracle BFS

The oracle runs BFS on a 21×79 grid for every step. BFS is O(V+E) ≈ O(1659) per call, called up to 500 times per episode. Profile whether this is actually slow — Python BFS on a small grid should be fast, but the overhead of creating deques and sets each call adds up.

**Quick wins:**
- Pre-allocate the visited array as a numpy boolean array instead of a Python set
- Use a C-level BFS via scipy.sparse.csgraph or a simple Cython extension
- Cache the BFS result and only recompute when the agent moves (the map doesn't change between steps in most MiniHack envs)

**Only pursue if oracle_rollout shows up as >20% of iteration time in profiling.**

### Candidate D: Mixed-precision training (AMP)

Add `torch.cuda.amp.autocast` to the gradient step and `GradScaler` for loss scaling. The transformer and CNN layers benefit from FP16, especially with batch_size=1024.

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(local, global_obs, noisy_input, t)
    loss = compute_loss(output, targets, mask)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
scaler.step(optimizer)
scaler.update()
```

**Only useful if on GPU and gradient steps are a significant fraction of iteration time.** The `Trainer` already has a `self.scaler` placeholder — wire it up.

**Config:**
```yaml
use_amp: true   # false on CPU
```

### Candidate E: Torch compile

`torch.compile(model)` can fuse operations and eliminate Python overhead in the forward pass. Test whether it works with the current model (some dynamic shapes in the diffusion loop may cause recompilation).

```python
if cfg.torch_compile and hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")
```

**Test on the gradient step first** (fixed shapes), then on inference (variable valid_actions might trigger recompilation).

**Config:**
```yaml
torch_compile: false   # experimental — enable and test
```

### Candidate F: Dataloader for gradient steps

Currently, `buffer.sample()` does random numpy indexing + array construction on every call. With 100 gradient steps per iteration at batch_size=1024, that's 102,400 samples constructed from Python lists.

Replace with a proper `torch.utils.data.DataLoader` with:
- `pin_memory=True` (for GPU)
- `num_workers=2` (prefetch next batch while GPU trains on current)
- `persistent_workers=True`

The buffer already stores data in memory, so the "dataset" is just an index sampler over the buffer's internal list. This eliminates the per-step numpy→torch conversion overhead.

**Only pursue if buffer.sample() shows up as >5% of gradient step time.**

### Candidate G: Reduce env creation overhead

The reference creates a **new** `AdvancedObservationEnv` (and underlying `gym.make`) for every episode — both for the model rollout and the oracle rollout. `gym.make` for MiniHack involves loading the NLE shared library and des file parsing. 

**Pool environments:** Create a pool of pre-initialised envs per env_id and reuse them via `env.reset(seed=X)` instead of `gym.make()` each time.

```python
class EnvPool:
    def __init__(self, env_ids, pool_size=2):
        self._pools = {eid: [make_env(eid) for _ in range(pool_size)] for eid in env_ids}
    
    def acquire(self, env_id):
        return self._pools[env_id].pop()
    
    def release(self, env_id, env):
        self._pools[env_id].append(env)
```

**Only pursue if env creation shows up as significant in profiling (>100ms per env.reset).**

---

## Phase 3 — Implement (top 3 only)

Based on your profiling numbers, **implement only the top 3 highest-impact optimisations**. For each one:

1. Implement behind a config flag so it can be disabled
2. Re-run the profiling script to measure the speedup
3. Run the smoke test to verify correctness is preserved
4. Commit with a message stating the measured speedup

**Do not implement all candidates.** More code means more bugs. Pick the 3 that move the needle most and leave the rest as documented `# TODO: Candidate X — <measured reason it's lower priority>`.

---

## Phase 4 — Validate

After implementing optimisations:

1. **Correctness check:** Run smoke test end-to-end. Compare final loss and win rate against the pre-optimisation smoke run (they should be statistically similar, not identical due to RNG).

2. **Speed check:** Run the profiling script again and produce a before/after comparison table:
   ```
   Component              | Before (s) | After (s) | Speedup
   Model rollout (10 eps) |      X.XX  |     X.XX  |   X.Xx
   Oracle rollout         |      X.XX  |     X.XX  |   X.Xx
   Gradient steps (100)   |      X.XX  |     X.XX  |   X.Xx
   TOTAL per iteration    |      X.XX  |     X.XX  |   X.Xx
   ```

3. **Memory check:** Confirm peak GPU memory hasn't increased beyond available VRAM.

4. **Smoke test wall-clock:** Time the full 30-iteration smoke test before and after. Report the total time reduction.

---

## Phase 5 — Document

Update `CLAUDE.md` with:
- New config keys added (with defaults and descriptions)
- Performance notes: which optimisations are active, expected speedups
- Any new dependencies or requirements (e.g. `multiprocessing` start method)

Update `defaults.yaml` with all new config keys and their defaults.

Add a section to `README.md` or `docs/performance.md`:
```markdown
## Performance Tuning

### Collection parallelism
Set `num_collection_workers: N` to run DAgger episodes in parallel.
Default: 4. Set to 0 for sequential collection (matches reference).

### Mixed precision
Set `use_amp: true` for FP16 training on GPU. Reduces memory ~40% and
speeds up gradient steps ~1.5-2x. Default: true on CUDA, false on CPU.

### Torch compile
Set `torch_compile: true` to use torch.compile for fused kernels.
Experimental — may cause issues with dynamic shapes in sampling.
```
