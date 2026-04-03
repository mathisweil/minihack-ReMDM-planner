# Fix collection bottleneck in DAgger training loop

## Goal

Investigate why data collection is the main bottleneck in my DAgger training loop, then implement concrete fixes to maximise throughput. This is about the core training pipeline (`src/planners/collect.py`, `src/planners/online.py`, `src/envs/minihack_env.py`), **not** the experiments directory.

## Hardware context

**Run A — RTX 3090 Ti (24GB VRAM):**
- `dagger_batch_size: 3584`, `grad_steps_per_iteration: 100`, `use_amp: false`, `num_collection_workers: 8`
- Iteration ~100-110s, train ~51-53s, samples/sec ~6900, GPU mem ~20GB
- Collection is the bottleneck; `env_steps_per_sec` unstable and often low

**Run B — H200 (only 8GB VRAM available):**
- `dagger_batch_size: 2048`, `grad_steps_per_iteration: 300`, `use_amp: true`, `num_collection_workers: 8`
- Iteration ~180-220s, collection ~120-165s, train oscillates 30-96s, samples/sec flips between ~19k and ~6.5k
- GPU mem only ~6.4GB — massively underutilised
- Increasing `grad_steps_per_iteration` to 300 made training slower without fixing the real problem

## Phase 1 — Investigate

Read and trace the full collection codepath. Start from `src/planners/online.py` (the DAgger loop), then follow into `src/planners/collect.py` (rollout logic), `src/envs/minihack_env.py` (env wrapper + oracle), and `src/diffusion/sampling.py` (plan generation during collection).

Answer these questions with evidence from the code:

1. **Collection architecture:** Is collection serial, multiprocess, or vectorised? Is there any async overlap with training, or is it strictly collect-then-train?

2. **`num_collection_workers`:** What do workers actually do? Are they truly parallel? What are they blocked on — env steps, resets, model inference, IPC serialisation?

3. **Environment parallelisation (`num_envs`):** Does the codebase use vectorised environments (e.g. `gym.vector.AsyncVectorEnv`)? If not, are envs stepped one at a time inside each worker? Would adding vectorisation produce larger batched forward passes or just more Python overhead?

4. **Policy inference batching:** During collection, is model inference batched across environments/workers, or does each worker run its own small forward pass? Are there many tiny CUDA kernel launches?

5. **Replanning cost:** How often does replanning trigger (`replan_every: 16`)? How expensive is each replan call (full diffusion sampling with K=10 denoising steps)? Is this the dominant cost inside collection?

6. **Reset / episode overhead:** Are MiniHack resets slow? Are episodes frequently hitting max horizon? Is there stuck-episode detection or early termination?

7. **Training/collection interaction:** Is the main loop `collect → train → repeat` with no overlap? Could a producer-consumer queue help?

## Phase 2 — Plan

Based on your findings, propose a ranked list of changes ordered by expected impact. For each change, specify:
- Which files and functions to modify
- What the change is
- Expected throughput improvement (rough estimate)
- Any risks or tradeoffs

Prioritise these categories (most likely high-impact first):
- Batched policy inference across environments during collection
- Vectorised environment stepping (AsyncVectorEnv or similar)
- Async collection/training overlap
- Reducing replanning frequency or cost
- Fixing any serial bottlenecks, unnecessary syncs, or Python-level overhead

Also recommend optimal configs for each hardware setup:

**3090 Ti 24GB** — current config is `batch_size: 3584, grad_steps: 100, amp: false`. Verify or improve.

**H200 8GB available** — current config is `batch_size: 2048, grad_steps: 300, amp: true`. The 300 grad steps seem counterproductive — recommend a fix.

## Phase 3 — Execute

Implement the top changes. After each significant change, briefly note what you changed and why.

Also add detailed profiling instrumentation so I can measure the impact. Add timing breakdowns for:
- Per-episode: env step time, policy inference time, reset time, replanning time
- Per-iteration: total collection time, total train time, worker idle/wait time, buffer write time
- Log these to the existing W&B logging under a `speed/` or `profile/` namespace

## Hypotheses to confirm or reject

I suspect these are true — confirm or reject each with code evidence:

1. Collection is not truly batched — each worker runs independent small forward passes
2. Environment parallelisation is weak or absent — envs are stepped serially within workers
3. Policy forward passes during collection are too small and too frequent
4. Replanning every 16 steps with K=10 denoising steps is expensive
5. Workers spend significant time waiting, serialising, or blocked on the GIL
6. The H200 run is underutilised because VRAM constraints prevent good batching
7. `grad_steps_per_iteration: 300` was a misguided attempt to compensate for slow collection

## Constraints

- Do not modify anything under `experiments/`
- Maintain backward compatibility with existing checkpoint format
- Keep the BFS oracle, efficiency filter, and curriculum sampling logic intact
- All config additions should have sensible defaults in `configs/defaults.yaml`
- Changes should not break `--mode smoke`, `--mode offline`, or `--mode inference`
