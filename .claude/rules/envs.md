---
paths: src/envs/**
---

# Environment Rules — src/envs/

> Also load @.claude/rules/python-idioms.md when editing this directory.

---

## Module responsibilities

| File | Responsibility |
|---|---|
| `minihack_env.py` | `AdvancedObservationEnv` wrapper + BFS oracle + reward shaping |

- **MUST NOT** add training logic, diffusion sampling, or model forward passes to this directory.
- **MUST NOT** add W&B logging here. Emit metrics through `src/planners/logging.py`.
- **MUST NOT** import from `minihack_reference/` — reimplement any needed behaviour in this file.

---

## Observation format

`AdvancedObservationEnv` wraps any MiniHack gym environment and returns a tuple observation at every `step()` and `reset()`:

| Stream | Shape | dtype | Content |
|---|---|---|---|
| `local_crop` | `(9, 9)` | `int16` | Glyph IDs in a 9×9 window centred on the agent `@`. Padded with `PAD_TOKEN = 13` at map borders. |
| `global_map` | `(21, 79)` | `int16` | Full-map glyph array from `obs["glyphs"]`. |

- **MUST** centre the local crop on the agent's `(row, col)` position. Use `obs["blstats"]` to find the agent.
- **MUST** pad with `PAD_TOKEN = 13` (not `0` or any glyph ID) when the crop window extends beyond the map boundary. PAD is a reserved token; using a real glyph ID here would corrupt observations near edges.
- **MUST** reset `visited_tiles` and `prev_bfs_dist` at every `reset()` call. Failing to do so leaks state across episodes.

---

## Action space

12 discrete actions (indices 0–11):

| Index | Direction |
|---|---|
| 0–3 | Cardinal: N, E, S, W |
| 4–7 | Diagonal: NE, SE, SW, NW |
| 8 | UP (go up stairs) |
| 9 | DWN (go down stairs) |
| 10 | WAIT |
| 11 | KICK (opens closed doors `+`) |

- **MUST** mask out actions beyond `env.action_space.n` at inference time (`logits[:, :, env.action_space.n:] = -inf`). Not all MiniHack envs expose all 12 actions.
- Masking **MUST** happen in `src/diffusion/sampling.py`, not inside this file. The environment wrapper does not touch logits.

---

## Reward shaping

`step()` augments the raw MiniHack reward with four shaped components applied in this order:

| Component | Value | Condition |
|---|---|---|
| Win bonus | `+20.0` | Episode won (staircase reached) |
| BFS progress | `+0.5 * (prev_bfs_dist - curr_bfs_dist)` | Positive when closer to staircase |
| Exploration | `+0.05` | New tile visited this step |
| Step penalty | `-0.01` | Every step |

- **MUST** preserve these exact coefficients. They are tuned to balance exploration vs. goal-directedness. Any change requires updating `README.md` and this rule.
- BFS distance is to the staircase `>` glyph. If no staircase is visible, BFS distance shaping is skipped (returns `0.0` for that component).
- The exploration bonus is computed against the `visited_tiles` set maintained across the episode.

---

## BFS oracle

The BFS oracle implements a 5-tier priority policy. **MUST** check tiers in this exact order:

1. **Kick adjacent doors:** if a closed door `+` is adjacent to the agent, return `KICK` action (11).
2. **BFS to staircase:** if staircase `>` is visible in `global_map`, BFS the shortest path and return the first step.
3. **BFS to frontier:** BFS to the nearest unexplored tile (adjacent to an explored tile but not yet visited).
4. **BFS to farthest explored tile:** BFS to the explored tile furthest from current position (maximises map coverage).
5. **Random cardinal:** choose uniformly from {N, E, S, W} as a fallback.

- **MUST NOT** reorder tiers — the priority is load-bearing for oracle quality.
- BFS operates on the walkable glyph set. Walls, water, lava, and `+` (closed doors) are impassable unless tier 1 is active.

---

## Staircase detection

```python
STAIRCASE_GLYPH = ord('>')   # or the integer glyph ID — verify against NLE glyph table
```

- **MUST** use the correct NLE glyph ID for `>`. Verify against the NLE glyph table if unsure — using the ASCII code `62` is a common mistake if NLE remaps glyphs.
- `find_staircase(global_map)` **MUST** return `None` (or `(-1, -1)`) when no staircase is visible. Callers must handle this gracefully.

---

## Environment registry

The 4 in-distribution and 3 OOD environment IDs are defined in `configs/defaults.yaml` under `id_envs` and `ood_envs`. **MUST NOT** hardcode environment ID strings in `src/envs/` — read them from `cfg`.

| Split | Environment IDs |
|---|---|
| In-distribution | `MiniHack-Room-Random-5x5-v0`, `MiniHack-Room-Random-15x15-v0`, `MiniHack-Corridor-R2-v0`, `MiniHack-MazeWalk-9x9-v0` |
| OOD (zero-shot) | `MiniHack-Room-Dark-15x15-v0`, `MiniHack-Corridor-R5-v0`, `MiniHack-MazeWalk-45x19-v0` |
