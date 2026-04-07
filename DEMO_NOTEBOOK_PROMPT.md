# Claude Code: Create Demo Notebook

## Objective

Create a self-contained Jupyter notebook (`demo.ipynb`) for our COMP0258 coursework submission. It will be **inside the submitted zip alongside the full source code**, so it can import from the codebase. The marker will upload it to Google Colab and run it top-to-bottom.

### Submission Guidelines (verbatim compliance required)

The guidelines state:
> "prepare a self-contained notebook demonstrating your approach so that it can be uploaded and executed on Colab"
> "If your project has a stronger machine learning component then please avoid training a model in the notebook. Instead, give us an **easy way** to test your system on unseen inputs."
> "In case you need to train models, do this training offline and load the model into Colab."
> "demonstrate that your findings are reproducible"

**Our project is both**: a strong ML component (diffusion planning model) AND explorative research (ablation study proving RL fine-tuning fails for discrete diffusion). The notebook must therefore:

1. **Load the pre-trained model into Colab** (trained offline, no training in notebook)
2. **Give the marker an easy way to test on unseen inputs** — they should be able to change environment names, seeds, episode counts, or even supply custom `.des` scenario files, and watch the model run
3. **Demonstrate our findings are reproducible** — live inference should approximately match reported win rates, and the ablation analysis should be clearly presented
4. **Demonstrate the approach** — show the ReMDM denoising process, the dual-stream architecture, and how the planner generates action sequences

---

## 1. Project Summary (You Already Know This)

We apply **ReMDM** (Remasking Discrete Diffusion Models) to action-sequence planning in **MiniHack** navigation environments. A dual-stream transformer generates 64-step action plans by iteratively denoising masked token sequences, conditioned on a 9×9 local crop and the full 21×79 dungeon map.

**Training pipeline:** DAgger with BFS oracle supervision → RL fine-tuning ablation study (25 ablations across 4 groups).

**Core finding:** DAgger-trained ReMDM achieves 67.5% ID win rate (vs <6% for PPO/A2C/DQN) and 10% OOD zero-shot transfer — but NO RL ablation meaningfully improves on the DAgger checkpoint. This reveals a "double intractability": (1) standard policy gradients are intractable for masked discrete diffusion (no closed-form log π_θ), and (2) the Return-Weighted ELBO surrogate degenerates because episode returns lack sufficient variance to discriminate trajectories.

---

## 2. Architecture (Exact Specification)

**`LocalDiffusionPlannerWithGlobal`** (~5.2M parameters):

```
Local stream:   9×9 glyphs → Embedding(6000,64) → CNN(64→32→64) → Linear → 1 token (256D)
Global stream:  21×79 glyphs → Embedding(6000,32) → CNN(32→32→64) → Pool(2,4) → 8 tokens (256D)
                Goal head: mean(global) → MLP → [B,2] staircase coords (auxiliary loss)
                Gate: sigmoid(learnable scalar, init=-3.0) * global_tokens
Action stream:  Embedding(14, 256) + timestep_emb(100, 256) + position_emb(64, 256)
Transformer:    concat [1 + 8 + 64 = 73 tokens] → 4-layer encoder (256D, 4 heads, pre-norm)
Output head:    last 64 tokens → Linear(256, 12) → action logits
```

Input: `(local_obs, global_obs, noisy_action_seq, t_discrete)` → Output: `{"actions": [B,64,12], "goal_pred": [B,2]}`

Action vocabulary: 12 actions (0-11) + MASK token (12) + PAD token (13) = 14 total embeddings.

---

## 3. Diffusion Process (Exact Specification)

**Forward (MDLM):** Each action token independently replaced with MASK (token 12) with probability `1 - alpha(t)`. Linear or cosine schedule. PAD tokens (13) never masked.

**Loss:** Cross-entropy on masked positions only, averaged globally across batch.

**Reverse sampling (ReMDM):** Over K denoising steps (default 10):
1. Model predicts logits → temperature scaling + top-K filtering
2. Sample predictions → per-token confidence scores
3. MaskGIT unmask: commit the `n_unmask` highest-confidence masked positions
4. ReMDM remask: stochastically re-mask committed positions with probability η·(1−r) where r is progress through chain
5. Final step: commit all remaining

**Greedy sampling** (used in DAgger collection): argmax decoding, no temperature/top-K/remasking, 5 denoising steps.

---

## 4. Environments (Exact Names)

**In-distribution (training):**
| Environment | Description |
|---|---|
| `MiniHack-Room-Random-5x5-v0` | Small random room |
| `MiniHack-Room-Random-15x15-v0` | Large random room |
| `MiniHack-Corridor-R2-v0` | Two-room corridor |
| `MiniHack-MazeWalk-9x9-v0` | Small maze |

**Out-of-distribution (zero-shot):**
| Environment | Description |
|---|---|
| `MiniHack-Room-Dark-15x15-v0` | Dark room (limited visibility) |
| `MiniHack-Corridor-R5-v0` | Five-room corridor |
| `MiniHack-MazeWalk-45x19-v0` | Large maze |

The codebase also supports custom `.des` scenario files via `--des` flag — the notebook should expose this.

---

## 5. Results to Reproduce (Exact Numbers From Paper)

### Table 1: MiniHack Imitation Learning Baselines
| Method | ID Win% | ID Steps | OOD Win% | OOD Steps |
|---|---|---|---|---|
| PPO | 0.5 | 400 | 0.0 | 767 |
| A2C | 6.0 | 384 | 0.5 | 760 |
| DQN | 4.5 | 388 | 1.3 | 762 |
| PPO-RNN | 4.0 | 386 | 0.5 | 765 |
| CNN+MLP (Offline BC) | 3.0 | 395 | 0.5 | 765 |
| ReMDM (Offline BC) | 58.2 | – | 7.0 | – |
| ReMDM (DAgger) | 67.5 | 132 | 10.0 | 402 |

### Table 2: MiniHack RL Ablation Results (from DAgger checkpoint, win rate 67.5%)
| Group | Method | Win(%) | Δ |
|---|---|---|---|
| – | DAgger (pretrained) | 67.5 | – |
| – | Baseline RL | 60 | −7.5 |
| A | LLRD | 62.5 | −5 |
| A | KL penalty (λ=1.0) | 61.3 | −6.2 |
| A | LoRA | 60 | −7.5 |
| B | BC on wins | 57.5 | −10 |
| B | Low-t only | 60.0 | −7.5 |
| C | Frozen backbone | 42.5 | −25 |

### Table 3: Per-Environment Win Rates (all 25 ablations)
```
Method                  Random-5x5  Random-15x15  Corridor-R2  MazeWalk-9x9
action_diversity        0.8500      0.8000        0.5500       0.2000
advantage_clip          0.9000      0.9500        0.2500       0.1500
attention_only          1.0000      0.9000        0.5000       0.3500
baseline_rl             1.0000      0.7000        0.3000       0.1000
bc_wins                 0.9000      0.7000        0.4000       0.1000
entropy_bonus           0.9000      0.4000        0.4500       0.1500
ewc                     1.0000      0.8500        0.6000       0.3000
ffn_only                1.0000      1.0000        0.3500       0.3000
frozen_backbone         0.9500      0.9000        0.5000       0.4000
gradient_surgery        1.0000      0.9000        0.4500       0.2500
head_only               1.0000      0.8000        0.3000       0.3000
kl_penalty              0.9500      1.0000        0.3000       0.3500
layer_ablation_top1     0.9000      0.7500        0.2000       0.2000
layer_ablation_top2     0.9500      0.9000        0.3500       0.2500
layer_ablation_top3     0.9500      0.7500        0.4000       0.4500
llrd                    0.8000      0.9000        0.4000       0.2500
lora                    1.0000      0.7500        0.2000       0.2000
low_t                   1.0000      0.6000        0.4500       0.1500
mixed_replay            0.9500      0.7500        0.4500       0.2000
normalized_adv          0.1000      0.0000        0.1000       0.1000
reward_filtering        0.8500      0.8000        0.4000       0.1500
reward_model            0.9500      0.9000        0.3500       0.2500
running_stats           0.9500      1.0000        0.3500       0.2000
t_curriculum            1.0000      0.8000        0.3000       0.1500
trust_region_kl         0.9500      0.7500        0.4500       0.2500
```

---

## 6. Checkpoint Format (Exact Schema)

The DAgger checkpoint at `checkpoint_final/` contains:
```python
{
    "model_state_dict":     ...,   # training weights
    "ema_state_dict":       ...,   # EMA weights (USE THESE for inference)
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "curriculum_state":     {...},
    "iteration":            int,
    "wandb_run_id":         str | None,
    "rng_states":           {"torch", "numpy", "python"},
}
```
**Always load `ema_state_dict` into the model for inference** (default behaviour of `main.py --mode inference`).

---

## 7. All 25 Ablations (Exact Registry)

| Group | Name | What it tests |
|---|---|---|
| Baseline | `baseline_rl` | Standard return-weighted ELBO |
| A: Regularisation | `kl_penalty` | Soft KL constraint vs pretrained |
| A | `ewc` | Elastic Weight Consolidation (Fisher diagonal) |
| A | `llrd` | Layer-wise Learning Rate Decay |
| A | `lora` | Low-Rank Adaptation of attention projections |
| A | `mixed_replay` | Offline data mixed into online batches |
| A | `trust_region_kl` | Hard KL trust region via quadratic barrier |
| B: Training Signal | `t_curriculum` | Anneal t range high-to-low over training |
| B | `entropy_bonus` | Entropy regularisation for action diversity |
| B | `gradient_surgery` | PCGrad: project conflicting RL/BC gradients |
| B | `advantage_clip` | PPO-style advantage clipping |
| B | `normalized_adv` | Std-normalised advantages (GRPO-style) |
| B | `bc_wins` | Uniform ELBO on win windows only |
| B | `low_t` | ELBO restricted to low-t regime |
| C: Architecture | `frozen_backbone` | Only train output head |
| C | `head_only` | Only train final linear projection |
| C | `attention_only` | Only train attention weights (Q/K/V/O) |
| C | `ffn_only` | Only train FFN layers |
| C | `layer_ablation_top1` | Only train top-1 transformer block |
| C | `layer_ablation_top2` | Only train top-2 transformer blocks |
| C | `layer_ablation_top3` | Only train top-3 transformer blocks |
| D: Data Quality | `reward_filtering` | Top-75th-percentile return windows only |
| D | `running_stats` | EMA running mean/std for advantage normalisation |
| D | `action_diversity` | Discard degenerate all-same-action plans |
| D | `reward_model` | MLP reward model soft-weighting |

---

## 8. Pre-Computed Output Files

All experiment outputs are in `experiments/rl_finetuning/outputs/minihack_final/`.

Run `ls` to confirm what's actually there. Only include the files that tell the story — do NOT dump everything.

### Figures to INCLUDE (key narrative):
- `score_comparison.png` — "No ablation exceeds pretrained 67.5%"
- `group_comparison.png` — "Group C worst; all groups below pretrained"
- `per_env_delta.png` — "Easy envs at ceiling, hard envs no improvement"
- `grad_alignment.png` — "RL gradients near-zero cosine similarity → uninformative"
- `score_delta.png` — "All deltas negative; normalized_adv catastrophic"

### Figures to INCLUDE if present and useful:
- `gradient_conflict_map.png`, `repr_drift.png`, `diagnosis_decision_tree.png`

### Figures to SKIP:
- Individual `train_{name}.png` per-ablation curves (25+ images, clutters notebook)
- `cka_similarity.png`, `t_bin_norms.png`, `t_ratio.png`, `win_rate.png` — too granular

### Tables to INCLUDE:
- `main_results.csv` — core quantitative results
- `hypothesis_verdicts.csv` — verdict per ablation (the punchline)

### Tables to SKIP:
- `group_summary.csv`, `gradient_diagnostics.csv`, `repr_drift.csv`, `forgetting_analysis.csv`, `per_env_win_rates.csv` — supplementary

---

## 9. Dependencies (From README)

Key deps: `nle>=1.2.0` (NetHack-LE fork), `minihack`, `torch>=2.11.0`, `wandb`, `polars`, `orjson`, `scipy`. Read `pyproject.toml` for full list.

Colab system packages needed for NLE:
```bash
apt-get install -y cmake build-essential bison flex libbz2-dev
```

---

## 10. Key Source Files You MUST Read

I cannot provide source code. Read these to understand the inference pipeline:

| File | Why |
|---|---|
| `src/models/denoiser.py` | Model constructor, `forward()`, `ModelEMA` |
| `src/diffusion/sampling.py` | ReMDM sampling function |
| `src/envs/minihack_env.py` | `AdvancedObservationEnv` wrapper |
| `src/planners/` | Planner class (model + sampling + env) — likely simplest inference path |
| `main.py` | The `inference` mode branch — exact evaluation sequence |
| `configs/defaults.yaml` | Default hyperparameters |
| `scripts/hf_upload.py` | HuggingFace Hub upload interface |
| `pyproject.toml` | Exact dependency list |

---

## 11. Checkpoint Distribution

Check `checkpoint_final/` for file sizes.
1. If under ~500MB → upload to HuggingFace Hub (read `scripts/hf_upload.py`). Notebook uses `hf_hub_download()`.
2. Fallback → notebook references local path with clear marker instructions.

**Goal: marker runs notebook, everything downloads automatically.**

---

## 12. Notebook Structure

The notebook sits **inside the zip alongside the source code**. It can do relative imports from the codebase (e.g. `from src.models.denoiser import ...`). The marker unzips, opens the notebook in Colab, and runs.

Front-load the live demo — that's what the guidelines emphasize ("easy way to test your system on unseen inputs"). Ablation analysis comes after as evidence for findings.

---

### Cell 0: Setup & Configuration

```python
#========== MARKER: CHANGE THESE TO TEST ON UNSEEN INPUTS ==========
QUICK_MODE = False          # True → 10 episodes, faster
EPISODES_PER_ENV = 10 if QUICK_MODE else 50
SEED = None                 # Set an integer for reproducibility

# Environments to evaluate (change these to test on different/unseen environments)
ID_ENVS = [
    "MiniHack-Room-Random-5x5-v0",
    "MiniHack-Room-Random-15x15-v0",
    "MiniHack-Corridor-R2-v0",
    "MiniHack-MazeWalk-9x9-v0",
]
OOD_ENVS = [
    "MiniHack-Room-Dark-15x15-v0",
    "MiniHack-Corridor-R5-v0",
    "MiniHack-MazeWalk-45x19-v0",
]

# Optional: path to a custom .des scenario file (set to None to skip)
CUSTOM_DES_FILE = None  # e.g. "environments/custom_level.des"
#====================================================================
```

Then: system deps (`apt-get`), pip install, download checkpoint. The source code is already in the zip, so no git clone needed — just make sure the working directory and `sys.path` are set correctly.

---

### Cell 1: Project Overview (Markdown)

Brief (1-2 paragraphs): problem (planning in procedurally-generated MiniHack), approach (ReMDM discrete diffusion planner with dual-stream CNN + transformer), training (DAgger with BFS oracle), research question (can RL fine-tuning improve beyond imitation learning?).

---

### Cell 2: Load Pre-Trained Model

- Read `src/models/denoiser.py` for constructor args, `configs/defaults.yaml` for values
- Load checkpoint from `checkpoint_final/`, extract `ema_state_dict`, load into model
- Print parameter count and brief architecture summary
- **This demonstrates:** "load the model into Colab" (guideline compliance)

---

### Cell 3: Test on Environments (Live Inference) ⭐

**This is what the guidelines ask for: "easy way to test your system on unseen inputs."**

- Use the existing inference pipeline (read `main.py --mode inference` and `src/planners/`)
- Evaluate on `ID_ENVS + OOD_ENVS` from the config cell, `EPISODES_PER_ENV` episodes each
- If `CUSTOM_DES_FILE` is set, also evaluate on that
- Print per-environment win rate + mean steps in a clean table
- Show side-by-side comparison with reported results (Table 1 hardcoded)
- Note: environments are procedurally generated so each run uses unseen layouts — the marker is already testing on unseen inputs by default. Different seeds produce different maps.
- **The marker can change `ID_ENVS`, `OOD_ENVS`, `SEED`, or provide a custom `.des` file in Cell 0 to test on truly novel inputs.**

---

### Cell 4: Visualise Agent Behaviour (Live) ⭐

- Run 1-2 episodes and capture observations at key timesteps
- Plot the 9×9 local crop and/or 21×79 global map (start → mid → win/lose)
- Read `src/envs/minihack_env.py` for observation structure
- **This demonstrates the approach:** the marker sees the agent actually navigating

---

### Cell 5: Visualise ReMDM Denoising (Live) ⭐

- Show how a fully-masked `[MASK]*64` action sequence gets iteratively unmasked over K=10 denoising steps
- Read `src/diffusion/sampling.py` to capture intermediate states
- Display as a grid: rows = steps, columns = action positions, colour = masked vs committed
- **This demonstrates the approach:** the marker sees the core mechanism (iterative refinement via remasking)

---

### Cell 6: Imitation Learning Context (Static)

- Hardcode Table 1 as pandas DataFrame (baselines trained separately, not part of this demo)
- Brief markdown: ReMDM DAgger 67.5% vs <6% for model-free baselines; 10% OOD zero-shot
- **This demonstrates findings are reproducible:** live results from Cell 3 should approximately match

---

### Cell 7: RL Ablation Findings (Pre-Computed Figures)

**This section demonstrates our research findings** — the ablation study is the core intellectual contribution.

Load and display key summary figures from `experiments/rl_finetuning/outputs/minihack_final/`:
- `score_comparison.png` — "No ablation exceeds pretrained 67.5%"
- `group_comparison.png` — "Group C (partial-parameter) worst; all groups below pretrained"
- `per_env_delta.png` — "Easy envs stay at ceiling; hard envs show no improvement"
- `grad_alignment.png` — "RL gradients have near-zero cosine similarity → uninformative signal"
- `score_delta.png` — "All deltas negative; normalized_adv catastrophically collapses"
- Plus 2-3 more if they exist and add value (`gradient_conflict_map.png`, `repr_drift.png`)

Each figure gets a brief markdown cell explaining the takeaway. Do NOT include per-ablation `train_{name}.png` curves.

---

### Cell 8: Ablation Results Tables (Pre-Computed)

- `main_results.csv` → pandas DataFrame (core quantitative results)
- `hypothesis_verdicts.csv` → pandas DataFrame (verdict per ablation — the punchline)

---

### Cell 9: Conclusions (Markdown)

1. ReMDM DAgger: strong imitation learner (67.5% ID, 10% OOD, 1000× faster convergence than offline BC)
2. No RL ablation (25 tested across regularisation, training signal, architecture, data quality) improves beyond DAgger
3. Double intractability: no closed-form log π_θ for policy gradients + near-uniform advantages in Return-Weighted ELBO
4. Partial-parameter methods (Group C) universally collapse — representations need full-parameter updates
5. Discrete diffusion planners are fundamentally imitation learning architectures in their current form
6. Open problem: tractable RL objective for masked discrete diffusion
7. Promising directions: Q-guided remasking (bias token selection at inference without modifying weights), SDE reformulations (treat each unmasking as a finite-horizon MDP action)

---

## 13. What Only You Can Do

Everything above is from the README, paper, and project knowledge. You MUST discover by reading source:

1. **Exact constructor arguments** for `LocalDiffusionPlannerWithGlobal` → read `src/models/denoiser.py`
2. **Exact inference call sequence** → read `main.py`, `src/planners/`, `src/diffusion/sampling.py`
3. **Environment wrapper instantiation** → read `src/envs/minihack_env.py`
4. **Actual files in** `experiments/rl_finetuning/outputs/minihack_final/` → run `ls`
5. **Actual files in** `checkpoint_final/` → run `ls` + check sizes
6. **Exact dependency versions** → read `pyproject.toml`
7. **Config defaults** → read `configs/defaults.yaml`
8. **HF upload script** → read `scripts/hf_upload.py`
9. **How to set up `sys.path`** for relative imports in Colab (the notebook is inside the zip at project root)

---

## 14. Execution Plan

1. `ls` project root, `checkpoint_final/`, `experiments/rl_finetuning/outputs/minihack_final/`
2. Read the 8 files in Section 10
3. Implement notebook cell by cell
4. Handle checkpoint distribution (HF upload or bundled)
5. Test the full notebook runs end-to-end
6. Save as `demo.ipynb` in project root

---

## 15. Critical Constraints

- **Do NOT train any model in the notebook.** Load pre-trained checkpoint only.
- **The marker must be able to change environments/seeds in one config cell** and re-run to test on unseen inputs.
- **Live inference is the priority** — Cells 3-5 are more important than pre-computed figures.
- **Pre-computed figures support the narrative** — include only key summary plots, not per-ablation detail.
- **The notebook is inside the zip** alongside source code — use relative imports, no git clone needed.
- **Do NOT hardcode absolute paths.**
- **Must run top-to-bottom on Colab** with no manual steps after the setup cell.
- **Target runtime**: <15 min GPU, <30 min CPU.
- **Environments are procedurally generated** — every run with a different seed produces unseen layouts, so the marker is inherently testing on unseen inputs.
