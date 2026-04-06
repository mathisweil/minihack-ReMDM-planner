# Ablation Diagnosis Report

## Hypothesis Ranking

### 1. Catastrophic Forgetting [*****] (100%)

**Description:** Pretrained representations are corrupted by RL gradients.

**Evidence:** 3/3 supporting ablations improved over baseline.

**Recommendation:** Implement strong parameter regularisation (EWC + LLRD) or use LoRA to restrict update space.

### 2. Distributional Shift [*****] (100%)

**Description:** Online data distribution diverges too far from offline pretraining distribution.

**Evidence:** 1/1 supporting ablations improved over baseline.

**Recommendation:** Maintain large offline replay buffer or apply importance sampling corrections.

### 3. Gradient Conflict [**] (50%)

**Description:** RL and BC gradients point in conflicting directions.

**Evidence:** 1/2 supporting ablations improved over baseline.

**Recommendation:** Apply PCGrad and investigate t-distribution bias.

### 4. t-Bias [**] (50%)

**Description:** High-t gradients dominate and carry misleading signal.

**Evidence:** 1/2 supporting ablations improved over baseline.

**Recommendation:** Restrict training to low-t regime or use t-curriculum.

### 5. Signal Sparsity [*] (0%)

**Description:** Returns are too sparse or noisy for useful training signal.

**Evidence:** 0/1 supporting ablations improved over baseline.

**Recommendation:** Increase episodes per iteration, use reward shaping, or apply curriculum-based episode selection.

### 6. Mode Collapse [*] (0%)

**Description:** Model collapses to degenerate distribution, losing action diversity.

**Evidence:** 0/3 supporting ablations improved over baseline.

**Recommendation:** Add strong entropy bonus and clip advantages.

## Individual Ablation Scores

| Ablation | Group | Score | Delta vs Pretrained |
|---|---|---|---|
| advantage_clip | B | 0.5250 | -0.0500 |
| baseline_rl | Baseline | 0.6000 | +0.0250 |
| bc_wins | B | 0.5750 | +0.0000 |
| entropy_bonus | B | 0.4500 | -0.1250 |
| ewc | A | 0.6500 | +0.0750 |
| kl_penalty | A | 0.6750 | +0.1000 |
| llrd | A | 0.6250 | +0.0500 |
| lora | A | 0.6000 | +0.0250 |
| low_t | B | 0.6000 | +0.0250 |
| mixed_replay | A | 0.6250 | +0.0500 |
| normalized_adv | B | 0.1250 | -0.4500 |
| t_curriculum | B | 0.6250 | +0.0500 |
| trust_region_kl | A | 0.5750 | +0.0000 |

| Ablation | Score | Delta vs Baseline | Verdict |
|---|---|---|---|
| advantage_clip | 0.5250 | -0.0750 | NEUTRAL |
| baseline_rl | 0.6000 | +0.0000 | NEUTRAL |
| bc_wins | 0.5750 | -0.0250 | NEUTRAL |
| entropy_bonus | 0.4500 | -0.1500 | COLLAPSE |
| ewc | 0.6500 | +0.0500 | NEUTRAL |
| kl_penalty | 0.6750 | +0.0750 | IMPROVEMENT |
| llrd | 0.6250 | +0.0250 | NEUTRAL |
| lora | 0.6000 | +0.0000 | NEUTRAL |
| low_t | 0.6000 | +0.0000 | NEUTRAL |
| mixed_replay | 0.6250 | +0.0250 | NEUTRAL |
| normalized_adv | 0.1250 | -0.4750 | COLLAPSE |
| t_curriculum | 0.6250 | +0.0250 | NEUTRAL |
| trust_region_kl | 0.5750 | -0.0250 | NEUTRAL |

*Pretrained score: 0.5750*
*Baseline RL score: 0.6000*

## Aggregate Verdict

Mixed results: 2/12 ablations collapsed. Check individual verdicts above.

**Gradient alignment** = +0.7119: RL gradient has useful signal.