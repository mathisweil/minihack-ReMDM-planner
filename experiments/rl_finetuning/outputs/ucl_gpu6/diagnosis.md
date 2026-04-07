# Ablation Diagnosis Report

## Hypothesis Ranking

### 1. Signal Sparsity [*] (33%)

**Description:** Returns are too sparse or noisy for useful training signal.

**Evidence:** 1/3 supporting ablations improved over baseline.

**Recommendation:** Increase episodes per iteration, use reward shaping, or apply curriculum-based episode selection.

### 2. Catastrophic Forgetting [*] (0%)

**Description:** Pretrained representations are corrupted by RL gradients.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Implement strong parameter regularisation (EWC + LLRD) or use LoRA to restrict update space.

### 3. Gradient Conflict [*] (0%)

**Description:** RL and BC gradients point in conflicting directions.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Apply PCGrad and investigate t-distribution bias.

### 4. Distributional Shift [*] (0%)

**Description:** Online data distribution diverges too far from offline pretraining distribution.

**Evidence:** 0/1 supporting ablations improved over baseline.

**Recommendation:** Maintain large offline replay buffer or apply importance sampling corrections.

### 5. Mode Collapse [*] (0%)

**Description:** Model collapses to degenerate distribution, losing action diversity.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Add strong entropy bonus and clip advantages.

### 6. t-Bias [*] (0%)

**Description:** High-t gradients dominate and carry misleading signal.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Restrict training to low-t regime or use t-curriculum.

## Individual Ablation Scores

| Ablation | Group | Score | Delta vs Pretrained |
|---|---|---|---|
| action_diversity | D | 0.6000 | -0.0125 |
| reward_filtering | D | 0.6000 | -0.0125 |
| reward_model | D | 0.6000 | -0.0125 |
| running_stats | D | 0.6375 | +0.0250 |

| Ablation | Score | Delta vs Baseline | Verdict |
|---|---|---|---|
| action_diversity | 0.6000 | -0.0125 | NEUTRAL |
| reward_filtering | 0.6000 | -0.0125 | NEUTRAL |
| reward_model | 0.6000 | -0.0125 | NEUTRAL |
| running_stats | 0.6375 | +0.0250 | NEUTRAL |

*Pretrained score: 0.6125*
*Baseline RL score: 0.6125*

## Aggregate Verdict

Mixed results: 0/4 ablations collapsed. Check individual verdicts above.

**Gradient alignment** = +0.8419: RL gradient has useful signal.