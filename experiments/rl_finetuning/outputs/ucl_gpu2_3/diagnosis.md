# Ablation Diagnosis Report

## Hypothesis Ranking

### 1. Mode Collapse [*****] (100%)

**Description:** Model collapses to degenerate distribution, losing action diversity.

**Evidence:** 1/1 supporting ablations improved over baseline.

**Recommendation:** Add strong entropy bonus and clip advantages.

### 2. t-Bias [*****] (100%)

**Description:** High-t gradients dominate and carry misleading signal.

**Evidence:** 1/1 supporting ablations improved over baseline.

**Recommendation:** Restrict training to low-t regime or use t-curriculum.

### 3. Catastrophic Forgetting [*] (0%)

**Description:** Pretrained representations are corrupted by RL gradients.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Implement strong parameter regularisation (EWC + LLRD) or use LoRA to restrict update space.

### 4. Gradient Conflict [*] (0%)

**Description:** RL and BC gradients point in conflicting directions.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Apply PCGrad and investigate t-distribution bias.

### 5. Signal Sparsity [*] (0%)

**Description:** Returns are too sparse or noisy for useful training signal.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Increase episodes per iteration, use reward shaping, or apply curriculum-based episode selection.

### 6. Distributional Shift [*] (0%)

**Description:** Online data distribution diverges too far from offline pretraining distribution.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Maintain large offline replay buffer or apply importance sampling corrections.

## Individual Ablation Scores

| Ablation | Group | Score | Delta vs Pretrained |
|---|---|---|---|
| entropy_bonus | B | 0.5875 | +0.0375 |
| t_curriculum | B | 0.6438 | +0.0938 |

| Ablation | Score | Delta vs Baseline | Verdict |
|---|---|---|---|
| entropy_bonus | 0.5875 | +0.0375 | NEUTRAL |
| t_curriculum | 0.6438 | +0.0938 | IMPROVEMENT |

*Pretrained score: 0.5500*
*Baseline RL score: 0.5500*

## Aggregate Verdict

Mixed results: 0/2 ablations collapsed. Check individual verdicts above.

**Gradient alignment** = +0.8161: RL gradient has useful signal.