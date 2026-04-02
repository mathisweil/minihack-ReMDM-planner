# Ablation Diagnosis Report

## Hypothesis Ranking

### 1. Catastrophic Forgetting [*] (0%)

**Description:** Pretrained representations are corrupted by RL gradients.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Implement strong parameter regularisation (EWC + LLRD) or use LoRA to restrict update space.

### 2. Gradient Conflict [*] (0%)

**Description:** RL and BC gradients point in conflicting directions.

**Evidence:** 0/1 supporting ablations improved over baseline.

**Recommendation:** Apply PCGrad and investigate t-distribution bias.

### 3. Signal Sparsity [*] (0%)

**Description:** Returns are too sparse or noisy for useful training signal.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Increase episodes per iteration, use reward shaping, or apply curriculum-based episode selection.

### 4. Distributional Shift [*] (0%)

**Description:** Online data distribution diverges too far from offline pretraining distribution.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Maintain large offline replay buffer or apply importance sampling corrections.

### 5. Mode Collapse [*] (0%)

**Description:** Model collapses to degenerate distribution, losing action diversity.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Add strong entropy bonus and clip advantages.

### 6. t-Bias [*] (0%)

**Description:** High-t gradients dominate and carry misleading signal.

**Evidence:** 0/1 supporting ablations improved over baseline.

**Recommendation:** Restrict training to low-t regime or use t-curriculum.

## Individual Ablation Scores

| Ablation | Group | Score | Delta vs Pretrained |
|---|---|---|---|
| low_t | B | 0.5500 | -0.0000 |

| Ablation | Score | Delta vs Baseline | Verdict |
|---|---|---|---|
| low_t | 0.5500 | -0.0000 | NEUTRAL |

*Pretrained score: 0.5500*
*Baseline RL score: 0.5500*

## Aggregate Verdict

Mixed results: 0/1 ablations collapsed. Check individual verdicts above.

**Gradient alignment** = +0.7456: RL gradient has useful signal.