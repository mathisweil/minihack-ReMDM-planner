# Ablation Diagnosis Report

## Hypothesis Ranking

### 1. Catastrophic Forgetting [*****] (100%)

**Description:** Pretrained representations are corrupted by RL gradients.

**Evidence:** 1/1 supporting ablations improved over baseline.

**Recommendation:** Implement strong parameter regularisation (EWC + LLRD) or use LoRA to restrict update space.

### 2. Distributional Shift [*****] (100%)

**Description:** Online data distribution diverges too far from offline pretraining distribution.

**Evidence:** 1/1 supporting ablations improved over baseline.

**Recommendation:** Maintain large offline replay buffer or apply importance sampling corrections.

### 3. Gradient Conflict [*] (0%)

**Description:** RL and BC gradients point in conflicting directions.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Apply PCGrad and investigate t-distribution bias.

### 4. Signal Sparsity [*] (0%)

**Description:** Returns are too sparse or noisy for useful training signal.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Increase episodes per iteration, use reward shaping, or apply curriculum-based episode selection.

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
| llrd | A | 0.6500 | +0.1000 |
| lora | A | 0.6000 | +0.0500 |
| mixed_replay | A | 0.6375 | +0.0875 |

| Ablation | Score | Delta vs Baseline | Verdict |
|---|---|---|---|
| llrd | 0.6500 | +0.1000 | IMPROVEMENT |
| lora | 0.6000 | +0.0500 | IMPROVEMENT |
| mixed_replay | 0.6375 | +0.0875 | IMPROVEMENT |

*Pretrained score: 0.5500*
*Baseline RL score: 0.5500*

## Aggregate Verdict

Mixed results: 0/3 ablations collapsed. Check individual verdicts above.

**Gradient alignment** = +0.6963: RL gradient has useful signal.