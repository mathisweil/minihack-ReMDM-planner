# Ablation Diagnosis Report

## Hypothesis Ranking

### 1. Gradient Conflict [*****] (100%)

**Description:** RL and BC gradients point in conflicting directions.

**Evidence:** 1/1 supporting ablations improved over baseline.

**Recommendation:** Apply PCGrad and investigate t-distribution bias.

### 2. Catastrophic Forgetting [*] (0%)

**Description:** Pretrained representations are corrupted by RL gradients.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Implement strong parameter regularisation (EWC + LLRD) or use LoRA to restrict update space.

### 3. Signal Sparsity [*] (0%)

**Description:** Returns are too sparse or noisy for useful training signal.

**Evidence:** 0/1 supporting ablations improved over baseline.

**Recommendation:** Increase episodes per iteration, use reward shaping, or apply curriculum-based episode selection.

### 4. Distributional Shift [*] (0%)

**Description:** Online data distribution diverges too far from offline pretraining distribution.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Maintain large offline replay buffer or apply importance sampling corrections.

### 5. Mode Collapse [*] (0%)

**Description:** Model collapses to degenerate distribution, losing action diversity.

**Evidence:** 0/2 supporting ablations improved over baseline.

**Recommendation:** Add strong entropy bonus and clip advantages.

### 6. t-Bias [*] (0%)

**Description:** High-t gradients dominate and carry misleading signal.

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Restrict training to low-t regime or use t-curriculum.

## Individual Ablation Scores

| Ablation | Group | Score | Delta vs Pretrained |
|---|---|---|---|
| advantage_clip | B | 0.5375 | -0.0875 |
| bc_wins | B | 0.5125 | -0.1125 |
| gradient_surgery | B | 0.6875 | +0.0625 |
| normalized_adv | B | 0.0500 | -0.5750 |

| Ablation | Score | Delta vs Baseline | Verdict |
|---|---|---|---|
| advantage_clip | 0.5375 | -0.0875 | NEUTRAL |
| bc_wins | 0.5125 | -0.1125 | COLLAPSE |
| gradient_surgery | 0.6875 | +0.0625 | IMPROVEMENT |
| normalized_adv | 0.0500 | -0.5750 | COLLAPSE |

*Pretrained score: 0.6250*
*Baseline RL score: 0.6250*

## Aggregate Verdict

Mixed results: 2/4 ablations collapsed. Check individual verdicts above.

**Gradient alignment** = +0.8528: RL gradient has useful signal.