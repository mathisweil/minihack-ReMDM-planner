# Ablation Diagnosis Report

## Hypothesis Ranking

### 1. Catastrophic Forgetting [*****] (100%)

**Description:** Pretrained representations are corrupted by RL gradients.

**Evidence:** 3/3 supporting ablations improved over baseline.

**Recommendation:** Implement strong parameter regularisation (EWC + LLRD) or use LoRA to restrict update space.

### 2. Gradient Conflict [*****] (100%)

**Description:** RL and BC gradients point in conflicting directions.

**Evidence:** 1/1 supporting ablations improved over baseline.

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

**Evidence:** 0/0 supporting ablations improved over baseline.

**Recommendation:** Restrict training to low-t regime or use t-curriculum.

## Individual Ablation Scores

| Ablation | Group | Score | Delta vs Pretrained |
|---|---|---|---|
| baseline_rl | Baseline | 0.6000 | +0.0250 |
| ewc | A | 0.6500 | +0.0750 |
| kl_penalty | A | 0.6750 | +0.1000 |
| llrd | A | 0.6250 | +0.0500 |
| lora | A | 0.6000 | +0.0250 |

| Ablation | Score | Delta vs Baseline | Verdict |
|---|---|---|---|
| baseline_rl | 0.6000 | +0.0000 | NEUTRAL |
| ewc | 0.6500 | +0.0500 | NEUTRAL |
| kl_penalty | 0.6750 | +0.0750 | IMPROVEMENT |
| llrd | 0.6250 | +0.0250 | NEUTRAL |
| lora | 0.6000 | +0.0000 | NEUTRAL |

*Pretrained score: 0.5750*
*Baseline RL score: 0.6000*

## Aggregate Verdict

Mixed results: 0/4 ablations collapsed. Check individual verdicts above.

**Gradient alignment** = +0.6994: RL gradient has useful signal.