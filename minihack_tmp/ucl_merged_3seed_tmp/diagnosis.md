# Ablation Diagnosis Report

## Hypothesis Ranking

### 1. Catastrophic Forgetting [***] (60%)

**Description:** Pretrained representations are corrupted by RL gradients.

**Evidence:** 3/5 supporting ablations improved over baseline.

**Recommendation:** Implement strong parameter regularisation (EWC + LLRD) or use LoRA to restrict update space.

### 2. Gradient Conflict [*] (33%)

**Description:** RL and BC gradients point in conflicting directions.

**Evidence:** 1/3 supporting ablations improved over baseline.

**Recommendation:** Apply PCGrad and investigate t-distribution bias.

### 3. Signal Sparsity [*] (0%)

**Description:** Returns are too sparse or noisy for useful training signal.

**Evidence:** 0/1 supporting ablations improved over baseline.

**Recommendation:** Increase episodes per iteration, use reward shaping, or apply curriculum-based episode selection.

### 4. Distributional Shift [*] (0%)

**Description:** Online data distribution diverges too far from offline pretraining distribution.

**Evidence:** 0/1 supporting ablations improved over baseline.

**Recommendation:** Maintain large offline replay buffer or apply importance sampling corrections.

### 5. Mode Collapse [*] (0%)

**Description:** Model collapses to degenerate distribution, losing action diversity.

**Evidence:** 0/3 supporting ablations improved over baseline.

**Recommendation:** Add strong entropy bonus and clip advantages.

### 6. t-Bias [*] (0%)

**Description:** High-t gradients dominate and carry misleading signal.

**Evidence:** 0/2 supporting ablations improved over baseline.

**Recommendation:** Restrict training to low-t regime or use t-curriculum.

## Individual Ablation Scores

| Ablation | Group | Score | Delta vs Pretrained |
|---|---|---|---|
| advantage_clip | B | 0.4958 | -0.0917 |
| attention_only | C | 0.6250 | +0.0375 |
| baseline_rl | Baseline | 0.5625 | -0.0250 |
| bc_wins | B | 0.5125 | -0.0750 |
| entropy_bonus | B | 0.5708 | -0.0167 |
| ewc | A | 0.6667 | +0.0792 |
| ffn_only | C | 0.6083 | +0.0208 |
| frozen_backbone | C | 0.6167 | +0.0292 |
| gradient_surgery | B | 0.6542 | +0.0667 |
| head_only | C | 0.5958 | +0.0083 |
| kl_penalty | A | 0.5583 | -0.0292 |
| layer_ablation_top1 | C | 0.6208 | +0.0333 |
| layer_ablation_top2 | C | 0.6458 | +0.0583 |
| layer_ablation_top3 | C | 0.5500 | -0.0375 |
| llrd | A | 0.6250 | +0.0375 |
| lora | A | 0.6042 | +0.0167 |
| low_t | B | 0.5500 | -0.0375 |
| mixed_replay | A | 0.5833 | -0.0042 |
| normalized_adv | B | 0.0625 | -0.5250 |
| t_curriculum | B | 0.5875 | -0.0000 |
| trust_region_kl | A | 0.5750 | -0.0125 |

| Ablation | Score | Delta vs Baseline | Verdict |
|---|---|---|---|
| advantage_clip | 0.4958 | -0.0667 | NEUTRAL |
| attention_only | 0.6250 | +0.0625 | IMPROVEMENT |
| baseline_rl | 0.5625 | +0.0000 | NEUTRAL |
| bc_wins | 0.5125 | -0.0500 | NEUTRAL |
| entropy_bonus | 0.5708 | +0.0083 | NEUTRAL |
| ewc | 0.6667 | +0.1042 | IMPROVEMENT |
| ffn_only | 0.6083 | +0.0458 | NEUTRAL |
| frozen_backbone | 0.6167 | +0.0542 | IMPROVEMENT |
| gradient_surgery | 0.6542 | +0.0917 | IMPROVEMENT |
| head_only | 0.5958 | +0.0333 | NEUTRAL |
| kl_penalty | 0.5583 | -0.0042 | NEUTRAL |
| layer_ablation_top1 | 0.6208 | +0.0583 | IMPROVEMENT |
| layer_ablation_top2 | 0.6458 | +0.0833 | IMPROVEMENT |
| layer_ablation_top3 | 0.5500 | -0.0125 | NEUTRAL |
| llrd | 0.6250 | +0.0625 | IMPROVEMENT |
| lora | 0.6042 | +0.0417 | NEUTRAL |
| low_t | 0.5500 | -0.0125 | NEUTRAL |
| mixed_replay | 0.5833 | +0.0208 | NEUTRAL |
| normalized_adv | 0.0625 | -0.5000 | COLLAPSE |
| t_curriculum | 0.5875 | +0.0250 | NEUTRAL |
| trust_region_kl | 0.5750 | +0.0125 | NEUTRAL |

*Pretrained score: 0.5875*
*Baseline RL score: 0.5625*

## Aggregate Verdict

Mixed results: 1/20 ablations collapsed. Check individual verdicts above.

**Gradient alignment** = +0.6841: RL gradient has useful signal.