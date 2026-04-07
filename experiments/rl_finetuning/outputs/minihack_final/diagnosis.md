# Ablation Diagnosis Report

## Hypothesis Ranking

### 1. Distributional Shift [**] (50%)

**Description:** Online data distribution diverges too far from offline pretraining distribution.

**Evidence:** 1/2 supporting ablations improved over baseline.

**Recommendation:** Maintain large offline replay buffer or apply importance sampling corrections.

### 2. Catastrophic Forgetting [**] (40%)

**Description:** Pretrained representations are corrupted by RL gradients.

**Evidence:** 2/5 supporting ablations improved over baseline.

**Recommendation:** Implement strong parameter regularisation (EWC + LLRD) or use LoRA to restrict update space.

### 3. Gradient Conflict [*] (33%)

**Description:** RL and BC gradients point in conflicting directions.

**Evidence:** 1/3 supporting ablations improved over baseline.

**Recommendation:** Apply PCGrad and investigate t-distribution bias.

### 4. Signal Sparsity [*] (25%)

**Description:** Returns are too sparse or noisy for useful training signal.

**Evidence:** 1/4 supporting ablations improved over baseline.

**Recommendation:** Increase episodes per iteration, use reward shaping, or apply curriculum-based episode selection.

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
| action_diversity | D | 0.6000 | -0.0054 |
| advantage_clip | B | 0.5375 | -0.0679 |
| attention_only | C | 0.6250 | +0.0196 |
| baseline_rl | Baseline | 0.5375 | -0.0679 |
| bc_wins | B | 0.5125 | -0.0929 |
| entropy_bonus | B | 0.5375 | -0.0679 |
| ewc | A | 0.6375 | +0.0321 |
| ffn_only | C | 0.6375 | +0.0321 |
| frozen_backbone | C | 0.5625 | -0.0429 |
| gradient_surgery | B | 0.6875 | +0.0821 |
| head_only | C | 0.5875 | -0.0179 |
| kl_penalty | A | 0.5500 | -0.0554 |
| layer_ablation_top1 | C | 0.6000 | -0.0054 |
| layer_ablation_top2 | C | 0.6625 | +0.0571 |
| layer_ablation_top3 | C | 0.5500 | -0.0554 |
| llrd | A | 0.6500 | +0.0446 |
| lora | A | 0.6000 | -0.0054 |
| low_t | B | 0.5375 | -0.0679 |
| mixed_replay | A | 0.6375 | +0.0321 |
| normalized_adv | B | 0.0500 | -0.5554 |
| reward_filtering | D | 0.6000 | -0.0054 |
| reward_model | D | 0.6000 | -0.0054 |
| running_stats | D | 0.6375 | +0.0321 |
| t_curriculum | B | 0.4750 | -0.1304 |
| trust_region_kl | A | 0.5500 | -0.0554 |

| Ablation | Score | Delta vs Baseline | Verdict |
|---|---|---|---|
| action_diversity | 0.6000 | +0.0625 | IMPROVEMENT |
| advantage_clip | 0.5375 | +0.0000 | NEUTRAL |
| attention_only | 0.6250 | +0.0875 | IMPROVEMENT |
| baseline_rl | 0.5375 | +0.0000 | NEUTRAL |
| bc_wins | 0.5125 | -0.0250 | NEUTRAL |
| entropy_bonus | 0.5375 | +0.0000 | NEUTRAL |
| ewc | 0.6375 | +0.1000 | IMPROVEMENT |
| ffn_only | 0.6375 | +0.1000 | IMPROVEMENT |
| frozen_backbone | 0.5625 | +0.0250 | NEUTRAL |
| gradient_surgery | 0.6875 | +0.1500 | IMPROVEMENT |
| head_only | 0.5875 | +0.0500 | IMPROVEMENT |
| kl_penalty | 0.5500 | +0.0125 | NEUTRAL |
| layer_ablation_top1 | 0.6000 | +0.0625 | IMPROVEMENT |
| layer_ablation_top2 | 0.6625 | +0.1250 | IMPROVEMENT |
| layer_ablation_top3 | 0.5500 | +0.0125 | NEUTRAL |
| llrd | 0.6500 | +0.1125 | IMPROVEMENT |
| lora | 0.6000 | +0.0625 | IMPROVEMENT |
| low_t | 0.5375 | +0.0000 | NEUTRAL |
| mixed_replay | 0.6375 | +0.1000 | IMPROVEMENT |
| normalized_adv | 0.0500 | -0.4875 | COLLAPSE |
| reward_filtering | 0.6000 | +0.0625 | IMPROVEMENT |
| reward_model | 0.6000 | +0.0625 | IMPROVEMENT |
| running_stats | 0.6375 | +0.1000 | IMPROVEMENT |
| t_curriculum | 0.4750 | -0.0625 | NEUTRAL |
| trust_region_kl | 0.5500 | +0.0125 | NEUTRAL |

*Pretrained score: 0.6054*
*Baseline RL score: 0.5375*

## Aggregate Verdict

Mixed results: 2/24 ablations collapsed. Check individual verdicts above.

**Gradient alignment** = +0.7088: RL gradient has useful signal.