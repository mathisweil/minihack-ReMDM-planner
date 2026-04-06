# Ablation Diagnosis Report

## Hypothesis Ranking

### 1. Catastrophic Forgetting [***] (60%)

**Description:** Pretrained representations are corrupted by RL gradients.

**Evidence:** 3/5 supporting ablations improved over baseline.

**Recommendation:** Implement strong parameter regularisation (EWC + LLRD) or use LoRA to restrict update space.

### 2. Gradient Conflict [**] (50%)

**Description:** RL and BC gradients point in conflicting directions.

**Evidence:** 1/2 supporting ablations improved over baseline.

**Recommendation:** Apply PCGrad and investigate t-distribution bias.

### 3. Distributional Shift [**] (50%)

**Description:** Online data distribution diverges too far from offline pretraining distribution.

**Evidence:** 1/2 supporting ablations improved over baseline.

**Recommendation:** Maintain large offline replay buffer or apply importance sampling corrections.

### 4. t-Bias [**] (50%)

**Description:** High-t gradients dominate and carry misleading signal.

**Evidence:** 1/2 supporting ablations improved over baseline.

**Recommendation:** Restrict training to low-t regime or use t-curriculum.

### 5. Signal Sparsity [*] (25%)

**Description:** Returns are too sparse or noisy for useful training signal.

**Evidence:** 1/4 supporting ablations improved over baseline.

**Recommendation:** Increase episodes per iteration, use reward shaping, or apply curriculum-based episode selection.

### 6. Mode Collapse [*] (0%)

**Description:** Model collapses to degenerate distribution, losing action diversity.

**Evidence:** 0/3 supporting ablations improved over baseline.

**Recommendation:** Add strong entropy bonus and clip advantages.

## Individual Ablation Scores

| Ablation | Group | Score | Delta vs Pretrained |
|---|---|---|---|
| action_diversity | D | 0.6000 | +0.0250 |
| advantage_clip | B | 0.5250 | -0.0500 |
| attention_only | C | 0.5250 | -0.0500 |
| baseline_rl | Baseline | 0.6000 | +0.0250 |
| bc_wins | B | 0.5750 | +0.0000 |
| entropy_bonus | B | 0.4500 | -0.1250 |
| ewc | A | 0.6500 | +0.0750 |
| ffn_only | C | 0.6500 | +0.0750 |
| frozen_backbone | C | 0.4250 | -0.1500 |
| head_only | C | 0.5750 | +0.0000 |
| kl_penalty | A | 0.6750 | +0.1000 |
| layer_ablation_top1 | C | 0.6000 | +0.0250 |
| layer_ablation_top2 | C | 0.5250 | -0.0500 |
| layer_ablation_top3 | C | 0.6250 | +0.0500 |
| llrd | A | 0.6250 | +0.0500 |
| lora | A | 0.6000 | +0.0250 |
| low_t | B | 0.6000 | +0.0250 |
| mixed_replay | A | 0.6250 | +0.0500 |
| normalized_adv | B | 0.1250 | -0.4500 |
| reward_filtering | D | 0.6000 | +0.0250 |
| reward_model | D | 0.4750 | -0.1000 |
| running_stats | D | 0.6250 | +0.0500 |
| t_curriculum | B | 0.6250 | +0.0500 |
| trust_region_kl | A | 0.5750 | +0.0000 |

| Ablation | Score | Delta vs Baseline | Verdict |
|---|---|---|---|
| action_diversity | 0.6000 | +0.0000 | NEUTRAL |
| advantage_clip | 0.5250 | -0.0750 | NEUTRAL |
| attention_only | 0.5250 | -0.0750 | NEUTRAL |
| baseline_rl | 0.6000 | +0.0000 | NEUTRAL |
| bc_wins | 0.5750 | -0.0250 | NEUTRAL |
| entropy_bonus | 0.4500 | -0.1500 | COLLAPSE |
| ewc | 0.6500 | +0.0500 | NEUTRAL |
| ffn_only | 0.6500 | +0.0500 | IMPROVEMENT |
| frozen_backbone | 0.4250 | -0.1750 | COLLAPSE |
| head_only | 0.5750 | -0.0250 | NEUTRAL |
| kl_penalty | 0.6750 | +0.0750 | IMPROVEMENT |
| layer_ablation_top1 | 0.6000 | +0.0000 | NEUTRAL |
| layer_ablation_top2 | 0.5250 | -0.0750 | NEUTRAL |
| layer_ablation_top3 | 0.6250 | +0.0250 | NEUTRAL |
| llrd | 0.6250 | +0.0250 | NEUTRAL |
| lora | 0.6000 | +0.0000 | NEUTRAL |
| low_t | 0.6000 | +0.0000 | NEUTRAL |
| mixed_replay | 0.6250 | +0.0250 | NEUTRAL |
| normalized_adv | 0.1250 | -0.4750 | COLLAPSE |
| reward_filtering | 0.6000 | +0.0000 | NEUTRAL |
| reward_model | 0.4750 | -0.1250 | COLLAPSE |
| running_stats | 0.6250 | +0.0250 | NEUTRAL |
| t_curriculum | 0.6250 | +0.0250 | NEUTRAL |
| trust_region_kl | 0.5750 | -0.0250 | NEUTRAL |

*Pretrained score: 0.5750*
*Baseline RL score: 0.6000*

## Aggregate Verdict

Mixed results: 3/23 ablations collapsed. Check individual verdicts above.

**Gradient alignment** = +0.6378: RL gradient has useful signal.