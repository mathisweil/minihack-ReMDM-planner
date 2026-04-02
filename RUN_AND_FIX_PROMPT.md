# Run & Fix All Ablations

Activate the environment and run each of the 25 ablations one by one. Fix every error you encounter until all 25 pass cleanly.

## Setup

```bash
source .venv/bin/activate.csh
```

## Procedure

For each ablation in this list:

```
baseline_rl kl_penalty ewc llrd lora mixed_replay trust_region_kl
low_t t_curriculum entropy_bonus gradient_surgery advantage_clip normalized_adv bc_wins
frozen_backbone head_only attention_only ffn_only layer_ablation_top1 layer_ablation_top2 layer_ablation_top3
reward_filtering running_stats action_diversity reward_model
```

1. Run it with a **4-minute timeout** using `--fast`:
   ```bash
   timeout 240 python experiments/rl_finetuning/run_ablations.py \
       --checkpoint checkpoints/iter1500.pth \
       --ablations <name> \
       --fast
   ```

2. If it **crashes or errors**: read the traceback, fix the bug in the source code, then re-run the same ablation. Repeat until it passes or you've exhausted the fix (explain why if stuck).

3. If it **times out** (exit code 124): that's fine, note it as TIMEOUT and move on — the point is it didn't crash.

4. If it **passes**: note the exit code, move to the next ablation.

After each fix, re-run only the ablation that failed — don't re-run ones that already passed.

## Output

After all 25 are done, print a summary table:

```
Ablation                 | Status  | Attempts | Notes
-------------------------|---------|----------|------
baseline_rl              | PASS    | 1        |
kl_penalty               | PASS    | 2        | Fixed missing ref_model kwarg
...
```

List every file you modified and what you changed.
