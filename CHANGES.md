# Correctness changes (CORRECTNESS_PLAN.md, approved 2026-08-09)

Baseline tag: `parity-baseline`. One change per commit; predicted fingerprint
movement stated in each commit message and verified after. Fingerprint
references in `parity/reference/` are the pre-change baseline and are not
regenerated until the Stage 4 retrain.

| Change | Commit | Source | Invalidates |
|---|---|---|---|
| FIX-4: sequential collection denoises with `diffusion_steps_collect` | fix/collection-steps | repo config contract (B-7) | nothing trained (paper runs used the GPU path) |
| FIX-5: `cosine` = MDLM eq (92); previous cos² renamed `cosine_sq` (eq 91) | fix/schedule-labels | MDLM App E.1 | nothing (no run ever used the label) |
| Group 3: annotations, pseudocode cross-refs | align/incidental | METHOD_PARITY 2.1/2.3/2.5 | nothing |
| FIX-3: eval sampler = ReMDM Algorithm 1 (Bernoulli posterior unmasking, stored-psi conf, 10% floor removed, greedy cleanup) | fix/eval-sampler | Wang Alg 1, Sec 4.1 | every evaluation table/figure/curve produced with the old sampler; no checkpoints |
| CH-1 (elected): nucleus top-p 0.9 replaces top-k 4; `top_k` legacy snapshot key | choice/nucleus-sampling | Wang Sec 5 (stated practice; a choice, not a correction) | same eval surface as FIX-3 |
| FIX-1: loss = `w(t) * sum_masked(CE) / L`, analytic alpha', weight mandatory; `use_importance_weighting` removed (legacy snapshot key) | fix/loss-estimator | MDLM eq (8)/(10); Shi eq (4), App (28) | all trained diffusion checkpoints (superseded, see ../RETRAIN_LOG.md) and every result derived from them |

Documented decisions (no code change): CH-2 optimiser stack stays AdamW +
constant DAgger LR (no source governs; per-benchmark tuning documented);
CH-3 EMA evaluation stays (source for the practice, Diffusion Policy, absent
from papers/); CH-6 physics-aware option and greedy collection sampler kept
as documented, unsourced engineering (default off / collection-only).

Old-weights, corrected-sampler evaluation fingerprints: `parity/stage2_eval/`.
