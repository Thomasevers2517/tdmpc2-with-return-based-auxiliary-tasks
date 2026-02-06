# BMPC Ablation Sweeps Plan

**Date**: February 2026  
**Baseline**: BMPC-equivalence config (sweep 1_test)  
**Location**: `sweep_list/7_bmpc_equivalence/`

---

## Common Settings (All Sweeps)

All sweeps inherit from the BMPC-equivalence baseline with these modifications:

| Setting | Value |
|---------|-------|
| Task | `dog-run` |
| Steps | `200000` |
| Seeds | `[1, 2]` |
| Baseline values | Already tested, excluded from sweeps |

---

## Sweep 2a: UTD Ratio

**Hypothesis**: Higher update-to-data ratios allow more gradient steps per environment step, potentially accelerating learning and improving sample efficiency.

**BMPC Context**: BMPC uses UTD=1 with 4 parallel environments.

| Parameter | Values |
|-----------|--------|
| `utd_ratio` | [2, 4] |

**Runs**: 2 configs × 2 seeds = **4 runs**

---

## Sweep 2b: Multi-Head Dynamics + Uncertainty

**Hypothesis**: Ensemble dynamics heads capture epistemic uncertainty. Different ways to use this uncertainty (optimism level, bootstrap strategy, LR scaling) may improve exploration or stability.

**BMPC Context**: BMPC uses single dynamics head, no uncertainty quantification.

| Parameter | Values | Notes |
|-----------|--------|-------|
| `planner_num_dynamics_heads` | 2 | Fixed |
| `planner_value_std_coef_train` | [0, 0.3, 1] | Optimism level: 0=mean, 0.3=mild, 1=full |
| `local_td_bootstrap` | [false, true] | Global vs per-head TD targets |
| `ensemble_lr_scaling` | true | Fixed (compensate for mean-reduced gradients) |

**Design Notes**:
- `planner_value_std_coef_train`: Controls how much ensemble std affects value estimates during planning. Higher = more optimistic exploration.
- `local_td_bootstrap=true`: Each V-head bootstraps from its own predictions (no cross-head signal).
- `local_td_bootstrap=false`: All V-heads get same TD target (mean across heads).

**Runs**: 3 × 2 = 6 configs × 2 seeds = **12 runs**

---

## Sweep 2c: Expert Mean / Initial Expert Source

**Hypothesis**: The source of expert targets for policy distillation affects learning. Using the actual chosen action (with noise) vs the planner distribution mean may have different effects.

**BMPC Context**: BMPC uses `expert_mean = mean` from final MPPI distribution.

| Parameter | Values | Description |
|-----------|--------|-------------|
| `reanalyze_use_chosen_action` | [true, false] | Expert mean = chosen action or distribution mean |
| `initial_expert_from_behavior` | [true, false] | Initial expert from act() or separate reanalyze |

**Design Notes**:
- `reanalyze_use_chosen_action=true`: Expert mean is the action actually executed (includes sampling noise).
- `initial_expert_from_behavior=true`: First expert target comes from act() during data collection, not a separate reanalyze call.

**Runs**: 2 × 2 = 4 configs × 2 seeds = **8 runs**

---

## Sweep 2d: Number of Rollouts

**Hypothesis**: Multiple action samples per starting state reduces variance in value learning gradients.

**BMPC Context**: BMPC uses single rollout.

| Parameter | Values |
|-----------|--------|
| `num_rollouts` | [4] |

**Design Notes**:
- Each starting state in imagination spawns N different action trajectories.
- Value loss averages over these rollouts.

**Runs**: 1 config × 2 seeds = **2 runs**

---

## Sweep 2e: Imagination Initial Source

**Hypothesis**: Starting imagination from true encoded states vs dynamics rollout affects V-learning. True states provide cleaner starting points but may not expose policy to dynamics errors.

**BMPC Context**: BMPC uses encoder states for TD targets (see TODO in code).

| Parameter | Values |
|-----------|--------|
| `imagine_initial_source` | [replay_true] |

**Design Notes**:
- `replay_true`: Imagination starts from true encoded observations.
- `replay_rollout`: Imagination starts from dynamics rollout (head 0).
- Currently, both V predictions and TD targets use the same imagined trajectory.

**Runs**: 1 config × 2 seeds = **2 runs**

---

## Sweep 2f: Encoder Consistency Loss

**Hypothesis**: Additional encoder training signal from consistency loss (encoder → dynamics direction reversed) may improve representation quality.

**BMPC Context**: BMPC only has dynamics-to-encoder consistency, not the reverse.

| Parameter | Values |
|-----------|--------|
| `encoder_consistency_coef` | [0.1, 1] |

**Design Notes**:
- Standard consistency: `(pred_dynamics - target_encoder.detach())²` trains dynamics.
- Encoder consistency: `(pred_dynamics.detach() - target_encoder)²` trains encoder.
- Both can be used simultaneously with different coefficients.

**Runs**: 2 configs × 2 seeds = **4 runs**

---

## Sweep 2g: Trust Region

**Hypothesis**: KL regularization to EMA policy prevents rapid policy changes, improving stability. EMA policy for TD targets provides more stable bootstrap values.

**BMPC Context**: BMPC has no trust region or EMA policy.

| Parameter | Values | Notes |
|-----------|--------|-------|
| `td_target_use_ema_policy` | true | Fixed (use EMA policy for imagination in TD) |
| `policy_trust_region_coef` | [0.1, 1] | KL penalty weight |
| `policy_ema_tau` | [0.003, 0.01, 0.03] | EMA update rate |

**Design Notes**:
- `policy_trust_region_coef`: Adds `coef × KL(EMA_policy || current_policy)` to policy loss.
- `policy_ema_tau`: Controls how fast EMA policy tracks current policy:
  - 0.003 = slow tracking, more conservative
  - 0.03 = fast tracking, less regularization
- `td_target_use_ema_policy`: Actions for TD target imagination come from EMA policy.

**Runs**: 2 × 3 = 6 configs × 2 seeds = **12 runs**

---

## Sweep 2h: Exploration (Dual Policy)

**Hypothesis**: Separate optimistic policy for exploration combined with ensemble uncertainty improves sample efficiency.

**BMPC Context**: BMPC has no dual policy or explicit exploration mechanism.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `planner_num_dynamics_heads` | 2 | Fixed |
| `planner_value_std_coef_train` | 1 | Full optimism |
| `local_td_bootstrap` | true | Per-head bootstrapping |
| `ensemble_lr_scaling` | true | Fixed |
| `dual_policy_enabled` | true | Key change |

**Design Notes**:
- Dual policy: Maintains two policies (pessimistic + optimistic).
- Optimistic policy used for exploration in planner.
- This sweep tests whether explicit exploration policy helps.

**Runs**: 1 config × 2 seeds = **2 runs**

---

## Summary Table

| Sweep | Focus | Configs | Runs | Key Question |
|-------|-------|---------|------|--------------|
| 2a | UTD Ratio | 2 | 4 | Does more training per step help? |
| 2b | Multi-Head + Uncertainty | 6 | 12 | How to use ensemble uncertainty? |
| 2c | Expert/Reanalyze | 4 | 8 | What's the best expert target source? |
| 2d | Num Rollouts | 1 | 2 | Does multiple rollouts reduce variance? |
| 2e | Imagination Source | 1 | 2 | True vs rollout states for imagination? |
| 2f | Encoder Consistency | 2 | 4 | Does reverse consistency help encoder? |
| 2g | Trust Region | 6 | 12 | Does KL regularization stabilize learning? |
| 2h | Exploration (Dual) | 1 | 2 | Does explicit exploration policy help? |
| **Total** | | **23** | **46** | |

---

## Directory Structure

```
sweep_list/7_bmpc_equivalence/
├── 1_test/              # BMPC baseline
├── 2a_utd/              # UTD ratio sweep
├── 2b_multihead/        # Multi-head dynamics sweep
├── 2c_expert_source/    # Expert target source sweep
├── 2d_num_rollouts/     # Number of rollouts sweep
├── 2e_imagine_source/   # Imagination source sweep
├── 2f_encoder_cons/     # Encoder consistency sweep
├── 2g_trust_region/     # Trust region sweep
├── 2h_exploration/      # Dual policy exploration sweep
└── ABLATION_SWEEPS.md   # This document
```

---

## Notes

1. **Interactions**: Some sweeps may interact (e.g., multi-head + trust region). We test independently first, then combine promising configs.

2. **BMPC Difference**: The current code uses same imagined trajectory for both V predictions and TD targets. BMPC uses rollout latents for V but true latents for TD target imagination. This is noted as a TODO in the code.

3. **Baseline**: The BMPC-equivalence baseline (sweep 1_test) is run separately and provides the comparison point for all ablations.
