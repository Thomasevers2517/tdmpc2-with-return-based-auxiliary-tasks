# Quadruped Asymptotic Tests

**Problem**: Training reward peaks ~880 then collapses. Policy shows random large dips in performance even when MPC is used. Instability late in training.

**Hypotheses & Tests**:

## 1. `policy_head_reduce_min/` (12 runs)
- **Hypothesis**: Policy optimizing for mean over heads is too optimistic
- **Test**: Compare `policy_head_reduce: mean` vs `min`
- **Expected**: min = slower but more stable, fewer sudden dips

## 2. `planner_value_disagreement/` (18 runs)
- **Hypothesis**: Planner selects overconfident trajectories in uncertain regions
- **Test**: `planner_lambda_value_disagreement: [0, 0.1, 0.3]`
- **Expected**: Penalizing disagreement avoids brittle high-value plans

## 3. `fixed_entropy/` (6 runs)
- **Hypothesis**: Entropy decay makes policy brittle late in training
- **Test**: Keep `start_entropy_coeff == end_entropy_coeff == 1e-4`
- **Expected**: Maintained exploration prevents collapse

## 4. `td_dynamics_heads/` (12 runs)
- **Hypothesis**: 2 dynamics heads for TD isn't pessimistic enough
- **Test**: `td_num_dynamics_heads: [2, 4]`
- **Expected**: 4 heads = more pessimistic TD targets = less overestimation

---

**Total runs**: 12 + 18 + 6 + 12 = **48 runs**

**Baseline config** (all sweeps):
- `task: quadruped-walk`
- `utd_ratio: 4`
- `tau: 0.003`
- `pi_update_freq: 1`
- `value_update_freq: -1`
- `num_q: 2`
- 6 seeds each
