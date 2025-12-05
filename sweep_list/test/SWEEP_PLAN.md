# Sweep Plan: V-Function Stability & Auxiliary Heads

## Base Configuration
All sweeps start from this baseline:
- `pi_lr_scale: 0.3`
- `multi_gamma_loss_weight: 0.01`
- `aux_value_source: imagine` (required for V-function - on-policy rewards)
- `planner_num_dynamics_heads: 4`
- `td_num_dynamics_heads: 2`
- `utd_ratio: 4`
- 6 seeds per configuration
- Task: `quadruped-walk`

---

## Observed Issues

### 1. Training Reward Shoots Up, Then Collapses
- Training reward rises quickly (~40-50K steps), then drops at end
- Evaluation performance lags behind training (optimism?)
- Both training and eval dip at the ending
- Not clear overfitting - validation loss doesn't spike dramatically

### 2. Policy Instability
- `log_std` switches a lot even with low `pi_lr_scale`
- Lower pi_lr_scale is more stable but learns slower initially
- Only clear advantage of lower pi_lr: doesn't collapse at ending

### 3. Evaluation Performance is Random
- High variance in eval, not stable
- Could be due to policy entropy or MPC interaction

---

## Sweep 1: Ensemble Configuration
**Goal**: Find optimal ensemble sizes and reduction methods

Sweep over:
- `num_q`: [1, 2]
- `policy_head_reduce`: [min, mean]
- `multi_gamma_head`: [joint, separate]

Fixed:
- `planner_num_dynamics_heads: 4`
- `td_num_dynamics_heads: 2`

**Total**: 2 × 2 × 2 × 6 seeds = 48 runs

---

## Sweep 2: Gamma Configuration
**Goal**: Test different auxiliary gamma schedules

Sweep over:
- `multi_gamma_gammas`: 
  - `[0.9, 0.95]` (current)
  - `[0.9, 0.97]` (higher auxiliary)
  - `[0.8, 0.9, 0.95, 0.97]` (more gammas)

Fixed:
- `num_q: 2`
- `multi_gamma_head: joint`

**Total**: 3 × 6 seeds = 18 runs

---

## Sweep 3: Update Frequency / Learning Rates
**Goal**: Find optimal relative update rates between policy, value, and world model

### Option A: Policy LR Scale
Sweep `pi_lr_scale`: [0.1, 0.3, 0.5, 1.0]

### Option B: Policy Update Frequency (NEW CONFIG NEEDED)
Add `pi_utd_ratio` or `pi_update_freq` to control how often policy is updated relative to critic.
- Example: `pi_update_freq: 2` means update policy every 2 critic updates (like TD3)

### Option C: Value Tau / EMA
Sweep `tau`: [0.005, 0.01, 0.02] (target network update rate)

**Questions to resolve**:
- Is learning rate or gradient update count the right knob?
- Should we add delayed policy updates (TD3-style)?

---

## Sweep 4: Entropy Schedule
**Goal**: Address end-of-training collapse via entropy

Sweep over:
- `start_entropy_coeff`: [1e-3, 1e-4, 1e-5]
- `end_entropy_coeff`: [1e-5, 1e-6, 1e-7]

Or try fixed entropy (no schedule):
- `start_entropy_coeff == end_entropy_coeff`

---

## Diagnostic Metrics to Watch

### Policy Stability
- `log_std` over time - should decrease smoothly, not oscillate
- `policy_entropy` - should decrease but not collapse to near-zero
- `pi_loss` variance - high variance = unstable updates

### Value Stability  
- `td_target_std_across_heads` - disagreement between V-heads
- `value_error_abs_mean` - should decrease
- `value_pred_mean` vs `td_target_mean` - optimism gap

### World Model
- `consistency_loss` - should stay low
- Validation consistency - shouldn't spike

### Heuristics for "Too Fast" Updates
1. **Policy too fast**: `log_std` oscillates, policy loss spiky, eval << train
2. **Value too fast**: TD targets unstable, value predictions overshoot
3. **WM too fast**: Consistency loss spikes, reward prediction degrades

---

## Implementation Notes

### Delayed Policy Updates (TD3-style)
Add to `config.yaml`:
```yaml
pi_update_freq: 1  # Update policy every N critic updates
```

Modify `_update()` to skip policy update unless `step % pi_update_freq == 0`.

### Policy EMA (alternative to lower LR)
Already have `policy_ema_enabled` and `policy_ema_tau` - could enable and sweep tau.

---

## Priority Order

1. **Sweep 1 (Ensemble)** - Most fundamental, affects all other decisions
2. **Sweep 3 (Update Freq)** - Directly addresses observed instability
3. **Sweep 2 (Gammas)** - Auxiliary-specific, can run in parallel
4. **Sweep 4 (Entropy)** - Fine-tuning once base is stable
