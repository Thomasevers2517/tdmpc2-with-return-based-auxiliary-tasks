# Planner-Based Training Targets Refactoring

## Executive Summary

This document describes a major architectural refactoring to replace the current `imagined_rollout` mechanism with **planner-based target generation**. Instead of sampling raw policy actions for training targets, we use the MPPI planner to iteratively refine action sequences, producing higher-quality targets for both value function and policy training.

### Core Motivation

The current approach samples actions directly from the policy and uses those for training targets. However, the planner (MPPI) can produce *better* actions through iterative refinement. By using planner-refined actions as targets, we can:

1. Train the policy to match higher-quality actions (distillation from planner to policy)
2. Compute more accurate value estimates based on refined action sequences
3. Unify the acting and training architectures around a single planner abstraction

---

## Table of Contents

1. [Current Architecture](#current-architecture)
2. [Proposed Architecture](#proposed-architecture)
3. [Detailed Design](#detailed-design)
4. [Replay Buffer Changes](#replay-buffer-changes)
5. [Reanalyze Mechanism](#reanalyze-mechanism)
6. [Policy Training](#policy-training)
7. [Value Training](#value-training)
8. [Implementation Plan](#implementation-plan)
9. [Configuration Schema](#configuration-schema)
10. [Code Changes](#code-changes)
11. [Open Questions](#open-questions)
12. [Testing Strategy](#testing-strategy)
13. [Expected Impact](#expected-impact)

---

## Current Architecture

### How `imagined_rollout` Works

The current training pipeline uses `imagined_rollout()` to generate training targets:

```python
def imagined_rollout(self, start_z, task, rollout_len=1):
    """
    Current implementation:
    1. Start from encoded replay observations (start_z)
    2. Sample N actions from pessimistic policy
    3. Roll out 1 step with dynamics model
    4. Compute rewards and value estimates
    5. Return actions, next_z, rewards, terminated flags
    """
```

#### Data Flow (Current)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CURRENT TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

Replay Buffer
     │
     ▼
Sample batch (obs, action, reward, terminated)
     │
     ▼
Encode observations ──► z_true [T+1, B, L]
     │
     ▼
imagined_rollout(start_z=z_true, rollout_len=1)
     │
     ├──► Sample N actions from policy ──► actions [T, 1, B*N, A]
     │
     ├──► Dynamics rollout ──► next_z [T, H, B*N, L]
     │
     ├──► Reward prediction ──► rewards [T, R, H, B*N, 1]
     │
     └──► Value estimates ──► V(next_z)
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           LOSS COMPUTATION                               │
│                                                                          │
│  Value Loss:                                                             │
│    TD targets = r + γ * (1-done) * V(next_z)                            │
│    Loss = soft_ce(V(z), TD_targets)                                     │
│                                                                          │
│  Policy Loss (AWR-style):                                                │
│    weights = softmax(Q / temperature)  over N actions                   │
│    Loss = -Σ(weights * log_prob(actions))                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Limitations of Current Approach

1. **Raw policy samples**: Actions come directly from policy, no refinement
2. **Quality ceiling**: Training targets are only as good as current policy
3. **No iterative improvement**: Single-shot sampling, no optimization over actions
4. **Disconnected from acting**: Acting uses planner, training uses raw policy

---

## Proposed Architecture

### Key Insight

The MPPI planner already exists for acting in the environment. It:
- Samples many action trajectories
- Evaluates them using the world model
- Iteratively refines the distribution toward high-value actions
- Produces a final action distribution that is *better* than raw policy

**Proposal**: Use the planner for generating training targets, not just for acting.

### Multiple Planner Instances

We create **separate planner instances** with different configurations:

| Planner | Purpose | Iterations | Warm Start | Horizon | Samples |
|---------|---------|------------|------------|---------|---------|
| **Acting Planner** | Environment interaction | 6 | Yes | 3 | 512 |
| **Training Planner** | Value/policy targets | Configurable (e.g., 3) | No | 1 | Configurable |

#### Why Separate Planners?

1. **Acting Planner**:
   - Needs real-time performance
   - Uses warm start from previous step (temporal coherence)
   - Longer horizon for look-ahead
   - Optimized for single action selection

2. **Training Planner**:
   - Can spend more compute (offline)
   - No warm start (each replay sample is independent)
   - Horizon = 1 (simplifies dynamics head selection)
   - Batched over many replay samples

#### Why Horizon = 1 for Training?

- **Dynamics head ambiguity**: With H > 1, we'd need to decide which dynamics head to use for each step
- **Consistency with current value loss**: Current TD targets use 1-step bootstrap
- **Policy distillation**: Only need first action for policy training
- **Simplicity**: Avoids multi-step trajectory management

### Data Flow (Proposed)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       PROPOSED TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

Replay Buffer (with stored planner targets)
     │
     ▼
Sample batch (obs, action, reward, terminated, planner_actions, planner_values)
     │
     ├──► If targets fresh: Use stored planner_actions, planner_values
     │
     └──► If targets stale OR on reanalyze:
          │
          ▼
     Encode observations ──► z [B, L]
          │
          ▼
     Training Planner (K iterations of MPPI)
          │
          ├──► Iteration 1: Sample from policy prior
          ├──► Iteration 2: Refine toward high-value actions
          ├──► ...
          └──► Iteration K: Final refined actions + values
          │
          ▼
     Store updated targets in replay buffer
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           LOSS COMPUTATION                               │
│                                                                          │
│  Value Loss:                                                             │
│    next_z = dynamics(z, planner_actions)                                │
│    TD targets = r + γ * (1-done) * V(next_z)                            │
│    weights = softmax(planner_values / temp)  [optional weighting]       │
│    Loss = weighted soft_ce(V(z), TD_targets)                            │
│                                                                          │
│  Policy Loss (AWR-style):                                                │
│    weights = softmax(planner_values / temperature)                      │
│    Loss = -Σ(weights * log_prob(planner_actions))                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Design

### Training Planner Specification

The training planner is a specialized MPPI instance:

```python
class TrainingPlanner:
    """
    MPPI planner configured for generating training targets.
    
    Key differences from acting planner:
    - No warm start (each sample independent)
    - Horizon = 1 (single-step planning)
    - Batched over replay samples
    - Returns all sampled actions + values (not just mean action)
    """
    
    def __init__(self, cfg, model):
        self.iterations = cfg.training_planner.iterations  # e.g., 3
        self.num_samples = cfg.training_planner.samples    # e.g., 128
        self.horizon = 1  # Always 1 for training
        self.temperature = cfg.training_planner.temperature
        self.model = model
    
    def plan(self, z, task=None):
        """
        Run MPPI planning from encoded state z.
        
        Args:
            z: Encoded state [B, L]
            task: Task identifier (for multitask)
        
        Returns:
            dict with:
                'actions': [B, N, A] - Final iteration's sampled actions
                'values': [B, N, 1] - Value estimates for each action
                'mean': [B, A] - Mean of final action distribution
                'std': [B, A] - Std of final action distribution
        """
        B, L = z.shape
        N = self.num_samples
        
        # Initialize action distribution from policy prior
        with torch.no_grad():
            _, pi_info = self.model.pi(z, task)
            mu = pi_info['mean']  # [B, A]
            std = pi_info['std']  # [B, A]
        
        # MPPI iterations
        for i in range(self.iterations):
            # Sample actions from current distribution
            # actions: [B, N, A]
            actions = mu.unsqueeze(1) + std.unsqueeze(1) * torch.randn(B, N, A)
            actions = actions.clamp(-1, 1)
            
            # Evaluate actions: roll out 1 step, get values
            # Expand z for N samples: [B, L] -> [B*N, L]
            z_expanded = z.unsqueeze(1).expand(B, N, L).reshape(B*N, L)
            actions_flat = actions.reshape(B*N, A)
            
            # Dynamics + value
            next_z = self.model.dynamics(z_expanded, actions_flat)  # [B*N, L]
            rewards = self.model.reward(z_expanded, actions_flat, next_z)  # [B*N, 1]
            v_next = self.model.V(next_z, task)  # [B*N, 1]
            
            # Q estimate
            Q = rewards + self.discount * v_next  # [B*N, 1]
            Q = Q.reshape(B, N, 1)  # [B, N, 1]
            
            # Softmax weights
            weights = torch.softmax(Q / self.temperature, dim=1)  # [B, N, 1]
            
            # Update distribution (weighted mean/std)
            mu = (weights * actions).sum(dim=1)  # [B, A]
            std = ((weights * (actions - mu.unsqueeze(1))**2).sum(dim=1)).sqrt()
            std = std.clamp(min=0.05)  # Minimum exploration
        
        # Final iteration: sample and return
        final_actions = mu.unsqueeze(1) + std.unsqueeze(1) * torch.randn(B, N, A)
        final_actions = final_actions.clamp(-1, 1)
        
        # Get final values
        z_exp = z.unsqueeze(1).expand(B, N, L).reshape(B*N, L)
        a_flat = final_actions.reshape(B*N, A)
        next_z = self.model.dynamics(z_exp, a_flat)
        rewards = self.model.reward(z_exp, a_flat, next_z)
        v_next = self.model.V(next_z, task)
        final_values = (rewards + self.discount * v_next).reshape(B, N, 1)
        
        return {
            'actions': final_actions,  # [B, N, A]
            'values': final_values,    # [B, N, 1]
            'mean': mu,                # [B, A]
            'std': std,                # [B, A]
        }
```

### Comparison: 1 Iteration vs K Iterations

| Iterations | Behavior | Quality | Compute |
|------------|----------|---------|---------|
| 1 | Equivalent to current `imagined_rollout` | Baseline | Low |
| 3 | Moderate refinement | Better | Medium |
| 6 | Strong refinement (like acting planner) | Best | High |

**Recommendation**: Start with K=3 iterations, tune empirically.

---

## Replay Buffer Changes

### Current Buffer Schema

```python
# Current fields per transition
{
    'obs': Tensor[...],           # Observation
    'action': Tensor[A],          # Action taken
    'reward': Tensor[1],          # Reward received
    'terminated': Tensor[1],      # Episode termination flag
    'truncated': Tensor[1],       # Episode truncation flag
}
```

### New Buffer Schema

```python
# New fields per transition (in addition to existing)
{
    # ... existing fields ...
    
    # Planner-generated targets
    'planner_actions': Tensor[N, A],   # N sampled actions from training planner
    'planner_values': Tensor[N, 1],    # Value estimates for each action
    'planner_timestamp': int,          # Training step when targets were generated
}
```

### Why Store Actions + Values (Not Distribution Params)?

**Option A**: Store distribution parameters (mean, std)
- Pro: Compact storage
- Con: Must resample on each use, different samples each time
- Con: Loses the specific action-value pairs

**Option B**: Store sampled actions + values (CHOSEN)
- Pro: Exact reproducibility
- Pro: Preserves action-value correspondence for AWR
- Pro: No resampling needed
- Con: More storage (N × A + N × 1 per transition)

**Decision**: Store actions + values. The AWR loss requires specific action-value pairs, and we want consistent targets across training steps (until reanalyze).

### Buffer Methods to Add

```python
class Buffer:
    def add_with_targets(self, obs, action, reward, terminated, truncated,
                         planner_actions, planner_values, planner_timestamp):
        """Add transition with planner-generated targets."""
        # Store all fields including planner targets
        ...
    
    def update_targets(self, indices, planner_actions, planner_values, timestamp):
        """Update planner targets for existing transitions (reanalyze)."""
        self._planner_actions[indices] = planner_actions
        self._planner_values[indices] = planner_values
        self._planner_timestamp[indices] = timestamp
    
    def sample(self, batch_size):
        """Sample batch including planner targets."""
        batch = super().sample(batch_size)
        batch['planner_actions'] = self._planner_actions[batch['indices']]
        batch['planner_values'] = self._planner_values[batch['indices']]
        batch['planner_timestamp'] = self._planner_timestamp[batch['indices']]
        return batch
```

---

## Reanalyze Mechanism

### Purpose

As the model improves during training, stored planner targets become **stale**. The planner with the updated model would produce different (better) targets. Reanalyze periodically refreshes targets.

### Reanalyze Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         REANALYZE MECHANISM                              │
└─────────────────────────────────────────────────────────────────────────┘

Training Step: 0    100   200   300   400   500   600   700   ...
                │     │     │     │     │     │     │     │
                ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
Normal Train:   ●     ●     ●     ●     ●     ●     ●     ●
                                              │
                                              ▼
Reanalyze:                                    ★  (every 500 steps)
                                              │
                                              ▼
                                    Sample batch from buffer
                                              │
                                              ▼
                                    Run training planner
                                              │
                                              ▼
                                    Update targets in buffer
```

### Reanalyze Implementation

```python
def maybe_reanalyze(self, step):
    """Periodically refresh planner targets in replay buffer."""
    if not self.cfg.reanalyze.enabled:
        return
    
    if step % self.cfg.reanalyze.interval != 0:
        return
    
    if step < self.cfg.reanalyze.min_step:
        return  # Don't reanalyze too early
    
    with torch.no_grad():
        # Sample batch (could be different from training batch)
        batch = self.buffer.sample(self.cfg.reanalyze.batch_size)
        
        # Encode observations
        z = self.model.encode(batch['obs'])
        
        # Run training planner
        planner_output = self.training_planner.plan(z, task=batch.get('task'))
        
        # Update targets in buffer
        self.buffer.update_targets(
            indices=batch['indices'],
            planner_actions=planner_output['actions'],
            planner_values=planner_output['values'],
            timestamp=step,
        )
    
    # Log reanalyze stats
    self.logger.log({
        'reanalyze/step': step,
        'reanalyze/batch_size': len(batch['indices']),
        'reanalyze/mean_target_age': step - batch['planner_timestamp'].float().mean(),
    })
```

### Reanalyze Frequency Considerations

| Interval | Pros | Cons |
|----------|------|------|
| Every step | Always fresh targets | Extremely expensive |
| Every 100 steps | Reasonably fresh | Moderate overhead |
| Every 500 steps | Low overhead | Some staleness |
| Every 1000 steps | Minimal overhead | Significant staleness |

**Recommendation**: Start with interval=500, tune based on compute budget.

### Target Staleness Tracking (Optional Enhancement)

```python
def sample_with_staleness_priority(self, batch_size):
    """Prioritize sampling transitions with stale targets."""
    # Compute staleness = current_step - planner_timestamp
    staleness = self._step - self._planner_timestamp
    
    # Prioritized sampling (higher staleness = higher priority)
    priorities = staleness ** self.cfg.staleness_exponent
    probs = priorities / priorities.sum()
    
    indices = np.random.choice(len(self), size=batch_size, p=probs)
    return self._get_batch(indices)
```

---

## Policy Training

### AWR-Style Regression (Unchanged Conceptually)

The policy training approach remains AWR-style regression, but now uses **planner-refined actions** as targets instead of raw policy samples.

### Before (Current)

```python
# Current: actions from policy
actions = policy.sample(z)  # Raw policy samples
Q = compute_Q(z, actions)
weights = softmax(Q / temperature)
loss = -(weights * policy.log_prob(actions)).sum()
```

### After (Proposed)

```python
# Proposed: actions from planner (stored or freshly computed)
planner_actions = batch['planner_actions']  # From buffer or fresh planner
planner_values = batch['planner_values']
weights = softmax(planner_values / temperature)
loss = -(weights * policy.log_prob(planner_actions)).sum()
```

### Key Insight: Policy Distillation

This is effectively **distilling the planner into the policy**:
- Planner produces high-quality actions (through iterative refinement)
- Policy is trained to match those actions (weighted by value)
- Over time, policy should approach planner quality
- But policy is faster (single forward pass vs. K iterations)

### Gradient Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GRADIENT FLOW (POLICY)                           │
└─────────────────────────────────────────────────────────────────────────┘

planner_actions ─────────────────────────────────────┐
     │                                               │
     │  (no grad - fixed target)                     │
     ▼                                               ▼
weights = softmax(planner_values / temp)      policy.log_prob(planner_actions)
     │                                               │
     │  (no grad - fixed weights)                    │ (WITH GRAD)
     ▼                                               ▼
                    loss = -(weights * log_probs).sum()
                                    │
                                    │ (backprop)
                                    ▼
                              Policy Parameters
```

### Entropy Regularization

The entropy bonus remains unchanged:

```python
# Entropy from current policy (encourages exploration)
entropy = policy.entropy()
loss = policy_loss - entropy_coeff * entropy
```

---

## Value Training

### TD Targets from Planner Actions

Instead of using raw policy actions, we compute TD targets using planner-refined actions:

```python
def compute_td_targets(self, z, planner_actions, rewards_actual, terminated, task):
    """
    Compute TD targets using planner's refined actions.
    
    Args:
        z: Current state encoding [B, L]
        planner_actions: Refined actions from training planner [B, N, A]
        rewards_actual: Actual rewards from replay [B, 1] (for grounding)
        terminated: Termination flags [B, 1]
        task: Task identifier
    
    Returns:
        td_targets: [B, N, 1] - TD target for each action sample
    """
    B, N, A = planner_actions.shape
    
    # Expand z for N samples
    z_exp = z.unsqueeze(1).expand(B, N, -1).reshape(B*N, -1)
    actions_flat = planner_actions.reshape(B*N, A)
    
    # Roll out with dynamics
    next_z = self.model.dynamics(z_exp, actions_flat)  # [B*N, L]
    
    # Predicted rewards (from model)
    rewards_pred = self.model.reward(z_exp, actions_flat, next_z)  # [B*N, 1]
    
    # Value at next state (target network)
    v_next = self.model.V(next_z, task, target=True)  # [B*N, 1]
    
    # TD targets
    rewards = rewards_pred.reshape(B, N, 1)
    v_next = v_next.reshape(B, N, 1)
    terminated_exp = terminated.unsqueeze(1).expand(B, N, 1)
    
    td_targets = rewards + self.discount * (1 - terminated_exp) * v_next
    
    return td_targets  # [B, N, 1]
```

### Weighted Value Loss (Optional)

We can optionally weight the value loss by the same softmax weights used for policy:

```python
def compute_value_loss_weighted(self, z, planner_actions, planner_values, task):
    """Value loss weighted by action quality."""
    # Compute TD targets
    td_targets = self.compute_td_targets(z, planner_actions, ...)
    
    # Current value predictions
    v_pred = self.model.V(z, task)  # [B, K] (K bins)
    
    # Expand for N samples
    v_pred_exp = v_pred.unsqueeze(1).expand(B, N, K)
    
    # Soft cross-entropy per sample
    ce_per_sample = soft_ce(v_pred_exp, td_targets)  # [B, N]
    
    # Weight by action quality (optional)
    if self.cfg.weighted_value_targets:
        weights = torch.softmax(planner_values / self.cfg.value_weight_temp, dim=1)
        loss = (weights.squeeze(-1) * ce_per_sample).sum(dim=1).mean()
    else:
        loss = ce_per_sample.mean()
    
    return loss
```

### Value Loss Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **Uniform** | Mean over all N samples | Simple, stable | Treats all actions equally |
| **Weighted** | Weight by softmax(Q/τ) | Focus on good actions | May ignore low-value regions |
| **Best-N** | Only use top-K actions | Strong signal | Loses diversity |

**Recommendation**: Start with weighted (same weights as policy loss).

---

## Implementation Plan

### Phase 1: Planner Abstraction (1-2 days)

**Goal**: Create reusable planner class that can be configured for acting vs. training.

**Tasks**:
1. [ ] Extract current planning logic into `Planner` class
2. [ ] Add configuration for iterations, samples, horizon, warm_start
3. [ ] Create `TrainingPlanner` subclass/config
4. [ ] Unit tests for planner outputs

**Files**:
- `tdmpc2/planner.py` (NEW)
- `tdmpc2/tdmpc2.py` (refactor `plan()` method)

### Phase 2: Replace `imagined_rollout` (2-3 days)

**Goal**: Use training planner instead of imagined_rollout for target generation.

**Tasks**:
1. [ ] Add `TrainingPlanner` instance to `TDMPC2.__init__`
2. [ ] Modify `_compute_loss_components` to use training planner
3. [ ] Update `calculate_value_loss` to accept planner outputs
4. [ ] Update `calculate_regression_pi_loss` to accept planner outputs
5. [ ] Remove/deprecate `imagined_rollout`
6. [ ] Integration tests

**Files**:
- `tdmpc2/tdmpc2.py`
- `tdmpc2/config.yaml`

### Phase 3: Replay Buffer Integration (2-3 days)

**Goal**: Store planner targets in replay buffer.

**Tasks**:
1. [ ] Add `planner_actions`, `planner_values`, `planner_timestamp` fields
2. [ ] Implement `Buffer.update_targets()` method
3. [ ] Modify `Buffer.sample()` to return planner targets
4. [ ] Modify data collection to run training planner before insertion
5. [ ] Handle buffer initialization (no targets yet)
6. [ ] Unit tests for buffer operations

**Files**:
- `tdmpc2/common/buffer.py`
- `tdmpc2/trainer/online_trainer.py`

### Phase 4: Reanalyze (1-2 days)

**Goal**: Implement periodic target refresh.

**Tasks**:
1. [ ] Add reanalyze config options
2. [ ] Implement `maybe_reanalyze()` method
3. [ ] Call from training loop
4. [ ] Add logging for reanalyze stats
5. [ ] Integration tests

**Files**:
- `tdmpc2/tdmpc2.py`
- `tdmpc2/config.yaml`

### Phase 5: Testing & Tuning (2-3 days)

**Goal**: Validate implementation and tune hyperparameters.

**Tasks**:
1. [ ] Run ablation: 1 vs. 3 vs. 6 planner iterations
2. [ ] Run ablation: with vs. without reanalyze
3. [ ] Run ablation: reanalyze interval (100, 500, 1000)
4. [ ] Compare training curves to baseline
5. [ ] Profile compute overhead

---

## Configuration Schema

### New Config Sections

```yaml
# =============================================================================
# TRAINING PLANNER CONFIG
# =============================================================================
training_planner:
  enabled: true              # Enable planner-based targets (vs. raw policy)
  iterations: 3              # Number of MPPI refinement iterations
  samples: 128               # Number of action samples per state
  horizon: 1                 # Planning horizon (always 1 for training)
  temperature: 0.5           # MPPI softmax temperature
  min_std: 0.05              # Minimum action std (exploration floor)

# =============================================================================
# REANALYZE CONFIG
# =============================================================================
reanalyze:
  enabled: true              # Enable periodic target refresh
  interval: 500              # Steps between reanalyze passes
  batch_size: 256            # Batch size for reanalyze (usually same as training)
  min_step: 1000             # Don't reanalyze before this step
  staleness_priority: false  # Prioritize stale targets in sampling

# =============================================================================
# EXISTING ACTING PLANNER CONFIG (unchanged)
# =============================================================================
mpc: true                    # Use MPC for acting
iterations: 6                # Planning iterations
samples: 512                 # Number of trajectory samples
horizon: 3                   # Planning horizon
temperature: 0.5             # Softmax temperature
```

### Backward Compatibility

```yaml
# To disable new features and use legacy imagined_rollout:
training_planner:
  enabled: false

# This falls back to current behavior
```

---

## Code Changes

### Files to Modify

| File | Changes | Complexity |
|------|---------|------------|
| `tdmpc2/tdmpc2.py` | Add training planner, modify losses | High |
| `tdmpc2/common/buffer.py` | Add target fields, update methods | Medium |
| `tdmpc2/config.yaml` | Add new config sections | Low |
| `tdmpc2/trainer/online_trainer.py` | Run planner on data collection | Medium |

### New Files

| File | Purpose |
|------|---------|
| `tdmpc2/planner.py` | Reusable planner abstraction |

### Methods to Add

```python
# tdmpc2/tdmpc2.py
class TDMPC2:
    def __init__(self, cfg):
        # ... existing init ...
        if cfg.training_planner.enabled:
            self.training_planner = TrainingPlanner(cfg, self.model)
    
    def compute_training_targets(self, z, task=None):
        """Run training planner to get refined actions + values."""
        return self.training_planner.plan(z, task)
    
    def maybe_reanalyze(self, step):
        """Periodically refresh targets in replay buffer."""
        ...

# tdmpc2/common/buffer.py
class Buffer:
    def update_targets(self, indices, planner_actions, planner_values, timestamp):
        """Update planner targets for given indices."""
        ...
    
    def get_target_staleness(self, indices, current_step):
        """Return staleness (age) of targets at given indices."""
        ...
```

### Methods to Remove/Deprecate

| Method | Action | Reason |
|--------|--------|--------|
| `imagined_rollout` | Deprecate/Remove | Replaced by training planner |
| `compute_imagination_td_targets` | Modify/Remove | Integrated into planner flow |

---

## Open Questions

### 1. Training Planner Iterations

**Question**: How many iterations provide the best quality-compute tradeoff?

**Options**:
- 1 iteration: Equivalent to current approach (baseline)
- 3 iterations: Moderate refinement, moderate compute
- 6 iterations: Strong refinement, higher compute

**Recommendation**: Start with 3, run ablation to compare.

### 2. Samples Per State

**Question**: How many action samples should training planner use?

**Current**: `num_rollouts` (e.g., 128)
**Acting planner**: 512

**Options**:
- Same as current (128): Direct comparison to baseline
- Same as acting planner (512): More coverage, higher compute
- Different (e.g., 256): Balance

**Recommendation**: Start with 128 (same as current), increase if needed.

### 3. Temperature

**Question**: Should training planner use same temperature as acting planner?

**Consideration**: Training might benefit from different temperature than acting:
- Higher temp → More exploration in targets
- Lower temp → More focused on best actions

**Recommendation**: Start with same (0.5), tune if needed.

### 4. Initial Targets (Bootstrap)

**Question**: How to handle transitions added before model is trained?

**Options**:
A. Run training planner even on initial data (correct but slow)
B. Use raw policy samples initially, reanalyze later
C. Fill with zeros/defaults, mark as "needs reanalyze"

**Recommendation**: Option A (always run planner before insertion). This ensures all data has valid targets from the start.

### 5. Interaction with UTD Ratio

**Question**: With high UTD ratio (e.g., 4), we train 4x per env step. Should we reanalyze more often?

**Consideration**: Higher UTD means model changes faster, targets become stale faster.

**Recommendation**: Scale reanalyze interval inversely with UTD: `effective_interval = base_interval / utd_ratio`

### 6. Multi-Task

**Question**: Should training planner be task-conditioned?

**Answer**: Yes, same as acting planner. Pass task identifier to planner.

---

## Testing Strategy

### Unit Tests

```python
# Test training planner outputs
def test_training_planner_shapes():
    planner = TrainingPlanner(cfg, model)
    z = torch.randn(32, 256)  # [B, L]
    output = planner.plan(z)
    
    assert output['actions'].shape == (32, 128, 4)  # [B, N, A]
    assert output['values'].shape == (32, 128, 1)   # [B, N, 1]
    assert output['mean'].shape == (32, 4)          # [B, A]
    assert output['std'].shape == (32, 4)           # [B, A]

def test_training_planner_values_reasonable():
    # Values should be bounded and finite
    output = planner.plan(z)
    assert output['values'].isfinite().all()
    assert output['values'].abs().max() < 1000  # Reasonable range

def test_buffer_update_targets():
    buffer = Buffer(cfg)
    buffer.add(obs, action, reward, terminated, truncated, 
               planner_actions, planner_values, timestamp=0)
    
    # Update targets
    new_actions = torch.randn_like(planner_actions)
    new_values = torch.randn_like(planner_values)
    buffer.update_targets([0], new_actions, new_values, timestamp=100)
    
    # Verify update
    batch = buffer.sample(1)
    assert torch.allclose(batch['planner_actions'], new_actions)
    assert batch['planner_timestamp'] == 100
```

### Integration Tests

```python
def test_training_loop_with_planner_targets():
    """Full training loop runs without errors."""
    agent = TDMPC2(cfg)
    trainer = OnlineTrainer(cfg, agent)
    
    # Run short training
    trainer.train(steps=1000)
    
    # Check no NaNs in losses
    assert not any(np.isnan(trainer.losses))

def test_reanalyze_updates_buffer():
    """Reanalyze actually updates targets in buffer."""
    agent = TDMPC2(cfg)
    agent.buffer.add(...)  # Add some data
    
    old_targets = agent.buffer._planner_actions.clone()
    agent.maybe_reanalyze(step=500)
    new_targets = agent.buffer._planner_actions
    
    # Targets should have changed
    assert not torch.allclose(old_targets, new_targets)
```

### Ablation Studies

1. **Planner iterations**: 1 vs. 3 vs. 6
2. **Reanalyze interval**: 100 vs. 500 vs. 1000 vs. disabled
3. **Samples**: 64 vs. 128 vs. 256
4. **Temperature**: 0.1 vs. 0.5 vs. 1.0

### Performance Benchmarks

- Training step time: baseline vs. with training planner
- Reanalyze overhead: time per reanalyze pass
- Memory usage: baseline vs. with stored targets
- GPU utilization: should remain high

---

## Expected Impact

### Benefits

1. **Higher Quality Targets**
   - Planner refinement produces better actions than raw policy
   - Policy learns from "expert" (planner) demonstrations
   - Value function trained on more accurate action-value pairs

2. **Unified Architecture**
   - Acting and training both use planner abstraction
   - Easier to reason about and maintain
   - Consistent behavior between train and eval

3. **Amortized Compute**
   - Reanalyze spreads planner compute over many training steps
   - Don't need to run planner on every training step
   - Can tune compute budget via reanalyze interval

4. **Cleaner Code**
   - Remove `imagined_rollout` complexity
   - Single planner abstraction handles both use cases
   - Better separation of concerns

### Costs

1. **Increased Compute**
   - K iterations of MPPI per target generation
   - Each iteration requires dynamics + value forward passes
   - Reanalyze adds additional compute periodically

2. **Increased Memory**
   - Store N actions + N values per transition
   - For N=128, A=4: 128×4 + 128×1 = 640 floats per transition
   - ~2.5KB per transition (float32)

3. **Increased Complexity**
   - Reanalyze mechanism adds bookkeeping
   - Target staleness tracking
   - More hyperparameters to tune

### Estimated Performance Impact

| Metric | Baseline | With Training Planner | Notes |
|--------|----------|----------------------|-------|
| Training step time | 1.0x | 1.2-1.5x | Depends on planner iterations |
| Memory per transition | 1.0x | 1.1x | Small overhead for stored targets |
| Final performance | Baseline | TBD (expected better) | Main goal |
| Sample efficiency | Baseline | TBD (expected better) | Better targets = faster learning |

---

## References

1. **MuZero Reanalyze**: Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (2020)
   - https://arxiv.org/abs/1911.08265
   - Introduced reanalyze for refreshing value targets

2. **TD-MPC2**: Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control" (2023)
   - https://arxiv.org/abs/2310.16828
   - Base architecture this builds on

3. **AWR**: Peng et al., "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning" (2019)
   - https://arxiv.org/abs/1906.04025
   - Policy learning via weighted regression

4. **MPPI**: Williams et al., "Model Predictive Path Integral Control" (2017)
   - https://arxiv.org/abs/1707.02342
   - Planning algorithm used

---

## Appendix A: Pseudocode

### Full Training Step (Proposed)

```python
def training_step(self, step):
    # 1. Sample batch from replay buffer
    batch = self.buffer.sample(self.cfg.batch_size)
    
    # 2. Encode observations
    z = self.model.encode(batch['obs'])
    
    # 3. Get planner targets (from buffer or fresh)
    if self.cfg.training_planner.enabled:
        planner_actions = batch['planner_actions']
        planner_values = batch['planner_values']
    else:
        # Legacy: compute fresh (no reuse)
        planner_output = self.training_planner.plan(z, batch.get('task'))
        planner_actions = planner_output['actions']
        planner_values = planner_output['values']
    
    # 4. Compute losses
    wm_loss = self.world_model_loss(z, batch)
    value_loss = self.value_loss(z, planner_actions, planner_values, batch)
    policy_loss = self.policy_loss(z, planner_actions, planner_values, batch)
    
    total_loss = wm_loss + value_loss + policy_loss
    
    # 5. Backward + optimize
    self.optimizer.zero_grad()
    total_loss.backward()
    self.optimizer.step()
    
    # 6. Maybe reanalyze
    self.maybe_reanalyze(step)
    
    return {'loss': total_loss.item()}
```

### Data Collection (Proposed)

```python
def collect_episode(self, env):
    obs = env.reset()
    episode = []
    
    while not done:
        # Act using acting planner
        action = self.acting_planner.plan(self.model.encode(obs))
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        episode.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
        })
        
        obs = next_obs
        done = terminated or truncated
    
    # Before adding to buffer: generate training planner targets
    with torch.no_grad():
        obs_batch = torch.stack([t['obs'] for t in episode])
        z = self.model.encode(obs_batch)
        planner_output = self.training_planner.plan(z)
    
    # Add to buffer with targets
    for i, transition in enumerate(episode):
        self.buffer.add(
            **transition,
            planner_actions=planner_output['actions'][i],
            planner_values=planner_output['values'][i],
            planner_timestamp=self._step,
        )
```

---

## Appendix B: Migration Guide

### For Users

1. **Enable new features**:
   ```yaml
   training_planner:
     enabled: true
     iterations: 3
   reanalyze:
     enabled: true
     interval: 500
   ```

2. **Disable for legacy behavior**:
   ```yaml
   training_planner:
     enabled: false
   ```

3. **Tune compute budget**:
   - More iterations = better targets, more compute
   - Lower reanalyze interval = fresher targets, more compute

### For Developers

1. **`imagined_rollout` is deprecated**
   - Use `training_planner.plan()` instead
   - Returns dict with `actions`, `values`, `mean`, `std`

2. **Buffer API changes**:
   - `add()` now accepts planner target fields
   - `sample()` returns planner targets
   - New `update_targets()` method for reanalyze

3. **Loss functions updated**:
   - Accept `planner_actions`, `planner_values` instead of computing internally
   - Weight computation moved to caller
