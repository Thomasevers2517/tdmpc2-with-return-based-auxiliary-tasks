# Policy Regression Refactoring Plan

## Overview

**Goal:** Replace the current Stochastic Value Gradient (SVG) policy update with Advantage-Weighted Regression (AWR) style policy optimization, while sharing computation between value and policy losses.

### Current Approach (SVG)
- Sample single action from policy: `a ~ π(·|z)`
- Compute Q-estimate by **backpropagating through** frozen world model: `Q = r(z,a) + γV(f(z,a))`
- Maximize Q + entropy bonus via gradient ascent through the policy
- Value and policy losses computed separately with different rollouts

### New Approach (Value-Weighted Regression)
- Roll out N actions from policy in imagination: `a_i ~ π(·|z)` for i=1..N
- Compute 1-step TD targets for each action (detached, no gradients): `Q_i = r(z,a_i) + γV(f(z,a_i))`
- **Share these TD targets** for both:
  - **Value loss:** Train value network to predict these estimates (with head reduction)
  - **Policy loss:** Softmax weights from Q-values, maximize weighted log-prob of actions
- Log probabilities and entropy come **directly from imagination** (for pessimistic policy)
- For optimistic policy, log-probs must be recomputed since actions were sampled from pessimistic

### Key Insight: Shared Computation
The Q-estimates (1-step TD targets) serve dual purpose:
1. **Value targets** for training the value network
2. **Action scores** for computing policy regression weights

We compute them once with all head combinations (R×H×Ve), then pass to separate loss functions that each apply their own head reduction logic.

### Key Insight: Gradient Flow Through Log-Probs
The policy learns by changing its parameters (mean μ, std σ) to assign higher probability to high-value actions. Gradients flow through the log-probability computation:
- `log π(a|z) = log_prob(a; μ, σ)` where μ, σ are outputs of the policy network
- The **actions are detached** (no gradient through sampling randomness)
- The **mean and std are attached** (gradients flow through them)
- Changing μ and σ changes the log-prob of the (fixed) sampled actions

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         _update()                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Encode observations → z_true [T+1, B, L]                        │
│                                                                      │
│  2. World model losses (consistency, reward, termination)            │
│     └── world_model_losses() → wm_loss, z_rollout                   │
│                                                                      │
│  3. Imagined rollout (samples N actions from pessimistic policy)     │
│     └── imagined_rollout() → z_seq, actions, rewards, log_probs,    │
│                               entropy, terminated                    │
│                                                                      │
│  4. Compute 1-step TD targets (all heads, detached)                  │
│     └── compute_imagination_td_targets() → Q [T, R, H, Ve, B*N, 1]  │
│                                                                      │
│  5. Value loss (reduce heads, CE against value network)              │
│     └── calculate_value_loss_from_targets(Q, ...) → value_loss      │
│                                                                      │
│  6. Policy loss (reduce heads → softmax weights → weighted NLL)      │
│     └── calculate_regression_pi_loss(Q, actions, log_probs, ...)    │
│         ├── Pessimistic: head_reduce='min', disagreement=-λσ        │
│         └── Optimistic:  head_reduce='max', disagreement=+λσ        │
│                                                                      │
│  7. Backward passes (separate optimizers)                            │
│     ├── optim: wm_loss + value_loss                                 │
│     └── pi_optim: pi_loss                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Method Specifications

### 1. `imagined_rollout` (MODIFIED)

**Location:** `tdmpc2.py`

**Purpose:** Roll out N action sequences from a starting latent using the policy. Now also returns log_probs and entropy from the policy (previously discarded).

**Current behavior:** Returns `z_seq`, `actions`, `rewards`, `terminated`

**New behavior:** Additionally returns `log_probs` and `scaled_entropy` from policy sampling

**Inputs:** (unchanged)
- `start_z`: Tensor[S, B, L] — Starting latents
- `task`: Optional task identifier
- `rollout_len`: int — Number of imagination steps (typically 1)

**Outputs:** (extended)
- `z_seq`: Tensor[T+1, H, B_exp, L] — Latent trajectory
- `actions`: Tensor[T, 1, B_exp, A] — Sampled actions (shared across H)
- `rewards`: Tensor[T, R, H, B_exp, 1] — Rewards per R×H heads
- `terminated`: Tensor[T, H, B_exp, 1] — Termination flags
- **NEW** `log_probs`: Tensor[T, 1, B_exp, 1] — Log π(a|z) for each sampled action
- **NEW** `scaled_entropy`: Tensor[T, 1, B_exp, 1] — Scaled entropy at each state

Where `B_exp = S * B * N` (S starting states × B batch × N num_rollouts)

**Implementation notes:**
- The policy's `pi()` method already returns `log_prob` and `scaled_entropy` in its info dict
- We just need to capture and return these instead of discarding them
- `rollout_latents` in world_model.py needs modification to return these

---

### 2. `compute_imagination_td_targets` (NEW)

**Location:** `tdmpc2.py`

**Purpose:** Compute 1-step TD targets for all head combinations from imagined rollout. Returns raw targets without any head reduction — that's left to the loss functions.

**Inputs:**
- `rewards`: Tensor[T, R, H, B_exp, 1] — Rewards from R reward heads × H dynamics heads
- `next_z`: Tensor[T, H, B_exp, L] — Next latent states from H dynamics heads
- `terminated`: Tensor[T, H, B_exp, 1] — Termination flags per dynamics head
- `task`: Optional task identifier
- `gamma`: float or Tensor — Discount factor

**Outputs:**
- `td_targets`: Tensor[T, R, H, Ve, B_exp, 1] — Raw TD targets for all head combinations
- `v_next`: Tensor[T, H, Ve, B_exp, 1] — V(next_z) for all heads (for disagreement computation)

**What it does:**
```python
with torch.no_grad():
    # Get V(next_z) for each dynamics head, all value ensemble heads
    # next_z: [T, H, B_exp, L]
    T, H, B_exp, L = next_z.shape
    
    # Flatten for V call: [T*H*B_exp, L]
    next_z_flat = next_z.view(T * H * B_exp, L)
    
    # Get all Ve value heads: return_type='all' gives [Ve, T*H*B_exp, K]
    v_logits = model.V(next_z_flat, task, return_type='all')  # [Ve, T*H*B_exp, K]
    v_values = two_hot_inv(v_logits)  # [Ve, T*H*B_exp, 1]
    
    # Reshape to [T, H, Ve, B_exp, 1]
    Ve = v_values.shape[0]
    v_next = v_values.view(Ve, T, H, B_exp, 1).permute(1, 2, 0, 3, 4)  # [T, H, Ve, B_exp, 1]
    
    # Compute TD targets: r + γ(1-term)V
    # rewards: [T, R, H, B_exp, 1], need to broadcast with Ve
    # terminated: [T, H, B_exp, 1], broadcast to [T, 1, H, 1, B_exp, 1]
    
    # Expand dimensions for broadcasting
    rewards_exp = rewards.unsqueeze(3)  # [T, R, H, 1, B_exp, 1]
    term_exp = terminated.unsqueeze(1).unsqueeze(3)  # [T, 1, H, 1, B_exp, 1]
    v_next_exp = v_next.unsqueeze(1)  # [T, 1, H, Ve, B_exp, 1]
    
    td_targets = rewards_exp + gamma * (1 - term_exp) * v_next_exp  # [T, R, H, Ve, B_exp, 1]

return td_targets, v_next
```

---

### 3. `calculate_value_loss_from_targets` (NEW or MODIFIED)

**Location:** `tdmpc2.py`

**Purpose:** Compute value loss from pre-computed TD targets. Applies head reduction and trains value network.

**Inputs:**
- `z`: Tensor[T, B_exp, L] — Latent states (head 0 from z_seq)
- `td_targets`: Tensor[T, R, H, Ve, B_exp, 1] — Raw TD targets
- `task`: Optional task identifier
- `rho`: float — Temporal discount for loss weighting

**Outputs:**
- `loss`: Tensor scalar — Value loss
- `info`: TensorDict — Logging info

**What it does:**
```python
T, B_exp, L = z.shape
rho_pows = rho ** arange(T)

# Get value predictions from network (all Ve heads)
vs = model.V(z, task, return_type='all')  # [Ve, T, B_exp, K]

# Reduce TD targets over R, H dimensions using configured reduction
# td_targets: [T, R, H, Ve, B_exp, 1]
# First reduce over R (reward heads)
td_r = reduce_heads(td_targets, dim=1, mode=cfg.td_reward_reduce)  # [T, H, Ve, B_exp, 1]
# Then reduce over H (dynamics heads)  
td_rh = reduce_heads(td_r, dim=1, mode=cfg.td_dynamics_reduce)  # [T, Ve, B_exp, 1]
# Permute to match vs shape: [Ve, T, B_exp, 1]
td_final = td_rh.permute(1, 0, 2, 3)  # [Ve, T, B_exp, 1]

# Compute soft cross-entropy loss
loss_per = soft_ce(vs, td_final, cfg)  # [Ve, T, B_exp]
loss_t = loss_per.mean(dim=(0, 2))  # [T] - mean over Ve and B_exp

# Temporal weighting
loss = (loss_t * rho_pows).mean()

return loss, info
```

**Note:** This replaces/refactors the existing `calculate_value_loss` to work with pre-computed targets.

---

### 4. `calculate_regression_pi_loss` (NEW)

**Location:** `tdmpc2.py`

**Purpose:** Compute AWR-style policy loss using softmax-weighted log-probabilities.

**Inputs:**
- `z`: Tensor[T, S*B, N, L] — Latent states (reshaped to expose N dimension)
- `actions`: Tensor[T, S*B, N, A] — Sampled actions (from pessimistic policy, DETACHED)
- `log_probs_pess`: Tensor[T, S*B, N, 1] — Log π_pess(a|z) from imagination (ATTACHED!)
- `scaled_entropy_pess`: Tensor[T, S*B, N, 1] — Scaled entropy from imagination (ATTACHED!)
- `td_targets`: Tensor[T, R, H, Ve, S*B, N, 1] — Raw TD targets (reshaped)
- `v_next`: Tensor[T, H, Ve, S*B, N, 1] — V(next_z) for disagreement
- `task`: Optional task identifier
- `optimistic`: bool — If True, use max reduction and add disagreement; else use policy_head_reduce and subtract
- `temperature`: float — Softmax temperature τ
- `entropy_coef`: float — Entropy bonus coefficient α
- `value_disagreement_coef`: float — Disagreement penalty/bonus coefficient λ

**Outputs:**
- `loss`: Tensor scalar — Policy loss
- `info`: TensorDict — Logging info

**Head Reduction Strategy:**
- **Pessimistic policy:** Use `cfg.policy_head_reduce` (default 'min', can be 'mean') over flattened R×H×Ve
- **Optimistic policy:** Always use 'max' over flattened R×H×Ve
- Flatten all head dimensions and reduce at once (not sequentially)

**What it does:**
```python
T, SB, N, L = z.shape
A = actions.shape[-1]
rho_pows = rho ** arange(T)

# 1. Reduce TD targets to get Q-estimate per action (DETACHED for weights)
#    td_targets: [T, R, H, Ve, SB, N, 1]
if optimistic:
    reduce_fn = torch.amax
    disagree_sign = +1
else:
    # Use policy_head_reduce for pessimistic (min or mean)
    reduce_fn = torch.amin if cfg.policy_head_reduce == 'min' else torch.mean
    disagree_sign = -1

# Reduce over R, H, Ve at once (flatten and reduce)
with torch.no_grad():
    # Flatten R, H, Ve dimensions: [T, R*H*Ve, SB, N, 1]
    td_flat = td_targets.view(T, -1, SB, N, 1)
    Q = reduce_fn(td_flat, dim=1)  # [T, SB, N, 1]
    
    # Value disagreement across dynamics heads
    v_flat = v_next.view(T, -1, SB, N, 1)  # [T, H*Ve, SB, N, 1]
    v_disagreement = v_flat.std(dim=1)  # [T, SB, N, 1]
    
    Q = Q + disagree_sign * value_disagreement_coef * v_disagreement
    
    # Update RunningScale with all Q values (before using it)
    self.scale.update(Q)
    
    # Scale Q-estimates and compute softmax weights
    Q_scaled = self.scale(Q)  # [T, SB, N, 1]
    weights = F.softmax(Q_scaled / temperature, dim=2)  # [T, SB, N, 1], softmax over N

# 2. Get log_probs and entropy WITH GRADIENTS
if optimistic:
    # Must recompute for optimistic policy - call pi() and extract params from info
    z_flat = z.view(T * SB * N, L)
    _, pi_info = model.pi(z_flat, task, optimistic=True)
    mu = pi_info['presquash_mean']  # [T*SB*N, A]
    log_std = pi_info['log_std']  # [T*SB*N, A]
    
    log_probs = compute_action_log_prob(actions.view(T * SB * N, A), mu, log_std)
    log_probs = log_probs.view(T, SB, N, 1)  # [T, SB, N, 1]
    
    # Use scaled_entropy from pi_info
    scaled_entropy = pi_info['scaled_entropy'].view(T, SB, N, 1)  # [T, SB, N, 1]
else:
    # Use from imagination (already has gradients to pessimistic policy)
    log_probs = log_probs_pess  # [T, SB, N, 1], ATTACHED
    scaled_entropy = scaled_entropy_pess  # [T, SB, N, 1], ATTACHED

# 3. Weighted negative log-likelihood
#    Loss = -Σ_i w_i log π(a_i|z) where w_i = softmax(Q_i / τ)
weighted_nll = -(weights * log_probs).sum(dim=2)  # [T, SB, 1]

# 4. Entropy bonus (mean over N samples)
entropy_mean = scaled_entropy.mean(dim=2)  # [T, SB, 1]

# 5. Per-timestep loss
loss_t = weighted_nll - entropy_coef * entropy_mean  # [T, SB, 1]

# 6. Temporal-weighted mean
loss_per_t = loss_t.mean(dim=(1, 2))  # [T]
loss = (loss_per_t * rho_pows).sum() / rho_pows.sum()

# Logging
info = TensorDict({
    'regression_loss': loss,
    'regression_weight_entropy': -(weights * (weights + 1e-8).log()).sum(dim=2).mean(),
    'regression_q_scaled_mean': Q_scaled.mean(),
    'regression_q_scaled_std': Q_scaled.std(),
    'regression_log_prob_mean': log_probs.mean(),
    'regression_entropy_mean': entropy_mean.mean(),
    'regression_weights_max': weights.max(),
    'regression_weights_min': weights.min(),
    'regression_nll_term': weighted_nll.mean(),
    'regression_entropy_term': (entropy_coef * entropy_mean).mean(),
})

return loss, info
```

**Key insights:**
1. **Pessimistic policy:** `log_probs` from imagination already have correct gradients to `_pi` parameters
2. **Optimistic policy:** Call `model.pi(optimistic=True)`, extract `presquash_mean` and `log_std` from info dict, compute log_prob via `compute_action_log_prob`
3. **Wasted sampling:** When computing optimistic log_probs, `pi()` samples an action we discard. This is acceptable (sampling is cheap).
4. **RunningScale:** Must call `self.scale.update(Q)` before `self.scale(Q)` to update statistics

---

### 5. `calc_value_and_pi_losses` (NEW)

**Location:** `tdmpc2.py`

**Purpose:** Unified method that computes both value and policy losses from shared imagined rollout. Supports both regression and legacy SVG modes.

**Inputs:**
- `obs`: Tensor[T+1, B, ...] — Observations
- `action`: Tensor[T, B, A] — Replay actions (for world model loss)
- `reward`: Tensor[T, B, 1] — Replay rewards
- `terminated`: Tensor[T, B, 1] — Termination flags
- `task`: Optional task identifier
- `update_value`: bool — Whether to compute value loss
- `update_pi`: bool — Whether to compute policy loss
- `update_world_model`: bool — Whether to compute WM losses

**Outputs:**
- `wm_loss`: Tensor scalar
- `value_loss`: Tensor scalar
- `pi_loss`: Tensor scalar
- `info`: TensorDict

**What it does:**
```python
# 1. Encode observations
z_true = encode(obs)  # [T+1, B, L]
z_target = encode(obs, use_ema=True) if encoder_ema_enabled else None

# 2. World model losses (if updating)
if update_world_model:
    wm_loss, wm_info, z_rollout, lat_all = world_model_losses(...)
else:
    wm_loss = torch.zeros(())
    ...

# 3. Imagined rollout (pessimistic policy samples N actions)
#    IMPORTANT: log_probs and scaled_entropy are ATTACHED (have gradients to pessimistic policy)
imagined = imagined_rollout(z_true, task, rollout_len=cfg.imagination_horizon)
# z_seq: [T_imag+1, H, B_exp, L]
# actions: [T_imag, 1, B_exp, A] — detached
# rewards: [T_imag, R, H, B_exp, 1] — detached
# terminated: [T_imag, H, B_exp, 1] — detached
# log_probs: [T_imag, 1, B_exp, 1] — ATTACHED (gradients to pessimistic policy)
# scaled_entropy: [T_imag, 1, B_exp, 1] — ATTACHED (gradients to pessimistic policy)

# 4. Compute 1-step TD targets (all heads, detached)
td_targets, v_next = compute_imagination_td_targets(
    imagined['rewards'], imagined['z_seq'][1:], imagined['terminated'], task, gamma
)
# td_targets: [T_imag, R, H, Ve, B_exp, 1]
# v_next: [T_imag, H, Ve, B_exp, 1]

# 5. Value loss
if update_value:
    z_for_value = imagined['z_seq'][:-1, 0]  # [T_imag, B_exp, L]
    value_loss, value_info = calculate_value_loss_from_targets(z_for_value, td_targets, task)
else:
    value_loss = torch.zeros(())

# 6. Policy loss
if update_pi:
    if cfg.pi_regression_enabled:
        # ========== NEW: Regression-style policy loss ==========
        # Reshape to expose N dimension: [T, S*B, N, ...]
        S, B, N = z_true.shape[0], z_true.shape[1], cfg.num_rollouts
        T_imag = cfg.imagination_horizon
        
        # Reshape helper
        def reshape_for_N(x, last_dim):
            return x.view(T_imag, S * B, N, last_dim)
        
        z_reshaped = reshape_for_N(imagined['z_seq'][:-1, 0], L)
        actions_reshaped = reshape_for_N(imagined['actions'][:, 0], A)
        log_probs_reshaped = reshape_for_N(imagined['log_probs'][:, 0], 1)  # ATTACHED
        entropy_reshaped = reshape_for_N(imagined['scaled_entropy'][:, 0], 1)  # ATTACHED
        td_reshaped = td_targets.view(T_imag, R, H, Ve, S * B, N, 1)
        v_next_reshaped = v_next.view(T_imag, H, Ve, S * B, N, 1)
        
        # Pessimistic policy loss (uses attached log_probs from imagination)
        pess_loss, pess_info = calculate_regression_pi_loss(
            z_reshaped, actions_reshaped,
            log_probs_reshaped, entropy_reshaped,  # Attached!
            td_reshaped, v_next_reshaped,
            task, optimistic=False,
            temperature=cfg.pi_regression_temperature,
            entropy_coef=self.entropy_coef,  # Use existing entropy coefficient
            value_disagreement_coef=cfg.policy_lambda_value_disagreement
        )
        pi_loss = pess_loss
        pi_info = pess_info
        
        # Optimistic policy loss (must recompute log_probs)
        if cfg.dual_policy_enabled:
            opt_loss, opt_info = calculate_regression_pi_loss(
                z_reshaped, actions_reshaped,
                log_probs_reshaped, entropy_reshaped,  # Will be ignored, recomputed inside
                td_reshaped, v_next_reshaped,
                task, optimistic=True,  # <- triggers recomputation
                temperature=cfg.pi_regression_temperature,
                entropy_coef=self.entropy_coef * cfg.optimistic_entropy_mult,
                value_disagreement_coef=cfg.optimistic_policy_lambda_value_disagreement
            )
            pi_loss = pi_loss + opt_loss
            for k, v in opt_info.items():
                pi_info[f'opti_{k}'] = v
    else:
        # ========== LEGACY: SVG-style policy loss ==========
        z_for_pi = imagined['z_seq'][1:, 0].detach()  # Detach for SVG
        pess_loss, pess_info = calc_pi_losses(z_for_pi, task, optimistic=False)
        pi_loss = pess_loss
        pi_info = pess_info
        
        if cfg.dual_policy_enabled:
            opt_loss, opt_info = calc_pi_losses(z_for_pi, task, optimistic=True)
            pi_loss = pi_loss + opt_loss
            for k, v in opt_info.items():
                pi_info[f'opti_{k}'] = v
else:
    pi_loss = torch.zeros(())
    pi_info = TensorDict({})

# Combine info
info = TensorDict({})
info.update(wm_info)
info.update(value_info)
info.update(pi_info)

return wm_loss, value_loss, pi_loss, info
```

**Key design choices:**
1. **Attached log_probs:** Log-probs from imagination have gradients to pessimistic policy — used directly
2. **Recompute for optimistic:** When `optimistic=True`, `calculate_regression_pi_loss` recomputes log_probs via `pi_params` + `compute_action_log_prob`
3. **SVG fallback:** When `pi_regression_enabled=False`, uses existing `calc_pi_losses` unchanged

---

### 6. Modifications to `_update` (in tdmpc2.py)

**Purpose:** Use unified method when `pi_regression_enabled`, keep legacy SVG path otherwise.

```python
def _update(self, obs, action, reward, terminated, update_value=True, update_pi=True, update_world_model=True, task=None):
    
    if self.cfg.pi_regression_enabled:
        # ============ NEW PATH: Unified value + policy regression ============
        wm_loss, value_loss, pi_loss, info = self.calc_value_and_pi_losses(
            obs, action, reward, terminated, task,
            update_value=update_value,
            update_pi=update_pi,
            update_world_model=update_world_model
        )
        
        # Backward world model + value loss
        total_wm_value = wm_loss + self.cfg.value_coef * value_loss
        self.optim.zero_grad(set_to_none=True)
        total_wm_value.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim_step()
        
        # Backward policy loss (separate optimizer)
        if update_pi:
            self.pi_optim.zero_grad(set_to_none=True)
            (pi_loss * self.cfg.policy_coef).backward()
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model._pi.parameters(), self.cfg.grad_clip_norm
            )
            self.pi_optim_step()
        
        # Soft updates
        if update_value:
            self.model.soft_update_target_V()
        # ... etc
        
    else:
        # ============ LEGACY PATH: SVG-style (unchanged) ============
        # ... existing code ...
```

---

### 7. Modifications to `rollout_latents` (in world_model.py)

**Purpose:** Return log_probs and entropy from policy sampling. Always returns 4 values now.

**Current signature:** Returns `(latents, actions)` — 2 values
**New signature:** Returns `(latents, actions, log_probs, scaled_entropy)` — 4 values

**When `use_policy=False`:** Returns `log_probs=None, scaled_entropy=None`
**When `use_policy=True`:** Returns attached tensors with gradients to policy

```python
# In the policy sampling loop:
if use_policy:
    z_for_pi = latents_steps[t][0].view(B * N, L)
    a_flat, pi_info = self.pi(z_for_pi, task, ...)
    a_t = a_flat.view(B, N, A).detach()  # Actions DETACHED
    
    # NEW: Capture log_prob and entropy (ATTACHED - gradients flow through μ, σ)
    log_prob_t = pi_info['log_prob'].view(B, N, 1)  # [B, N, 1] - ATTACHED
    entropy_t = pi_info['scaled_entropy'].view(B, N, 1)  # [B, N, 1] - ATTACHED
    log_probs_steps.append(log_prob_t)
    entropy_steps.append(entropy_t)

# At the end:
if use_policy:
    log_probs_out = torch.stack(log_probs_steps, dim=2)  # [B, N, T, 1]
    entropy_out = torch.stack(entropy_steps, dim=2)  # [B, N, T, 1]
else:
    log_probs_out = None
    entropy_out = None

return latents, actions_out, log_probs_out, entropy_out
```

**Callers to update (unpack 4 values):**
- `tdmpc2.py` line ~741: `lat_all, _, _, _ = self.model.rollout_latents(...)`
- `tdmpc2.py` line ~1002: `latents, actions, log_probs, scaled_entropy = self.model.rollout_latents(...)`
- `planner.py` line ~95: `latents_p, actions_p, _, _ = self.world_model.rollout_latents(...)`
- `planner.py` line ~153: `latents_s, actions_s, _, _ = self.world_model.rollout_latents(...)`
- `info_types.py` line ~14: `lat_all, _, _, _ = world_model.rollout_latents(...)`

**Key insight on attach/detach:**
- **Actions are detached:** We don't want gradients flowing through the sampling randomness (ε ~ N(0,1))
- **Log-probs are attached:** Gradients flow through μ and σ, which are policy network outputs
- The log-prob depends on (action, μ, σ). When action is fixed (detached), changing μ/σ changes log-prob.

Then `imagined_rollout` in tdmpc2.py needs to handle and reshape these new outputs.

---

## Config Flags

### Config Removals
The following config flags will be **removed** (logic hardcoded instead):
- `ac_source` — was used to select value loss source, now always 'imagine'
- `aux_value_source` — was used for aux value loss source, now always 'imagine'  
- `actor_source` — was used for policy loss source, now hardcoded per mode
- `imagination_horizon` — hardcoded to 1 (always single-step imagination)

### Config Kept
- `imagine_initial_source` — controls starting states for imagination ('replay_true' vs 'replay_rollout')

### Config Additions
Add to `config.yaml`:

```yaml
# Policy Regression (AWR-style) - alternative to SVG
pi_regression_enabled: false       # Toggle regression vs SVG
pi_regression_temperature: 1.0     # Softmax temperature τ (tune via sweep)
```

### Existing Flags Reused
- `policy_head_reduce` — reduction over R, H, Ve heads for pessimistic policy ('min' or 'mean'); optimistic always uses 'max'
- `policy_lambda_value_disagreement` — disagreement penalty coefficient (pessimistic)
- `optimistic_policy_lambda_value_disagreement` — disagreement bonus coefficient (optimistic)
- `num_rollouts` — number of action samples N
- `rho` — temporal weighting for loss aggregation
- Existing entropy coefficient logic (`dynamic_entropy_coef`, `start_entropy_coeff`, etc.) — reused as-is

**Note on entropy coefficient:** The loss magnitude differs between regression and SVG, so the same entropy coefficient may have different effective strength. This is expected and will require hyperparameter tuning when switching between modes.

---

## Existing Code Reused (No Changes)

| Component | Location | How Used |
|-----------|----------|----------|
| `model.V()` | world_model.py | Get value estimates for TD targets |
| `model.reward()` | world_model.py | Already called in imagined_rollout |
| `model.pi()` | world_model.py | Samples actions, returns log_prob & entropy |
| `self.scale` | tdmpc2.py (RunningScale) | Scale Q-estimates before softmax |
| `soft_ce` | common/math.py | Cross-entropy for value loss |
| `two_hot_inv` | common/math.py | Convert logits to scalar values |
| `world_model_losses` | tdmpc2.py | Consistency, reward, termination losses |

---

## Concerns & Resolutions

### Resolved: Log-Prob Gradient Flow

**Key Insight:** Gradients flow through `log_prob` to update the policy. Different handling for each policy:

**Pessimistic policy:**
- Actions are sampled from pessimistic policy during imagination
- The `log_probs` from imagination already have gradients to pessimistic policy parameters
- **Solution:** Keep `log_probs` attached (don't detach), use directly for backprop
- Entropy: similarly, `scaled_entropy` from imagination has correct gradients

**Optimistic policy:**
- Actions are sampled from pessimistic policy, but we need gradients to optimistic policy
- The `log_probs` from imagination have no connection to optimistic policy parameters
- **Solution:** Call `model.pi(z, task, optimistic=True)`, extract `presquash_mean` and `log_std` from info dict, compute log_prob via `compute_action_log_prob`

```python
# For pessimistic policy: use imagination outputs directly (keep attached)
log_probs_pess = imagined['log_probs']  # Already has gradients to pessimistic policy
entropy_pess = imagined['scaled_entropy']  # Already has gradients

# For optimistic policy: must recompute via model.pi()
_, pi_info = model.pi(z, task, optimistic=True)
mu_opt = pi_info['presquash_mean']
log_std_opt = pi_info['log_std']
log_probs_opt = compute_action_log_prob(actions, mu_opt, log_std_opt)  # Gradients to optimistic
entropy_opt = pi_info['scaled_entropy']  # Gradients to optimistic
```

---

### Resolved: Keep Legacy SVG Path

**Requirement:** The old SVG-style policy loss should remain available as an alternative.

**Implementation:** In `calc_value_and_pi_losses`, after computing value loss:
```python
if update_pi:
    if cfg.pi_regression_enabled:
        # NEW: Regression-style policy loss
        pi_loss, pi_info = calculate_regression_pi_loss(...)
    else:
        # LEGACY: SVG-style policy loss (existing calc_pi_losses)
        pi_loss, pi_info = calc_pi_losses(z_for_pi, task, optimistic=False)
        if cfg.dual_policy_enabled:
            opti_loss, opti_info = calc_pi_losses(z_for_pi, task, optimistic=True)
            pi_loss = pi_loss + opti_loss
```

This keeps both options available via config toggle.

---

### Resolved: Detach/Attach Strategy

**Principle:** Keep gradients attached by default during imagination. Detach only where needed (e.g., for TD targets, for weights computation).

**In `rollout_latents`:**
- `actions` — detached (we don't want policy gradients through dynamics)
- `log_probs` — **attached** (needed for policy gradient)
- `scaled_entropy` — **attached** (needed for entropy bonus gradient)
- `rewards`, `z_seq[1:]` — detached (no gradients through world model for policy)

**In `calculate_regression_pi_loss`:**
- `weights` — detached (computed from scaled Q, no gradient through world model)
- `log_probs` — attached (gradient flows to policy)
- `entropy` — attached (gradient flows to policy)

---

### Resolved: Entropy Computation

**Pessimistic:** Use `scaled_entropy` from imagination (already has correct gradients)

**Optimistic:** Use `scaled_entropy` from `model.pi(optimistic=True)` info dict:
```python
# After calling model.pi(z, task, optimistic=True):
_, pi_info = model.pi(z, task, optimistic=True)
entropy_opt = pi_info['scaled_entropy']  # Already correctly scaled, has gradients
```

---

### Remaining: Reshape Correctness

**Question:** Is the reshape from `[T, 1, B_exp, ...]` to `[T, S*B, N, ...]` correct?

**Analysis:** 
- `B_exp = S * B * N` where S = T+1 (starting states), B = batch_size, N = num_rollouts
- The order in `imagined_rollout` is: for each starting state, for each batch item, for each rollout
- Need to verify the memory layout matches this assumption

**Action:** Add assertions to verify shapes during testing.

---

### Summary of Resolutions

1. ✅ **Pessimistic log_probs:** Use from imagination, keep attached
2. ✅ **Optimistic log_probs:** Recompute via `compute_action_log_prob`
3. ✅ **Entropy:** Same pattern — imagination for pessimistic, recompute for optimistic
4. ✅ **Legacy SVG:** Keep as alternative via config toggle
5. ✅ **Temperature:** Put in config, tune later  
6. ✅ **Value disagreement sign:** Symmetric (subtract for pessimistic, add for optimistic)
7. ✅ **Off-policy concern:** Acceptable, different scoring via head reduction

---

## Summary of New/Modified Code

### New Methods in `tdmpc2.py`:
1. `compute_imagination_td_targets()` — Compute 1-step TD targets for all heads (R×H×Ve)
2. `compute_imagination_aux_td_targets()` — Compute 1-step aux TD targets for all gammas (R×H×G_aux)
3. `calculate_value_loss_from_targets()` — Value loss from pre-computed targets
4. `calculate_regression_pi_loss()` — AWR-style policy loss
5. `calc_value_and_pi_losses()` — Unified entry point (replaces complex `_compute_loss_components`)

### New Methods in `common/math.py`:
1. `compute_action_log_prob(actions, mu, log_std)` — Log-prob for given actions under Gaussian with tanh squashing

### Modified Methods:
1. `rollout_latents()` in world_model.py — Return 4 values: `(latents, actions, log_probs, scaled_entropy)`
2. `imagined_rollout()` in tdmpc2.py — Handle new returns, keep log_probs/entropy attached
3. `_update()` in tdmpc2.py — Add regression path with config toggle
4. `_compute_loss_components()` in tdmpc2.py — Simplified (remove `fetch_source` closure)

### Removed Code:
1. `fetch_source()` closure in `_compute_loss_components` — replaced with hardcoded logic
2. Config flags: `ac_source`, `aux_value_source`, `actor_source`, `imagination_horizon`

### Config Additions:
1. `pi_regression_enabled: false` — toggle between regression and SVG
2. `pi_regression_temperature: 1.0` — softmax temperature τ

---

## Auxiliary Value Loss

The auxiliary value loss uses different gamma values (multi-gamma). It's computed **separately** from main value loss but uses the same imagined rollout.

### `compute_imagination_aux_td_targets` (NEW)

**Purpose:** Compute 1-step TD targets for auxiliary values with different gammas.

**Key differences from main TD targets:**
- Uses `V_aux` instead of `V`
- No Ve ensemble (aux values have G_aux heads for different gammas)
- Output shape: `[T, R, H, G_aux, B_exp, 1]` instead of `[T, R, H, Ve, B_exp, 1]`

**Implementation follows existing `_td_target_aux` pattern:**
```python
with torch.no_grad():
    # next_z: [T, H, B_exp, L]
    T, H, B_exp, L = next_z.shape
    
    # Flatten for V_aux call
    next_z_flat = next_z.view(T, H * B_exp, L)  # [T, H*B_exp, L]
    
    # Get aux values for all gammas using target network
    v_aux_flat = model.V_aux(next_z_flat, task, return_type='min', target=True)  # [G_aux, T, H*B_exp, 1]
    
    # Reshape to [G_aux, T, H, B_exp, 1]
    G_aux = v_aux_flat.shape[0]
    v_aux = v_aux_flat.view(G_aux, T, H, B_exp, 1)
    
    # Compute TD targets for each gamma
    # gammas_aux: [G_aux] tensor of gamma values
    gammas_aux = torch.tensor(cfg.multi_gamma_gammas, device=next_z.device)
    gammas_aux = gammas_aux.view(G_aux, 1, 1, 1, 1)  # [G_aux, 1, 1, 1, 1]
    
    # rewards: [T, R, H, B_exp, 1] -> expand for G_aux
    rewards_exp = rewards.unsqueeze(0)  # [1, T, R, H, B_exp, 1]
    terminated_exp = terminated.unsqueeze(0).unsqueeze(2)  # [1, T, 1, H, B_exp, 1]
    v_aux_exp = v_aux.unsqueeze(2)  # [G_aux, T, 1, H, B_exp, 1]
    
    aux_td_targets = rewards_exp + gammas_aux * discount * (1 - terminated_exp) * v_aux_exp
    # Shape: [G_aux, T, R, H, B_exp, 1]
    # Permute to match main TD target convention: [T, R, H, G_aux, B_exp, 1]
    aux_td_targets = aux_td_targets.permute(1, 2, 3, 0, 4, 5)

return aux_td_targets
```

---

## Simplification: Remove `fetch_source` Logic

### Current `_compute_loss_components` Complexity
The current implementation uses a `fetch_source()` closure that lazily computes different source data:
- `'replay_true'` — uses `z_true` directly
- `'replay_rollout'` — interleaved head selection from `z_rollout`
- `'imagine'` — calls `imagined_rollout()`

This is controlled by `ac_source`, `aux_value_source`, `actor_source` config flags.

### New Simplified Flow
**Hardcode the sources:**
- Value loss → always uses imagined rollout
- Aux value loss → always uses imagined rollout
- Legacy SVG policy loss → uses `z_rollout` from world model loss
- Regression policy loss → uses imagined rollout (shared with value)

**Remove:**
- `fetch_source()` closure
- `source_cache` dictionary
- `ac_source`, `aux_value_source`, `actor_source` config flags
- `imagination_horizon` config (hardcode to 1)

**Keep:**
- `imagine_initial_source` config — controls whether imagination starts from `z_true` (replay_true) or `z_rollout` (replay_rollout)

---

## Method Specifications for New Helper Functions

### `compute_action_log_prob` (in common/math.py)

**Purpose:** Compute log π(a|z) for given squashed actions under a Gaussian policy with tanh squashing.

**Inputs:**
- `actions`: Tensor[..., A] — Squashed actions in [-1, 1]
- `mu`: Tensor[..., A] — Policy mean (pre-squash), from `pi_info['presquash_mean']`
- `log_std`: Tensor[..., A] — Policy log standard deviation, from `pi_info['log_std']`
- `clamp_bound`: float — Clamp bound for atanh stability (default 0.9999)

**Outputs:**
- `log_prob`: Tensor[..., 1] — Log probability (summed over action dimensions)

**What it does:**
```python
def compute_action_log_prob(actions, mu, log_std, clamp_bound=0.9999):
    """Compute log-probability of squashed actions under Gaussian policy.
    
    Args:
        actions: Squashed actions in [-1, 1], float32[..., A]
        mu: Pre-squash mean from policy (presquash_mean), float32[..., A]
        log_std: Log std from policy, float32[..., A]
        clamp_bound: Clamp actions to [-bound, bound] for atanh stability
        
    Returns:
        log_prob: Log probability, float32[..., 1]
    """
    # Clamp actions for numerical stability (atanh undefined at ±1)
    actions_clamped = actions.clamp(-clamp_bound, clamp_bound)
    
    # Transform to pre-squash space: a_presquash = atanh(a)
    actions_presquash = torch.atanh(actions_clamped)
    
    # Compute Gaussian log-prob in pre-squash space
    std = log_std.exp()
    eps = (actions_presquash - mu) / std
    log_prob_presquash = gaussian_logprob(eps, log_std)  # [..*, 1]
    
    # Jacobian correction for tanh squashing: -Σ log(1 - a²)
    # This accounts for the change of variables from pre-squash to squashed space
    jacobian = torch.log(1 - actions_clamped.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    
    return log_prob_presquash - jacobian
```

**Note:** We use `model.pi()` directly to get distribution parameters. The `pi()` method returns an info dict containing:
- `presquash_mean`: Pre-tanh mean (μ)
- `log_std`: Log standard deviation
- `scaled_entropy`: Entropy with action dimension scaling
- `log_prob`: Log probability of the sampled action

For optimistic policy log-prob recomputation, we call `model.pi(z, task, optimistic=True)` and extract `presquash_mean`, `log_std` from the info dict, then call `compute_action_log_prob` with these and the original (detached) actions.

---

## Next Steps

### Implementation Order
1. **`compute_action_log_prob`** in common/math.py — simple utility function
2. **Modify `rollout_latents`** in world_model.py — return 4 values, keep log_probs attached
3. **Update all `rollout_latents` callers** — unpack 4 values (5 call sites)
4. **Modify `imagined_rollout`** in tdmpc2.py — capture and reshape new returns
5. **`compute_imagination_td_targets`** in tdmpc2.py — main TD targets
6. **`compute_imagination_aux_td_targets`** in tdmpc2.py — aux TD targets (separate)
7. **`calculate_value_loss_from_targets`** in tdmpc2.py — refactor existing
8. **`calculate_regression_pi_loss`** in tdmpc2.py — new AWR-style loss
9. **Simplify `_compute_loss_components`** — remove `fetch_source`, hardcode sources
10. **Modify `_update`** with config toggle
11. **Add config flags** — `pi_regression_enabled`, `pi_regression_temperature`
12. **Remove old config flags** — `ac_source`, `aux_value_source`, `actor_source`, `imagination_horizon`
13. **Clean up sweep configs** — remove references to deleted flags

### Testing Strategy
1. **First test with `pi_regression_enabled: false`** — should behave identically to before
2. **Then toggle on** and verify:
   - Training runs without errors
   - Gradients flow correctly (check via gradient norms)
   - Metrics are being logged
3. **Add assertions for shape checking** during development

### Gradient Flow Verification
After implementation, verify:
- [ ] Pessimistic: gradients flow through `log_probs` → `_pi` parameters
- [ ] Optimistic: gradients flow through recomputed log_probs → `_pi_optimistic` parameters
- [ ] Weights: fully detached (no gradient through Q-estimates or world model)
- [ ] Both policies in `pi_optim` get non-zero gradients

### Hyperparameter Tuning
- `pi_regression_temperature`: Start with 1.0, monitor `regression_weight_entropy`
  - If weights collapse to one-hot (entropy → 0), increase temperature
  - If weights are nearly uniform, decrease temperature
- Entropy coefficient: Reuse existing logic, but may need different magnitude for regression

---

## Watchpoints and Potential Issues

### 1. Compile Compatibility
Keep code compatible with `torch.compile`. Avoid:
- Dynamic control flow based on tensor values
- Excessive Python-side data-dependent branching
- Unsupported operations

The regression vs SVG branch is config-based (static per run), so should compile fine.

### 2. Memory Usage
Keeping log_probs attached (with gradients) for N×B×T tensors increases memory slightly. With typical `num_rollouts=8`, this should be negligible compared to latent states. Monitor if OOM occurs with large N.

### 3. RunningScale Update Timing
Must call `self.scale.update(Q)` before `self.scale(Q)` to update statistics. The current SVG code updates with first-timestep Q only; for regression, update with all N Q-values.

### 4. Softmax Numerical Stability
If Q-values have extreme variance, softmax could saturate. Temperature helps, but monitor:
- `regression_q_scaled_std` — should be reasonable (not exploding)
- `regression_weight_entropy` — should not collapse to 0

### 5. Dual Policy Optimizer
`pi_optim` contains both `_pi` and `_pi_optimistic` parameters. With regression:
- Pessimistic gets gradients via attached log_probs from imagination
- Optimistic gets gradients via recomputed log_probs from `model.pi(optimistic=True)`

Verify both parameter groups receive gradients.

### 6. Config Removal Ripple Effects
When removing `ac_source`, `aux_value_source`, `actor_source`, `imagination_horizon`:
- Check all YAML config files
- Check sweep configs in `sweep_list/`
- Check any logging code that references these
- Update documentation

### 7. Terminal State Handling
When a state is terminated, TD target = reward (no bootstrap). The policy still learns from terminal state actions — this is intentional (the action that led to termination still matters).

### 8. T Dimension
Even though imagination_horizon=1, keep the T dimension in code:
- `z_seq` has shape `[T_imag+1, H, B_exp, L]` = `[2, H, B_exp, L]` (initial + next)
- `td_targets` has shape `[T_imag, R, H, Ve, B_exp, 1]` = `[1, R, H, Ve, B_exp, 1]`
- This allows potential future extension to multi-step imagination

### 9. Logging Metrics
New metrics to monitor regression behavior:
- `regression_weight_entropy` — should be positive (not collapsed)
- `regression_q_scaled_mean/std` — Q-value distribution
- `regression_nll_term` — weighted NLL magnitude
- `regression_entropy_term` — entropy bonus magnitude
- Ratio of entropy_term / nll_term — useful for understanding balance

Log at same frequency as existing policy metrics.

---

## Summary of All Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Imagination horizon | Always 1 | Simplicity, single-step TD |
| Config removals | `ac_source`, `aux_value_source`, `actor_source`, `imagination_horizon` | Hardcode instead |
| Head reduction (pessimistic) | Use `policy_head_reduce` (min or mean) | Configurable pessimism |
| Head reduction (optimistic) | Always max | Optimism for exploration |
| Reduction order | Flatten R×H×Ve, reduce at once | Simpler than sequential |
| Aux value | Compute separately | Different gammas |
| `pi_params` function | Don't add, use `model.pi()` directly | Reuse existing code |
| Wasted sampling | Acceptable | Sampling is cheap |
| Actions | Detached | No gradient through randomness |
| Log_probs | Attached | Gradients through μ, σ |
| Entropy | Attached | Gradients through σ |
| RunningScale | Update with all N Q-values | Consistent statistics |
| Terminal states | Policy still learns | Action before terminal matters |
| `rollout_latents` return | Always 4 values | Consistent API |
| Entropy coefficient | Reuse existing logic | May need different magnitude |
| Temperature | Start at 1.0, config flag | Tune via sweep |
