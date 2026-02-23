# BMPC JAX vs TDMPC2 Codebase Comparison

This document compares the BMPC JAX implementation (`bmpc_jax_orig/bmpc-jax/bmpc_jax/`) with the tdmpc2 codebase (`tdmpc2/`) to identify functional differences in architecture, losses, and training procedures.

**Focus Configuration** (BMPC-style settings):
- tau: 0.001–0.01
- td_target_use_ema_policy: false
- policy_trust_region_coef: 0
- utd_ratio: 1
- encoder_consistency_coef: 0
- value_update_freq: 1
- end_entropy_coeff: 0.0001
- reanalyze_slice_mode: true
- num_rollouts: 1
- dropout: 0.01
- All std_coef settings: 0 (no optimism/pessimism)
- planner_num_dynamics_heads: 1
- num_q (value heads): 2
- num_reward_heads: 1
- num_reward_layers: 2
- local_td_bootstrap: false
- dual_policy_enabled: false

---

## 1. Environment Wrappers

### BMPC JAX
**File**: [bmpc_jax_orig/bmpc-jax/bmpc_jax/envs/dmcontrol.py](../bmpc_jax_orig/bmpc-jax/bmpc_jax/envs/dmcontrol.py)

```python
# Wrappers applied (in order):
env = suite.load(domain, task, ...)
env = ActionDTypeWrapper(env, np.float32)
env = ActionRepeatWrapper(env, 2)  # Action repeat = 2
env = action_scale.Wrapper(env, minimum=-1., maximum=1.)  # Rescale to [-1, 1]
env = ExtendedTimeStepWrapper(env)
env = TimeStepToGymWrapper(env, domain, task)  # Converts to Gym API
```

- **Action Repeat**: 2 (accumulates rewards with discounting)
- **Action Scale**: [-1, +1]
- **Episode Length**: Fixed 500 steps (after action repeat = 1000 raw env steps)
- **Termination**: Episodes never terminate early (`terminated=False` always), only truncate

### TDMPC2
**File**: [tdmpc2/envs/dmcontrol.py](../tdmpc2/envs/dmcontrol.py)

```python
# Wrappers applied:
env = suite.load(domain, task, ...)
env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
env = DMControlWrapper(env, domain)  # Custom wrapper with action repeat inside
env = Timeout(env, max_episode_steps=500)
```

- **Action Repeat**: 2 (hardcoded in `DMControlWrapper.step()`)
- **Action Scale**: [-1, +1]
- **Episode Length**: Fixed 500 steps

### Key Differences
| Aspect | BMPC JAX | TDMPC2 |
|--------|----------|--------|
| Action repeat location | Separate wrapper | Inside DMControlWrapper |
| Reward accumulation | `reward += r * discount; discount *= d` | `reward += r` (simple sum) |
| Returns observation | numpy array | torch tensor |

**⚠️ DIFFERENCE**: BMPC JAX applies discount during action repeat for reward accumulation, TDMPC2 does simple sum.

---

## 2. World Model Architecture

### BMPC JAX
**File**: [bmpc_jax_orig/bmpc-jax/bmpc_jax/world_model.py](../bmpc_jax_orig/bmpc-jax/bmpc_jax/world_model.py)  
**Layers**: [bmpc_jax_orig/bmpc-jax/bmpc_jax/networks/mlp.py](../bmpc_jax_orig/bmpc-jax/bmpc_jax/networks/mlp.py)

**Architecture Overview**:
```python
# Encoder (state)
encoder = Sequential([
    NormedLinear(encoder_dim, activation=mish)  # x2 layers
    NormedLinear(latent_dim, activation=None)   # Output layer
])
# Then SimNorm is applied: simnorm(z, simplex_dim=8)

# Dynamics (latent prediction)
dynamics = Sequential([
    NormedLinear(latent_dim, activation=mish),
    NormedLinear(latent_dim, activation=mish),
    NormedLinear(latent_dim, activation=None),  # No activation
])
# Output: simnorm(z) applied in encode() method

# Reward Head
reward = Sequential([
    NormedLinear(latent_dim, activation=mish),
    NormedLinear(latent_dim, activation=mish),
    Dense(num_bins, kernel_init=zeros)  # Zero-init output
])

# Value Head (Ensemble of 5)
value = Ensemble([
    Sequential([
        NormedLinear(latent_dim, activation=mish, dropout=0.01),
        NormedLinear(latent_dim, activation=mish, dropout=0.01),
        Dense(num_bins, kernel_init=zeros)
    ])
], num=5)

# Policy Head
policy = Sequential([
    NormedLinear(latent_dim, activation=mish),
    NormedLinear(latent_dim, activation=mish),
    Dense(2*action_dim, kernel_init=truncated_normal(0.02))
])
```

**NormedLinear Layer**:
```python
# Dense -> LayerNorm -> Activation [-> Dropout]
Dense(features) -> LayerNorm() -> activation() [-> Dropout(rate)]
```

**SimNorm** (simplicial normalization):
```python
def simnorm(x, simplex_dim=8):
    x = rearrange(x, '...(L V) -> ... L V', V=simplex_dim)  # Split into groups of 8
    x = softmax(x, axis=-1)  # Softmax per group
    return rearrange(x, '... L V -> ... (L V)')  # Flatten back
```

**Configs**:
- latent_dim: 512
- encoder_dim: 256
- num_enc_layers: 2
- num_bins: 101
- symlog_min: -10, symlog_max: +10
- num_value_nets: 5
- value_dropout: 0.01
- simnorm_dim: 8

### TDMPC2
**File**: [tdmpc2/common/world_model.py](../tdmpc2/common/world_model.py)  
**Layers**: [tdmpc2/common/layers.py](../tdmpc2/common/layers.py)

**Architecture Overview**:
```python
# Encoder
encoder = mlp(obs_dim + task_dim, [enc_dim]*(num_enc_layers-1), latent_dim, 
              act=SimNorm(cfg), dropout=encoder_dropout, dropout_layer=-1)

# Dynamics (with optional prior for diversity)
dynamics = DynamicsHeadWithPrior(
    in_dim=latent_dim + action_dim + task_dim,
    mlp_dims=[mlp_dim] * dynamics_num_layers,  # Default: 2 layers
    out_dim=latent_dim,
    prior_hidden_div=16,
    prior_scale=0.1,  # Optional frozen prior
    dropout=dynamics_dropout,
)
# Output: SimNorm applied after MLP

# Reward Head (Ensemble via vmap)
reward = Ensemble([
    MLPWithPrior(
        in_dim=latent_dim + action_dim + task_dim,
        hidden_dims=[reward_hidden_dim] * num_reward_layers,
        out_dim=num_bins,
        prior_hidden_div=16,
        prior_scale=0.1,
        distributional=True,
    )
], num=num_reward_heads)

# Value Head (Ensemble)
Vs = Ensemble([
    MLPWithPrior(
        in_dim=latent_dim + task_dim,  # V(s) not V(s,a)
        hidden_dims=[v_mlp_dim] * num_value_layers,
        out_dim=num_bins,
        prior_hidden_div=16,
        prior_scale=0.1,
        distributional=True,
        dropout=dropout,
    )
], num=num_q)

# Policy Head
pi = mlp(latent_dim + task_dim, [mlp_dim, mlp_dim], 2*action_dim)
```

**NormedLinear Layer** (same concept):
```python
class NormedLinear(nn.Linear):
    def forward(self, x):
        x = super().forward(x)  # Linear
        if self.dropout: x = self.dropout(x)
        return self.act(self.ln(x))  # LayerNorm -> Activation
```

**Configs** (your sweep settings applied):
- latent_dim: 512
- mlp_dim: 512
- num_bins: 101
- vmin: -10, vmax: +10
- num_q: 2 (value heads)
- dropout: 0.01
- simnorm_dim: 8
- num_reward_heads: 1
- num_reward_layers: 2

### Key Differences
| Aspect | BMPC JAX | TDMPC2 |
|--------|----------|--------|
| Value function | V(s) with ensemble of 5 | V(s) with ensemble of 2 (your config) |
| Reward input | z concat a | z concat a |
| Dynamics input | z concat a | z concat a concat task_emb |
| Prior networks | None | Optional frozen prior for diversity |
| Dropout location | Value only (in NormedLinear) | Encoder (last hidden), Value, optionally Reward |
| Encoder output | encoder -> simnorm | mlp with SimNorm as final activation |
| Multi-head dynamics | 1 head | 1 head (your config) |
| Weight init (output) | zeros for reward/value | zeros for reward/value main MLP |

**⚠️ FUNCTIONAL**: Both use V(s) state-value function (not Q(s,a)), which is identical.

---

## 3. Losses - World Model

### 3.1 Consistency Loss

**BMPC JAX** ([bmpc.py L160-170](../bmpc_jax_orig/bmpc-jax/bmpc_jax/bmpc.py#L160)):
```python
# MSE between predicted latent and encoded next observation
consistency_loss = 0
for t in range(horizon):
    z_pred = model.next(latent_zs[t], actions[t], dynamics_params)
    # sg() = stop_gradient
    consistency_loss += lam[t] * mean((z_pred - sg(next_zs[t]))**2, where=~finished)
```

**Math**: 
$$\mathcal{L}_{consist} = \sum_{t=0}^{T-1} \rho^t \cdot \frac{1}{B} \sum_b \| f_\theta(z_t, a_t) - \text{sg}(\text{enc}(o_{t+1})) \|_2^2$$

**TDMPC2** ([tdmpc2.py world_model_losses](../tdmpc2/tdmpc2.py#L1370)):
```python
# Multi-head rollout, average MSE across heads
pred_TBL = lat_all[:, :, 0, 1:, :].permute(0, 2, 1, 3)  # [H,T,B,L]
target_TBL = z_consistency_target[1:].unsqueeze(0)  # [1,T,B,L]
delta = pred_TBL - target_TBL.detach()
consistency_losses = delta.pow(2).mean(dim=(0, 2, 3))  # [T]
consistency_loss = (rho_pows * consistency_losses).mean()
```

**Math** (identical):
$$\mathcal{L}_{consist} = \frac{1}{T} \sum_{t=0}^{T-1} \rho^t \cdot \frac{1}{H \cdot B} \sum_{h,b} \| f_\theta^h(z_t, a_t) - \text{sg}(\text{enc}(o_{t+1})) \|_2^2$$

### 3.2 Reward Loss

**BMPC JAX**:
```python
# Soft cross-entropy with two-hot targets
_, reward_logits = model.reward(latent_zs, actions, reward_params)
reward_loss = sum(
    lam * soft_crossentropy(reward_logits, rewards, symlog_min, symlog_max, num_bins),
    axis=0, where=~finished
).mean()
```

```python
def soft_crossentropy(pred_logits, target, low, high, num_bins):
    pred = log_softmax(pred_logits, axis=-1)
    target = two_hot(target, low, high, num_bins)
    return -(pred * target).sum(axis=-1)
```

**Math**:
$$\mathcal{L}_{reward} = -\sum_{t=0}^{T-1} \rho^t \cdot \frac{1}{B} \sum_b \sum_k \hat{r}_k^{(t,b)} \cdot \log p_k^{(t,b)}$$

where $\hat{r}$ is the two-hot encoding of symlog(reward).

**TDMPC2**:
```python
# Same soft cross-entropy formulation
reward_logits_all = self.model.reward(lat_flat, actions_flat, task, head_mode='all')
rew_ce = math.soft_ce(logits, reward_target, self.cfg)  # [R*T*H*B]
reward_loss = (rho_pows * rew_ce.mean(dim=(0,2,3))).mean()  # Average over R, H, B
```

**Math** (identical):
$$\mathcal{L}_{reward} = -\sum_{t=0}^{T-1} \rho^t \cdot \frac{1}{R \cdot H \cdot B} \sum_{r,h,b} \sum_k \hat{r}_k \cdot \log p_k^{(r,h,t,b)}$$

### Key Differences
| Aspect | BMPC JAX | TDMPC2 |
|--------|----------|--------|
| Consistency target grad | stop_gradient on target | detach() on target |
| Multi-head reduction | N/A (1 head) | Mean over H heads |
| Reward heads | 1 | Configurable (1 in your config) |
| Loss weighting | λ = ρ^t / Σρ^t (normalized) | λ = ρ^t (unnormalized, then mean) |

**⚠️ MATH**: BMPC normalizes ρ weights (`lam /= sum(lam)`), TDMPC2 uses raw ρ^t then takes mean.

---

## 4. Losses - Value/Q

### TD Target Computation

**BMPC JAX** ([bmpc.py td_target](../bmpc_jax_orig/bmpc-jax/bmpc_jax/bmpc.py#L293)):
```python
def td_target(self, z, num_td_steps=1, *, key):
    G, discount = 0, 1
    for t in range(num_td_steps):  # Default: 1 step
        action = model.sample_actions(z, params, key)[0]  # Policy action
        reward, _ = model.reward(z, action, reward_params)
        z = model.next(z, action, dynamics_params)
        G += discount * reward
        discount *= self.discount
    
    # Use TARGET value network
    Vs, _ = model.V(z, target_value_model.params, key)
    V = Vs.mean(axis=0)  # Mean over 5 value heads
    td_target = G + discount * V
    return td_target
```

**Math** (1-step imagination):
$$y = r(z, \pi(z)) + \gamma \cdot \bar{V}(f(z, \pi(z)))$$

where $\bar{V}$ is the EMA target network, mean over value ensemble.

**TDMPC2** ([tdmpc2.py _td_target](../tdmpc2/tdmpc2.py#L997)):
```python
def _td_target(self, next_z, reward, terminated, task):
    # next_z: [T, H, B, L] - from imagined rollout
    # reward: [T, R, H, B, 1] - from all reward heads and dynamics heads
    
    # Get V from TARGET network for all value heads
    v_logits_flat = self.model.V(next_z_flat, task, return_type='all', target=True)
    v_values = two_hot_inv(v_logits_flat, cfg)  # [Ve, T, H*B, 1]
    
    # Mean across reward heads per dynamics head
    r_mean_per_h = reward.mean(dim=1)  # [T, H, B, 1]
    
    # Global bootstrap: mean across value heads
    v_mean_per_h = v_values.mean(dim=0)  # [T, H, B, 1]
    
    # TD target per dynamics head
    td_mean_per_h = r_mean + discount * (1 - terminated) * v_mean_per_h
    
    # Reduce over dynamics heads (mean when std_coef=0)
    td_targets = td_mean_per_h.mean(dim=1)  # [T, B, 1]
    
    # All Ve heads get same target (global bootstrap)
    td_targets = td_targets.expand(Ve, T, B, 1)
```

**Math** (with your config: local_td_bootstrap=false, std_coefs=0):
$$y = \frac{1}{R \cdot H} \sum_{r,h} r_{r,h} + \gamma (1-d) \cdot \frac{1}{V_e \cdot H} \sum_{v,h} V_v(z'_h)$$

Reduced over dynamics and value heads by mean.

### Value Loss

**BMPC JAX**:
```python
# Soft CE between predicted V and TD target
_, V_logits = model.V(latent_zs, value_params, key)  # [Ve, T, B, K]
td_targets = td_target(encoder_zs, key)  # [T, B]
value_loss = sum(
    lam * soft_crossentropy(V_logits, sg(td_targets), symlog_min, symlog_max, num_bins),
    axis=1, where=~finished
).mean()
```

**TDMPC2**:
```python
# Same soft CE, but computed on imagined rollout
z_for_v = z_seq[:-1, 0]  # Head 0 for V prediction
vs = model.V(z_for_v_flat, task, return_type='all')  # [Ve, T*S*BN, K]
val_ce = soft_ce(vs, td_targets, cfg)  # [Ve, T, S, BN]
# Rho weighting on S (replay buffer steps used as imagination starts)
loss = (val_ce_per_s * rho_pows).mean()
```

### Key Differences
| Aspect | BMPC JAX | TDMPC2 |
|--------|----------|--------|
| TD target source | Imagined from encoder states | Imagined from rollout/encoder states |
| Value heads | 5, mean reduction | 2 (your config), mean reduction |
| Target network | EMA (tau=0.01) | EMA (tau=0.001-0.01) |
| local_td_bootstrap | N/A (always global) | Configurable (false = global) |
| Imagination policy | Online policy | EMA policy if td_target_use_ema_policy |
| Multi-head dynamics | N/A | 1 head (your config) |

**⚠️ KEY**: Both use 1-step TD with imagination. BMPC imagines from encoder states directly, TDMPC2 from either encoder or dynamics rollout states.

---

## 5. Losses - Policy

### BMPC JAX (KL Distillation)
**File**: [bmpc.py update_policy](../bmpc_jax_orig/bmpc-jax/bmpc_jax/bmpc.py#L306)

```python
def update_policy(self, zs, expert_mean, expert_std, finished, key):
    _, mean, log_std, log_probs = model.sample_actions(z=zs, params=actor_params, key=key)
    
    # Policy distribution
    action_dist = MultivariateNormalDiag(mean, exp(log_std))
    # Expert distribution (from planner)
    expert_dist = MultivariateNormalDiag(expert_mean, expert_std)
    
    # KL divergence: KL(policy || expert)
    kl_div = kl_divergence(action_dist, expert_dist)
    
    # Percentile normalization of KL scale
    kl_scale = percentile_normalization(kl_div[0], prev_kl_scale).clip(1, None)
    
    # Policy loss = normalized KL
    policy_loss = sum(lam * kl_div / kl_scale, axis=0, where=~finished).mean()
```

**Math**:
$$\mathcal{L}_{\pi} = \sum_{t=0}^{T-1} \rho^t \cdot \frac{1}{s} \cdot \text{KL}(\pi(\cdot|z_t) \| \text{expert}(\cdot|z_t))$$

where $s$ is the running percentile-normalized scale.

**Policy Parameterization** (BMPC style):
```python
mean = tanh(mean_raw)  # Squash mean first
log_std = MIN + (MAX - MIN) * 0.5 * (tanh(log_std_raw) + 1)
std = exp(log_std)
action = (mean + eps * std).clip(-1, 1)  # Clamp instead of tanh
log_prob = gaussian_logprob(eps, log_std)  # No Jacobian correction
```

- MIN_LOG_STD = -5, MAX_LOG_STD = 1

### TDMPC2

**Policy Optimization Methods**:
1. **SVG (Stochastic Value Gradient)** - backprop through model
2. **Distillation** - KL to expert planner targets
3. **Both** - weighted combination

**For BMPC-like config** (`policy_optimization_method: distillation`):
```python
def calc_pi_distillation_losses(self, z, expert_action_dist, task, optimistic=False):
    _, info = model.pi(z[:-1], task)
    policy_mean = info["mean"]           # [T, B, A]
    policy_std = info["log_std"].exp()   # [T, B, A]
    
    expert_mean = expert_action_dist[..., 0]  # [T, B, A]
    expert_std = expert_action_dist[..., 1]   # [T, B, A]
    
    # KL(expert || policy) when fix_kl_order=True, else KL(policy || expert)
    kl_per_dim = kl_div_gaussian(expert_mean, expert_std, policy_mean, policy_std)
    kl_loss = kl_per_dim.mean(dim=-1, keepdim=True)  # [T, B, 1]
    
    # Running scale normalization
    self.kl_scale.update(kl_loss[0])
    kl_scaled = self.kl_scale(kl_loss)
    
    # Entropy bonus
    entropy_term = info["scaled_entropy"]
    
    # Final loss with rho weighting
    objective = kl_scaled - entropy_coeff * entropy_term
    pi_loss = (objective.mean(dim=(1,2)) * rho_pows).mean()
```

**Math**:
$$\mathcal{L}_{\pi} = \sum_{t=0}^{T-1} \rho^t \left[ \frac{1}{s} \text{KL}(\text{expert} \| \pi) - \alpha \cdot H(\pi) \right]$$

**Policy Parameterization** (`bmpc_policy_parameterization: true`):
```python
mean = tanh(mean)  # Squash mean first
action = (mean + eps * log_std.exp()).clamp(-1, 1)
log_prob = gaussian_logprob(eps, log_std)  # No Jacobian correction
```

### Key Differences
| Aspect | BMPC JAX | TDMPC2 (BMPC config) |
|--------|----------|----------------------|
| KL direction | KL(policy \|\| expert) | KL(expert \|\| policy) with fix_kl_order=true |
| Entropy bonus | None | Yes, with configurable coefficient |
| Scale normalization | Percentile (5-95) | Running scale with min_scale |
| log_std range | [-5, +1] | [-10, +2] (configurable) |
| Policy param | tanh(mean), clamp action | Same with bmpc_policy_parameterization=true |

**⚠️ KEY DIFFERENCE**: BMPC has no entropy bonus in policy loss! TDMPC2 adds entropy regularization by default.

To match BMPC exactly: set `end_entropy_coeff: 0`, or close to 0 like `0.0001`.

---

## 6. Planner/MPC

### BMPC JAX
**File**: [bmpc.py plan](../bmpc_jax_orig/bmpc-jax/bmpc_jax/bmpc.py#L88)

```python
def plan(self, z, horizon, prev_plan=None, deterministic=False, train=False, *, key):
    actions = zeros((batch, population_size, horizon, action_dim))
    
    # Policy prior samples (24 trajectories)
    z_t = z.repeat(policy_prior_samples, axis=-2)
    for t in range(horizon):
        policy_actions[..., t, :] = model.sample_actions(z_t, params, key)[0]
        z_t = model.next(z_t, policy_actions[..., t, :], params)
    actions[..., :policy_prior_samples, :, :] = policy_actions
    
    # Initialize distribution
    mean = zeros((batch, horizon, action_dim))
    std = full((batch, horizon, action_dim), max_plan_std)
    if prev_plan is not None:
        mean[..., :-1, :] = prev_plan[0][..., 1:, :]  # Warm start
    
    # MPPI iterations
    for i in range(mppi_iterations):  # 6 iterations
        # Sample from distribution
        actions[..., policy_prior_samples:, :, :] = (
            mean[..., None, :, :] + std[..., None, :, :] * noise[..., i, :, :]
        ).clip(-1, 1)
        
        # Compute values
        values = estimate_value(z_t, actions, horizon, key)
        
        # Select elites (64)
        elite_values, elite_inds = top_k(values, num_elites)
        elite_actions = take_along_axis(actions, elite_inds, axis=-3)
        
        # Update distribution
        score = softmax(temperature * elite_values)  # temp=0.5
        mean = sum(score * elite_actions, axis=-3)
        std = sqrt(sum(score * (elite_actions - mean)**2, axis=-3) + 1e-6)
        std = std.clip(min_plan_std, max_plan_std)
    
    # Final action selection
    if deterministic:
        action_ind = argmax(elite_values, axis=-1)
    else:
        action_ind = categorical(log(score), shape=batch_shape)
    action = take_along_axis(elite_actions, action_ind, axis=-3).squeeze(-3)
    
    if train:
        final_action = action[..., 0, :] + std[..., 0, :] * randn(...)
```

**Value Estimation**:
```python
def estimate_value(self, z, actions, horizon, key):
    G, discount = 0.0, 1.0
    for t in range(horizon):
        reward, _ = model.reward(z, actions[..., t, :], params)
        z = model.next(z, actions[..., t, :], params)
        G += discount * reward
        discount *= self.discount
    
    Vs, _ = model.V(z, value_model.params, key)  # Online network
    V = Vs.mean(axis=0)
    return G + discount * V
```

### TDMPC2
**File**: [tdmpc2/common/planner/planner.py](../tdmpc2/common/planner/planner.py)

```python
def plan(self, z0, eval_mode=False, ...):
    # Initialize with warm start
    mean = shifted_prev_mean(batch_size=1) if use_warm_start else zeros(B, T, A)
    std = full((B, T, A), max_std)
    
    # Policy-seeded trajectories (24 by default)
    latents_p, actions_p = world_model.rollout_latents(
        z0, use_policy=True, horizon=T, num_rollouts=S,
        head_mode=head_mode, policy_action_noise_std=0.05
    )
    vals_p = compute_values(latents_p, actions_p, ...)
    
    for it in range(iterations):  # 6 iterations
        # Sample action sequences from distribution
        actions_s = sample_action_sequences(mean, std, N)
        
        # World model rollout
        latents_s, actions_s = world_model.rollout_latents(
            z0, actions=actions_s, use_policy=False, head_mode=head_mode
        )
        
        # Compute values (with optional std_coef for optimism/pessimism)
        vals_s = compute_values(latents_s, actions_s, value_std_coef=0, ...)
        
        # Combine policy and sampled candidates
        scores = concat([policy_cache['scores'], scores_s], dim=1)
        
        # Elite selection
        elite_scores, elite_indices = topk(scores, K)
        
        # Update distribution (BMPC-style: subtract max, scale by temp)
        max_elite = elite_scores.max(dim=1)
        score_delta = elite_scores - max_elite
        w = exp(temp * score_delta)  # mult_by_temp=true
        w = w / w.sum()
        mean = (w * elite_actions).sum(dim=1)
        std = sqrt((w * (elite_actions - mean)**2).sum(dim=1)).clamp(min_std, max_std)
    
    # Final selection
    if eval_mode or greedy_train_action_selection:
        chosen_idx = scores.argmax(dim=1)
    else:
        probs = softmax(temp * (elite_scores - max_elite))
        chosen_idx = multinomial(probs, 1)
```

**Value Computation** (scoring.py):
```python
def compute_values(latents, actions, world_model, value_std_coef=0, ...):
    # latents: [H, B, N, T+1, L]
    H, B, N, T_plus_1, L = latents.shape
    
    # Compute rewards for all timesteps
    rewards = world_model.reward(z_flat, a_flat, head_mode='all')  # [R, T*H*B*N, K]
    rewards = two_hot_inv(rewards, cfg)  # [R, T*H*B*N, 1]
    
    # Compute terminal values
    v_terminal = world_model.V(z_final, return_type='all', target=use_ema_value)  # [Ve, H*B*N, K]
    
    # Discounted returns: G = sum_t(gamma^t * r_t) + gamma^T * V
    G = rewards_sum + discount_factor * v_terminal
    
    # With std_coef=0: just return mean
    return G.mean()  # Average over value/reward heads
```

### Config Comparison
| Parameter | BMPC JAX | TDMPC2 (your config) |
|-----------|----------|----------------------|
| horizon | 3 | 3 |
| mppi_iterations | 6 | 6 |
| population_size | 512 | 512 |
| policy_prior_samples | 24 | 24 |
| num_elites | 64 | 64 |
| temperature | 0.5 | 0.5 |
| min_plan_std | 0 | 0.05 |
| max_plan_std | 2 | 1 |
| Value network | Online | Online (ema_value_planning=false) |
| Warm start | Yes (shifted prev mean) | Yes |
| Score formula | softmax(temp * scores) | softmax(temp * (scores - max)) |
| std_coef | N/A (no optimism) | 0 (neutral) |

### Key Differences
| Aspect | BMPC JAX | TDMPC2 |
|--------|----------|--------|
| Score normalization | Raw scores | Subtract max before softmax |
| min_plan_std | 0 | 0.05 (prevents collapse) |
| max_plan_std | 2 | 1 |
| Policy noise in seeding | No explicit noise | 0.05 std noise added |
| Multi-head dynamics | N/A | Configurable (1 head in your config) |
| Value ensemble for planning | Mean of 5 heads | Mean of heads, configurable |

**⚠️ DIFFERENCE**: TDMPC2 subtracts max score before softmax for numerical stability, BMPC doesn't.

---

## 7. Optimizers

### BMPC JAX
```python
# All components use same optimizer settings
tx = optax.chain(
    optax.zero_nans(),
    optax.clip_by_global_norm(max_grad_norm=20),
    optax.adamw(learning_rate=3e-4),
)
```

**Settings**:
- Learning rate: 3e-4 for all components
- Optimizer: AdamW
- Grad clip: 20 (global norm)

### TDMPC2
```python
# Per-component param groups with different LRs
param_groups = [
    {'params': encoder.parameters(), 'lr': lr * enc_lr_scale},      # 3e-4 * 0.3
    {'params': dynamics.parameters(), 'lr': lr},                     # 3e-4
    {'params': reward.parameters(), 'lr': lr},                       # 3e-4
    {'params': termination.parameters(), 'lr': lr},                  # 3e-4
    {'params': Vs.parameters(), 'lr': lr},                           # 3e-4
]
self.optim = Adam(param_groups, lr=3e-4, capturable=True)

# Separate policy optimizer
self.pi_optim = Adam(pi_params, lr=lr * pi_lr_scale, eps=1e-5)     # 3e-4 * 1.0
```

**Settings** (your config with ensemble_lr_scaling=false):
| Component | BMPC JAX | TDMPC2 |
|-----------|----------|--------|
| Encoder LR | 3e-4 | 3e-4 * 0.3 = 9e-5 |
| Dynamics LR | 3e-4 | 3e-4 |
| Reward LR | 3e-4 | 3e-4 |
| Value LR | 3e-4 | 3e-4 |
| Policy LR | 3e-4 | 3e-4 |
| Grad clip | 20 | 20 |
| Optimizer | AdamW | Adam (your config) |

**⚠️ DIFFERENCE**: TDMPC2 uses lower encoder LR by default (0.3x). Set `enc_lr_scale: 1.0` to match BMPC.

---

## 8. Training Loop

### BMPC JAX
**File**: [bmpc_jax_orig/bmpc-jax/bmpc_jax/train.py](../bmpc_jax_orig/bmpc-jax/bmpc_jax/train.py)

```python
# Main loop structure
for global_step in range(max_steps):
    # Environment step
    if global_step <= seed_steps:
        action = env.action_space.sample()
    else:
        action, plan = agent.act(obs, prev_plan=plan, mpc=True, train=True)
    
    # Store transition + expert targets
    replay_buffer.insert({
        observation, action, reward, next_obs, terminated, truncated,
        expert_mean, expert_std  # From planner
    })
    
    # Training
    if global_step >= seed_steps:
        num_updates = max(1, int(num_envs * utd_ratio))  # 1 with utd_ratio=1
        
        for iupdate in range(num_updates):
            batch = replay_buffer.sample(batch_size=256, horizon=3)
            
            # Update world model (single call)
            agent, train_info = agent.update_world_model(
                observations, actions, rewards, next_observations,
                terminated, truncated
            )
            
            # Reanalyze every 10 updates
            if total_num_updates % reanalyze_interval == 0:
                _, reanalyzed_plan = agent.plan(encoder_zs[:, :reanalyze_batch_size, :], horizon=3)
                # Update buffer with new expert targets
                replay_buffer.data['expert_mean'][...] = reanalyze_mean
                replay_buffer.data['expert_std'][...] = reanalyze_std
            
            # Update policy with (possibly reanalyzed) expert targets
            agent, policy_info = agent.update_policy(
                zs=latent_zs,
                expert_mean=batch['expert_mean'],
                expert_std=batch['expert_std'] * policy_std_scale,  # 1.5x
            )
```

**Batch Structure**:
- All tensors are `[T, B, ...]` where T=horizon, B=batch_size
- Observations/actions offset by 1 (obs[t] aligned with action[t])

### TDMPC2
**File**: [tdmpc2/trainer/online_trainer.py](../tdmpc2/trainer/online_trainer.py)

```python
# Main loop structure
while step < total_steps:
    # Collect experience
    for _ in range(buffer_update_interval):
        action, planner_info = agent.act(obs, eval_mode=False, mpc=train_mpc)
        obs, reward, terminated, info = env.step(action)
        buffer.add(obs, action, reward, expert_action_dist)
    
    # Training (utd_ratio updates per env step)
    for _ in range(utd_ratio):
        # World model update
        if step % value_update_freq != 0:
            info = agent.update(buffer, step, update_value=False)
        else:
            info = agent.update(buffer, step, update_value=True)
```

**agent.update() internals** ([tdmpc2.py update](../tdmpc2/tdmpc2.py#L2117)):
```python
def update(self, buffer, step, update_value=True, update_pi=True):
    # Lazy reanalyze (before _update)
    if step % reanalyze_interval == 0:
        if reanalyze_slice_mode:
            # BMPC style: reanalyze all T timesteps for fewer slices
            obs_reanalyze = obs[:T, :num_slices].reshape(-1, *obs_shape)
        else:
            # Independent: sample B independent t=0 observations
            obs_reanalyze = obs[0, :reanalyze_batch_size]
        
        expert_action_dist_new, _, _ = self.reanalyze(obs_reanalyze)
        buffer.update_expert_data(indices, expert_action_dist_new)
    
    # Core update
    info = self._update(obs, action, reward, terminated, expert_action_dist, ...)
```

### Key Differences
| Aspect | BMPC JAX | TDMPC2 |
|--------|----------|--------|
| UTD ratio | 1 (default) | 4 (default), 1 (your config) |
| Value update freq | Every update | Every update (value_update_freq=1) |
| Reanalyze mode | Fixed: first reanalyze_batch_size in batch | Configurable: slice (BMPC) or independent |
| Expert std scaling | 1.5x * clip(min=0.1) | None by default |
| Policy update | Every world model update | Configurable (pi_update_freq=1) |
| Seed steps | 5 * episode_len * num_envs * utd_ratio | 5 * episode_len |

**⚠️ DIFFERENCE**: BMPC scales expert std by 1.5x and clips to min 0.1. TDMPC2 doesn't by default.

---

## 9. Config Defaults Comparison

### Critical Parameters (BMPC defaults → TDMPC2 equivalents)

| Parameter | BMPC JAX Default | TDMPC2 Default | Your Config |
|-----------|------------------|----------------|-------------|
| **Training** |
| utd_ratio | 1 | 4 | 1 ✓ |
| batch_size | 256 | 256 | 256 ✓ |
| horizon | 3 | 3 | 3 ✓ |
| discount | 0.99 | Task-dependent | ~0.99 |
| rho | 0.5 | 0.5 | 0.5 ✓ |
| tau (EMA) | 0.01 | 0.003 | 0.001-0.01 ✓ |
| lr | 3e-4 | 3e-4 | 3e-4 ✓ |
| enc_lr_scale | 1.0 | 0.3 | **Set to 1.0** |
| **Architecture** |
| latent_dim | 512 | 512 | 512 ✓ |
| mlp_dim | 512 | 512 | 512 ✓ |
| num_value_nets | 5 | 4 (num_q) | 2 ✓ |
| num_bins | 101 | 101 | 101 ✓ |
| dropout | 0.01 (value only) | 0 | 0.01 ✓ |
| simnorm_dim | 8 | 8 | 8 ✓ |
| **Losses** |
| consistency_coef | 10 | 20 | **Keep 20** |
| reward_coef | 0.1 | 0.1 | 0.1 ✓ |
| value_coef | 0.1 | 0.1 | 0.1 ✓ |
| encoder_consistency_coef | N/A (included in main) | 1 | 0 ✓ |
| **Policy** |
| policy method | Distillation only | SVG/Distillation | Distillation |
| entropy_coeff | 0 | 1e-6 → 1e-4 | 0.0001 ✓ |
| bmpc_policy_param | Yes | Configurable | true ✓ |
| fix_kl_order | N/A (KL(pi\|\|exp)) | false | **Set to true** |
| **Planner** |
| iterations | 6 | 6 | 6 ✓ |
| num_samples | 512 | 512 | 512 ✓ |
| num_pi_trajs | 24 | 24 | 24 ✓ |
| num_elites | 64 | 64 | 64 ✓ |
| temperature | 0.5 | 0.5 | 0.5 ✓ |
| min_std | 0 | 0.05 | **Set to 0** |
| max_std | 2 | 1 | **Set to 2** |
| policy_seed_noise | 0 | 0.05 | **Set to 0** |
| **Reanalyze** |
| interval | 10 | 10 | 10 ✓ |
| batch_size | 20 | 64 | **Set to 20** |
| horizon | 3 | inherit (3) | 3 ✓ |
| slice_mode | Yes (implicit) | Configurable | true ✓ |
| use_chosen_action | Yes | Configurable | true ✓ |

---

## 10. Summary: Key Differences to Address

### Must Change for BMPC Equivalence:

1. **Encoder LR**: 
   ```yaml
   enc_lr_scale: 1.0  # Currently 0.3
   ```

2. **KL Direction**:
   ```yaml
   fix_kl_order: true  # KL(expert || policy) instead of KL(policy || expert)
   ```

3. **Planner Std Range**:
   ```yaml
   min_std: 0       # Currently 0.05
   max_std: 2       # Currently 1
   ```

4. **Policy Seed Noise**:
   ```yaml
   policy_seed_noise_std: 0  # Currently 0.05
   ```

5. **Reanalyze Batch Size**:
   ```yaml
   reanalyze_batch_size: 20  # Currently 64
   ```

### Remaining Differences (Minor/Structural):

1. **Consistency Coefficient**: BMPC uses 10, TDMPC2 uses 20. The separate `encoder_consistency_coef` in TDMPC2 adds extra gradients to encoder.

2. **Expert Std Scaling**: BMPC scales by 1.5x and clips to min 0.1 during policy update. TDMPC2 doesn't by default.

3. **Score Softmax**: TDMPC2 subtracts max score before softmax for numerical stability.

4. **Action Repeat Rewards**: BMPC discounts rewards during action repeat, TDMPC2 sums directly.

5. **Observation Return Type**: BMPC returns numpy, TDMPC2 returns torch tensor.

### Unclear Areas Requiring Further Investigation:

1. **rho Normalization**: BMPC normalizes ρ weights (λ = ρ^t / Σρ^t), TDMPC2 uses raw ρ^t then takes mean. Impact unclear.

2. **Value Ensemble Reduction**: Both use mean, but TDMPC2 has additional features for std-based optimism/pessimism which don't exist in BMPC.

3. **Expert Std Scaling**: The exact impact of BMPC's 1.5x scaling + 0.1 min clipping needs evaluation.

---

## Appendix: File Paths Reference

### BMPC JAX
| Component | File |
|-----------|------|
| Config | `bmpc_jax_orig/bmpc-jax/bmpc_jax/config.yaml` |
| World Model | `bmpc_jax_orig/bmpc-jax/bmpc_jax/world_model.py` |
| Agent/Training | `bmpc_jax_orig/bmpc-jax/bmpc_jax/bmpc.py` |
| Main Train Loop | `bmpc_jax_orig/bmpc-jax/bmpc_jax/train.py` |
| Losses | `bmpc_jax_orig/bmpc-jax/bmpc_jax/common/loss.py` |
| Utils (two_hot) | `bmpc_jax_orig/bmpc-jax/bmpc_jax/common/util.py` |
| Layers | `bmpc_jax_orig/bmpc-jax/bmpc_jax/networks/mlp.py` |
| Activations | `bmpc_jax_orig/bmpc-jax/bmpc_jax/common/activations.py` |
| Scale | `bmpc_jax_orig/bmpc-jax/bmpc_jax/common/scale.py` |
| DMControl Env | `bmpc_jax_orig/bmpc-jax/bmpc_jax/envs/dmcontrol.py` |

### TDMPC2
| Component | File |
|-----------|------|
| Config | `tdmpc2/config.yaml` |
| Agent | `tdmpc2/tdmpc2.py` |
| World Model | `tdmpc2/common/world_model.py` |
| Layers | `tdmpc2/common/layers.py` |
| Math Utils | `tdmpc2/common/math.py` |
| Planner | `tdmpc2/common/planner/planner.py` |
| Scoring | `tdmpc2/common/planner/scoring.py` |
| DMControl Env | `tdmpc2/envs/dmcontrol.py` |
| Online Trainer | `tdmpc2/trainer/online_trainer.py` |
