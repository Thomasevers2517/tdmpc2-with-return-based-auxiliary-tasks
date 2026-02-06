# TDMPC2 Implementation Deep Dive

This document provides detailed documentation of the TDMPC2 implementation losses and training components with exact code locations and mathematical formulas.

---

## 1. TD Target Computation

**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L966-L1100)  
**Method:** `_td_target()`

### Mathematical Formula

The TD target is computed via:

$$\text{TD}_h = \mu_h + \text{td\_target\_std\_coef} \times \sigma_h$$

Where for each dynamics head $h$:
- **Local bootstrapping** (`local_td_bootstrap=True`, default):
  - $\mu_{ve,h} = r_{\text{mean},h} + \gamma (1 - \text{term}) \cdot v_{ve,h}$
  - $\sigma_h = r_{\text{std},h}$ (no value std since each Ve head bootstraps itself)
  - Each value ensemble member Ve gets its own target
  
- **Global bootstrapping** (`local_td_bootstrap=False`):
  - $\mu_h = r_{\text{mean},h} + \gamma (1 - \text{term}) \cdot v_{\text{mean},h}$
  - $\sigma_h = r_{\text{std},h} + \gamma \cdot v_{\text{std},h}$
  - All Ve heads get the same target

### Dynamics Head Reduction

After computing TD per dynamics head, reduction happens via `td_target_dynamics_reduction`:
```python
# From tdmpc2.py L1043-1053
dyn_reduction = self.cfg.td_target_dynamics_reduction
if dyn_reduction == "from_std_coef":
    dyn_reduction = "max" if std_coef > 0 else ("min" if std_coef < 0 else "mean")
if dyn_reduction == "max":
    td_targets, _ = td_per_ve_h.max(dim=2)  # float32[Ve, T, B, 1]
elif dyn_reduction == "min":
    td_targets, _ = td_per_ve_h.min(dim=2)
else:  # "mean"
    td_targets = td_per_ve_h.mean(dim=2)
```

### Key Config Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `td_target_std_coef` | `0` | Pessimism/optimism coefficient (0 = neutral) |
| `td_target_dynamics_reduction` | `"mean"` | How to reduce over dynamics heads (`mean`/`min`/`max`) |
| `local_td_bootstrap` | `true` | Each Ve head bootstraps itself (no cross-head variance) |
| `td_target_use_all_dynamics_heads` | `true` | Use all H heads or single random head |

### Does it use n-step returns?
**No.** The implementation uses **1-step TD targets**: $r_t + \gamma V(s_{t+1})$

### Does it use TD(λ)?
**No.** There is no λ-return or eligibility trace mechanism. Pure 1-step bootstrapping.

### td_target_use_ema_policy Effect
**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L1516-L1519)

When `td_target_use_ema_policy=True`, the imagined trajectories for TD targets use the EMA target policy instead of the online policy:
```python
imagined = self.imagined_rollout(
    start_z,
    task=task,
    rollout_len=T_imag,
    use_target_policy=self.cfg.td_target_use_ema_policy,  # <-- Here
)
```

This provides more stable targets by using a slower-moving policy for imagination.

### local_td_bootstrap Effect
**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L1016-1060)

- `local_td_bootstrap=True` (default): Each value ensemble head Ve bootstraps from its own value predictions. No cross-head value variance is used.
- `local_td_bootstrap=False`: All Ve heads receive the same target computed from mean-reduced values. Value uncertainty σ_v is included.

---

## 2. Policy Loss Computation

### 2.1 SVG Policy Loss
**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L518-L693)  
**Method:** `calc_pi_losses()`

**Formula:**
$$\mathcal{L}_\pi = -\mathbb{E}_{z,a\sim\pi}\left[\frac{Q(z,a)}{\text{scale}} + \alpha \cdot H[\pi]\right]$$

Where:
- $Q_h = \mu_h + \text{policy\_value\_std\_coef} \times \sigma_h$
- $\sigma_h = r_{\text{std}} + \gamma \cdot v_{\text{std},h}$ (per dynamics head)
- Reduction over dynamics heads follows `policy_value_std_coef` sign

```python
# From tdmpc2.py L640-648
if value_std_coef > 0:
    # Optimistic: max over dynamics heads
    q_estimate_flat, _ = q_per_h.max(dim=0)
elif value_std_coef < 0:
    # Pessimistic: min over dynamics heads
    q_estimate_flat, _ = q_per_h.min(dim=0)
else:
    # Neutral: mean over dynamics heads
    q_estimate_flat = q_per_h.mean(dim=0)
```

### 2.2 KL Distillation Policy Loss
**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L695-L791)  
**Method:** `calc_pi_distillation_losses()`

**Formula:**
$$\mathcal{L}_\pi = \text{KL}(\pi_\text{expert} \| \pi) - \alpha \cdot H[\pi]$$

(when `fix_kl_order=True`)

### KL Order: `fix_kl_order`
**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L755-L759)

```python
if self.cfg.fix_kl_order:
    # KL(expert || policy) - minimizing makes policy match expert
    kl_per_dim = math.kl_div_gaussian(expert_mean, expert_std, policy_mean, policy_std)
else:
    # KL(policy || expert) - legacy behavior (mode-seeking)
    kl_per_dim = math.kl_div_gaussian(policy_mean, policy_std, expert_mean, expert_std)
```

| Setting | KL Direction | Behavior |
|---------|--------------|----------|
| `fix_kl_order=True` | KL(expert \|\| policy) | Mean-covering: policy covers expert's modes |
| `fix_kl_order=False` | KL(policy \|\| expert) | Mode-seeking: policy collapses to expert's mode |

### Entropy Handling
**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L652-L660)

Entropy is computed with configurable action dimension power:
```python
# From world_model.py L515-520
entropy = -log_prob
entropy_multiplier = torch.tensor(
    action_dim, dtype=log_prob.dtype, device=log_prob.device
).pow(self.cfg.entropy_action_dim_power)
scaled_entropy = entropy * entropy_multiplier
```

### policy_trust_region_coef
**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L793-L846)  
**Method:** `calc_trust_region_kl_loss()`

When `policy_trust_region_coef > 0`, adds KL regularization to EMA policy:
$$\mathcal{L}_\text{TR} = \text{KL}(\pi_\text{EMA} \| \pi_\text{current})$$

This prevents rapid policy updates by penalizing divergence from the EMA policy.

---

## 3. World Model Losses

**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L1183-L1403)  
**Method:** `world_model_losses()`

### 3.1 Consistency Loss
**Lines:** [L1229-L1233](../tdmpc2/tdmpc2.py#L1229-L1233)

$$\mathcal{L}_\text{cons} = \mathbb{E}_t[\rho^t \cdot \|z^{pred}_t - z^{target}_t\|_2^2]$$

```python
delta = pred_TBL - target_TBL.detach()
consistency_losses = (delta.pow(2).mean(dim=(0, 2, 3)))  # float32[T]
consistency_loss = (rho_pows * consistency_losses).mean()
```

The consistency loss trains the dynamics model to predict accurate next-latent states.

### 3.2 Encoder Consistency Loss
**Lines:** [L1230-L1233](../tdmpc2/tdmpc2.py#L1230-L1233)

```python
delta_enc = pred_TBL.detach() - z_consistency_target[1:].unsqueeze(0)
encoder_consistency_losses = (delta_enc.pow(2).mean(dim=(0, 2, 3)))
encoder_consistency_loss = (rho_pows * encoder_consistency_losses).mean()
```

This is the Dreamer-style encoder consistency: gradients flow through encoder but dynamics predictions are detached.

**Effect of `encoder_consistency_coef`:** Controls gradient flow to encoder from consistency loss. Higher values increase encoder updates from dynamics prediction errors.

**Warmup:** `encoder_consistency_warmup_ratio` disables encoder consistency for the first X% of training.

### 3.3 Reward Loss
**Lines:** [L1281-L1301](../tdmpc2/tdmpc2.py#L1281-L1301)

$$\mathcal{L}_r = \mathbb{E}_{t,h}[\rho^t \cdot \text{CE}(r^{pred}_{h,t}, r^{target}_t)]$$

```python
rew_ce_flat = math.soft_ce(logits_flat, reward_target_flat, self.cfg)
rew_ce_all = rew_ce_flat.view(R, T, H_dyn, B)  # [R, T, H, B]
rew_ce = rew_ce_all.mean(dim=(0, 2, 3))  # float32[T]
reward_loss_branch = (rho_pows * rew_ce).mean()
```

Uses **distributional soft cross-entropy** with two-hot encoding.

### 3.4 Value Loss
**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L1571-L1692)  
**Method:** `calculate_value_loss()`

$$\mathcal{L}_V = \mathbb{E}_{s,ve}[\rho^s \cdot \text{CE}(V^{pred}_{ve,s}, \text{TD}^{target}_{ve,s})]$$

```python
# From L1643-1654
val_ce_flat = math.soft_ce(vs_flat_ce, td_flat_ce, self.cfg)
val_ce = val_ce_flat.view(Ve, T_imag, S, BN)

# Average over Ve, T_imag, BN; keep S for rho weighting
val_ce_per_s = val_ce.mean(dim=(0, 1, 3))  # float32[S]

# Apply rho weighting on S dimension
weighted = val_ce_per_s * rho_pows  # float32[S]
loss = weighted.mean()
```

Note: Rho weighting is applied on the **S dimension** (replay buffer starting states), not T_imag.

### 3.5 Termination Loss
**Lines:** [L1312-L1321](../tdmpc2/tdmpc2.py#L1312-L1321)

$$\mathcal{L}_\text{term} = \text{BCE}(\text{term}^{pred}, \text{term}^{target})$$

Binary cross-entropy for episodic termination prediction.

---

## 4. ρ (Rho) Time Weighting

**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L1199), [common/parser.py](../tdmpc2/common/parser.py#L193-L196)

### Computation
```python
rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=dtype))  # float32[T]
```

This creates exponential time decay: $[1, \rho, \rho^2, ..., \rho^{T-1}]$

### final_rho Override
**File:** [common/parser.py](../tdmpc2/common/parser.py#L193-L196)

```python
if cfg.final_rho != -1:
    cfg.rho = _math.pow(cfg.final_rho, 1 / cfg.horizon)
```

When `final_rho` is set, ρ is computed such that $\rho^T = \text{final\_rho}$.

**Default:** `rho=0.5`  
**Default:** `final_rho=0.125` → sets rho so that $\rho^3 = 0.125$

### Application
Rho weighting is applied in:
1. **Consistency loss** - [L1232](../tdmpc2/tdmpc2.py#L1232)
2. **Reward loss** - [L1301](../tdmpc2/tdmpc2.py#L1301)
3. **Policy loss** - [L660](../tdmpc2/tdmpc2.py#L660)
4. **Value loss** - [L1653](../tdmpc2/tdmpc2.py#L1653) (on S dimension)

---

## 5. Reanalysis (Planner Distillation)

**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L385-L439)  
**Method:** `reanalyze()`

### What is Reanalyze?
Reanalyze re-runs the planner on observations from the replay buffer to generate expert targets for policy distillation. This enables the policy to learn from the planner's action distribution.

### reanalyze_slice_mode
**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L2118-L2145)

| Mode | Setting | Behavior |
|------|---------|----------|
| Independent | `reanalyze_slice_mode=False` | Sample B_re independent t=0 observations |
| Slice | `reanalyze_slice_mode=True` | Sample fewer slices, reanalyze ALL timesteps (BMPC style) |

```python
if slice_mode:
    # BMPC slice mode: fewer slices, all T_exp timesteps per slice
    num_slices = max(1, reanalyze_batch_size // T_exp)
    obs_reanalyze = obs[:T_exp, :num_slices].reshape(-1, *obs.shape[2:])
else:
    # Independent mode: B_re independent t=0 observations
    obs_reanalyze = obs[0, :reanalyze_batch_size]
```

### Key Config Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `reanalyze_interval` | `10` | Steps between reanalyze (0 = disabled) |
| `reanalyze_batch_size` | `64` | Samples to reanalyze per interval |
| `reanalyze_use_chosen_action` | `true` | Use chosen action as expert mean (BMPC style) |
| `reanalyze_value_std_coef` | `0.0` | Value std coef for reanalyze (does NOT inherit) |

---

## 6. MPPI/Planning Algorithm

**File:** [tdmpc2/common/planner/planner.py](../tdmpc2/common/planner/planner.py)

### Algorithm Overview
The planner uses CEM/MPPI-style iterative refinement:

1. **Initialize** mean/std from warm-start (if B=1)
2. **Sample** N action sequences from N(mean, std)
3. **Policy seeding**: Also include S trajectories from policy prior
4. **Rollout** through world model (all dynamics heads)
5. **Score** trajectories: `value + λ × latent_disagreement`
6. **Select elites**: Top K by score
7. **Update** mean/std via weighted average of elites
8. **Repeat** for `iterations` times

### Value Scoring
**File:** [scoring.py](../tdmpc2/common/planner/scoring.py#L7-L152)

Per dynamics head h:
$$Q_h = \sum_t \alpha_t \cdot r_h(t) + \alpha_T \cdot V_h(z_T) + \text{std\_coef} \times \sigma_h$$

Where:
- $\alpha_t$ = alive probability at step t (episodic only)
- $\sigma_h = \sqrt{\sum_t (\alpha_t \cdot \sigma^r_{h,t})^2 + (\alpha_T \cdot \sigma^v_h)^2}$

### Dynamics Head Reduction in Planner
```python
# From scoring.py L162-170
if value_std_coef > 0:
    values_unscaled, _ = q_per_h.max(dim=0)  # Optimistic
elif value_std_coef < 0:
    values_unscaled, _ = q_per_h.min(dim=0)  # Pessimistic
else:
    values_unscaled = q_per_h.mean(dim=0)    # Neutral
```

### Elite Selection
**File:** [planner.py](../tdmpc2/common/planner/planner.py#L254-L270)

```python
elite_scores, elite_indices = torch.topk(scores, K, dim=1)
max_elite = elite_scores.max(dim=1, keepdim=True).values
score_delta = elite_scores - max_elite
if bool(self.cfg.mult_by_temp):
    w = torch.exp(temp * score_delta)  # Multiply by temp (softer)
else:
    w = torch.exp(score_delta / temp)  # Divide by temp (sharper)
w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
```

### Key Planner Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_rollouts` | `4` | Parallel imagination rollouts per state |
| `planner_num_dynamics_heads` | `2` | Number of dynamics ensemble heads |
| `num_samples` | `512` | Action sequences sampled per iteration |
| `num_elites` | `64` | Top-K for mean/std update |
| `num_pi_trajs` | `24` | Policy-seeded trajectories |
| `horizon` | `3` | Planning horizon T |
| `iterations` | `6` | MPPI iterations |
| `temperature` | `0.5` | Softmax temperature for elite weighting |
| `planner_value_std_coef_train` | `1` | Std coef during training (optimistic) |
| `planner_value_std_coef_eval` | `-1` | Std coef during eval (pessimistic) |
| `planner_lambda_disagreement` | `0` | Weight for latent disagreement bonus |
| `planner_aggregate_value` | `false` | λ-style compound return over horizons |

---

## 7. Optimizer Settings

**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L84-L175)

### Learning Rates
```python
lr_encoder = self.cfg.lr * self.cfg.enc_lr_scale  # 3e-4 * 0.3 = 9e-5
if ensemble_lr_scaling:
    lr_dynamics = self.cfg.lr * num_dynamics_heads
    lr_reward = self.cfg.lr * num_reward_heads
    lr_value = self.cfg.lr * num_q / 5
    lr_aux_value = self.cfg.lr * num_aux_heads / 5
```

### enc_lr_scale
Encoder learning rate = `lr * enc_lr_scale = 3e-4 * 0.3 = 9e-5`

### ensemble_lr_scaling
**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L103-L117)

When `ensemble_lr_scaling=True` (default), LRs are scaled by ensemble size to compensate for mean-reduced gradients:
- Dynamics: `lr * num_dynamics_heads`
- Reward: `lr * num_reward_heads`  
- Value: `lr * num_q / 5` (normalized to original 5 heads)

### Encoder LR Step Schedule
**File:** [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py#L2090-L2095)

```python
enc_lr_cutoff = int((1 - self.cfg.enc_lr_step_ratio) * self.cfg.steps)
if not self._enc_lr_stepped and self._step >= enc_lr_cutoff:
    new_enc_lr = self._enc_lr_initial * self.cfg.enc_lr_step_scale
```

At `(1 - enc_lr_step_ratio) * steps`, encoder LR is multiplied by `enc_lr_step_scale`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enc_lr_step_ratio` | `0.5` | Fraction after which to step LR (0.5 = halfway) |
| `enc_lr_step_scale` | `1` | Multiplier for encoder LR (0 = freeze) |

### Gradient Clipping
```python
grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
```
**Default:** `grad_clip_norm=20`

### Adam vs AdamW
```python
optim_type = self.cfg.optimizer_type.lower()  # 'adam' or 'adamw'
weight_decay = float(self.cfg.weight_decay)   # 0.01 for AdamW
```

---

## 8. Action Normalization and Clipping

### Action Sampling (Planner)
**File:** [sampling.py](../tdmpc2/common/planner/sampling.py#L6-L28)

```python
def sample_action_sequences(..., clamp_low=-1.0, clamp_high=1.0):
    actions = mean.unsqueeze(1) + eps * std.unsqueeze(1)
    return actions.clamp(clamp_low, clamp_high)
```

Actions are always clamped to [-1, 1].

### Policy Action (BMPC vs Standard)
**File:** [world_model.py](../tdmpc2/common/world_model.py#L486-L515)

**BMPC style** (`bmpc_policy_parameterization=True`, default):
```python
mean = torch.tanh(mean)  # Squash mean only
action = (mean + eps * log_std.exp()).clamp(-1, 1)
log_prob = log_prob_presquash  # No Jacobian correction
```

**Standard squashed Gaussian** (`bmpc_policy_parameterization=False`):
```python
action = mean + eps * log_std.exp()
mean, action, log_prob = math.squash(mean, action, log_prob_presquash, jacobian_scale)
```

---

## 9. Target Network Update (τ)

**File:** [world_model.py](../tdmpc2/common/world_model.py#L323-L331)  
**Method:** `soft_update_target_V()`

```python
self._target_Vs_params.lerp_(self._detach_Vs_params, self.cfg.tau)
```

Polyak averaging: $\theta_\text{target} \leftarrow (1-\tau)\theta_\text{target} + \tau\theta_\text{online}$

**Default:** `tau=0.003`

### Policy Target Update
**File:** [world_model.py](../tdmpc2/common/world_model.py#L398-L410)  
**Method:** `soft_update_target_pi()`

Same Polyak averaging for policy targets (when trust region or EMA TD target is enabled):
```python
def soft_update_target_pi(self):
    if self._target_pi is None:
        return
    self._soft_update_module(self._target_pi, self._pi, self.cfg.policy_ema_tau)
```

---

## 10. World Model Architecture

**File:** [world_model.py](../tdmpc2/common/world_model.py#L15-L299)

### Components
1. **Encoder** (`_encoder`): State/pixel → latent
2. **Dynamics heads** (`_dynamics_heads`): ModuleList of H MLP heads with prior
3. **Reward heads** (`_Rs`): Ensemble of R reward MLPs with prior
4. **Termination head** (`_termination`): Single MLP (episodic only)
5. **Policy** (`_pi`): MLP → (mean, log_std)
6. **Value ensemble** (`_Vs`): Ensemble of Ve value MLPs with prior
7. **Auxiliary V heads** (`_aux_separate_Vs` or `_aux_joint_Vs`): Multi-gamma heads

### MLPWithPrior
**File:** [layers.py](../tdmpc2/common/layers.py#L9-L143)

Each ensemble head uses `MLPWithPrior`:
- **Main MLP**: Trainable, initialized with Xavier
- **Prior MLP**: Frozen (via detach), provides diversity
- Output: `main(x) + detach(prior(x)) * prior_scale`

For distributional outputs, prior outputs scalar → two-hot encoded.

---

## Summary Table: Key Loss Formulas

| Loss | Formula | Weighted By |
|------|---------|-------------|
| Consistency | $\|\|z^{pred} - z^{target}\|\|_2^2$ | `consistency_coef=20`, ρ |
| Encoder Consistency | $\|\|z^{pred}.detach() - z^{target}\|\|_2^2$ | `encoder_consistency_coef=1`, ρ |
| Reward | $\text{CE}(r^{pred}, \text{two\_hot}(r^{true}))$ | `reward_coef=0.1`, ρ |
| Value | $\text{CE}(V^{pred}, \text{two\_hot}(\text{TD}^{target}))$ | `value_coef=0.1`, ρ |
| Termination | $\text{BCE}(\text{term}^{pred}, \text{term}^{true})$ | `termination_coef=1` |
| Policy (SVG) | $-Q_\text{scaled} - \alpha H[\pi]$ | `policy_coef=1`, ρ |
| Policy (KL) | $\text{KL}(\pi_\text{expert} \|\| \pi) - \alpha H[\pi]$ | `policy_coef=1`, ρ |
| Trust Region | $\text{KL}(\pi_\text{EMA} \|\| \pi)$ | `policy_trust_region_coef` |
| Aux Value | $\text{CE}(V^{aux,\gamma}, \text{two\_hot}(\text{TD}^{aux}))$ | `multi_gamma_loss_weight` |

---

## References

- Main agent: [tdmpc2/tdmpc2.py](../tdmpc2/tdmpc2.py)
- World model: [tdmpc2/common/world_model.py](../tdmpc2/common/world_model.py)
- Planner: [tdmpc2/common/planner/planner.py](../tdmpc2/common/planner/planner.py)
- Scoring: [tdmpc2/common/planner/scoring.py](../tdmpc2/common/planner/scoring.py)
- Math utilities: [tdmpc2/common/math.py](../tdmpc2/common/math.py)
- Config defaults: [tdmpc2/config.yaml](../tdmpc2/config.yaml)
- Online trainer: [tdmpc2/trainer/online_trainer.py](../tdmpc2/trainer/online_trainer.py)
