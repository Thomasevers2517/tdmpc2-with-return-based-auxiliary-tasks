# Dual Policy Architecture: Optimistic + Pessimistic

## Overview

Add a second "optimistic" policy network (`_pi_optimistic`) alongside the existing pessimistic policy (`_pi`). The optimistic policy seeds the planner search during training, while the pessimistic policy defines value targets.

**Key Insight**: For value estimation we want pessimism (conservative). For seeding exploration we want optimism (seek out uncertain/high-upside regions).

---

## Config

```yaml
dual_policy_enabled: false        # If true, create pessimistic + optimistic policies
optimistic_entropy_mult: 1.0      # Multiply entropy coeff by this for optimistic policy
```

---

## Changes by File

### 1. `tdmpc2/common/world_model.py`

**Add `_pi_optimistic` network:**
```python
# In __init__, after _pi creation:
if cfg.dual_policy_enabled:
    self._pi_optimistic = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
```

**Modify `pi()` method to accept `optimistic` param:**
```python
def pi(self, z, task, optimistic=False):
    """Sample action from policy.
    
    Args:
        z: Latent state
        task: Task embedding
        optimistic: If True, use optimistic policy; else pessimistic
    """
    if optimistic and self.cfg.dual_policy_enabled:
        module = self._pi_optimistic
    else:
        module = self._pi
    # ... rest of existing logic using module
```

**Modify `rollout_latents()` to accept `use_optimistic_policy` param:**
```python
def rollout_latents(
    self,
    z0,
    use_policy=True,
    use_optimistic_policy=False,  # NEW
    horizon=...,
    ...
):
    # When sampling actions from policy:
    if use_policy:
        a_flat, _ = self.pi(z_for_pi, task, optimistic=use_optimistic_policy)
```

### 2. `tdmpc2/tdmpc2.py`

**Extend `pi_optim` with both policies' params:**
```python
# In __init__:
if self.cfg.dual_policy_enabled:
    import itertools
    pi_params = itertools.chain(
        self.model._pi.parameters(),
        self.model._pi_optimistic.parameters()
    )
else:
    pi_params = self.model._pi.parameters()
self.pi_optim = torch.optim.Adam(pi_params, lr=lr_pi, eps=1e-5, capturable=True)
```

**Modify `calc_pi_losses()` to accept `optimistic` param:**
```python
def calc_pi_losses(self, z, task, optimistic=False):
    """Compute policy loss.
    
    Args:
        optimistic: If True, use optimistic policy with max reduction;
                    else use pessimistic policy with min reduction.
    """
    # Select reduction mode
    policy_reduce = 'max' if optimistic else self.cfg.policy_head_reduce  # 'min' by default
    
    # Select entropy coefficient
    if optimistic:
        entropy_coeff = self.dynamic_entropy_coeff * self.cfg.optimistic_entropy_mult
    else:
        entropy_coeff = self.dynamic_entropy_coeff
    
    # Sample action from appropriate policy
    action, info = self.model.pi(z, task, optimistic=optimistic)
    
    # ... rest of existing logic, using policy_reduce for head aggregation
    # ... and entropy_coeff for entropy bonus
```

**Modify `update_pi()` to call twice and sum losses:**
```python
def update_pi(self, z, task):
    # Pessimistic policy loss
    pi_loss, pi_info = self.calc_pi_losses(z, task, optimistic=False)
    
    if self.cfg.dual_policy_enabled:
        # Optimistic policy loss
        opti_pi_loss, opti_pi_info = self.calc_pi_losses(z, task, optimistic=True)
        
        # Prefix optimistic info keys
        for k, v in opti_pi_info.items():
            pi_info[f'opti_{k}'] = v
        
        # Sum losses (equal weight)
        total_pi_loss = pi_loss + opti_pi_loss
    else:
        total_pi_loss = pi_loss
    
    return total_pi_loss, pi_info
```

**Pass `use_optimistic_policy=True` in training planner calls:**
```python
# In plan() or act(), when calling rollout_latents for policy seeding:
if use_policy and self.cfg.dual_policy_enabled and not eval_mode:
    use_optimistic_policy = True
else:
    use_optimistic_policy = False

latents_p, actions_p = self.world_model.rollout_latents(
    z0,
    use_policy=True,
    use_optimistic_policy=use_optimistic_policy,
    ...
)
```

### 3. `tdmpc2/config.yaml`

Add under `# actor` section:
```yaml
# Dual policy (optimistic + pessimistic)
dual_policy_enabled: false        # If true, create pessimistic + optimistic policies
optimistic_entropy_mult: 1.0      # Multiply entropy coeff by this for optimistic policy
```

---

## Behavior Summary

| Context | Policy Used | Head Reduction |
|---------|-------------|----------------|
| Training: planner seeding | Optimistic (`_pi_optimistic`) | N/A (action sampling) |
| Training: policy loss (pessimistic) | `_pi` | `min` |
| Training: policy loss (optimistic) | `_pi_optimistic` | `max` |
| Eval: planner seeding | Pessimistic (`_pi`) | N/A |
| Eval: no MPC | Pessimistic (`_pi`) | N/A |

---

## Design Decisions

| Concern | Decision |
|---------|----------|
| Hinge loss | Apply to both policies (same `hinge_coef`) |
| Gradient clipping | Single clip over combined params |
| Eval seeding | Use pessimistic policy (not optimistic) |
| Actor loss states | Both policies trained on same latent states `z` |
| Env actions (no MPC) | Use pessimistic policy |
| Initialization | Independent random init for each policy |
| Loss weighting | Equal weight (1:1) for both policy losses |
| Entropy | Shared schedule, multiplied by `optimistic_entropy_mult` for optimistic |
| Encoder gradients | Detached during policy optimization (no concern) |

---

## Logging

Info dict from optimistic policy gets keys prefixed with `opti_`:
- `opti_pi_loss`
- `opti_pi_entropy`
- `opti_pi_log_std`
- etc.

---

## Future Extensions

1. **Separate entropy schedules**: Could add `optimistic_start_entropy_coeff` and `optimistic_end_entropy_coeff` for fully independent entropy decay.
2. **Loss weighting**: Could add `optimistic_policy_coef` to weight the optimistic loss differently.
3. **Mixed seeding**: Could seed with both policies (50/50 split) instead of only optimistic.
