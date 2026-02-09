# Plan: Dynamics Ensemble with Split-Data Support

## TL;DR

Add a `split_data` flag to `Ensemble.forward()` so the first dim of input can be split across members. Wrap dynamics heads in `Ensemble` (replacing `nn.ModuleList`). Update `next()` to support `head_mode='all'` (broadcast) and `head_mode='split'` (per-member). Refactor `rollout_latents()` to call `next()` with `'split'` instead of looping over individual heads.

## Context

### Current State
- **Reward/Value heads**: Already wrapped in `Ensemble` (vmap + functional_call). Efficient, compile-friendly.
- **Dynamics heads**: `nn.ModuleList` of `DynamicsHeadWithPrior`. Forward passes use sequential Python for-loops.

### Two Data Patterns
1. **Broadcast (same data → all heads)**: Used in `calc_pi_losses()` where the same `(z, a)` goes to all H dynamics heads for 1-step Q-estimates.
2. **Split (per-head data)**: Used in `rollout_latents()` where each dynamics head maintains its own latent trajectory — head h at time t uses the latent it produced at time t-1, not the latents from other heads.

### Call Sites

**`next()` (single-step dynamics):**
| Call site | head_mode | Data pattern |
|---|---|---|
| `tdmpc2.py` L618 (`calc_pi_losses`) | `'all'` | Same `z, a` to all heads → `[H, T*B*N, L]` |

**`rollout_latents()` (multi-step dynamics):**
| Call site | Context |
|---|---|
| `tdmpc2.py` L1260 (`world_model_losses`) | Replay actions, `head_mode='all'` |
| `tdmpc2.py` L1539 (`imagined_rollout`) | Policy actions, `head_mode='all'` or `'random'` |
| `planner.py` L157 | Policy-seeded rollout |
| `planner.py` L217 | CEM iteration rollout |
| `info_types.py` L14 | Post-noise diagnostics |

No other dynamics head call sites exist. `_td_target` and `_td_target_aux` receive already-computed `next_z` — they don't call dynamics heads.

---

## Steps

### 1. Extend `Ensemble.forward()` — `layers.py` L193-213

Add `split_data: bool = False` parameter.

- **`split_data=False`** (default, current behavior): vmap over params only, args broadcast to all members.
  - `in_dims` for vmap: `(0,)` for params, `None` for all args.
  - Input: `[*batch, in_dim]` → Output: `[H, *batch, out_dim]`.
- **`split_data=True`**: vmap over both params (dim 0) and first positional arg (dim 0).
  - `in_dims` for vmap: `(0,)` for params, `(0,)` for first arg, `None` for rest.
  - Input: `[H, *batch, in_dim]` → Output: `[H, *batch, out_dim]`.
  - Assert first arg's dim 0 matches `len(self)`.

The only change is `in_dims` for the first positional arg: `None` (broadcast) vs `0` (split).

### 2. Wrap dynamics heads in `Ensemble` — `world_model.py` L59-72

- Build list of `DynamicsHeadWithPrior` (same as now).
- Call `head.apply(init.weight_init)` on each **before** wrapping (because `self.apply()` at L171 won't reach Ensemble internals — same pattern as reward/value heads).
- `self._dynamics_heads = layers.Ensemble(dyn_heads)` — **keep same attribute name** so optimizer param groups (`tdmpc2.py` L121) and gradient logging (`tdmpc2.py` L482) work unchanged.
- Update `self._dynamics` alias: point to `self._dynamics_heads` for `__repr__` (`world_model.py` L285).
- Remove `nn.ModuleList`.

Note: `len(self._dynamics_heads)` still works — `Ensemble` has `__len__()`.

### 3. Update `next()` method — `world_model.py` L409-446

Two head modes only:

- **`head_mode='all'`**: `self._dynamics_heads(za)` with `split_data=False` (broadcast).
  - Input: `za [B, L+A]` → Output: `[H, B, L]`.
  - Same data to all heads.
- **`head_mode='split'`**: `self._dynamics_heads(za, split_data=True)`.
  - Input: `za [H, B, L+A]` → Output: `[H, B, L]`.
  - Each head gets its own slice.

Remove `head_mode=None` and `head_mode='single'` — not needed.

Apply autocast and `.float()` in both paths.

### 4. Refactor `rollout_latents()` inner loop — `world_model.py` L770-810

- Remove `head_mode` parameter entirely — always uses all H heads.
- Remove `head_indices` mechanism.
- Remove `for h in range(H_sel)` loop.
- At each timestep `t`:
  1. `z_all = latents_steps[t]` → `[H, B, N, L]`, reshape to `[H, B*N, L]`.
  2. `a_all = a_t.view(1, B*N, A).expand(H, -1, -1)` → `[H, B*N, A]`.
  3. Call `self.next(z_all, a_all, task, head_mode='split')` → `[H, B*N, L]`.
  4. Reshape to `[H, B, N, L]`, append to `latents_steps`.
- At t=0 the data is already stacked as `[H, B*N, L]` (copies of z0), so use `'split'` uniformly for all timesteps. No branching.
- H is always `len(self._dynamics_heads)` (all heads).

### 5. Update `rollout_latents()` callers

Remove `head_mode` argument from all call sites:

- `tdmpc2.py` L1260 — `world_model_losses()`: remove `head_mode='all'`
- `tdmpc2.py` L1539 — `imagined_rollout()`: remove `head_mode='all'`/`'random'` config dispatch
- `planner.py` L157 — policy-seed rollout: remove `head_mode` param
- `planner.py` L217 — CEM iteration: remove `head_mode` param  
- `info_types.py` L14 — post-noise effects: remove `head_mode` param
- Remove any config params for `dynamics_head_mode` if they exist

### 6. `calc_pi_losses()` — `tdmpc2.py` L618

Currently calls `self.model.next(z_flat, action_flat, task_flat, head_mode='all')`.
No change needed — `head_mode='all'` (broadcast) is exactly what we want.

### 7. Weight initialization handling

Current flow: `self.apply(init.weight_init)` at L171 traverses all submodules.

After wrapping in `Ensemble`, dynamics params live in `TensorDictParams` which `nn.Module.apply()` does NOT traverse. Solution: apply `weight_init` per-head **before** wrapping (step 2), same pattern as reward/value heads already use.

---

## Verification

```bash
# Debug (no compile, multiple heads)
CUDA_VISIBLE_DEVICES=1 python tdmpc2/train.py task=quadruped-walk steps=5000 compile=false planner_num_dynamics_heads=3

# Compile test
CUDA_VISIBLE_DEVICES=1 python tdmpc2/train.py task=quadruped-walk steps=5000 compile=true planner_num_dynamics_heads=3

# Single head default (backward compat)
CUDA_VISIBLE_DEVICES=1 python tdmpc2/train.py task=quadruped-walk steps=5000 compile=true planner_num_dynamics_heads=1
```

Check: `rollout_latents` output shape = `[H, B, N, T+1, L]`.

## Decisions

- **Single forward method** with `split_data` flag — simpler than two methods.
- **No `forward_subset` / single-head selection** — always run all heads. Simplifies code.
- **`rollout_latents()` loses `head_mode` param** — always uses all H heads.
- **Keep attribute name `_dynamics_heads`** — no ripple on optimizer/gradient code.
- **Ignore task embedding handling** for simplicity (not used in current experiments).
