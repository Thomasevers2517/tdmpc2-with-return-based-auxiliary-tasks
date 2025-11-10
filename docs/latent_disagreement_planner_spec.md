# Train Planner with Latent Disagreement (Design Spec)

Owner: tdmpc2
Branch: latent-disagreement
Status: Draft (implementation next)

## Summary

Add a training-only planner (`plan_train`) that uses an ensemble of dynamics heads and optimizes a raw score that combines optimistic value and trajectory-wide latent disagreement. Evaluation planning (`plan_eval`) remains unchanged and uses dynamics head 0 only.

- Ensemble: shared encoder; H dynamics heads (H = cfg.dynamics_heads).
- Training planner: particle-style search over action sequences; per-parent local children; noise scale updated from elite children.
- Score for a candidate sequence a[0..T-1]:
  - Value: for each head h, rollout latents/rewards with that head; compute discounted return + Q bootstrap using the last action; take max over heads.
  - Disagreement: for each time step, compute variance across heads of latent vectors; average over latent dims; mean over time steps.
  - raw_score = max_h Value_h + λ · Disagreement.
- Termination (planning-only): optimistic: continue unless all heads predict termination.
- Action selection in act():
  - Training: sample a parent according to softmax over final parent raw_scores (temperature cfg.temperature), then take the first action from that sequence (optionally add very small exploration noise via existing cfg.train_act_std_coeff).
  - Evaluation: current MPPI unchanged; dynamics head 0; existing behavior.

## Integration with existing code

This section maps the design to the current codebase to maximize reuse and keep evaluation behavior intact.

### Touch points

- `tdmpc2/tdmpc2.py` (class `TDMPC2`)
  - Keep current `_plan` as evaluation planner (alias: `plan_eval = _plan`).
  - Add `plan_train` implementing particle search and raw-score.
  - Update `act()` dispatch:
    - `eval_mode=True` → `plan_eval` (unchanged).
    - else if `cfg.enable_train_planner` → `plan_train`.
    - else → `plan_eval` (current training MPPI).
  - Add non-persistent buffers for train planner warm-start:
    - `_train_parents_actions: (T,P,A)` and `_train_parents_std: (T,P,A)`; extend `reset_planner_state()` to zero them.
  - Validate required config with `_validate_train_planner_cfg()` when `enable_train_planner` is true (raise on any missing key; no defaults).
  - Add `_score_sequences_with_ensemble(...)` used by `plan_train` to compute value/disagreement/raw_score.

- `tdmpc2/common/world_model.py` (class `WorldModel`)
  - Add optional ensemble for dynamics:
  - When `cfg.dynamics_heads >= 2`, create `self._dynamics_heads = nn.ModuleList([...])` with identical architecture to current `_dynamics` but different init.
    - Keep `self._dynamics` for single-head (and define head 0 = `_dynamics` for eval compatibility).
    - Add `next_head(z, a, task, head:int)`; keep `next(z,a,task)` as-is for eval (uses `_dynamics` or head 0).
  - No change to reward/termination/Q modules.
  - Optimizer param groups in `TDMPC2.__init__`: include parameters of all dynamics heads when present; otherwise include `_dynamics`.

- `tdmpc2/common/logger.py` (class `Logger`)
  - Add `log_planner(planning_info: dict, step: int)` to process detailed planner payloads and log under `planner/` namespace. Keep simple at first.

- `tdmpc2/trainer/online_trainer.py`
  - After `action, info = agent.act(...)` in data collection:
    - If planner basics exist in `info`, log them every step via `logger.log({...}, 'train')` with keys prefixed `planner/`.
    - If heavy `planning_info` exists and detailed logging is active, call `logger.log_planner(planning_info, step)`.

### Reuse of existing utilities

- Keep `math.two_hot_inv` for reward/Q inversion and `math.gumbel_softmax_sample` for final parent sampling (after computing softmax of raw scores with `cfg.temperature`).
- Retain action clamping to [-1,1].
- Use `maybe_range` for NVTX-scoped profiling around planner phases.

### Compilation hooks

- Optionally mirror `plan` compilation: cache and (later) compile `plan_train` when `cfg.compile` is enabled. Start uncompiled for initial bring-up.

### Multitask

- For this phase, raise `NotImplementedError` inside `plan_train` if `cfg.multitask` is True; leave current multitask machinery untouched elsewhere.

### Evaluation compatibility

- Ensure `act(eval_mode=True)` uses `plan_eval` and dynamics head 0 only, preserving current evaluation semantics and cost.

## Implementation plan (step-by-step)

1) WorldModel ensemble scaffolding
  - Add `self._dynamics_heads` when `cfg.dynamics_heads >= 2`; add `next_head()`; keep `next()` behavior for single-head/eval.
   - Modify `TDMPC2.__init__` optimizer param groups to include all dynamics heads when present.

2) Config validation
   - Implement `_validate_train_planner_cfg()` in `TDMPC2`; call it when `enable_train_planner` is true. Enforce presence/ranges of all required keys and `cfg.temperature`.

3) Train planner state & `plan_train`
   - Allocate `_train_parents_actions/std` on first use (shapes `(T,P,A)`). Warm start by shifting one step between env steps.
   - Iteration 0: per parent, seed `C` children using policy ratio and Gaussian `particle_init_gaussian_std`.
   - Iterations: sample children around parent using per-(t,a) `parent_std`; evaluate; select best child as parent; update `parent_std` from top-`elite_k` children (where `elite_k = ceil(ratio*C)`); clamp by `cfg.min_std`.
   - Final selection: compute softmax over parent raw scores with `cfg.temperature`, sample parent index (gumbel), and return its first action. Also return `plan_info_basic` and, optionally, `planning_info` when detailed logging is active.

4) Ensemble scoring `_score_sequences_with_ensemble`
   - Input: `z0`, `actions` as `(T,S,A)`; output per-sequence `value_max`, `disagreement`, `raw_score` and (optionally) diagnostic extras.
   - Stream heads: for each head, rollout latents via `next_head`, compute rewards and Q bootstrap; keep running (per-step, per-sequence) moments to derive latent variance; build optimistic mask across heads for discounting.

5) Logger integration
   - Add `Logger.log_planner(...)` to accept heavy payload; log curated subsets to WandB. Use `logger.log(...,'train')` for step-wise basics.

6) Keep eval planner unchanged
   - Alias `plan_eval = _plan`; no changes to `_estimate_value` or existing MPPI loop used for evaluation.


## Out of scope (for this phase)

- Decoder for visualizing imagined observations (not implemented yet; planned as future TODO).
- Multitask action masking and per-task discounts (assume single-task for train planner initial implementation).

## Config (REQUIRED keys; no defaults)

The following keys MUST exist when `enable_train_planner` is true; otherwise raise an explicit error on agent init or on planner call.

- enable_train_planner: bool
- dynamics_heads: int >= 2
- latent_disagreement_lambda: float  (λ)
- latent_disagreement_metric: string; must be 'variance'
- particle_parents: int (P)
- particle_children: int (C)
- particle_iterations: int (I >= 1)
- particle_use_policy_ratio: float in [0,1] (fraction of initial children seeded from policy)
- particle_child_elite_ratio: float in (0,1]  (fraction of each parent’s children used to compute next-iteration std)
- particle_init_gaussian_std: float > 0 (std for Gaussian children initialization; actions clipped to [-1,1])
- min_std: float > 0  (reuse existing cfg.min_std to clamp per-(t,a) std)

Validation rules:
- If `enable_train_planner` and `dynamics_heads < 2` → error.
- `latent_disagreement_metric` must be exactly 'variance' (other metrics not supported yet).
- `particle_parents * particle_children` defines the number of evaluated sequences per iteration (no alternative `num_samples` allowed in train planner).
- `particle_child_elite_ratio` ∈ (0,1]; compute elite_k = ceil(ratio * C); enforce elite_k >= 1.

Notes:
- `cfg.temperature` from existing planner is reused for the final softmax sampling across parents. If absent, raise an error (no implicit defaults).
- Use existing `cfg.discount_…` settings and `cfg.min_std` as defined elsewhere in the codebase; do not create silent fallbacks.

## Planner contracts

### plan_eval (unchanged)
- Uses current MPPI/CEM behavior and dynamics head 0 only.

### plan_train (new)
Inputs:
- obs: (obs_dim,) or batched; encoded to latent z0 via shared encoder.
- task: optional; for now assume single-task in train planner; raise if multitask is on.
- eval_mode: must be False to use `plan_train` (otherwise `plan_eval`).

Outputs:
- score_parents: Tensor (P,) softmax weights over final parents (`exp(raw/temperature)` normalized); raw scores are returned inside `info['parents/raw_scores']`.
- actions_parents: Tensor (T, P, A) final parent action sequences.
- info: dict with at least the following keys for basic logging:
  - chosen_parent_index: int
  - chosen_parent_weight: float (softmax weight for sampled parent)
  - chosen_value_max_head: int (max value head for chosen parent)
  - chosen_value_max: float
  - chosen_disagreement: float
  - parents_value_max_mean: float
  - parents_value_max_std: float
  - parents_disagreement_mean: float
  - parents_disagreement_std: float
  - parents_softmax: Tensor (P,) temperature-weighted probabilities for logging
  - ensemble_size: int (H)
  - particle_parents, particle_children, particle_iterations, particle_use_policy_ratio, particle_child_elite_ratio
  - temperature_used: float (cfg.temperature)

If `log_detailed` is True at act-time, `info` additionally contains the complete planning payload, suitable for `logger.log_planner`:
- parents (list length P): for each parent p:
  - best_child_actions: (T, A)
  - best_child_raw_score: float
  - best_child_value_max: float
  - best_child_disagreement: float
  - best_child_value_max_head: int
  - children_table (C rows): for each child j
    - raw_score, value_max, disagreement, value_max_head (scalars)
- per_step_disagreement_for_chosen_parent: (T,) variances
- optionally per-step per-head latent magnitude summaries (mean/std), not full tensors

The planner retains warm state on device:
- parents_actions: (T, P, A)
- parents_std: (T, P, A)
for seeding the next environment step (shifted by one step to warm start).

## Scoring details

Let H be ensemble size, T horizon, A action dimensions.

### Value per head
For a candidate sequence a[0..T-1]:
- z_0 = encode(obs)
- For each head h ∈ {1..H}:
  - For t=0..T-1:
    - r_t^h = two_hot_inv(reward(z_t^h, a_t), cfg)
    - z_{t+1}^h = next_h(z_t^h, a_t)
    - discount_t accumulates using the same discount heuristic as the agent, multiplied by an optimistic alive mask M_t (see below).
  - Bootstrap: use last action a_{T-1} for Q bootstrap: Q(z_T^h, a_{T-1}). No policy bootstrap.
  - Value_h = Σ_t (discount_t · r_t^h) + discount_T · Q(z_T^h, a_{T-1}).

### Optimistic termination mask
- For each step t, compute per-head termination probability p_term_t^h.
- Alive^h_t = 1 - 𝟙[p_term_t^h > 0.5].
- Optimistic mask across heads: M_t = 1 if any head is alive (i.e., max_h Alive^h_t = 1), else 0.
- Apply M_t multiplicatively to discount accumulation (i.e., continue discounting as long as M_t==1; when M_t goes 0, subsequent contributions are nil).

### Latent disagreement (variance metric)
- For each t = 1..T, with latents {z_t^h ∈ ℝ^L} across heads:
  - μ_t = mean_h z_t^h
  - var_t = mean_d ( (1/H) Σ_h (z_t^h[d] - μ_t[d])^2 )   # average over latent dimensions
- Disagreement(sequence) = mean_t var_t (uniform mean over t; no discount).

### Raw score
- raw_score(sequence) = max_h Value_h(sequence) + λ · Disagreement(sequence)
- λ = cfg.latent_disagreement_lambda (required).

## Particle planner algorithm (per environment step)

Parameters: P=particle_parents, C=particle_children, I=particle_iterations, elite_ratio ∈ (0,1], elite_k = ceil(elite_ratio·C).

Warm start state:
- If available, parents_actions (T,P,A) and parents_std (T,P,A) from previous step are shifted by one step and used to initialize parents.

Iteration 0: initialize children per parent
- For each parent p in 1..P:
  - Create C children actions.
  - For a fraction `particle_use_policy_ratio` of C:
    - Generate actions by rolling policy prior (use `pi(..., search_noise=True)` per step). You may scale policy sampling std later (configurable), but initially use policy’s own std.
  - For the remaining children:
    - Sample Gaussian with std = particle_init_gaussian_std per (t,a); clip to [-1,1].
  - Evaluate all C children via the scoring procedure above.
  - Select best child as the new parent p’s sequence; set parent_std[p] = std over actions of the top `elite_k` children within this parent (per (t,a)); clamp by cfg.min_std.

Iterations 1..I-1:
- For each parent p:
  - Spawn C children by sampling: child_actions = parent_actions + Normal(0, parent_std) per (t,a); clip to [-1,1].
  - Evaluate children; pick best child as new parent.
  - Update parent_std from the top `elite_k` children (per (t,a)); clamp by cfg.min_std.

Final selection (feeding act()):
- Compute raw_scores for the final P parents.
- Sample a parent index using softmax(raw_scores / temperature), where `temperature = cfg.temperature` (required), to promote exploration across different promising modes.
- The action sent to the environment is the first action of the sampled parent’s sequence. Optionally add very small exploration noise following existing `cfg.train_act_std_coeff`.

## Interfaces

### Agent
- Add `plan_train` method and route inside `act()` when `eval_mode=False and cfg.enable_train_planner=True`.
- Keep existing `_plan` for evaluation planner; use train planner only for training path.

### Logger
- Add `logger.log_planner(planning_info: dict, step: int)`
  - planning_info is the `info` payload returned by `plan_train` when `log_detailed` is True.
  - Inside `log_planner`, start simple: log a curated subset to WandB under a `planner/` namespace (scalars, small vectors). The raw payload is available for richer processing later.

## Logging

Basic (every train step while using train planner):
- Scalars:
  - planner/type = 'particle'
  - ensemble/size = H
  - particle/parents = P, particle/children = C, particle/iterations = I
  - planner/lambda = λ
  - planner/temperature = cfg.temperature
  - planner/chosen_parent_index
  - planner/chosen_parent_weight
  - planner/chosen_value_max_head
  - planner/chosen_value_max
  - planner/chosen_disagreement
  - planner/parents_value_max_mean, planner/parents_value_max_std
  - planner/parents_disagreement_mean, planner/parents_disagreement_std

Detailed (only when `log_detailed` is True):
- For top K parents (K configurable later; start with all available in info and downselect in `log_planner`):
  - parent[p]/best_child_actions (T,A)
  - parent[p]/best_child_raw_score, value_max, disagreement, value_max_head
  - parent[p]/children_table: per child j, raw_score, value_max, disagreement, value_max_head
- Chosen parent per-step disagreement (T,)
- Optional: pairwise latent distances per step can be added later.

Implementation note: array-heavy artifact logging can be gated and/or written via `wandb.Table` or saved tensor summaries; we start light and can extend.

## Efficiency and memory

- Stream across heads to avoid materializing (H,T,P*C,L) tensors when computing disagreement:
  - Keep running sums per step for mean and second moment to compute variance over heads.
  - For detailed logging, we may additionally keep summaries (not full latents) unless explicitly requested.
- Parents and std kept on device between steps for warm start.

## Error handling

- If `enable_train_planner` is True and any required key is missing → raise `RuntimeError` with a precise message (no silent defaults).
- If multitask is enabled in config when using train planner (for this phase) → raise `NotImplementedError`.
- If `dynamics_heads < 2` with train planner → error.
- If `latent_disagreement_metric != 'variance'` → error.
- If `cfg.temperature` is missing → error.

## Future TODOs

- Add optional decoder (stop-grad) for visualizing imagined observations; integrate into detailed `log_planner`.
- Add alternative disagreement metrics (pairwise L2, reward disagreement, termination disagreement).
- Consider percentile aggregation instead of max over heads for robustness (ablation).
- Optional λ schedule; disagreement running scale if needed.
- Multitask support.

## Acceptance criteria

- Training uses `plan_train` only when explicitly enabled via config.
- Evaluation behavior is unchanged.
- Planner composes raw_score exactly as specified and uses optimistic termination.
- Logging emits the basic scalars every train step; detailed payload is routed to `logger.log_planner` when enabled.
- All required config keys are validated early; no fallback/default values are silently assumed.
