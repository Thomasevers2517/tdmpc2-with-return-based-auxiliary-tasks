Plan: Planner Refactor With Disagreement

Refactor planning into common/planner/ and add multi-head disagreement scoring. Keep config flat; delegate all planning from TDMPC2.act to a compiled call site. World model exposes a single rollout_latents that returns both latents and actions and supports policy-driven rollouts.

World Model Rollout API
- Method: world_model.rollout_latents(z0, actions=None, use_policy=False, horizon=None, num_rollouts=None, head_mode='single', head_idx=None, task=None)
- Preconditions:
  - Assert not multitask for now: assert not self.cfg.multitask
  - If use_policy: require horizon and num_rollouts, and actions is None.
  - If not use_policy: require actions with shape float32[N, T, A]; infer horizon=T and num_rollouts=N.
- Returns:
  - latents:
    - head_mode == 'all': float32[H, N, T+1, L]
    - head_mode == 'single': float32[1, N, T+1, L] (unsqueezed head dim at 0)
    - head_mode == 'random': NotImplementedError (stub)
  - actions_out: float32[N, T, A] (policy-generated or provided actions)
- Notes:
  - Head 0 used for value; all heads for disagreement during training.
  - No scaling/logging; vectorized and compile-friendly.

Planner Info Dataclasses (common/planner/info_types.py)
- PlannerBasicInfo:
  - value_chosen: float
  - disagreement_chosen: float | None
  - value_elite_mean: float
  - value_elite_std: float
  - value_elite_max: float
  - disagreement_elite_mean: float | None
  - disagreement_elite_std: float | None
  - disagreement_elite_max: float | None
  - num_elites: int
  - elite_indices: int64[E]
- PlannerAdvancedInfo(PlannerBasicInfo):
  - actions_all: float32[E, T, A]
  - latents_all: float32[H, E, T+1, L]
  - values_all_unscaled: float32[E]
  - values_all_scaled: float32[E]  # currently equals unscaled (scaling disabled)
  - disagreements_all: float32[E] | None
  - raw_scores: float32[E]

Sampling Utility (common/planner/sampling.py)
- sample_action_sequences(mean: float32[T, A], std: float32[T, A], num_samples: int) -> float32[N, T, A]
  - Clamp to [-1, 1]; std clipped to [cfg.std_min, cfg.std_max].

Scoring Utilities (common/planner/scoring.py)
- compute_values_head0(latents_head0: float32[N, T+1, L], actions: float32[N, T, A], world_model, task=None) -> (values_unscaled: float32[N], values_scaled: float32[N])
  - For now: values_scaled = values_unscaled * 1.0  # TODO: enable scaling later without updating scale during planner.
- compute_disagreement(final_latents_all: float32[H, N, L]) -> float32[N]
  - var = final_latents_all.var(dim=0, unbiased=False) -> float32[N, L]; disagreement = var.mean(dim=1) -> float32[N].
- combine_scores(values_scaled: float32[N], disagreements: float32[N] | None, lambda_coeff: float) -> float32[N]
  - If disagreements is None or lambda_coeff == 0: return values_scaled; else return values_scaled + lambda_coeff * disagreements.

Planner Class (common/planner/planner.py)
- class Planner(torch.nn.Module)
- Buffers/state:
  - prev_mean: float32[T, A]  # warm-start; zeros initially
- Methods:
  - reset_warm_start() -> None  # zero prev_mean
  - plan(z0: float32[L] or float32[1, L], task=None, eval_mode=False, step: int | None=None) -> (chosen_action: float32[A], info: PlannerBasicInfo | PlannerAdvancedInfo)
    - Shapes: T=cfg.horizon; A=cfg.action_dim
    - Config aliases: iterations=cfg.iterations; N=cfg.num_samples; K=cfg.num_elites; S=cfg.planner_num_seed_policy_trajectories; temp=(cfg.planner_temperature_eval if eval_mode else cfg.planner_temperature_train); lambda_d=(0.0 if eval_mode else cfg.planner_lambda_disagreement)
    - Initialization:
      - mean = prev_mean.clone()  # float32[T, A]
      - std = torch.full((T, A), cfg.std_max, device=..., dtype=...)
      - Policy seeds (if S>0, iteration 0 only): world_model.rollout_latents(z0, use_policy=True, horizon=T, num_rollouts=S, head_mode=('single' if eval_mode else 'all')) -> (latents_p, actions_p); frozen across iterations.
    - Iteration loop (i in 1..iterations):
      1) actions_s = sample_action_sequences(mean, std, N)  # float32[N,T,A]
      2) latents_s, actions_s = world_model.rollout_latents(z0, actions=actions_s, use_policy=False, head_mode=('single' if eval_mode else 'all'))
      3) Concatenate candidates with policy seeds if present:
         - actions_cat: float32[E, T, A] where E = S + N
         - latents_cat: float32[H, E, T+1, L] (H=1 for single head)
      4) latents_head0 = latents_cat[0]  # float32[E, T+1, L]
      5) (vals_unscaled, vals_scaled) = compute_values_head0(latents_head0, actions_cat, world_model, task)
      6) dis = compute_disagreement(latents_cat[:, :, -1, :]) if (not eval_mode and H>1) else None
      7) scores = combine_scores(vals_scaled, dis, lambda_d)
      8) elite_indices = topk(scores, K)  # descending
      9) w = softmax(scores[elite_indices] / temp)
      10) elite_actions = actions_cat[elite_indices]  # float32[K,T,A]
          mean = (w.view(K,1,1) * elite_actions).sum(dim=0)  # float32[T,A]
          std = ((w.view(K,1,1) * (elite_actions - mean).pow(2)).sum(dim=0)).sqrt().clamp(cfg.std_min, cfg.std_max)
    - Post-iterations:
      - Action selection:
        - Eval: chosen_idx = scores.argmax()
        - Train: sample elite index via categorical softmax(scores[elite_indices] / temp), then map to global index
      - chosen_action = actions_cat[chosen_idx][0]  # float32[A]
      - Build PlannerBasicInfo (stats over final elite set). If log_detailed_every>0 and step%log_detailed_every==0, build PlannerAdvancedInfo with actions_all[E,T,A], latents_all[H,E,T+1,L], values_all_unscaled[E], values_all_scaled[E], disagreements_all[E], raw_scores[E], elite_indices[E].
      - Update prev_mean = mean.detach()  # warm-start retained across environment steps
      - Return (chosen_action, info)
    - Assertions:
      - assert not cfg.multitask
      - assert latents_cat.shape[0] in {1, cfg.planner_num_dynamics_heads}

Logger Integration (common/logger.py)
- Add Logger.log_planner_info(self, info, step, category='train')
  - Extract fields from dataclass and log scalars with keys:
    - planner/value_chosen
    - planner/disagreement_chosen (if available)
    - planner/value_elite_mean, planner/value_elite_std, planner/value_elite_max
    - planner/disagreement_elite_mean, planner/disagreement_elite_std, planner/disagreement_elite_max (if available)
    - planner/num_elites
  - Advanced payload: keep empty initially (structure only); frequency gating handled by caller via cfg.log_detailed_every.

Agent Integration (tdmpc2/tdmpc2.py)
- __init__: self.planner = Planner(cfg, self.model, self.scale)
- reset_planner_state(): self.planner.reset_warm_start()
- act(obs, eval_mode=False, ...):
  - Encode obs -> z0: float32[L]
  - action, info = self.planner.plan(z0, task, eval_mode=eval_mode, step=self._step)
  - self.logger.log_planner_info(info, step=self._step, category=('eval' if eval_mode else 'train'))
  - Return action; remove legacy planner logic (_plan, _estimate_value, warm-start std, planner logging).

Flat Config Keys (new or reused)
- New:
  - planner_num_dynamics_heads: int
  - planner_head_mode_eval: str = 'single'
  - planner_lambda_disagreement: float = 0.0  # TODO: may schedule later
  - planner_temperature_train: float
  - planner_temperature_eval: float
  - planner_num_seed_policy_trajectories: int  # 0 disables seeding
- Reuse existing:
  - horizon, iterations, num_samples, num_elites, std_min, std_max, log_detailed_every
- Assumptions:
  - use_prev_mean always on; only mean is persisted, std reset each call to std_max.

Key Notes / Decisions
- Value scaling deferred: values_scaled = values_unscaled * 1.0 (placeholder for future RunningScale). Do not update scale during planner.
- Policy seeds frozen after first iteration; included in candidates and eligible for elites each iteration.
- Elite updates happen each iteration; chosen action taken only after final iteration.
- Training sampling uses elite softmax weights; evaluation uses argmax over all candidates.
- Disagreement only computed in training with multi-head latents (H>1); evaluation uses single head and lambda=0.
- No multitask support: early assert in planner and rollout.
- Reset hook to zero prev_mean for new episodes (call from trainer when episode starts).
- Random head mode stubbed (NotImplementedError) for future.
