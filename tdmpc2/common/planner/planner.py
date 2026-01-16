from typing import Optional, Tuple
import torch
import torch._dynamo as dynamo

from .sampling import sample_action_sequences
from .scoring import compute_values, compute_disagreement, combine_scores
from .info_types import PlannerBasicInfo, PlannerAdvancedInfo
from common.logger import get_logger

log = get_logger(__name__)


class Planner(torch.nn.Module):
    """CEM/MPPI-style planner with optional latent disagreement scoring.

    Uses world_model.rollout_latents for both policy-seeded and sampled trajectories.
    Persists a warm-start mean across environment steps via prev_mean.
    """

    def __init__(self, cfg, world_model, scale=None, discount=None):
        super().__init__()
        self.cfg = cfg
        self.world_model = world_model
        self.scale = scale  # reserved; scaling currently disabled
        T, A = cfg.horizon, cfg.action_dim
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.register_buffer('prev_mean', torch.zeros(T, A, device=device))  # float32[T,A]
        # Store discount for compute_values (can be scalar or tensor for multitask)
        if discount is not None:
            if isinstance(discount, torch.Tensor):
                self.register_buffer('discount', discount.clone())
            else:
                self.register_buffer('discount', torch.tensor(discount, device=device))
        else:
            self.register_buffer('discount', torch.tensor(0.99, device=device))

    def reset_warm_start(self) -> None:
        self.prev_mean.zero_()

    def shifted_prev_mean(self) -> torch.Tensor:
        """Return previous mean shifted left by one step with zeroed tail.

        Returns:
            Tensor[T, A]: Warm-start mean after temporal shift.
        """
        shifted = torch.zeros_like(self.prev_mean)  # float32[T,A]
        if shifted.shape[0] > 1:
            shifted[:-1].copy_(self.prev_mean[1:])
        return shifted

    @torch.no_grad()
    def plan(self, z0: torch.Tensor, task: Optional[torch.Tensor] = None, eval_mode: bool = False, log_detailed: bool = False, train_noise_multiplier: Optional[float] = None, value_std_coef_override: Optional[float] = None) -> Tuple[torch.Tensor, PlannerBasicInfo, torch.Tensor, torch.Tensor]:
        """Plan an action sequence and return the first action.

        Args:
            z0 (Tensor[L] or Tensor[1, L]): Initial latent.
            task: Optional multitask id (unsupported; asserted off).
            eval_mode: If True, use value-only scoring, single head, argmax selection, eval temperature.
            log_detailed: If True, return detailed planner info for logging.
            value_std_coef_override: If provided, overrides the default value_std_coef for this call.
                Use 0.0 for mean-only reduction (e.g., eval_mean_head_reduce mode).

        Returns:
            (Tensor[A], PlannerBasicInfo | PlannerAdvancedInfo, Tensor[T,A], Tensor[T,A])
        """
        assert not getattr(self.cfg, 'multitask', False), 'Planner currently does not support multitask.'
        device = self.prev_mean.device
        dtype = self.prev_mean.dtype
        T, A = self.prev_mean.shape

        iterations = int(self.cfg.iterations)
        N = int(self.cfg.num_samples)
        K = int(self.cfg.num_elites)
        S = int(self.cfg.num_pi_trajs)

        temp = float(self.cfg.temperature)

        lambda_latent = 0.0 if eval_mode else float(self.cfg.planner_lambda_disagreement)

        # Head mode and value_std_coef depend on eval vs train
        # Train: always use all heads with planner_value_std_coef_train (optimistic by default: +1.0)
        # Eval: use all heads or single head based on planner_use_all_heads_eval
        #       value_std_coef from planner_value_std_coef_eval (pessimistic by default: -1.0)
        if eval_mode:
            use_all_heads_eval = bool(self.cfg.planner_use_all_heads_eval)
            head_mode = 'all' if use_all_heads_eval else 'single'
            value_std_coef = float(self.cfg.planner_value_std_coef_eval) if use_all_heads_eval else 0.0
            reward_head_mode = head_mode  # Match dynamics head mode
        else:
            head_mode = 'all'
            value_std_coef = float(self.cfg.planner_value_std_coef_train)
            reward_head_mode = 'all'
        
        # Allow caller to override value_std_coef (e.g., for mean-only eval mode)
        if value_std_coef_override is not None:
            value_std_coef = value_std_coef_override

        # Whether to use EMA (target) network for V in planning
        use_ema_value = bool(self.cfg.ema_value_planning)

        policy_elites_first_iter_only = bool(self.cfg.planner_policy_elites_first_iter_only)

        mean = self.shifted_prev_mean()  # float32[T,A]
        std = torch.full((T, A), self.cfg.max_std, device=device, dtype=dtype)

        # Prepare policy-seeded candidates (frozen across iterations)
        policy_cache = None
        if S > 0:
            # Use optimistic policy for seeding during training if dual_policy_enabled
            use_optimistic = (not eval_mode) and self.cfg.dual_policy_enabled
            latents_p, actions_p = self.world_model.rollout_latents(
                z0,
                use_policy=True,
                horizon=T,
                num_rollouts=S,
                head_mode=head_mode,
                task=task,
                policy_action_noise_std=float(self.cfg.policy_seed_noise_std),
                use_optimistic_policy=use_optimistic,
            )
            # Squeeze singleton batch dim: latents [H,1,S,T+1,L] -> [H,S,T+1,L], actions [1,S,T,A] -> [S,T,A]
            latents_p = latents_p[:, 0]
            actions_p = actions_p[0]
            vals_unscaled_p, vals_scaled_p, vals_std_p, val_dis_p = compute_values(
                latents_p,
                actions_p,
                self.world_model,
                task,
                value_std_coef=value_std_coef,
                reward_head_mode=reward_head_mode,
                use_ema_value=use_ema_value,
                discount=float(self.discount),
            )
            latent_dis_p = None
            if not eval_mode and latents_p.shape[0] > 1:
                final_policy = latents_p[:, :, -1, :]
                latent_dis_p = compute_disagreement(final_policy)
            scores_p = combine_scores(vals_scaled_p, latent_dis_p, lambda_latent)
            weighted_latent_dis_p = (lambda_latent * latent_dis_p) if latent_dis_p is not None else None
            policy_cache = dict(
                latents=latents_p,
                actions=actions_p,
                vals_unscaled=vals_unscaled_p,
                vals_scaled=vals_scaled_p,
                vals_std=vals_std_p,
                latent_disagreement=latent_dis_p,
                value_disagreement=val_dis_p,
                weighted_latent_dis=weighted_latent_dis_p,
                scores=scores_p,
            )

        # Containers for per-iteration histories (for advanced logging)
        # actions_hist: float32[I, E, T, A]; E varies only after concat (policy + sampled)
        # latents_hist: float32[I, H_used, E, T+1, L]
        # values_unscaled_hist / values_scaled_hist / latent_disagreement_hist / value_disagreement_hist / scores_hist: float32[I, E]
        actions_hist = []
        latents_hist = []
        values_unscaled_hist = []
        values_scaled_hist = []
        latent_disagreement_hist = []  # may remain empty if not computed
        value_disagreement_hist = []
        scores_hist = []

        # Iterative refinement
        for it in range(iterations):
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            actions_s = sample_action_sequences(mean, std, N).detach()
            # World model expects actions [B,N,T,A]; here B=1
            latents_s, actions_s = self.world_model.rollout_latents(
                z0, actions=actions_s.unsqueeze(0), use_policy=False, head_mode=head_mode, task=task
            )
            # Squeeze singleton batch dim back to planner shapes
            latents_s = latents_s[:, 0]
            actions_s = actions_s[0]

            # Values for sampled trajectories using configured head mode and reduction.
            vals_unscaled_s, vals_scaled_s, vals_std_s, val_dis_s = compute_values(
                latents_s,
                actions_s,
                self.world_model,
                task,
                value_std_coef=value_std_coef,
                reward_head_mode=reward_head_mode,
                use_ema_value=use_ema_value,
                discount=float(self.discount),
            )
            latent_dis_s = None
            if not eval_mode and latents_s.shape[0] > 1:
                # Multiple dynamics heads are present; measure disagreement at final latent state.
                final_s = latents_s[:, :, -1, :]
                latent_dis_s = compute_disagreement(final_s)
            scores_s = combine_scores(vals_scaled_s, latent_dis_s, lambda_latent)
            weighted_latent_dis_s = (lambda_latent * latent_dis_s) if latent_dis_s is not None else None

            include_policy = (
                (policy_cache is not None) and
                ((it == 0) or (not policy_elites_first_iter_only))
            )

            if include_policy:
                latents_cat = torch.cat([policy_cache['latents'], latents_s], dim=1)
                actions_cat = torch.cat([policy_cache['actions'], actions_s], dim=0)
                vals_unscaled = torch.cat([policy_cache['vals_unscaled'], vals_unscaled_s], dim=0)
                vals_scaled = torch.cat([policy_cache['vals_scaled'], vals_scaled_s], dim=0)
                # Latent disagreement concatenation
                if policy_cache['latent_disagreement'] is not None and latent_dis_s is not None:
                    latent_dis = torch.cat([policy_cache['latent_disagreement'], latent_dis_s], dim=0)
                elif policy_cache['latent_disagreement'] is not None:
                    latent_dis = policy_cache['latent_disagreement']
                else:
                    latent_dis = latent_dis_s
                # Value disagreement concatenation
                if policy_cache['value_disagreement'] is not None and val_dis_s is not None:
                    val_dis = torch.cat([policy_cache['value_disagreement'], val_dis_s], dim=0)
                elif policy_cache['value_disagreement'] is not None:
                    val_dis = policy_cache['value_disagreement']
                else:
                    val_dis = val_dis_s
                # Weighted latent disagreement concatenation
                if policy_cache['weighted_latent_dis'] is not None and weighted_latent_dis_s is not None:
                    weighted_latent_dis_all = torch.cat([policy_cache['weighted_latent_dis'], weighted_latent_dis_s], dim=0)
                elif policy_cache['weighted_latent_dis'] is not None:
                    weighted_latent_dis_all = policy_cache['weighted_latent_dis']
                else:
                    weighted_latent_dis_all = weighted_latent_dis_s
                scores = torch.cat([policy_cache['scores'], scores_s], dim=0)
            else:
                latents_cat = latents_s
                actions_cat = actions_s
                vals_unscaled = vals_unscaled_s
                vals_scaled = vals_scaled_s
                latent_dis = latent_dis_s
                val_dis = val_dis_s
                weighted_latent_dis_all = weighted_latent_dis_s
                scores = scores_s

            elite_scores, elite_indices = torch.topk(scores, K, largest=True, sorted=True)
            # Softmax weights over elite scores
            w = torch.softmax(elite_scores / max(temp, 1e-8), dim=0)  # [K]
            elite_actions = actions_cat[elite_indices]  # [K,T,A]
            mean = (w.view(K, 1, 1) * elite_actions).sum(dim=0)
            var = (w.view(K, 1, 1) * (elite_actions - mean).pow(2)).sum(dim=0)
            std = var.clamp_min(0).sqrt().clamp(self.cfg.min_std, self.cfg.max_std)

            # Append iteration snapshots
            actions_hist.append(actions_cat.detach())          # [E,T,A]
            latents_hist.append(latents_cat.detach())          # [H,E,T+1,L]
            values_unscaled_hist.append(vals_unscaled.detach())# [E]
            values_scaled_hist.append(vals_scaled.detach())    # [E]
            scores_hist.append(scores.detach())                # [E]
            if latent_dis is not None:
                latent_disagreement_hist.append(latent_dis.detach())  # [E]
            if val_dis is not None:
                value_disagreement_hist.append(val_dis.detach())      # [E]

        # Final selection
        if eval_mode:
            # Keep as tensor to avoid graph breaks
            chosen_idx = scores.argmax().unsqueeze(0)  # [1]
        else:
            # Sample proportionally over elite scores
            probs = torch.softmax(elite_scores / max(temp, 1e-8), dim=0)
            elite_pick = torch.multinomial(probs, 1)              # [1]
            chosen_idx = elite_indices.gather(0, elite_pick)      # [1]
        # Select chosen sequence and first action
        chosen_seq = actions_cat.index_select(0, chosen_idx).squeeze(0)  # [T,A]
        chosen_action = chosen_seq[0]  # [A]

        # Prepare first-step std for potential noise (planner-owned)
        std_first = std[0]  # [A]
        noise_vec_first = None
        # Legacy: std-scaled noise via train_noise_multiplier (train_act_std_coeff)
        if (not eval_mode) and (train_noise_multiplier is not None) and (float(train_noise_multiplier) > 0.0):
            noise_vec_first = torch.randn_like(chosen_action) * std_first * float(train_noise_multiplier)
            chosen_action = (chosen_action + noise_vec_first).clamp(-1.0, 1.0)
        # Fixed-std action noise for ablation (planner_action_noise_std)
        fixed_action_noise_std = float(getattr(self.cfg, 'planner_action_noise_std', 0.0))
        if (not eval_mode) and fixed_action_noise_std > 0.0:
            fixed_noise = torch.randn_like(chosen_action) * fixed_action_noise_std
            chosen_action = (chosen_action + fixed_noise).clamp(-1.0, 1.0)
            # Merge into noise_vec_first for logging/info if needed
            if noise_vec_first is None:
                noise_vec_first = fixed_noise
            else:
                noise_vec_first = noise_vec_first + fixed_noise

        # Build info
        value_elite_mean = elite_scores.mean()
        value_elite_std = elite_scores.std(unbiased=False) if K > 1 else elite_scores.new_zeros(())
        value_elite_max = elite_scores.max()

        # Latent disagreement stats
        latent_disagreement_chosen = None
        latent_disagreement_elite_mean = None
        latent_disagreement_elite_std = None
        latent_disagreement_elite_max = None
        if latent_dis is not None:
            latent_disagreement_chosen = latent_dis.index_select(0, chosen_idx).squeeze(0)
            elite_latent_dis = latent_dis.index_select(0, elite_indices)
            latent_disagreement_elite_mean = elite_latent_dis.mean()
            latent_disagreement_elite_std = elite_latent_dis.std(unbiased=False) if K > 1 else elite_latent_dis.new_zeros(())
            latent_disagreement_elite_max = elite_latent_dis.max()

        # Value disagreement stats
        value_disagreement_chosen = None
        value_disagreement_elite_mean = None
        value_disagreement_elite_std = None
        value_disagreement_elite_max = None
        if val_dis is not None:
            value_disagreement_chosen = val_dis.index_select(0, chosen_idx).squeeze(0)
            elite_val_dis = val_dis.index_select(0, elite_indices)
            value_disagreement_elite_mean = elite_val_dis.mean()
            value_disagreement_elite_std = elite_val_dis.std(unbiased=False) if K > 1 else elite_val_dis.new_zeros(())
            value_disagreement_elite_max = elite_val_dis.max()

        info_basic = PlannerBasicInfo(
            value_chosen=vals_scaled.index_select(0, chosen_idx).squeeze(0),
            latent_disagreement_chosen=latent_disagreement_chosen,
            value_disagreement_chosen=value_disagreement_chosen,
            score_chosen=scores.index_select(0, chosen_idx).squeeze(0),
            weighted_latent_disagreement_chosen=(weighted_latent_dis_all.index_select(0, chosen_idx).squeeze(0)) if weighted_latent_dis_all is not None else None,
            value_elite_mean=value_elite_mean,
            value_elite_std=value_elite_std,
            value_elite_max=value_elite_max,
            latent_disagreement_elite_mean=latent_disagreement_elite_mean,
            latent_disagreement_elite_std=latent_disagreement_elite_std,
            latent_disagreement_elite_max=latent_disagreement_elite_max,
            value_disagreement_elite_mean=value_disagreement_elite_mean,
            value_disagreement_elite_std=value_disagreement_elite_std,
            value_disagreement_elite_max=value_disagreement_elite_max,
            num_elites=K,
            elite_indices=elite_indices.to(torch.long),
            scores_all=scores,
            values_scaled_all=vals_scaled,
            weighted_latent_disagreements_all=weighted_latent_dis_all,
        )


        # Use the log_detailed flag passed by the caller
        if log_detailed:
            # Stack iteration histories
            actions_all = torch.stack(actions_hist, dim=0)               # [I,E,T,A]
            latents_all = torch.stack(latents_hist, dim=0)               # [I,H,E,T+1,L]
            values_all_unscaled = torch.stack(values_unscaled_hist, dim=0)  # [I,E]
            values_all_scaled = torch.stack(values_scaled_hist, dim=0)      # [I,E]
            raw_scores = torch.stack(scores_hist, dim=0)                  # [I,E]
            latent_disagreements_all = torch.stack(latent_disagreement_hist, dim=0) if len(latent_disagreement_hist) > 0 else None  # [I,E]
            value_disagreements_all = torch.stack(value_disagreement_hist, dim=0) if len(value_disagreement_hist) > 0 else None  # [I,E]

            # Build advanced info (includes context for post-noise analysis)
            info_adv = PlannerAdvancedInfo(
                **vars(info_basic),
                actions_all=actions_all,
                latents_all=latents_all,
                values_all_unscaled=values_all_unscaled,
                values_all_scaled=values_all_scaled,
                latent_disagreements_all=latent_disagreements_all,
                value_disagreements_all=value_disagreements_all,
                raw_scores=raw_scores,
                # Context for analysis
                action_seq_chosen=chosen_seq.detach(),
                action_noise=(noise_vec_first.detach() if noise_vec_first is not None else None),
                std_first_action=std_first.detach(),
                z0=z0.detach() if z0.dim() == 1 else z0.squeeze(0).detach(),
                task=task.detach() if (task is not None and torch.is_tensor(task)) else task,
                lambda_latent=float(self.cfg.planner_lambda_disagreement) if not eval_mode else 0.0,
                head_mode=head_mode,
                value_std_coef=value_std_coef,
                T=T,
                use_ema_value=use_ema_value,
                discount=float(self.discount),
            )
            # Compute post-noise effects once (no-grad), available for logger + wandb
            with torch.no_grad():
                info_adv.compute_post_noise_effects(self.world_model)
            # Update warm-start mean for next call
            self.prev_mean.copy_(mean)
            # Ensure bounds
            return chosen_action.detach(), info_adv, mean.detach(), std.detach()

        # Update warm-start mean for next call
        self.prev_mean.copy_(mean)
        # Ensure bounds
        chosen_action = chosen_action.clamp(-1.0, 1.0)
        return chosen_action.detach(), info_basic, mean.detach(), std.detach()
