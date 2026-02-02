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
    
    All operations support batch dimension B. Shapes are always [B, ...] throughout.
    """

    def __init__(self, cfg, world_model, scale=None):
        super().__init__()
        self.cfg = cfg
        self.world_model = world_model
        self.scale = scale  # reserved; scaling currently disabled
        T, A = cfg.horizon, cfg.action_dim
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.register_buffer('prev_mean', torch.zeros(T, A, device=device))  # float32[T,A]

    def reset_warm_start(self) -> None:
        self.prev_mean.zero_()

    def shifted_prev_mean(self, batch_size: int = 1) -> torch.Tensor:
        """Return previous mean shifted left by one step with zeroed tail.

        Args:
            batch_size: Batch size B. If >1, returns zeros (no warm-start for batch).

        Returns:
            Tensor[B, T, A]: Warm-start mean after temporal shift, broadcast to batch.
        """
        T, A = self.prev_mean.shape
        if batch_size > 1:
            # No warm-start for batch planning
            return torch.zeros(batch_size, T, A, device=self.prev_mean.device, dtype=self.prev_mean.dtype)
        shifted = torch.zeros_like(self.prev_mean)  # float32[T, A]
        if shifted.shape[0] > 1:
            shifted[:-1].copy_(self.prev_mean[1:])
        return shifted.unsqueeze(0)  # float32[1, T, A]

    @torch.no_grad()
    def plan(
        self,
        z0: torch.Tensor,
        task: Optional[torch.Tensor] = None,
        eval_mode: bool = False,
        log_detailed: bool = False,
        train_noise_multiplier: Optional[float] = None,
        value_std_coef_override: Optional[float] = None,
        use_warm_start: bool = True,
        update_warm_start: bool = True,
        reanalyze: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PlannerBasicInfo], torch.Tensor, torch.Tensor]:
        """Plan action sequences and return the first action for each batch element.
        
        All operations support batch dimension B throughout. When B>1, warm-start
        is disabled and detailed logging is skipped.

        Args:
            z0 (Tensor[B, L] or Tensor[L]): Initial latent states.
            task: Optional multitask id (unsupported; asserted off).
            eval_mode: If True, use value-only scoring, single head, argmax selection.
            log_detailed: If True, return PlannerAdvancedInfo with full iteration history.
            value_std_coef_override: Override default value_std_coef for this call.
            use_warm_start: If True, initialize mean from shifted prev_mean (only for B=1).
            update_warm_start: If True, update prev_mean after planning (only for B=1).
            reanalyze: If True, use reanalyze-specific config (reanalyze_num_pi_trajs, etc.).

        Returns:
            Tuple of:
                - Tensor[B, A]: First action for each batch element.
                - PlannerBasicInfo | PlannerAdvancedInfo | None: Planning info (None if B>1).
                - Tensor[B, T, A]: Final mean action sequences.
                - Tensor[B, T, A]: Final std action sequences.
        """
        assert not getattr(self.cfg, 'multitask', False), 'Planner currently does not support multitask.'
        
        # Ensure z0 has batch dimension: [L] -> [1, L]
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)  # float32[1, L]
        B = z0.shape[0]
        
        device = self.prev_mean.device
        dtype = self.prev_mean.dtype
        _, A = self.prev_mean.shape

        # Planning mode config: reanalyze vs eval vs train
        if reanalyze:
            # Reanalyze mode: use reanalyze-specific config
            T = int(self.cfg.reanalyze_horizon)
            iterations = int(self.cfg.reanalyze_iterations)
            N = int(self.cfg.reanalyze_num_samples)
            S = int(self.cfg.reanalyze_num_pi_trajs)
            K = int(self.cfg.reanalyze_num_elites)
            temp = float(self.cfg.reanalyze_temperature)
            lambda_latent = 0.0  # No exploration bonus during reanalyze
            head_mode = 'all'
            reward_head_mode = 'all'
            value_std_coef = float(self.cfg.reanalyze_value_std_coef)
        elif eval_mode:
            # Eval mode: value-only scoring, optionally single head
            T = int(self.cfg.horizon)
            iterations = int(self.cfg.iterations)
            N = int(self.cfg.num_samples)
            S = int(self.cfg.num_pi_trajs)
            K = int(self.cfg.num_elites)
            temp = float(self.cfg.temperature)
            lambda_latent = 0.0
            use_all_heads_eval = bool(self.cfg.planner_use_all_heads_eval)
            head_mode = 'all' if use_all_heads_eval else 'single'
            reward_head_mode = head_mode
            value_std_coef = float(self.cfg.planner_value_std_coef_eval) if use_all_heads_eval else 0.0
        else:
            # Training mode: full ensemble with exploration bonus
            T = int(self.cfg.horizon)
            iterations = int(self.cfg.iterations)
            N = int(self.cfg.num_samples)
            S = int(self.cfg.num_pi_trajs)
            K = int(self.cfg.num_elites)
            temp = float(self.cfg.temperature)
            lambda_latent = float(self.cfg.planner_lambda_disagreement)
            head_mode = 'all'
            reward_head_mode = 'all'
            value_std_coef = float(self.cfg.planner_value_std_coef_train)

        # Allow explicit override (rare, mostly for testing)
        if value_std_coef_override is not None:
            value_std_coef = value_std_coef_override

        use_ema_value = bool(self.cfg.ema_value_planning)
        policy_elites_first_iter_only = bool(self.cfg.planner_policy_elites_first_iter_only)
        aggregate_horizon = bool(self.cfg.planner_aggregate_value)

        # Initialize mean/std with batch dimension [B, T, A]
        if use_warm_start and B == 1:
            mean = self.shifted_prev_mean(batch_size=1)  # float32[1, T, A]
        else:
            mean = torch.zeros(B, T, A, device=device, dtype=dtype)  # float32[B, T, A]
        std = torch.full((B, T, A), self.cfg.max_std, device=device, dtype=dtype)  # float32[B, T, A]

        # Prepare policy-seeded candidates (frozen across iterations)
        policy_cache = None
        if S > 0:
            use_optimistic = (not eval_mode) and self.cfg.dual_policy_enabled
            # rollout_latents: z0 [B, L] -> latents [H, B, S, T+1, L], actions [B, S, T, A]
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
            # Shapes: latents_p [H, B, S, T+1, L], actions_p [B, S, T, A]
            vals_unscaled_p, vals_scaled_p, vals_std_p, val_dis_p = compute_values(
                latents_p,     # [H, B, S, T+1, L]
                actions_p,     # [B, S, T, A]
                self.world_model,
                task,
                value_std_coef=value_std_coef,
                reward_head_mode=reward_head_mode,
                use_ema_value=use_ema_value,
                aggregate_horizon=aggregate_horizon,
            )
            # vals_*: [B, S]
            latent_dis_p = None
            if not eval_mode and latents_p.shape[0] > 1:
                final_policy = latents_p[:, :, :, -1, :]  # [H, B, S, L]
                latent_dis_p = compute_disagreement(final_policy)  # [B, S]
            scores_p = combine_scores(vals_scaled_p, latent_dis_p, lambda_latent)  # [B, S]
            weighted_latent_dis_p = (lambda_latent * latent_dis_p) if latent_dis_p is not None else None
            policy_cache = dict(
                latents=latents_p,                    # [H, B, S, T+1, L]
                actions=actions_p,                    # [B, S, T, A]
                vals_unscaled=vals_unscaled_p,        # [B, S]
                vals_scaled=vals_scaled_p,            # [B, S]
                vals_std=vals_std_p,                  # [B, S]
                latent_disagreement=latent_dis_p,     # [B, S] or None
                value_disagreement=val_dis_p,         # [B, S]
                weighted_latent_dis=weighted_latent_dis_p,  # [B, S] or None
                scores=scores_p,                      # [B, S]
            )

        # Containers for per-iteration histories (for advanced logging, only for B=1)
        enable_detailed_logging = (B == 1)
        actions_hist = [] if enable_detailed_logging else None
        latents_hist = [] if enable_detailed_logging else None
        values_unscaled_hist = [] if enable_detailed_logging else None
        values_scaled_hist = [] if enable_detailed_logging else None
        latent_disagreement_hist = [] if enable_detailed_logging else None
        value_disagreement_hist = [] if enable_detailed_logging else None
        scores_hist = [] if enable_detailed_logging else None
        mean_hist = [] if enable_detailed_logging else None
        std_hist = [] if enable_detailed_logging else None

        # Iterative refinement
        for it in range(iterations):
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)  # [B, T, A]
            # Sample action sequences: [B, T, A] mean/std -> [B, N, T, A] actions
            actions_s = sample_action_sequences(mean, std, N).detach()  # float32[B, N, T, A]
            
            # World model rollout: z0 [B, L], actions [B, N, T, A]
            # Returns latents [H, B, N, T+1, L], actions [B, N, T, A]
            latents_s, actions_s = self.world_model.rollout_latents(
                z0, actions=actions_s, use_policy=False, head_mode=head_mode, task=task
            )
            # latents_s: [H, B, N, T+1, L], actions_s: [B, N, T, A]

            # Values for sampled trajectories
            vals_unscaled_s, vals_scaled_s, vals_std_s, val_dis_s = compute_values(
                latents_s,   # [H, B, N, T+1, L]
                actions_s,   # [B, N, T, A]
                self.world_model,
                task,
                value_std_coef=value_std_coef,
                reward_head_mode=reward_head_mode,
                use_ema_value=use_ema_value,
                aggregate_horizon=aggregate_horizon,
            )
            # vals_*: [B, N]
            
            latent_dis_s = None
            if not eval_mode and latents_s.shape[0] > 1:
                final_s = latents_s[:, :, :, -1, :]  # [H, B, N, L]
                latent_dis_s = compute_disagreement(final_s)  # [B, N]
            scores_s = combine_scores(vals_scaled_s, latent_dis_s, lambda_latent)  # [B, N]
            weighted_latent_dis_s = (lambda_latent * latent_dis_s) if latent_dis_s is not None else None

            # Include policy candidates
            include_policy = (
                (policy_cache is not None) and
                ((it == 0) or (not policy_elites_first_iter_only))
            )

            if include_policy:
                # Concatenate policy and sampled candidates on N dimension
                # Policy: [H, B, S, T+1, L], Sampled: [H, B, N, T+1, L] -> [H, B, S+N, T+1, L]
                latents_cat = torch.cat([policy_cache['latents'], latents_s], dim=2)
                actions_cat = torch.cat([policy_cache['actions'], actions_s], dim=1)  # [B, S+N, T, A]
                vals_unscaled = torch.cat([policy_cache['vals_unscaled'], vals_unscaled_s], dim=1)  # [B, S+N]
                vals_scaled = torch.cat([policy_cache['vals_scaled'], vals_scaled_s], dim=1)
                
                # Latent disagreement concatenation
                if policy_cache['latent_disagreement'] is not None and latent_dis_s is not None:
                    latent_dis = torch.cat([policy_cache['latent_disagreement'], latent_dis_s], dim=1)
                elif policy_cache['latent_disagreement'] is not None:
                    latent_dis = policy_cache['latent_disagreement']
                else:
                    latent_dis = latent_dis_s
                
                # Value disagreement concatenation
                if policy_cache['value_disagreement'] is not None and val_dis_s is not None:
                    val_dis = torch.cat([policy_cache['value_disagreement'], val_dis_s], dim=1)
                elif policy_cache['value_disagreement'] is not None:
                    val_dis = policy_cache['value_disagreement']
                else:
                    val_dis = val_dis_s
                
                # Weighted latent disagreement concatenation
                if policy_cache['weighted_latent_dis'] is not None and weighted_latent_dis_s is not None:
                    weighted_latent_dis_all = torch.cat([policy_cache['weighted_latent_dis'], weighted_latent_dis_s], dim=1)
                elif policy_cache['weighted_latent_dis'] is not None:
                    weighted_latent_dis_all = policy_cache['weighted_latent_dis']
                else:
                    weighted_latent_dis_all = weighted_latent_dis_s
                    
                scores = torch.cat([policy_cache['scores'], scores_s], dim=1)  # [B, S+N]
            else:
                latents_cat = latents_s      # [H, B, N, T+1, L]
                actions_cat = actions_s      # [B, N, T, A]
                vals_unscaled = vals_unscaled_s  # [B, N]
                vals_scaled = vals_scaled_s
                latent_dis = latent_dis_s
                val_dis = val_dis_s
                weighted_latent_dis_all = weighted_latent_dis_s
                scores = scores_s            # [B, N]

            # Elite selection per batch element: [B, E] -> [B, K]
            elite_scores, elite_indices = torch.topk(scores, K, dim=1, largest=True, sorted=True)  # [B, K]
            
            # Compute weights over elite scores per batch (BMPC-style: subtract max, scale by temp)
            max_elite = elite_scores.max(dim=1, keepdim=True).values  # [B, 1]
            score_delta = elite_scores - max_elite  # [B, K]
            if bool(self.cfg.mult_by_temp):
                w = torch.exp(temp * score_delta)  # [B, K] - multiply: higher temp = softer
            else:
                w = torch.exp(score_delta / temp)  # [B, K] - divide: higher temp = softer
            w = w / (w.sum(dim=1, keepdim=True) + 1e-9)  # [B, K]
            
            # Gather elite actions: [B, E, T, A] -> [B, K, T, A]
            elite_indices_expanded = elite_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, A)  # [B, K, T, A]
            elite_actions = actions_cat.gather(1, elite_indices_expanded)  # [B, K, T, A]
            
            # Compute new mean and std per batch
            w_expanded = w.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
            mean = (w_expanded * elite_actions).sum(dim=1)  # [B, T, A]
            var = (w_expanded * (elite_actions - mean.unsqueeze(1)).pow(2)).sum(dim=1)  # [B, T, A]
            std = var.clamp_min(0).sqrt().clamp(self.cfg.min_std, self.cfg.max_std)  # [B, T, A]

            # Append iteration snapshots (only for B=1)
            if enable_detailed_logging:
                actions_hist.append(actions_cat.detach())           # [1, E, T, A]
                latents_hist.append(latents_cat.detach())           # [H, 1, E, T+1, L]
                values_unscaled_hist.append(vals_unscaled.detach()) # [1, E]
                values_scaled_hist.append(vals_scaled.detach())     # [1, E]
                scores_hist.append(scores.detach())                 # [1, E]
                mean_hist.append(mean.detach())                     # [B, T, A]
                std_hist.append(std.detach())                       # [B, T, A]
                if latent_dis is not None:
                    latent_disagreement_hist.append(latent_dis.detach())  # [1, E]
                if val_dis is not None:
                    value_disagreement_hist.append(val_dis.detach())      # [1, E]

        # Final selection: per batch element
        # Greedy (argmax) selection: used in eval mode OR when greedy_train_action_selection is enabled
        use_greedy = eval_mode or bool(getattr(self.cfg, 'greedy_train_action_selection', False))
        if use_greedy:
            chosen_idx = scores.argmax(dim=1, keepdim=True)  # [B, 1]
        else:
            # BMPC-style: subtract max, scale by temp
            max_elite = elite_scores.max(dim=1, keepdim=True).values  # [B, 1]
            score_delta = elite_scores - max_elite  # [B, K]
            if bool(self.cfg.mult_by_temp):
                probs = torch.exp(temp * score_delta)  # [B, K] - multiply: higher temp = softer
            else:
                probs = torch.exp(score_delta / temp)  # [B, K] - divide: higher temp = softer
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-9)  # [B, K]
            elite_pick = torch.multinomial(probs, 1)  # [B, 1]
            chosen_idx = elite_indices.gather(1, elite_pick)  # [B, 1]
        
        # Select chosen sequence and first action per batch
        chosen_idx_expanded = chosen_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, A)  # [B, 1, T, A]
        chosen_seq = actions_cat.gather(1, chosen_idx_expanded).squeeze(1)  # [B, T, A]
        chosen_action = chosen_seq[:, 0, :]  # [B, A]

        # Prepare first-step std for potential noise
        std_first = std[:, 0, :]  # [B, A]
        noise_vec_first = None
        
        # Legacy: std-scaled noise via train_noise_multiplier (only for B=1)
        if (not eval_mode) and (train_noise_multiplier is not None) and (float(train_noise_multiplier) > 0.0) and (B == 1):
            noise_vec_first = torch.randn_like(chosen_action) * std_first * float(train_noise_multiplier)
            chosen_action = (chosen_action + noise_vec_first).clamp(-1.0, 1.0)
        
        # Fixed-std action noise for ablation (only for B=1)
        fixed_action_noise_std = float(getattr(self.cfg, 'planner_action_noise_std', 0.0))
        if (not eval_mode) and fixed_action_noise_std > 0.0 and (B == 1):
            fixed_noise = torch.randn_like(chosen_action) * fixed_action_noise_std
            chosen_action = (chosen_action + fixed_noise).clamp(-1.0, 1.0)
            if noise_vec_first is None:
                noise_vec_first = fixed_noise
            else:
                noise_vec_first = noise_vec_first + fixed_noise

        # Build info (only for B=1)
        info_basic = None
        if B == 1:
            elite_scores_b0 = elite_scores[0]  # [K]
            value_elite_mean = elite_scores_b0.mean()
            value_elite_std = elite_scores_b0.std(unbiased=False) if K > 1 else elite_scores_b0.new_zeros(())
            value_elite_max = elite_scores_b0.max()

            latent_disagreement_chosen = None
            latent_disagreement_elite_mean = None
            latent_disagreement_elite_std = None
            latent_disagreement_elite_max = None
            if latent_dis is not None:
                latent_dis_b0 = latent_dis[0]  # [E]
                latent_disagreement_chosen = latent_dis_b0[chosen_idx[0, 0]]
                elite_latent_dis = latent_dis_b0[elite_indices[0]]  # [K]
                latent_disagreement_elite_mean = elite_latent_dis.mean()
                latent_disagreement_elite_std = elite_latent_dis.std(unbiased=False) if K > 1 else elite_latent_dis.new_zeros(())
                latent_disagreement_elite_max = elite_latent_dis.max()

            value_disagreement_chosen = None
            value_disagreement_elite_mean = None
            value_disagreement_elite_std = None
            value_disagreement_elite_max = None
            if val_dis is not None:
                val_dis_b0 = val_dis[0]  # [E]
                value_disagreement_chosen = val_dis_b0[chosen_idx[0, 0]]
                elite_val_dis = val_dis_b0[elite_indices[0]]  # [K]
                value_disagreement_elite_mean = elite_val_dis.mean()
                value_disagreement_elite_std = elite_val_dis.std(unbiased=False) if K > 1 else elite_val_dis.new_zeros(())
                value_disagreement_elite_max = elite_val_dis.max()

            vals_scaled_b0 = vals_scaled[0]  # [E]
            scores_b0 = scores[0]  # [E]
            weighted_latent_dis_b0 = weighted_latent_dis_all[0] if weighted_latent_dis_all is not None else None
            
            info_basic = PlannerBasicInfo(
                value_chosen=vals_scaled_b0[chosen_idx[0, 0]],
                latent_disagreement_chosen=latent_disagreement_chosen,
                value_disagreement_chosen=value_disagreement_chosen,
                score_chosen=scores_b0[chosen_idx[0, 0]],
                weighted_latent_disagreement_chosen=(weighted_latent_dis_b0[chosen_idx[0, 0]]) if weighted_latent_dis_b0 is not None else None,
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
                elite_indices=elite_indices[0].to(torch.long),  # [K]
                scores_all=scores_b0,
                values_scaled_all=vals_scaled_b0,
                weighted_latent_disagreements_all=weighted_latent_dis_b0,
            )

        # Detailed logging: upgrade info_basic to info_adv if requested (only for B=1)
        info_out = info_basic
        if log_detailed and enable_detailed_logging and info_basic is not None:
            actions_all = torch.stack(actions_hist, dim=0).squeeze(1)               # [I, E, T, A]
            latents_all = torch.stack(latents_hist, dim=0).squeeze(2)               # [I, H, E, T+1, L]
            values_all_unscaled = torch.stack(values_unscaled_hist, dim=0).squeeze(1)  # [I, E]
            values_all_scaled = torch.stack(values_scaled_hist, dim=0).squeeze(1)      # [I, E]
            raw_scores = torch.stack(scores_hist, dim=0).squeeze(1)                  # [I, E]
            mean_all = torch.stack(mean_hist, dim=0).squeeze(1)                      # [I, T, A]
            std_all = torch.stack(std_hist, dim=0).squeeze(1)                        # [I, T, A]
            latent_disagreements_all = torch.stack(latent_disagreement_hist, dim=0).squeeze(1) if len(latent_disagreement_hist) > 0 else None
            value_disagreements_all = torch.stack(value_disagreement_hist, dim=0).squeeze(1) if len(value_disagreement_hist) > 0 else None

            info_out = PlannerAdvancedInfo(
                **vars(info_basic),
                actions_all=actions_all,
                latents_all=latents_all,
                values_all_unscaled=values_all_unscaled,
                values_all_scaled=values_all_scaled,
                latent_disagreements_all=latent_disagreements_all,
                value_disagreements_all=value_disagreements_all,
                raw_scores=raw_scores,
                mean_hist=mean_all,
                std_hist=std_all,
                action_seq_chosen=chosen_seq[0].detach(),  # [T, A]
                action_noise=(noise_vec_first[0].detach() if noise_vec_first is not None else None),
                std_first_action=std_first[0].detach(),
                z0=z0[0].detach(),
                task=task.detach() if (task is not None and torch.is_tensor(task)) else task,
                lambda_latent=float(self.cfg.planner_lambda_disagreement) if not eval_mode else 0.0,
                head_mode=head_mode,
                value_std_coef=value_std_coef,
                T=T,
                use_ema_value=use_ema_value,
            )
            with torch.no_grad():
                info_out.compute_post_noise_effects(self.world_model)

        # Update warm-start mean (only for B=1)
        if update_warm_start and B == 1:
            self.prev_mean.copy_(mean[0])
        
        # Final clamp and return (always executed)
        chosen_action = chosen_action.clamp(-1.0, 1.0)
        return chosen_action.detach(), info_out, mean.detach(), std.detach()
