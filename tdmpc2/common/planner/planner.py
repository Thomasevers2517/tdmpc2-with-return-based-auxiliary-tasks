from typing import Optional, Tuple
import torch
import torch._dynamo as dynamo

from .sampling import sample_action_sequences
from .scoring import compute_values_head0, compute_disagreement, combine_scores
from .info_types import PlannerBasicInfo, PlannerAdvancedInfo
from common.logger import get_logger

log = get_logger(__name__)


class Planner(torch.nn.Module):
    """CEM/MPPI-style planner with optional latent disagreement scoring.

    Uses world_model.rollout_latents for both policy-seeded and sampled trajectories.
    Persists a warm-start mean across environment steps via prev_mean.
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
    def plan(self, z0: torch.Tensor, task: Optional[torch.Tensor] = None, eval_mode: bool = False, step: Optional[int] = None, train_noise_multiplier: Optional[float] = None) -> Tuple[torch.Tensor, PlannerBasicInfo, torch.Tensor, torch.Tensor]:
        """Plan an action sequence and return the first action.

        Args:
            z0 (Tensor[L] or Tensor[1, L]): Initial latent.
            task: Optional multitask id (unsupported; asserted off).
            eval_mode: If True, use value-only scoring, single head, argmax selection, eval temperature.
            step: Global step for detailed logging frequency gating.

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
        # Temperature: prefer new keys; fallback to legacy cfg.temperature
        if eval_mode:
            temp = getattr(self.cfg, 'planner_temperature_eval', None)
        else:
            temp = getattr(self.cfg, 'planner_temperature_train', None)
        if temp is None:
            temp = getattr(self.cfg, 'temperature', 1.0)
        temp = float(temp)
        lambda_d = 0.0 if eval_mode else self.cfg.planner_lambda_disagreement
        # Std bounds: prefer std_min/std_max; fallback to legacy min_std/max_std


        head_mode = 'single' if eval_mode else 'all'

        mean = self.shifted_prev_mean()  # float32[T,A]
        std = torch.full((T, A), self.cfg.max_std, device=device, dtype=dtype)

        # Prepare policy-seeded candidates (frozen across iterations)
        latents_p = actions_p = None
        if S > 0:
            latents_p, actions_p = self.world_model.rollout_latents(
                z0, use_policy=True, horizon=T, num_rollouts=S, head_mode=head_mode, task=task
            )
            # Squeeze singleton batch dim: latents [H,1,S,T+1,L] -> [H,S,T+1,L], actions [1,S,T,A] -> [S,T,A]
            latents_p = latents_p[:, 0]
            actions_p = actions_p[0]

        # Containers for per-iteration histories (for advanced logging)
        # actions_hist: float32[I, E, T, A]; E varies only after concat (policy + sampled)
        # latents_hist: float32[I, H_used, E, T+1, L]
        # values_unscaled_hist / values_scaled_hist / disagreement_hist / scores_hist: float32[I, E]
        actions_hist = []
        latents_hist = []
        values_unscaled_hist = []
        values_scaled_hist = []
        disagreement_hist = []  # may remain empty if not computed
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

            # Concatenate frozen policy seeds (if any)
            if S > 0:
                # latents: [H,S,T+1,L], actions: [S,T,A]
                latents_cat = torch.cat([latents_p, latents_s], dim=1)
                actions_cat = torch.cat([actions_p, actions_s], dim=0)
            else:
                latents_cat, actions_cat = latents_s, actions_s

            # Head 0 values
            latents_head0 = latents_cat[0]  # [E,T+1,L]
            vals_unscaled, vals_scaled = compute_values_head0(latents_head0, actions_cat, self.world_model, task)
            # Disagreement only when training with multi-head
            dis = None
            if not eval_mode and latents_cat.shape[0] > 1:
                final_all = latents_cat[:, :, -1, :]  # [H,E,L]
                dis = compute_disagreement(final_all)
                # Scale disagreement by the average of rho^t over the planning horizon
                # to roughly match the magnitude of the logged consistency loss.
                t_idx = torch.arange(T, device=dis.device, dtype=dis.dtype)
                scale = torch.pow(torch.tensor(self.cfg.rho, device=dis.device, dtype=dis.dtype), t_idx).mean()
                dis = dis * scale
            scores = combine_scores(vals_scaled, dis, lambda_d)
            weighted_dis_all = (lambda_d * dis) if dis is not None else None

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
            if dis is not None:
                disagreement_hist.append(dis.detach())         # [E]

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
        if (not eval_mode) and (train_noise_multiplier is not None) and (float(train_noise_multiplier) > 0.0):
            noise_vec_first = torch.randn_like(chosen_action) * std_first * float(train_noise_multiplier)
            chosen_action = (chosen_action + noise_vec_first).clamp(-1.0, 1.0)

        # Build info
        value_elite_mean = elite_scores.mean()
        value_elite_std = elite_scores.std(unbiased=False) if K > 1 else elite_scores.new_zeros(())
        value_elite_max = elite_scores.max()

        disagreement_chosen = None
        disagreement_elite_mean = None
        disagreement_elite_std = None
        disagreement_elite_max = None
        if dis is not None:
            # Tensor-safe selections and stats
            disagreement_chosen = dis.index_select(0, chosen_idx).squeeze(0)
            elite_dis = dis.index_select(0, elite_indices)
            disagreement_elite_mean = elite_dis.mean()
            disagreement_elite_std = elite_dis.std(unbiased=False) if K > 1 else elite_dis.new_zeros(())
            disagreement_elite_max = elite_dis.max()

        info_basic = PlannerBasicInfo(
            value_chosen=vals_scaled.index_select(0, chosen_idx).squeeze(0),
            disagreement_chosen=disagreement_chosen,
            score_chosen=scores.index_select(0, chosen_idx).squeeze(0),
            weighted_disagreement_chosen=(weighted_dis_all.index_select(0, chosen_idx).squeeze(0)) if weighted_dis_all is not None else None,
            value_elite_mean=value_elite_mean,
            value_elite_std=value_elite_std,
            value_elite_max=value_elite_max,
            disagreement_elite_mean=disagreement_elite_mean,
            disagreement_elite_std=disagreement_elite_std,
            disagreement_elite_max=disagreement_elite_max,
            num_elites=K,
            elite_indices=elite_indices.to(torch.long),
            scores_all=scores,
            values_scaled_all=vals_scaled,
            weighted_disagreements_all=weighted_dis_all,
        )


        # Use existing global log_detail_freq instead of new key if available.
        log_detailed = self.cfg.log_detail_freq > 0 and (step is not None) and (step % self.cfg.log_detail_freq == 0)
        if log_detailed:
            # Stack iteration histories
            actions_all = torch.stack(actions_hist, dim=0)               # [I,E,T,A]
            latents_all = torch.stack(latents_hist, dim=0)               # [I,H,E,T+1,L]
            values_all_unscaled = torch.stack(values_unscaled_hist, dim=0)  # [I,E]
            values_all_scaled = torch.stack(values_scaled_hist, dim=0)      # [I,E]
            raw_scores = torch.stack(scores_hist, dim=0)                  # [I,E]
            disagreements_all = torch.stack(disagreement_hist, dim=0) if len(disagreement_hist) > 0 else None  # [I,E]

            # Build advanced info (includes context for post-noise analysis)
            info_adv = PlannerAdvancedInfo(
                **vars(info_basic),
                actions_all=actions_all,
                latents_all=latents_all,
                values_all_unscaled=values_all_unscaled,
                values_all_scaled=values_all_scaled,
                disagreements_all=disagreements_all,
                raw_scores=raw_scores,
                # Context for analysis
                action_seq_chosen=chosen_seq.detach(),
                action_noise=(noise_vec_first.detach() if noise_vec_first is not None else None),
                std_first_action=std_first.detach(),
                z0=z0.detach() if z0.dim() == 1 else z0.squeeze(0).detach(),
                task=task.detach() if (task is not None and torch.is_tensor(task)) else task,
                lambda_d=float(self.cfg.planner_lambda_disagreement) if not eval_mode else 0.0,
                head_mode=('single' if eval_mode else 'all'),
                T=T,
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
