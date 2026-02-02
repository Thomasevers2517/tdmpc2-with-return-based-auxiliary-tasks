from typing import Optional, Tuple
import torch
from common import math


def compute_values(
    latents_all: torch.Tensor,  # float32[H, B, N, T+1, L]
    actions: torch.Tensor,      # float32[B, N, T, A]
    world_model,
    task=None,
    value_std_coef: float = 0.0,
    reward_head_mode: str = "all",
    use_ema_value: bool = False,
    aggregate_horizon: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trajectory values using dynamics heads and reward heads.
    
    With V-function (state-only value), we bootstrap using V(z_last) directly
    without needing to sample or provide an action.
    
    For episodic tasks, rewards and bootstrap are weighted by alive probability:
      Value = sum_t(alive_probs[t] * r_t) + alive_probs[T] * V(z_last)
    where alive_probs[t] = Π_{s=0}^{t-1}(1 - p_term(z_s))
    
    For non-episodic tasks: Value = sum(rewards) + V(z_last)
    
    CORRECT OPTIMISM: For each dynamics head h, compute:
      σ_h = sqrt(sum_t(alive[t]² × σ^r_{h,t}²) + alive[T]² × σ^v_h²)
      Q_h = μ_h + value_std_coef × σ_h
    Then reduce over dynamics heads:
      value_std_coef > 0: max over H (optimistic)
      value_std_coef < 0: min over H (pessimistic)
      value_std_coef = 0: mean over H (neutral)
    
    AGGREGATE HORIZON (λ-style compound return):
    When aggregate_horizon=True, instead of only bootstrapping at z_T, we compute
    returns at each intermediate horizon τ ∈ {1, ..., T}:
      G_τ = Σ_{t=0}^{τ-1}(α_t × r_t) + α_τ × V(z_τ)
    And take the uniform average: G = (1/T) Σ_τ G_τ
    
    This reduces model exploitation by averaging over different bootstrap depths.
    Uncertainty is propagated correctly using cumulative variance.

    Args:
        latents_all (Tensor[H, B, N, T+1, L]): Latent rollouts for all dynamics heads.
            H = dynamics heads, B = batch, N = candidates per batch, T+1 = timesteps, L = latent_dim.
        actions (Tensor[B, N, T, A]): Action sequences aligned with latents.
        world_model: WorldModel exposing reward() and V().
        task: Optional task index for multitask setups.
        value_std_coef: Coefficient for mean + coef × std reduction.
            >0 = optimistic (max over dynamics), <0 = pessimistic (min over dynamics), 0 = neutral (mean).
        reward_head_mode: 'single' uses only head 0, 'all' uses all reward heads.
            During eval, use 'single' to match single dynamics head for fair comparison.
        use_ema_value: If True, use EMA target network for V; otherwise use online network.
        aggregate_horizon: If True, compute λ-style compound return averaging over all horizons.

    Returns:
        Tuple[Tensor[B, N], Tensor[B, N], Tensor[B, N], Tensor[B, N]]: 
            (values_unscaled, values_scaled, values_std, value_disagreement).
            values_std is the aggregated std per candidate after dynamics reduction.
    """
    H, B, N, Tp1, L = latents_all.shape  # H=dynamics heads, B=batch, N=candidates, Tp1=T+1, L=latent_dim
    T = Tp1 - 1
    cfg = world_model.cfg
    Ve = cfg.num_q  # number of V ensemble heads
    device = latents_all.device
    dtype = latents_all.dtype

    # =========== STEP 1: Rewards & alive probs (SHARED) ===========
    # Reward computation across all dynamics heads in parallel.
    z_t = latents_all[:, :, :, :-1, :]                      # float32[H, B, N, T, L]
    # Broadcast actions to all dynamics heads: [B, N, T, A] -> [H, B, N, T, A]
    a_t = actions.unsqueeze(0).expand(H, -1, -1, -1, -1)    # float32[H, B, N, T, A]

    # Flatten (dynamics head, batch, candidate) into batch for reward model.
    z_flat = z_t.contiguous().view(H * B * N, T, L)         # float32[H*B*N, T, L]
    a_flat = a_t.contiguous().view(H * B * N, T, -1)        # float32[H*B*N, T, A]

    # Compute cumulative alive probabilities for episodic tasks
    # alive_probs[t] = probability of being alive at timestep t = Π_{s=0}^{t-1}(1 - p_term(z_s))
    # alive_probs[0] = 1 (always alive at start), alive_probs[T] = prob alive for bootstrap
    if cfg.episodic:
        # Get termination probabilities at each step
        z_for_term = z_flat.view(H * B * N * T, L)
        term_probs_flat = world_model.termination(z_for_term, task)  # float32[H*B*N*T, 1]
        term_probs = term_probs_flat.view(H, B, N, T)  # float32[H, B, N, T]
        
        # Cumulative product of survival: cumprod(1 - term_probs)
        cumulative_survival = torch.cumprod(1 - term_probs, dim=3)  # float32[H, B, N, T]
        
        # Prepend 1 to get alive_probs at each timestep (including t=0 and t=T)
        ones = torch.ones(H, B, N, 1, device=device, dtype=dtype)
        alive_probs = torch.cat([ones, cumulative_survival], dim=3)  # float32[H, B, N, T+1]
    else:
        # Non-episodic: always alive
        alive_probs = torch.ones(H, B, N, T + 1, device=device, dtype=dtype)

    # Get reward logits from reward heads: [R, H*B*N, T, K]
    # R=1 if reward_head_mode='single', R=num_reward_heads if 'all'
    rew_logits_all = world_model.reward(z_flat, a_flat, task, head_mode=reward_head_mode)
    R = rew_logits_all.shape[0]  # number of reward heads
    
    # Convert to scalar rewards: [R, H*B*N, T, K] -> [R, H*B*N, T, 1] -> [R, H, B, N, T]
    r_all = math.two_hot_inv(rew_logits_all, cfg).squeeze(-1)  # float32[R, H*B*N, T]
    r_all = r_all.view(R, H, B, N, T)  # float32[R, H, B, N, T]
    
    # Per dynamics head h, compute reward mean and std across R reward heads at each timestep
    # r_all: [R, H, B, N, T]
    r_mean_per_h = r_all.mean(dim=0)  # float32[H, B, N, T] - mean over R
    r_std_per_h = r_all.std(dim=0, unbiased=(R > 1))  # float32[H, B, N, T] - std over R

    # =========== STEP 2: Get V values ===========
    if aggregate_horizon:
        # Aggregate horizon: V at ALL intermediate states z_1, ..., z_T (single batched call)
        # z_inter contains states after each action: z_1=next(z_0,a_0), z_2=next(z_1,a_1), etc.
        z_inter = latents_all[:, :, :, 1:, :]                   # float32[H, B, N, T, L]
        z_inter_flat = z_inter.contiguous().view(H * B * N * T, L)  # float32[H*B*N*T, L]
        
        # Get V logits for ALL ensemble heads at all horizons: [Ve, H*B*N*T, K]
        v_logits_all = world_model.V(z_inter_flat, task, return_type='all', target=use_ema_value)
        v_all = math.two_hot_inv(v_logits_all, cfg).squeeze(-1)  # float32[Ve, H*B*N*T]
        v_all = v_all.view(Ve, H, B, N, T)  # float32[Ve, H, B, N, T]
        
        # Per dynamics head h, compute value mean and std across Ve value heads at each horizon
        v_mean_per_h = v_all.mean(dim=0)  # float32[H, B, N, T] - mean over Ve
        v_std_per_h = v_all.std(dim=0, unbiased=(Ve > 1))  # float32[H, B, N, T] - std over Ve
    else:
        # Single horizon: V only at terminal z_T (original behavior)
        z_last = latents_all[:, :, :, -1, :]                    # float32[H, B, N, L]
        z_last_flat = z_last.contiguous().view(H * B * N, L)    # float32[H*B*N, L]
        
        # Get V logits for ALL ensemble heads: [Ve, H*B*N, K] -> [Ve, H*B*N] -> [Ve, H, B, N]
        v_logits_all = world_model.V(z_last_flat, task, return_type='all', target=use_ema_value)
        v_all = math.two_hot_inv(v_logits_all, cfg).squeeze(-1)  # float32[Ve, H*B*N]
        v_all = v_all.view(Ve, H, B, N)  # float32[Ve, H, B, N]
        
        # Per dynamics head h, compute value mean and std across Ve value heads
        v_mean_per_h = v_all.mean(dim=0)  # float32[H, B, N] - mean over Ve
        v_std_per_h = v_all.std(dim=0, unbiased=(Ve > 1))  # float32[H, B, N] - std over Ve

    # =========== STEP 3: Compute returns ===========
    if aggregate_horizon:
        # Aggregate horizon: compute G_τ for each horizon τ ∈ {1, ..., T}, then average
        # G_τ = Σ_{t=0}^{τ-1}(α_t × r_t) + α_τ × V(z_τ)
        
        # Cumulative weighted rewards: cumsum_r[τ] = Σ_{t=0}^{τ-1}(α_t × r_t)
        # Index τ-1 in 0-indexed array gives sum of rewards up to step τ-1
        weighted_r = r_mean_per_h * alive_probs[:, :, :, :T]  # float32[H, B, N, T]
        cumsum_r = weighted_r.cumsum(dim=3)  # float32[H, B, N, T] - cumsum_r[τ-1] = Σ_{t=0}^{τ-1}(...)
        
        # Alive probs for bootstrap at each horizon: α_1, α_2, ..., α_T
        alive_bootstrap = alive_probs[:, :, :, 1:T+1]  # float32[H, B, N, T]
        
        # Returns mean at each horizon: G_τ = cumsum_r[τ-1] + α_τ × V(z_τ)
        # cumsum_r[τ-1] is cumsum_r[:,:,:,τ-1], and we want τ=1..T, so indices 0..T-1
        returns_mean_per_h_per_tau = cumsum_r + alive_bootstrap * v_mean_per_h  # float32[H, B, N, T]
        
        # Variance at each horizon: cumulative reward variance + bootstrap variance
        # σ²_G_τ = Σ_{t=0}^{τ-1}(α_t × σ^r_t)² + (α_τ × σ^v_τ)²
        weighted_r_var = (r_std_per_h * alive_probs[:, :, :, :T]) ** 2  # float32[H, B, N, T]
        cumvar_r = weighted_r_var.cumsum(dim=3)  # float32[H, B, N, T]
        total_var_per_h_per_tau = cumvar_r + (alive_bootstrap * v_std_per_h) ** 2  # float32[H, B, N, T]
        total_std_per_h_per_tau = total_var_per_h_per_tau.sqrt()  # float32[H, B, N, T]
        
        # Q_{h,τ} = G_{h,τ} + value_std_coef × σ_{h,τ}
        q_per_h_per_tau = returns_mean_per_h_per_tau + value_std_coef * total_std_per_h_per_tau  # float32[H, B, N, T]
        
        # Average over horizons: Q_h = (1/T) Σ_τ Q_{h,τ}
        q_per_h = q_per_h_per_tau.mean(dim=3)  # float32[H, B, N]
        total_std_per_h = total_std_per_h_per_tau.mean(dim=3)  # float32[H, B, N]
    else:
        # Single horizon: original computation
        # μ_h = sum_t(alive_probs[t] * r_mean_{h,t}) + alive_probs[T] * v_mean_h
        # alive_probs[:,:,:,:T] for rewards, alive_probs[:,:,:,T] for bootstrap
        returns_mean_per_h = (r_mean_per_h * alive_probs[:, :, :, :T]).sum(dim=3)  # float32[H, B, N]
        total_mean_per_h = returns_mean_per_h + alive_probs[:, :, :, T] * v_mean_per_h  # float32[H, B, N]
        
        # σ_h = sqrt(sum_t(alive_probs[t]² * σ^r_{h,t}²) + alive_probs[T]² * σ^v_h²)
        reward_var_sum_per_h = ((r_std_per_h * alive_probs[:, :, :, :T]) ** 2).sum(dim=3)  # float32[H, B, N]
        total_var_per_h = reward_var_sum_per_h + (alive_probs[:, :, :, T] * v_std_per_h) ** 2  # float32[H, B, N]
        total_std_per_h = total_var_per_h.sqrt()  # float32[H, B, N]
        
        # Q_h = μ_h + value_std_coef × σ_h
        q_per_h = total_mean_per_h + value_std_coef * total_std_per_h  # float32[H, B, N]

    # =========== STEP 4: Reduce over dynamics heads (SHARED) ===========
    if value_std_coef > 0:
        # Optimistic: max over dynamics heads
        values_unscaled, _ = q_per_h.max(dim=0)  # float32[B, N]
    elif value_std_coef < 0:
        # Pessimistic: min over dynamics heads
        values_unscaled, _ = q_per_h.min(dim=0)  # float32[B, N]
    else:
        # Neutral: mean over dynamics heads
        values_unscaled = q_per_h.mean(dim=0)  # float32[B, N]
    
    values_scaled = values_unscaled  # Same for now (scaling handled elsewhere)
    
    # Value disagreement: std of Q_h across dynamics heads (after std_coef adjustment)
    value_disagreement = q_per_h.std(dim=0, unbiased=(H > 1))  # float32[B, N]
    
    # Return the aggregated std (mean over dynamics heads) for logging
    values_std = total_std_per_h.mean(dim=0)  # float32[B, N]

    return values_unscaled, values_scaled, values_std, value_disagreement


def compute_disagreement(final_latents_all: torch.Tensor) -> torch.Tensor:
    """Mean variance across heads at final step.

    Args:
        final_latents_all (Tensor[H, B, N, L]): Final-step latents for all heads.

    Returns:
        Tensor[B, N]: Disagreement per candidate (mean over latent dims of variance across heads).
    """
    var = final_latents_all.var(dim=0, unbiased=False)  # float32[B, N, L]
    return var.mean(dim=2)                              # float32[B, N]


def combine_scores(
    values_scaled: torch.Tensor,
    latent_disagreement: Optional[torch.Tensor],
    lambda_latent: float,
) -> torch.Tensor:
    """Combine scaled values with optional latent disagreement signal into final score.
    
    score = values_scaled + lambda_latent * latent_disagreement
    
    Latent disagreement captures uncertainty from dynamics model (different predicted 
    latent states). Value disagreement is now incorporated directly into values_scaled
    via the value_std_coef parameter in compute_values.

    Args:
        values_scaled (Tensor[B, N]): Scaled values (already includes value_std_coef × value_std).
        latent_disagreement (Tensor[B, N] or None): Latent disagreement signal (variance of latents).
        lambda_latent (float): Weight for latent disagreement term.

    Returns:
        Tensor[B, N]: Combined scores.
    """
    score = values_scaled  # float32[B, N]
    if latent_disagreement is not None and lambda_latent != 0.0:
        score = score + lambda_latent * latent_disagreement
    return score
