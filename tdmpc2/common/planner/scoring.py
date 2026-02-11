from typing import Optional, Tuple
import torch
from common import math


def compute_values(
    latents_all: torch.Tensor,  # float32[H, B, N, T+1, L]
    actions: torch.Tensor,      # float32[B, N, T, A]
    world_model,
    task=None,
    value_std_coef: float = 0.0,
    use_ema_value: bool = False,
    aggregate_horizon: bool = False,
    return_group_detail: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict]]:
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
        use_ema_value: If True, use EMA target network for V; otherwise use online network.
        aggregate_horizon: If True, compute λ-style compound return averaging over all horizons.
        return_group_detail: If True and G > 1, return per-group Q/V detail for disagreement.

    Returns:
        Tuple[Tensor[B, N], ..., Optional[dict]]:
            (values_unscaled, values_scaled, values_std, value_disagreement, group_detail).
            values_std is the aggregated std per candidate after dynamics reduction.
            group_detail is None unless return_group_detail=True and G > 1, containing:
                q_per_group (Tensor[G, B, N]): Mean Q per group (over H_g within-group heads).
                v_mean_per_group (Tensor[G, B, N]): Mean V per group (over H_g within-group heads).
    """
    H, B, N, Tp1, L = latents_all.shape  # H=dynamics heads, B=batch, N=candidates, Tp1=T+1, L=latent_dim
    T = Tp1 - 1
    cfg = world_model.cfg
    Ve = cfg.num_q  # number of V ensemble heads
    R = cfg.num_reward_heads  # number of reward heads
    G = cfg.num_groups
    H_g = cfg.heads_per_group        # H // G
    R_g = cfg.reward_heads_per_group  # R // G
    Ve_g = cfg.value_heads_per_group  # Ve // G
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

    # =========== Efficient per-group reward (split_data=True) ===========
    # Each reward head r in group g sees only latents from group g's H_g dynamics heads.
    # z_t: [H, B, N, T, L] = [G*H_g, B, N, T, L]
    # Reshape to [G, H_g, B, N, T, L] → [G, H_g*B*N, T, L] → expand to [R, H_g*B*N, T, L]
    z_grouped = z_t.view(G, H_g, B, N, T, L).permute(0, 1, 2, 3, 4, 5)  # [G, H_g, B, N, T, L]
    z_grouped = z_grouped.reshape(G, H_g * B * N, T, L)  # float32[G, H_g*B*N, T, L]
    z_for_R = (
        z_grouped
        .unsqueeze(1)
        .expand(G, R_g, H_g * B * N, T, L)
        .reshape(R, H_g * B * N, T, L)
    )  # float32[R, H_g*B*N, T, L]

    a_grouped = a_t.view(G, H_g, B, N, T, -1).reshape(G, H_g * B * N, T, -1)  # float32[G, H_g*B*N, T, A]
    a_for_R = (
        a_grouped
        .unsqueeze(1)
        .expand(G, R_g, H_g * B * N, T, -1)
        .reshape(R, H_g * B * N, T, -1)
    )  # float32[R, H_g*B*N, T, A]

    # Each reward head processes only its group's latents (no waste)
    rew_logits_all = world_model.reward(z_for_R, a_for_R, task, head_mode='all', split_data=True)
    # float32[R, H_g*B*N, T, K]
    
    # Convert to scalar rewards and reshape to per-group format
    r_all = math.two_hot_inv(rew_logits_all, cfg).squeeze(-1)  # float32[R, H_g*B*N, T]
    r_within = r_all.view(G, R_g, H_g, B, N, T)  # float32[G, R_g, H_g, B, N, T]

    # Per dynamics head h, compute reward mean and std across R_g (within-group) reward heads
    r_mean_per_h = r_within.mean(dim=1).reshape(H, B, N, T)     # float32[H, B, N, T]
    r_std_per_h = r_within.std(dim=1, unbiased=(R_g > 1)).reshape(H, B, N, T)  # float32[H, B, N, T]

    # =========== STEP 2: Get V values (efficient per-group) ===========
    if aggregate_horizon:
        # Aggregate horizon: V at ALL intermediate states z_1, ..., z_T (single batched call)
        # z_inter contains states after each action: z_1=next(z_0,a_0), z_2=next(z_1,a_1), etc.
        z_inter = latents_all[:, :, :, 1:, :]                   # float32[H, B, N, T, L]
        
        # Efficient per-group V: reshape to [Ve, H_g*B*N*T, L] for split_data call
        z_inter_grouped = z_inter.view(G, H_g, B, N, T, L).reshape(G, H_g * B * N * T, L)
        z_for_Ve = (
            z_inter_grouped
            .unsqueeze(1)
            .expand(G, Ve_g, H_g * B * N * T, L)
            .reshape(Ve, H_g * B * N * T, L)
        )  # float32[Ve, H_g*B*N*T, L]
        
        v_logits_all = world_model.V(z_for_Ve, task, return_type='all', target=use_ema_value, split_data=True)
        v_all = math.two_hot_inv(v_logits_all, cfg).squeeze(-1)  # float32[Ve, H_g*B*N*T]
        v_within = v_all.view(G, Ve_g, H_g, B, N, T)  # float32[G, Ve_g, H_g, B, N, T]
        
        v_mean_per_h = v_within.mean(dim=1).reshape(H, B, N, T)  # float32[H, B, N, T]
        v_std_per_h = v_within.std(dim=1, unbiased=(Ve_g > 1)).reshape(H, B, N, T)
    else:
        # Single horizon: V only at terminal z_T (original behavior)
        z_last = latents_all[:, :, :, -1, :]                    # float32[H, B, N, L]
        
        # Efficient per-group V: reshape to [Ve, H_g*B*N, L] for split_data call
        z_last_grouped = z_last.view(G, H_g, B, N, L).reshape(G, H_g * B * N, L)
        z_for_Ve = (
            z_last_grouped
            .unsqueeze(1)
            .expand(G, Ve_g, H_g * B * N, L)
            .reshape(Ve, H_g * B * N, L)
        )  # float32[Ve, H_g*B*N, L]
        
        v_logits_all = world_model.V(z_for_Ve, task, return_type='all', target=use_ema_value, split_data=True)
        v_all = math.two_hot_inv(v_logits_all, cfg).squeeze(-1)  # float32[Ve, H_g*B*N]
        v_within = v_all.view(G, Ve_g, H_g, B, N)  # float32[G, Ve_g, H_g, B, N]
        
        v_mean_per_h = v_within.mean(dim=1).reshape(H, B, N)     # float32[H, B, N]
        v_std_per_h = v_within.std(dim=1, unbiased=(Ve_g > 1)).reshape(H, B, N)

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

    # Optionally compute per-group detail for between-group disagreement metrics
    group_detail = None
    if return_group_detail and G > 1:
        # q_per_group: mean Q per group, collapsing H_g dynamics heads within each group
        q_per_group = q_per_h.view(G, H_g, B, N).mean(dim=1)  # float32[G, B, N]
        # v_mean_per_group: mean V per group (single-horizon or averaged over T for aggregate)
        if aggregate_horizon:
            v_per_group = v_mean_per_h.mean(dim=3).view(G, H_g, B, N).mean(dim=1)  # float32[G, B, N]
        else:
            v_per_group = v_mean_per_h.view(G, H_g, B, N).mean(dim=1)  # float32[G, B, N]
        group_detail = dict(
            q_per_group=q_per_group,          # float32[G, B, N]
            v_mean_per_group=v_per_group,     # float32[G, B, N]
        )

    return values_unscaled, values_scaled, values_std, value_disagreement, group_detail


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
