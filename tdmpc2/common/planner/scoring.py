from typing import Optional, Tuple
import torch
from common import math


def compute_values(
    latents_all: torch.Tensor,  # float32[H, E, T+1, L]
    actions: torch.Tensor,      # float32[E, T, A]
    world_model,
    task=None,
    value_std_coef: float = 0.0,
    reward_head_mode: str = "all",
    use_ema_value: bool = False,
    discount: float = 0.99,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trajectory values using dynamics heads and reward heads.
    
    With V-function (state-only value), we bootstrap using V(z_last) directly
    without needing to sample or provide an action.
    
    Value = sum(rewards over T steps) + γ^T * V(z_last)
    
    CORRECT OPTIMISM: For each dynamics head h, compute:
      σ_h = sum_t(γ^(t-1) * σ^r_{h,t}) + γ^T * σ^v_h
      Q_h = μ_h + value_std_coef × σ_h
    Then reduce over dynamics heads:
      value_std_coef > 0: max over H (optimistic)
      value_std_coef < 0: min over H (pessimistic)
      value_std_coef = 0: mean over H (neutral)

    Args:
        latents_all (Tensor[H, E, T+1, L]): Latent rollouts for all dynamics heads.
        actions (Tensor[E, T, A]): Action sequences aligned with latents.
        world_model: WorldModel exposing reward() and V().
        task: Optional task index for multitask setups.
        value_std_coef: Coefficient for mean + coef × std reduction.
            >0 = optimistic (max over dynamics), <0 = pessimistic (min over dynamics), 0 = neutral (mean).
        reward_head_mode: 'single' uses only head 0, 'all' uses all reward heads.
            During eval, use 'single' to match single dynamics head for fair comparison.
        use_ema_value: If True, use EMA target network for V; otherwise use online network.

    Returns:
        Tuple[Tensor[E], Tensor[E], Tensor[E], Tensor[E]]: 
            (values_unscaled, values_scaled, values_std, value_disagreement).
            values_std is the aggregated std per candidate after dynamics reduction.
    """
    H, E, Tp1, L = latents_all.shape  # H=dynamics heads, E=candidates, Tp1=T+1, L=latent_dim
    T = Tp1 - 1
    cfg = world_model.cfg
    Ve = cfg.num_q  # number of V ensemble heads

    # Precompute discount powers: γ^0, γ^1, ..., γ^(T-1) for rewards, γ^T for bootstrap
    # discount_powers[t] = γ^t for t in [0, T-1]
    device = latents_all.device
    dtype = latents_all.dtype
    discount_powers = torch.pow(torch.tensor(discount, device=device, dtype=dtype), 
                                 torch.arange(T, device=device, dtype=dtype))  # float32[T]
    discount_T = discount ** T  # γ^T for bootstrap value

    # Reward computation across all dynamics heads in parallel.
    z_t = latents_all[:, :, :-1, :]                         # float32[H, E, T, L]
    # Broadcast actions to all dynamics heads.
    a_t = actions.unsqueeze(0).expand(H, -1, -1, -1)        # float32[H, E, T, A]

    # Flatten (dynamics head, candidate) into batch for reward model.
    z_flat = z_t.contiguous().view(H * E, T, L)             # float32[H*E, T, L]
    a_flat = a_t.contiguous().view(H * E, T, -1)            # float32[H*E, T, A]

    # Get reward logits from reward heads: [R, H*E, T, K]
    # R=1 if reward_head_mode='single', R=num_reward_heads if 'all'
    rew_logits_all = world_model.reward(z_flat, a_flat, task, head_mode=reward_head_mode)
    R = rew_logits_all.shape[0]  # number of reward heads
    
    # Convert to scalar rewards: [R, H*E, T, K] -> [R, H*E, T, 1] -> [R, H, E, T]
    r_all = math.two_hot_inv(rew_logits_all, cfg).squeeze(-1)  # float32[R, H*E, T]
    r_all = r_all.view(R, H, E, T)  # float32[R, H, E, T]
    
    # Per dynamics head h, compute reward mean and std across R reward heads at each timestep
    # r_all: [R, H, E, T]
    r_mean_per_h = r_all.mean(dim=0)  # float32[H, E, T] - mean over R
    r_std_per_h = r_all.std(dim=0, unbiased=(R > 1))  # float32[H, E, T] - std over R

    # Get V values for ALL Ve ensemble heads at terminal state
    z_last = latents_all[:, :, -1, :]                       # float32[H, E, L]
    z_last_flat = z_last.contiguous().view(H * E, L)        # float32[H*E, L]
    
    # Get V logits for ALL ensemble heads: [Ve, H*E, K] -> [Ve, H*E] -> [Ve, H, E]
    v_logits_all = world_model.V(z_last_flat, task, return_type='all', target=use_ema_value)  # float32[Ve, H*E, K]
    v_all = math.two_hot_inv(v_logits_all, cfg).squeeze(-1)  # float32[Ve, H*E]
    v_all = v_all.view(Ve, H, E)  # float32[Ve, H, E]
    
    # Per dynamics head h, compute value mean and std across Ve value heads
    # v_all: [Ve, H, E]
    v_mean_per_h = v_all.mean(dim=0)  # float32[H, E] - mean over Ve
    v_std_per_h = v_all.std(dim=0, unbiased=(Ve > 1))  # float32[H, E] - std over Ve
    
    # Compute discounted return mean per dynamics head
    # μ_h = sum_t(γ^(t-1) * r_mean_{h,t}) + γ^T * v_mean_h
    # discount_powers: [T], r_mean_per_h: [H, E, T]
    returns_mean_per_h = (r_mean_per_h * discount_powers).sum(dim=2)  # float32[H, E]
    total_mean_per_h = returns_mean_per_h + discount_T * v_mean_per_h  # float32[H, E]
    
    # Compute aggregated std per dynamics head (sum of discounted stds)
    # σ_h = sum_t(γ^(t-1) * σ^r_{h,t}) + γ^T * σ^v_h
    # discount_powers: [T], r_std_per_h: [H, E, T]
    reward_std_sum_per_h = (r_std_per_h * discount_powers).sum(dim=2)  # float32[H, E]
    total_std_per_h = reward_std_sum_per_h + discount_T * v_std_per_h  # float32[H, E]
    
    # Compute Q_h = μ_h + value_std_coef × σ_h per dynamics head
    q_per_h = total_mean_per_h + value_std_coef * total_std_per_h  # float32[H, E]
    
    # Reduce over dynamics heads based on sign of value_std_coef
    if value_std_coef > 0:
        # Optimistic: max over dynamics heads
        values_unscaled, _ = q_per_h.max(dim=0)  # float32[E]
    elif value_std_coef < 0:
        # Pessimistic: min over dynamics heads
        values_unscaled, _ = q_per_h.min(dim=0)  # float32[E]
    else:
        # Neutral: mean over dynamics heads
        values_unscaled = q_per_h.mean(dim=0)  # float32[E]
    
    values_scaled = values_unscaled  # Same for now (scaling handled elsewhere)
    
    # Value disagreement: std of Q_h across dynamics heads (after std_coef adjustment)
    value_disagreement = q_per_h.std(dim=0, unbiased=(H > 1))  # float32[E]
    
    # Return the aggregated std (mean over dynamics heads) for logging
    values_std = total_std_per_h.mean(dim=0)  # float32[E]

    return values_unscaled, values_scaled, values_std, value_disagreement


def compute_disagreement(final_latents_all: torch.Tensor) -> torch.Tensor:
    """Mean variance across heads at final step.

    Args:
        final_latents_all (Tensor[H, E, L]): Final-step latents for all heads.

    Returns:
        Tensor[E]: Disagreement per candidate (mean over latent dims of variance across heads).
    """
    var = final_latents_all.var(dim=0, unbiased=False)  # float32[E, L]
    return var.mean(dim=1)                             # float32[E]


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
        values_scaled (Tensor[E]): Scaled values (already includes value_std_coef × value_std).
        latent_disagreement (Tensor[E] or None): Latent disagreement signal (variance of latents).
        lambda_latent (float): Weight for latent disagreement term.

    Returns:
        Tensor[E]: Combined scores.
    """
    score = values_scaled  # float32[E]
    if latent_disagreement is not None and lambda_latent != 0.0:
        score = score + lambda_latent * latent_disagreement
    return score
