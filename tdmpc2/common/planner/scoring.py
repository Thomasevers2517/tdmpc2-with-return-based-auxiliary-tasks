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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trajectory values using dynamics heads and reward heads.
    
    With V-function (state-only value), we bootstrap using V(z_last) directly
    without needing to sample or provide an action.
    
    Value = sum(rewards over T steps) + V(z_last)
    
    We compute values for ALL (R × H × Ve) combinations, then reduce via:
      reduced_value = mean + value_std_coef × std
    
    This provides gradients through the std term, allowing the planner to
    be uncertainty-aware with value_std_coef > 0 (optimistic) or < 0 (pessimistic).
    
    Value disagreement is computed as std across all (R × H × Ve) combinations.

    Args:
        latents_all (Tensor[H, E, T+1, L]): Latent rollouts for all dynamics heads.
        actions (Tensor[E, T, A]): Action sequences aligned with latents.
        world_model: WorldModel exposing reward() and V().
        task: Optional task index for multitask setups.
        value_std_coef: Coefficient for mean + coef × std reduction.
            +1.0 = optimistic (mean + std), -1.0 = pessimistic (mean - std), 0.0 = mean only.
        reward_head_mode: 'single' uses only head 0, 'all' uses all reward heads.
            During eval, use 'single' to match single dynamics head for fair comparison.
        use_ema_value: If True, use EMA target network for V; otherwise use online network.

    Returns:
        Tuple[Tensor[E], Tensor[E], Tensor[E], Tensor[E]]: 
            (values_unscaled, values_scaled, values_std, value_disagreement).
            values_std and value_disagreement are both std across all (R × H × Ve) combinations.
    """
    H, E, Tp1, L = latents_all.shape  # H=dynamics heads, E=candidates, Tp1=T+1, L=latent_dim
    T = Tp1 - 1
    cfg = world_model.cfg
    Ve = cfg.num_q  # number of V ensemble heads

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
    
    # Sum rewards over T per (R, H) combination: [R, H, E]
    returns_rhe = r_all.sum(dim=3)  # float32[R, H, E]

    # Get V values for ALL Ve ensemble heads at terminal state
    z_last = latents_all[:, :, -1, :]                       # float32[H, E, L]
    z_last_flat = z_last.contiguous().view(H * E, L)        # float32[H*E, L]
    
    # Get V logits for ALL ensemble heads: [Ve, H*E, K] -> [Ve, H*E] -> [Ve, H, E]
    # use_ema_value=True uses target network (slower-moving, more stable)
    # use_ema_value=False uses online network (current estimates)
    v_logits_all = world_model.V(z_last_flat, task, return_type='all', target=use_ema_value)  # float32[Ve, H*E, K]
    v_all = math.two_hot_inv(v_logits_all, cfg).squeeze(-1)  # float32[Ve, H*E]
    v_all = v_all.view(Ve, H, E)  # float32[Ve, H, E]
    
    # Compute total value for ALL (R × H × Ve) combinations
    # returns_rhe: [R, H, E] -> [1, R, H, E] for broadcasting
    # v_all: [Ve, H, E] -> [Ve, 1, H, E] for broadcasting
    returns_exp = returns_rhe.unsqueeze(0)  # float32[1, R, H, E]
    v_exp = v_all.unsqueeze(1)  # float32[Ve, 1, H, E]
    
    # Total value = reward returns + bootstrap V for each (Ve, R, H) combination
    values_all = returns_exp + v_exp  # float32[Ve, R, H, E] via broadcasting
    
    # Flatten (Ve × R × H) and compute mean + std_coef × std reduction
    values_flat = values_all.view(Ve * R * H, E)  # float32[Ve*R*H, E]
    
    values_mean = values_flat.mean(dim=0)  # float32[E]
    values_std = values_flat.std(dim=0, unbiased=False)  # float32[E]
    
    # Reduce: mean + value_std_coef × std
    values_unscaled = values_mean + value_std_coef * values_std  # float32[E]
    values_scaled = values_unscaled  # Same for now (scaling handled elsewhere)
    
    # Value disagreement = std across all (Ve × R × H) combinations
    value_disagreement = values_std  # Same as values_std now

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
