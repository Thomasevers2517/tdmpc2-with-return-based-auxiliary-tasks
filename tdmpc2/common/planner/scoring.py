from typing import Optional, Tuple
import torch
from common import math


def compute_values(
    latents_all: torch.Tensor,  # float32[H, E, T+1, L]
    actions: torch.Tensor,      # float32[E, T, A]
    world_model,
    task=None,
    head_reduce: str = "mean",
    reward_head_mode: str = "all",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trajectory values using dynamics heads and reward heads.
    
    With V-function (state-only value), we bootstrap using V(z_last) directly
    without needing to sample or provide an action.
    
    Value = sum(rewards over T steps) + V(z_last)
    
    Reduction is applied to both reward heads (R) and dynamics heads (H) using
    the same head_reduce mode (typically 'max' for optimistic planning).
    
    Value disagreement is computed as std across all (dynamics head × V head) 
    combinations, capturing joint uncertainty from both sources. This is most
    meaningful when planner_num_dynamics_heads > 1 and/or num_q > 1.

    Args:
        latents_all (Tensor[H, E, T+1, L]): Latent rollouts for all dynamics heads.
        actions (Tensor[E, T, A]): Action sequences aligned with latents.
        world_model: WorldModel exposing reward() and V().
        task: Optional task index for multitask setups.
        head_reduce: Aggregation over heads: 'mean' or 'max'. Applied to both R and H.
        reward_head_mode: 'single' uses only head 0, 'all' uses all reward heads.
            During eval, use 'single' to match single dynamics head for fair comparison.

    Returns:
        Tuple[Tensor[E], Tensor[E], Tensor[E], Tensor[E]]: 
            (values_unscaled, values_scaled, values_std_across_dynamics_heads, value_disagreement).
            value_disagreement is std across all (H × Q) value estimates per candidate.
    """
    H, E, Tp1, L = latents_all.shape  # H=dynamics heads, E=candidates, Tp1=T+1, L=latent_dim
    T = Tp1 - 1
    cfg = world_model.cfg
    Q = cfg.num_q  # number of V ensemble heads

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
    
    # Convert to scalar rewards: [R, H*E, T, K] -> [R, H*E, T]
    r_all = math.two_hot_inv(rew_logits_all, cfg).squeeze(-1)  # float32[R, H*E, T]
    # Reshape to [R, H, E, T]
    r_all = r_all.view(R, H, E, T)  # float32[R, H, E, T]
    
    # Reduce over reward heads (R) per timestep using head_reduce
    # When R=1 (single head mode), reduction is a no-op
    if head_reduce == "mean":
        r_t = r_all.mean(dim=0)  # float32[H, E, T]
    elif head_reduce == "max":
        r_t = r_all.max(dim=0).values  # float32[H, E, T]
    else:
        raise ValueError(f"Invalid head_reduce '{head_reduce}'. Expected 'mean' or 'max'.")

    # Sum rewards (undiscounted) over T and bootstrap with V(z_last).
    # With V-function, no action needed for bootstrapping.
    returns_he = r_t.sum(dim=2)                             # float32[H, E] sum of rewards
    z_last = latents_all[:, :, -1, :]                       # float32[H, E, L]
    z_last_flat = z_last.contiguous().view(H * E, L)        # float32[H*E, L]
    
    # Get V logits for ALL ensemble heads to compute value disagreement
    v_logits_all = world_model.V(z_last_flat, task, return_type='all', target=True)  # float32[Q, H*E, K]
    v_all_qhe = math.two_hot_inv(v_logits_all, cfg).squeeze(-1)  # float32[Q, H*E]
    
    # Reshape to [Q, H, E] then flatten to [Q*H, E] for disagreement computation
    v_all_qhe = v_all_qhe.view(Q, H, E)  # float32[Q, H, E]
    v_all_flat = v_all_qhe.view(Q * H, E)  # float32[Q*H, E]
    
    # Value disagreement: std across all Q*H estimates per candidate
    value_disagreement = v_all_flat.std(dim=0, unbiased=False)  # float32[E]
    
    # Bootstrap value: reduce over Q (V heads) using head_reduce, keep H dimension
    if head_reduce == "mean":
        v_boot_he = v_all_qhe.mean(dim=0)  # float32[H, E]
    elif head_reduce == "max":
        v_boot_he = v_all_qhe.max(dim=0).values  # float32[H, E]
    else:
        raise ValueError(f"Invalid head_reduce '{head_reduce}'. Expected 'mean' or 'max'.")

    values_unscaled_he = returns_he + v_boot_he             # float32[H, E]
    values_scaled_he = values_unscaled_he * 1.0             # float32[H, E]

    # Std across dynamics heads (independent of reduction mode).
    values_std = values_unscaled_he.std(dim=0, unbiased=False)  # float32[E]

    # Reduce over dynamics heads (H) using head_reduce
    if head_reduce == "mean":
        values_unscaled = values_unscaled_he.mean(dim=0)        # float32[E]
        values_scaled = values_scaled_he.mean(dim=0)            # float32[E]
    elif head_reduce == "max":
        values_unscaled = values_unscaled_he.max(dim=0).values  # float32[E]
        values_scaled = values_scaled_he.max(dim=0).values      # float32[E]
    else:
        raise ValueError(f"Invalid head_reduce '{head_reduce}'. Expected 'mean' or 'max'.")

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
    value_disagreement: Optional[torch.Tensor],
    lambda_latent: float,
    lambda_value: float,
) -> torch.Tensor:
    """Combine scaled values with optional disagreement signals into final score.
    
    score = values_scaled + lambda_latent * latent_disagreement + lambda_value * value_disagreement
    
    Latent disagreement captures uncertainty from dynamics model (different predicted 
    latent states). Value disagreement captures uncertainty from value estimates across 
    all dynamics×V head combinations.

    Args:
        values_scaled (Tensor[E]): Scaled values.
        latent_disagreement (Tensor[E] or None): Latent disagreement signal (variance of latents).
        value_disagreement (Tensor[E] or None): Value disagreement signal (std of V estimates).
        lambda_latent (float): Weight for latent disagreement term.
        lambda_value (float): Weight for value disagreement term.

    Returns:
        Tensor[E]: Combined scores.
    """
    score = values_scaled  # float32[E]
    if latent_disagreement is not None and lambda_latent != 0.0:
        score = score + lambda_latent * latent_disagreement
    if value_disagreement is not None and lambda_value != 0.0:
        score = score + lambda_value * value_disagreement
    return score
