from typing import Optional, Tuple
import torch
from common import math


def compute_values(
    latents_all: torch.Tensor,  # float32[H, E, T+1, L]
    actions: torch.Tensor,      # float32[E, T, A]
    world_model,
    task=None,
    head_reduce: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trajectory values using all dynamics heads in parallel.
    
    With V-function (state-only value), we bootstrap using V(z_last) directly
    without needing to sample or provide an action.
    
    Value = sum(rewards over T steps) + V(z_last)

    Args:
        latents_all (Tensor[H, E, T+1, L]): Latent rollouts for all heads.
        actions (Tensor[E, T, A]): Action sequences aligned with latents.
        world_model: WorldModel exposing reward() and V().
        task: Optional task index for multitask setups.
        head_reduce: Aggregation over heads: 'mean' or 'max'.

    Returns:
        Tuple[Tensor[E], Tensor[E], Tensor[E]]: (values_unscaled, values_scaled,
        values_std_across_heads).
    """
    H, E, Tp1, L = latents_all.shape  # H=heads, E=candidates, Tp1=T+1, L=latent_dim
    T = Tp1 - 1
    cfg = world_model.cfg

    # Reward computation across all heads in parallel.
    z_t = latents_all[:, :, :-1, :]                         # float32[H, E, T, L]
    # Broadcast actions to all heads.
    a_t = actions.unsqueeze(0).expand(H, -1, -1, -1)        # float32[H, E, T, A]

    # Flatten (head, candidate) into batch for reward model.
    z_flat = z_t.contiguous().view(H * E, T, L)             # float32[H*E, T, L]
    a_flat = a_t.contiguous().view(H * E, T, -1)            # float32[H*E, T, A]

    rew_logits = world_model.reward(z_flat, a_flat, task)   # float32[H*E, T, K] or [H*E, T, 1]
    r_flat = math.two_hot_inv(rew_logits, cfg).squeeze(-1)  # float32[H*E, T]
    r_t = r_flat.view(H, E, T)                              # float32[H, E, T]

    # Sum rewards (undiscounted) and bootstrap with V(z_last).
    # With V-function, no action needed for bootstrapping.
    returns_he = r_t.sum(dim=2)                             # float32[H, E] sum of rewards
    z_last = latents_all[:, :, -1, :]                       # float32[H, E, L]
    z_last_flat = z_last.contiguous().view(H * E, L)        # float32[H*E, L]
    
    v_boot_flat = world_model.V(z_last_flat, task, return_type=head_reduce, target=True)
    v_boot_flat = v_boot_flat.squeeze(-1).squeeze(-1)       # float32[H*E]
    v_boot_he = v_boot_flat.view(H, E)                      # float32[H, E]

    values_unscaled_he = returns_he + v_boot_he             # float32[H, E]
    values_scaled_he = values_unscaled_he * 1.0        # float32[H, E]

    # Std across heads (independent of reduction mode).
    values_std = values_unscaled_he.std(dim=0, unbiased=False)  # float32[E]

    if head_reduce == "mean":
        values_unscaled = values_unscaled_he.mean(dim=0)        # float32[E]
        values_scaled = values_scaled_he.mean(dim=0)            # float32[E]
    elif head_reduce == "max":
        values_unscaled = values_unscaled_he.max(dim=0).values  # float32[E]
        values_scaled = values_scaled_he.max(dim=0).values      # float32[E]
    else:
        raise ValueError(f"Invalid head_reduce '{head_reduce}'. Expected 'mean' or 'max'.")

    return values_unscaled, values_scaled, values_std


def compute_disagreement(final_latents_all: torch.Tensor) -> torch.Tensor:
    """Mean variance across heads at final step.

    Args:
        final_latents_all (Tensor[H, E, L]): Final-step latents for all heads.

    Returns:
        Tensor[E]: Disagreement per candidate (mean over latent dims of variance across heads).
    """
    var = final_latents_all.var(dim=0, unbiased=False)  # float32[E, L]
    return var.mean(dim=1)                             # float32[E]


def combine_scores(values_scaled: torch.Tensor, disagreements: Optional[torch.Tensor], lambda_coeff: float) -> torch.Tensor:
    """Combine scaled values and (optional) disagreement into final score.

    Args:
        values_scaled (Tensor[E]): Scaled values.
        disagreements (Tensor[E] or None): Disagreement signal.
        lambda_coeff (float): Weight for disagreement term.

    Returns:
        Tensor[E]: Combined scores.
    """
    if disagreements is None or lambda_coeff == 0.0:
        return values_scaled  # float32[E]
    return values_scaled + lambda_coeff * disagreements  # float32[E]
