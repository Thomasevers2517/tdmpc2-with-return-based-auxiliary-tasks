from typing import Optional, Tuple
import torch
from common import math


def compute_values_head0(latents_head0: torch.Tensor, actions: torch.Tensor, world_model, task=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute trajectory values using head 0 latents and bootstrap via Q.

    Args:
        latents_head0 (Tensor[E, T+1, L]): Latent rollout for head 0 only.
        actions (Tensor[E, T, A]): Action sequences aligned with latents.
        world_model: WorldModel instance exposing reward(), pi(), and Q().
        task: Optional task index for multitask setups (unsupported here).

    Returns:
        Tuple[Tensor[E], Tensor[E]]: (values_unscaled, values_scaled). Currently scaled == unscaled.
    """
    # Shapes
    E, Tp1, L = latents_head0.shape
    T = Tp1 - 1
    device = latents_head0.device
    dtype = latents_head0.dtype

    # Sum per-step rewards (distributional -> scalar via two_hot_inv)
    z_t = latents_head0[:, :-1, :]  # [E,T,L]
    a_t = actions  # [E,T,A]
    rew_logits = world_model.reward(z_t, a_t, task)  # expect [E,T,K] or [E,T,1]
    # two_hot_inv expects shape [*, K]; we let it handle trailing dims
    r_t = math.two_hot_inv(rew_logits, world_model.cfg).squeeze(-1)  # [E,T]
    returns = r_t.sum(dim=1)  # [E]

    # Bootstrap from last latent via policy prior + target Q if available
    z_last = latents_head0[:, -1, :]  # [E,L]
    #TODO Fix bootstrapping with policy task, should be an action that is optimized....
    a_boot, _ = world_model.pi(z_last, task, use_ema=getattr(world_model.cfg, 'policy_ema_enabled', False))
    q_boot = world_model.Q(z_last, a_boot, task, return_type='min', target=True)
    q_boot = q_boot.squeeze(-1).squeeze(-1) if q_boot.ndim > 1 else q_boot  # [E]

    values_unscaled = returns + q_boot
    values_scaled = values_unscaled * 1.0  # TODO: optional scaling via RunningScale; keep as identity for now
    return values_unscaled, values_scaled


def compute_disagreement(final_latents_all: torch.Tensor) -> torch.Tensor:
    """Mean variance across heads at final step.

    Args:
        final_latents_all (Tensor[H, E, L]): Final-step latents for all heads.

    Returns:
        Tensor[E]: Disagreement per candidate (mean over latent dims of variance across heads).
    """
    var = final_latents_all.var(dim=0, unbiased=False)  # [E,L]
    return var.mean(dim=1)  # [E]


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
        return values_scaled
    return values_scaled + lambda_coeff * disagreements
