from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class PlannerBasicInfo:
    """Basic planner summary metrics.

    Attributes:
        value_chosen: Scalar value of the chosen sequence.
        disagreement_chosen: Latent disagreement of chosen (None if not computed).
        score_chosen: Combined score of the chosen sequence.
        weighted_disagreement_chosen: Weighted disagreement (lambda_d * disagreement) of chosen (None if not computed).
        value_elite_mean: Mean value over elite set.
        value_elite_std: Std of value over elite set.
        value_elite_max: Max value over elite set.
        disagreement_elite_mean: Mean disagreement over elites (None if not computed).
        disagreement_elite_std: Std disagreement over elites (None if not computed).
        disagreement_elite_max: Max disagreement over elites (None if not computed).
        num_elites: Number of elites used for updates.
        elite_indices: Indices of elites within candidate set; int64[E].
        scores_all: Combined scores for all candidates; Tensor[E].
        values_scaled_all: Scaled values for all candidates; Tensor[E].
        weighted_disagreements_all: Weighted disagreements for all candidates (None if not computed); Tensor[E].
    """
    value_chosen: torch.Tensor  # 0-dim tensor
    disagreement_chosen: Optional[torch.Tensor]
    score_chosen: torch.Tensor  # 0-dim tensor
    weighted_disagreement_chosen: Optional[torch.Tensor]
    value_elite_mean: torch.Tensor
    value_elite_std: torch.Tensor
    value_elite_max: torch.Tensor
    disagreement_elite_mean: Optional[torch.Tensor]
    disagreement_elite_std: Optional[torch.Tensor]
    disagreement_elite_max: Optional[torch.Tensor]
    num_elites: int
    elite_indices: torch.Tensor  # int64[E]
    scores_all: torch.Tensor  # (E,)
    values_scaled_all: torch.Tensor  # (E,)
    weighted_disagreements_all: Optional[torch.Tensor]  # (E,)


@dataclass
class PlannerAdvancedInfo(PlannerBasicInfo):
    """Full planner payload for detailed logging across iterations.

    Iteration-stacked tensors capture the candidate set at each refinement
    step. Shapes below use I=iterations, E=candidates (N+S), H=heads,
    T=horizon, A=action_dim, L=latent_dim.

    Attributes:
        actions_all: float32[I, E, T, A]
        latents_all: float32[I, H, E, T+1, L]
        values_all_unscaled: float32[I, E]
        values_all_scaled: float32[I, E]
        disagreements_all: Optional[torch.Tensor]  # float32[I, E]
        raw_scores: float32[I, E]
    """
    actions_all: torch.Tensor
    latents_all: torch.Tensor
    values_all_unscaled: torch.Tensor
    values_all_scaled: torch.Tensor
    disagreements_all: Optional[torch.Tensor]
    raw_scores: torch.Tensor
