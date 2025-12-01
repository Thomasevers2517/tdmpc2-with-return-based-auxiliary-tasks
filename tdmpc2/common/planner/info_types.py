from dataclasses import dataclass
from typing import Optional
import torch
from .scoring import compute_values, compute_disagreement, combine_scores
import torch._dynamo as dynamo


def _post_noise_effects_impl(world_model, z0: torch.Tensor, noisy_seq: torch.Tensor, head_mode: str, task, lambda_d: float):
    """Core kernel to roll out a single noisy sequence and score it.

    Returns (value_scaled: Tensor[1], disagreement: Optional[Tensor[1]], score: Tensor[1]).
    """
    lat_all, _ = world_model.rollout_latents(
        z0,
        actions=noisy_seq.unsqueeze(0).unsqueeze(0),  # [1,1,T,A]
        use_policy=False,
        head_mode=head_mode,
        task=task,
    )  # [H,1,1,T+1,L]
    lat_all = lat_all[:, 0]            # [H,1,T+1,L]
    # compute_values expects latents_all [H,E,T+1,L] and actions [E,T,A]; here E=1
    vals_unscaled, vals_scaled, vals_std = compute_values(
        lat_all,
        noisy_seq.unsqueeze(0),
        world_model,
        task,
        head_reduce=world_model.cfg.planner_head_reduce,
    )
    dis = None
    if lat_all.shape[0] > 1:
        final_all = lat_all[:, 0, -1, :]  # [H,L]
        # Compute disagreement with E=1; returns shape [1]
        dis = compute_disagreement(final_all.unsqueeze(1))
    score = combine_scores(vals_scaled, dis, lambda_d)
    return vals_scaled, dis, score


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

    # Analysis context (advanced-only)
    action_seq_chosen: torch.Tensor  # (T,A)
    action_noise: Optional[torch.Tensor]  # (A,) applied at t=0; None if no noise
    std_first_action: torch.Tensor  # (A,)
    z0: torch.Tensor  # (L,)
    task: Optional[torch.Tensor]
    lambda_d: float
    head_mode: str
    T: int

    # Outputs of post-noise analysis (set by compute_post_noise_effects)
    value_chosen_post_noise: Optional[torch.Tensor] = None
    disagreement_chosen_post_noise: Optional[torch.Tensor] = None
    score_chosen_post_noise: Optional[torch.Tensor] = None

    def compute_post_noise_effects(self, world_model) -> None:
        """Compute value/disagreement/score for the chosen sequence after noise.

        Runs under no-grad on the current CUDA device. Uses world_model.rollout_latents
        to evaluate the full noisy sequence across heads, then computes scores via
        planner scoring utilities.
        """
        if self.action_noise is None:
            # No noise to analyze
            return
        device = self.action_seq_chosen.device
        # Build noisy sequence: add noise to first step only
        T, A = self.action_seq_chosen.shape
        noisy_seq = self.action_seq_chosen.clone()
        noisy_seq[0] = (noisy_seq[0] + self.action_noise).clamp(-1.0, 1.0)

        # Ensure z0 has shape (L,) and on correct device
        z0 = self.z0
        if z0.dim() != 1:
            z0 = z0.view(-1)
        z0 = z0.to(device)
        task = self.task
        # Use compiled helper when available/desired
        impl = _post_noise_effects_impl
        try:
            cfg = getattr(world_model, 'cfg')
            if getattr(cfg, 'compile', False):
                impl = torch.compile(_post_noise_effects_impl, mode=getattr(cfg, 'compile_type', 'reduce-overhead'), fullgraph=False)
        except Exception:
            pass
        vals_scaled, dis, score = impl(world_model, z0, noisy_seq, self.head_mode, task, float(self.lambda_d))

        # Store 0-dim tensors
        self.value_chosen_post_noise = vals_scaled.squeeze(0)
        self.disagreement_chosen_post_noise = (dis.squeeze(0) if dis is not None else None)
        self.score_chosen_post_noise = score.squeeze(0)

    def to_text_summary(self, num_random: int = 5, num_action: int = 4) -> str:
        """Create a concise, human-readable summary of planning across iterations.

        For each iteration: sample `num_random` candidates uniformly, plus best and worst
        by `raw_scores`. For each selected candidate, print value, disagreement (if any),
        weighted disagreement, total score, and a slice of its action sequence limited to
        the first `num_action` action dimensions.
        """
        lines = []
        I = self.actions_all.shape[0]
        E = self.actions_all.shape[1]
        T, A = self.actions_all.shape[2], self.actions_all.shape[3]
        has_dis = (self.disagreements_all is not None)

        def _fmt_vec(x: torch.Tensor) -> str:
            # Small vector formatter for action slices
            return ", ".join([f"{float(v):+.3f}" for v in x.tolist()])

        for i in range(I):
            lines.append(f"Iteration {i} | candidates={E}")
            scores_i = self.raw_scores[i]            # (E,)
            vals_i = self.values_all_scaled[i]       # (E,)
            dis_i = self.disagreements_all[i] if has_dis else None  # (E,)
            # Pick indices: best, worst, and num_random random
            best_idx = int(torch.argmax(scores_i).item())
            worst_idx = int(torch.argmin(scores_i).item())
            rand_count = max(0, num_random)
            rand_idx = torch.randint(low=0, high=E, size=(rand_count,), device=scores_i.device)
            sel = [best_idx, worst_idx] + [int(x.item()) for x in rand_idx]

            for tag, idx in zip(["best", "worst"] + [f"rand{j}" for j in range(rand_count)], sel):
                val = float(vals_i[idx].item())
                dis = float(dis_i[idx].item()) if has_dis else 0.0
                wd = float(self.lambda_d * dis) if has_dis else 0.0
                sc = float(scores_i[idx].item())
                act = self.actions_all[i, idx]  # (T,A)
                # Slice first num_action dims
                sl = act[:, :min(num_action, A)]  # (T,num_action)
                act_str = "; ".join([f"t{t}:[{_fmt_vec(sl[t])}]" for t in range(T)])
                lines.append(f"  {tag:<6} idx={idx:>4} | val={val:+.3f} dis={dis:+.3f} wd={wd:+.3f} sc={sc:+.3f} | actions {act_str}")

        # Final chosen (pre-noise) and post-noise summary
        val_pre = float(self.value_chosen.item())
        dis_pre = float(self.disagreement_chosen.item()) if self.disagreement_chosen is not None else 0.0
        wd_pre = float(self.weighted_disagreement_chosen.item()) if self.weighted_disagreement_chosen is not None else (self.lambda_d * dis_pre)
        sc_pre = float(self.score_chosen.item())
        lines.append("Chosen (pre-noise): "
                     f"val={val_pre:+.3f} dis={dis_pre:+.3f} wd={wd_pre:+.3f} sc={sc_pre:+.3f}")
        if self.value_chosen_post_noise is not None:
            val_post = float(self.value_chosen_post_noise.item())
            dis_post = float(self.disagreement_chosen_post_noise.item()) if self.disagreement_chosen_post_noise is not None else 0.0
            wd_post = float(self.lambda_d * dis_post)
            sc_post = float(self.score_chosen_post_noise.item())
            # Noise stats
            nz = self.action_noise
            nz_norm = float(nz.norm().item()) if nz is not None else 0.0
            nz_max = float(nz.abs().max().item()) if nz is not None else 0.0
            std0 = self.std_first_action
            std_mean = float(std0.mean().item())
            std_max = float(std0.max().item())
            lines.append("Chosen (post-noise): "
                         f"val={val_post:+.3f} dis={dis_post:+.3f} wd={wd_post:+.3f} sc={sc_post:+.3f} | noise ||·||={nz_norm:.3f} max|·|={nz_max:.3f} | std0 mean={std_mean:.3f} max={std_max:.3f}")

        return "\n".join(lines)
