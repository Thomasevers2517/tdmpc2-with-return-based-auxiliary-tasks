from dataclasses import dataclass
from typing import Optional
import torch
from .scoring import compute_values, compute_disagreement, combine_scores
import torch._dynamo as dynamo


def _post_noise_effects_impl(world_model, z0: torch.Tensor, noisy_seq: torch.Tensor, head_mode: str, value_std_coef: float, task, lambda_latent: float, use_ema_value: bool = False, discount: torch.Tensor = None):
    """Core kernel to roll out a single noisy sequence and score it.

    Args:
        discount: float32[] scalar tensor on device.

    Returns (value_scaled: Tensor[1], latent_disagreement: Optional[Tensor[1]], 
             value_disagreement: Tensor[1], score: Tensor[1]).
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
    vals_unscaled, vals_scaled, vals_std, val_dis = compute_values(
        lat_all,
        noisy_seq.unsqueeze(0),
        world_model,
        task,
        value_std_coef=value_std_coef,
        use_ema_value=use_ema_value,
        discount=discount,
    )
    latent_dis = None
    if lat_all.shape[0] > 1:
        final_all = lat_all[:, 0, -1, :]  # [H,L]
        # Compute disagreement with E=1; returns shape [1]
        latent_dis = compute_disagreement(final_all.unsqueeze(1))
    score = combine_scores(vals_scaled, latent_dis, lambda_latent)
    return vals_scaled, latent_dis, val_dis, score


@dataclass
class PlannerBasicInfo:
    """Basic planner summary metrics.

    Attributes:
        value_chosen: Scalar value of the chosen sequence.
        latent_disagreement_chosen: Latent disagreement of chosen (None if not computed).
        value_disagreement_chosen: Value disagreement (std across H×Q heads) of chosen (None if not computed).
        score_chosen: Combined score of the chosen sequence.
        weighted_latent_disagreement_chosen: Weighted latent disagreement (lambda * dis) of chosen.
        value_elite_mean: Mean value over elite set.
        value_elite_std: Std of value over elite set.
        value_elite_max: Max value over elite set.
        latent_disagreement_elite_mean: Mean latent disagreement over elites (None if not computed).
        latent_disagreement_elite_std: Std latent disagreement over elites (None if not computed).
        latent_disagreement_elite_max: Max latent disagreement over elites (None if not computed).
        value_disagreement_elite_mean: Mean value disagreement over elites (None if not computed).
        value_disagreement_elite_std: Std value disagreement over elites (None if not computed).
        value_disagreement_elite_max: Max value disagreement over elites (None if not computed).
        num_elites: Number of elites used for updates.
        elite_indices: Indices of elites within candidate set; int64[E].
        scores_all: Combined scores for all candidates; Tensor[E].
        values_scaled_all: Scaled values for all candidates; Tensor[E].
        weighted_latent_disagreements_all: Weighted latent disagreements for all candidates; Tensor[E].
    """
    value_chosen: torch.Tensor  # 0-dim tensor
    latent_disagreement_chosen: Optional[torch.Tensor]
    value_disagreement_chosen: Optional[torch.Tensor]
    score_chosen: torch.Tensor  # 0-dim tensor
    weighted_latent_disagreement_chosen: Optional[torch.Tensor]
    value_elite_mean: torch.Tensor
    value_elite_std: torch.Tensor
    value_elite_max: torch.Tensor
    latent_disagreement_elite_mean: Optional[torch.Tensor]
    latent_disagreement_elite_std: Optional[torch.Tensor]
    latent_disagreement_elite_max: Optional[torch.Tensor]
    value_disagreement_elite_mean: Optional[torch.Tensor]
    value_disagreement_elite_std: Optional[torch.Tensor]
    value_disagreement_elite_max: Optional[torch.Tensor]
    num_elites: int
    elite_indices: torch.Tensor  # int64[E]
    scores_all: torch.Tensor  # (E,)
    values_scaled_all: torch.Tensor  # (E,)
    weighted_latent_disagreements_all: Optional[torch.Tensor]  # (E,)


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
        latent_disagreements_all: Optional[torch.Tensor]  # float32[I, E]
        value_disagreements_all: Optional[torch.Tensor]  # float32[I, E]
        raw_scores: float32[I, E]
        mean_all: float32[I, T, A] - Distribution mean per iteration
        std_all: float32[I, T, A] - Distribution std per iteration
    """
    actions_all: torch.Tensor
    latents_all: torch.Tensor
    values_all_unscaled: torch.Tensor
    values_all_scaled: torch.Tensor
    latent_disagreements_all: Optional[torch.Tensor]
    value_disagreements_all: Optional[torch.Tensor]
    raw_scores: torch.Tensor
    mean_all: torch.Tensor  # [I, T, A] - Distribution mean per iteration
    std_all: torch.Tensor   # [I, T, A] - Distribution std per iteration

    # Analysis context (advanced-only)
    action_seq_chosen: torch.Tensor  # (T,A)
    action_noise: Optional[torch.Tensor]  # (A,) applied at t=0; None if no noise
    std_first_action: torch.Tensor  # (A,)
    z0: torch.Tensor  # (L,)
    task: Optional[torch.Tensor]
    lambda_latent: float
    head_mode: str
    value_std_coef: float
    T: int
    use_ema_value: bool = False  # Whether to use EMA target network for V in planning
    discount: torch.Tensor = None  # float32[] scalar tensor on device for compute_values

    # Outputs of post-noise analysis (set by compute_post_noise_effects)
    value_chosen_post_noise: Optional[torch.Tensor] = None
    latent_disagreement_chosen_post_noise: Optional[torch.Tensor] = None
    value_disagreement_chosen_post_noise: Optional[torch.Tensor] = None
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
        vals_scaled, latent_dis, val_dis, score = impl(world_model, z0, noisy_seq, self.head_mode, self.value_std_coef, task, float(self.lambda_latent), self.use_ema_value, self.discount)

        # Store 0-dim tensors
        self.value_chosen_post_noise = vals_scaled.squeeze(0)
        self.latent_disagreement_chosen_post_noise = (latent_dis.squeeze(0) if latent_dis is not None else None)
        self.value_disagreement_chosen_post_noise = (val_dis.squeeze(0) if val_dis is not None else None)
        self.score_chosen_post_noise = score.squeeze(0)

    def to_text_summary(self, num_random: int = 5, num_action: int = 4) -> str:
        """Create a concise, human-readable summary of planning across iterations.

        For each iteration: sample `num_random` candidates uniformly, plus best and worst
        by `raw_scores`. For each selected candidate, print value, disagreement (if any),
        weighted disagreement, total score, and a slice of its action sequence limited to
        the first `num_action` action dimensions.
        
        Also shows the distribution mean and std for the first `num_action` action dims
        at each timestep.
        """
        lines = []
        I = self.actions_all.shape[0]
        E = self.actions_all.shape[1]
        T, A = self.actions_all.shape[2], self.actions_all.shape[3]
        has_latent_dis = (self.latent_disagreements_all is not None)
        has_val_dis = (self.value_disagreements_all is not None)
        num_action_show = min(num_action, A)

        def _fmt_vec(x: torch.Tensor) -> str:
            # Small vector formatter for action slices
            return ", ".join([f"{float(v):+.3f}" for v in x.tolist()])

        for i in range(I):
            lines.append(f"Iteration {i} | candidates={E}")
            
            # Show distribution mean/std for first num_action dims at each timestep
            mean_i = self.mean_all[i]  # [T, A]
            std_i = self.std_all[i]    # [T, A]
            dist_parts = []
            for t in range(T):
                m_slice = mean_i[t, :num_action_show]
                s_slice = std_i[t, :num_action_show]
                dist_parts.append(f"t{t}: μ=[{_fmt_vec(m_slice)}] σ=[{_fmt_vec(s_slice)}]")
            lines.append(f"  dist: {' | '.join(dist_parts)}")
            
            scores_i = self.raw_scores[i]            # (E,)
            vals_i = self.values_all_scaled[i]       # (E,)
            latent_dis_i = self.latent_disagreements_all[i] if has_latent_dis else None  # (E,)
            val_dis_i = self.value_disagreements_all[i] if has_val_dis else None  # (E,)
            # Pick indices: best, worst, and num_random random
            best_idx = int(torch.argmax(scores_i).item())
            worst_idx = int(torch.argmin(scores_i).item())
            rand_count = max(0, num_random)
            rand_idx = torch.randint(low=0, high=E, size=(rand_count,), device=scores_i.device)
            sel = [best_idx, worst_idx] + [int(x.item()) for x in rand_idx]

            for tag, idx in zip(["best", "worst"] + [f"rand{j}" for j in range(rand_count)], sel):
                val = float(vals_i[idx].item())
                lat_dis = float(latent_dis_i[idx].item()) if has_latent_dis else 0.0
                v_dis = float(val_dis_i[idx].item()) if has_val_dis else 0.0
                wd_lat = float(self.lambda_latent * lat_dis) if has_latent_dis else 0.0
                sc = float(scores_i[idx].item())
                act = self.actions_all[i, idx]  # (T,A)
                # Slice first num_action dims
                sl = act[:, :min(num_action, A)]  # (T,num_action)
                act_str = "; ".join([f"t{t}:[{_fmt_vec(sl[t])}]" for t in range(T)])
                lines.append(f"  {tag:<6} idx={idx:>4} | val={val:+.3f} lat_dis={lat_dis:+.3f} val_dis={v_dis:+.3f} wd_lat={wd_lat:+.3f} sc={sc:+.3f} | actions {act_str}")

        # Final chosen (pre-noise) and post-noise summary
        val_pre = float(self.value_chosen.item())
        lat_dis_pre = float(self.latent_disagreement_chosen.item()) if self.latent_disagreement_chosen is not None else 0.0
        val_dis_pre = float(self.value_disagreement_chosen.item()) if self.value_disagreement_chosen is not None else 0.0
        wd_lat_pre = float(self.weighted_latent_disagreement_chosen.item()) if self.weighted_latent_disagreement_chosen is not None else (self.lambda_latent * lat_dis_pre)
        sc_pre = float(self.score_chosen.item())
        lines.append("Chosen (pre-noise): "
                     f"val={val_pre:+.3f} lat_dis={lat_dis_pre:+.3f} val_dis={val_dis_pre:+.3f} wd_lat={wd_lat_pre:+.3f} sc={sc_pre:+.3f}")
        if self.value_chosen_post_noise is not None:
            val_post = float(self.value_chosen_post_noise.item())
            lat_dis_post = float(self.latent_disagreement_chosen_post_noise.item()) if self.latent_disagreement_chosen_post_noise is not None else 0.0
            val_dis_post = float(self.value_disagreement_chosen_post_noise.item()) if self.value_disagreement_chosen_post_noise is not None else 0.0
            wd_lat_post = float(self.lambda_latent * lat_dis_post)
            sc_post = float(self.score_chosen_post_noise.item())
            # Noise stats
            nz = self.action_noise
            nz_norm = float(nz.norm().item()) if nz is not None else 0.0
            nz_max = float(nz.abs().max().item()) if nz is not None else 0.0
            std0 = self.std_first_action
            std_mean = float(std0.mean().item())
            std_max = float(std0.max().item())
            lines.append("Chosen (post-noise): "
                         f"val={val_post:+.3f} lat_dis={lat_dis_post:+.3f} val_dis={val_dis_post:+.3f} wd_lat={wd_lat_post:+.3f} sc={sc_post:+.3f} | noise ||·||={nz_norm:.3f} max|·|={nz_max:.3f} | std0 mean={std_mean:.3f} max={std_max:.3f}")

        return "\n".join(lines)
