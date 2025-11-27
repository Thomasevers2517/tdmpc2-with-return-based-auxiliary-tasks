## Confirmed Analyses

- **Detach encoder ratio vs consistency loss (Acrobot swingup)**  
  Compare runs with identical configs but different `detach_encoder_ratio` (e.g. 0.0 vs 0.5).  
  Plot consistency loss over training (train + validation_recent + validation_all) and highlight the step where the encoder is detached.  
  Start with Acrobot Swingup; later generalize to a small set of other tasks (e.g. cartpole, quadruped).

## Candidate / Future Analyses

- **Entropy schedule vs policy exploration**  
  For sweeps changing entropy schedule / coefficients (`start_entropy_coeff`, `end_entropy_coeff`, `dynamic_entropy_schedule`), plot action std (per-dimension or mean) over training to show how exploration decays.

- **UTD and AC UTD vs learning speed**  
  For sweeps varying `utd_ratio` and/or `ac_utd_multiplier`, plot evaluation reward vs environment steps and optionally steps-per-second / updates-per-second to quantify compute vs performance.

- **Imagination rollout depth vs performance**  
  For sweeps over `imagination_horizon` and/or `num_rollouts`, compare evaluation reward, model loss, and value loss to see how deeper imagination affects sample efficiency and stability.

- **Hot buffer effect on stability**  
  For sweeps toggling `hot_buffer_enabled` / `hot_buffer_ratio`, visualize training loss curves (value, consistency, policy) and evaluation reward to assess whether the hot buffer stabilizes or destabilizes training.

- **Auxiliary multi-gamma value heads**  
  For sweeps enabling / disabling `multi_gamma_gammas` and changing `multi_gamma_loss_weight`, plot auxiliary value losses per gamma and compare evaluation performance vs baseline (no aux heads).

- **Detach encoder ratio sweep (global)**  
  Beyond the Acrobot case above, systematically sweep `detach_encoder_ratio` across tasks and show both consistency loss and evaluation reward to understand where representation non-stationarity is the bottleneck vs data scarcity.

- **Planning temperature and policy seeding**  
  For sweeps over `planner_temperature_train` / `planner_temperature_eval` and `planner_policy_elites_first_iter_only`, plot evaluation reward and planning statistics (e.g. distribution of elite values, policy vs sampled trajectory usage) where available.

- **Imagination source for critic / actor**  
  For sweeps switching `ac_source`, `aux_value_source`, and `actor_source`, compare value loss, policy loss, and evaluation reward to understand when imagined vs replay rollouts are beneficial.

- **TD target style (SAC-style vs baseline)**  
  For sweeps flipping `sac_style_td`, compare Q-value calibration (e.g. via return vs predicted value plots) and evaluation reward.

- **EMA variants for encoder / policy / value**  
  For sweeps toggling `policy_ema_enabled`, `encoder_ema_enabled`, and `ema_value_planning`, study stability of losses, evaluation reward, and possibly target/online divergence metrics.

- **Model size and capacity**  
  For sweeps over `model_size`, `mlp_dim`, `latent_dim`, and `num_q`, visualize compute vs performance (steps-per-second vs final eval reward) and overfitting signals (train vs validation losses).

- **Distracting control robustness**  
  Using `distracted_dynamic` and `distracted_difficulty` variants, compare performance and consistency loss across difficulty levels to quantify robustness to visual distractors.

- **Planner configuration ablations**  
  For `iterations`, `num_samples`, `num_elites`, `num_pi_trajs`, and `horizon`, plot evaluation reward vs compute cost to identify efficient planning regimes.

- **Buffer configuration and recency effects**  
  For sweeps that change `buffer_size`, `buffer_update_interval`, or potential PER-related settings, visualize how sample age distribution and performance interact.

---

Implementation notes (for future notebooks/tools):

- Use the existing `analysis.tools` modules (`wandb_io`, `selection`, `aggregations`, `plotting`, etc.) to fetch runs from Weights & Biases, filter them by sweep and config, and generate figures.
- For each confirmed analysis, create a dedicated notebook under `analysis/notebooks/` (e.g. `detach_encoder_consistency.ipynb`) that:
  - Encodes the sweep IDs / run filters in a small, editable config cell.
  - Uses shared helper functions for querying W&B and producing standardized figures.
  - Saves key figures to disk (e.g. under `assets/` or a new `analysis/figures/` directory) for inclusion in the thesis.
- When possible, match or compare against the legacy baselines under `results/` (Dreamer, old TD-MPC2) to contextualize new runs.
