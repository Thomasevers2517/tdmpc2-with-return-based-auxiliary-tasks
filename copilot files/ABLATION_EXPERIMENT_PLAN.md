# Ablation Experiment Plan (sweep_list/18_ablations/)

## Goal
Isolate the effect of each contribution compared to original TD-MPC2, then show the full system.

## Baseline (Original TD-MPC2)
All sweeps share this base config:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `utd_ratio` | 1 | Original |
| `value_update_freq` | 1 | Every step |
| `planner_num_dynamics_heads` | 1 | Single head |
| `num_q` | 5 | Original |
| `tau` | 0.01 | Original |
| `consistency_coef` | 20 | Original |
| `rollout_head_strategy` | split | Current default |
| `jacobian_correction_scale` | 0 | No fix |
| `buffer_update_interval` | -1 | Disabled (episode-end only) |
| `local_td_bootstrap` | false | Standard bootstrap |
| `start_entropy_coeff` | 1e-4 | Default |
| `end_entropy_coeff` | 0 | |
| `min_std` | 0.05 | Original TD-MPC2 |
| `max_std` | 2 | |
| `log_std_min` | -10 | Original TD-MPC2 |
| `log_std_max` | 2 | Original TD-MPC2 |
| `dynamics/value/policy std coefs` | all 0 | |
| `save_video` | true | |
| `save_train_video` | true | |

## Sweeps — Contributions Isolated

| # | Name | Changes from baseline |
|---|------|-----------------------|
| **1** | `baseline` | Nothing — pure original TD-MPC2 |
| **2** | `buffer_fix` | `buffer_update_interval: 1` |
| **3** | `jacobian_fix` | `jacobian_correction_scale: 1`, `start_entropy_coeff: 3e-5` |
| **4** | `local_bootstrap` | `local_td_bootstrap: true` |
| **5** | `multi_dynamics` | `planner_num_dynamics_heads: 4` |
| **6** | `all_contributions` | Sweeps 2+3+4+5 combined (entropy stays 1e-4) |
| **7** | `all_utd4` | Sweep 6 + `utd_ratio: 4` |
| **8** | `all_plus_hyperparams` | Sweep 7 + `value_update_freq: 2`, `num_q: 8`, `tau: 0.003`, `consistency_coef: 10` |

Each sweep is split into **a** (easy, 100k) and **b** (hard, 300k) subsweeps.

## Task Groups

| Group | Subsweep | Tasks | Steps | Seeds |
|-------|----------|-------|-------|-------|
| Easy | a | hopper-hop, quadruped-walk, humanoid-stand, cheetah-run, finger-turn_hard | 100k | 8 |
| Hard | b | dog-run | 300k | 8 |

## Run Counts

| | Runs/subsweep | Subsweeps/sweep | Runs/sweep | × 8 sweeps | Total |
|---|--------------|-----------------|-----------|------------|-------|
| Easy (5 tasks × 8 seeds) | 40 | 1 | | | |
| Hard (1 task × 8 seeds) | 8 | 1 | | | |
| **Per sweep** | | | **48** | | |
| **Grand total** | | | | | **384** |

## What This Tells Us
- **1→2**: Effect of replay buffer fix (sub-episode buffer updates)
- **1→3**: Effect of Jacobian correction for tanh squashing (with tuned entropy)
- **1→4**: Effect of local TD bootstrap
- **1→5**: Effect of multi-head dynamics ensemble
- **1→6**: Combined effect of all 4 contributions
- **6→7**: Marginal value of higher UTD ratio (4×)
- **7→8**: Marginal value of hyperparameter tuning (tau, num_q, etc.) on top of all contributions + UTD4
