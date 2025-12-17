# Analysis Plan

Step-by-step plan for creating analysis notebooks and visualizations for ICML 2026 submission.

---

## Status Overview

| # | Notebook | Status | Sweep ID | Baselines Needed |
|---|----------|--------|----------|------------------|
| 07 | Optimistic Exploration Ablation | ✅ Done | `mpjo8uje` | None |
| 08 | Humanoid Bench Baseline | ⏳ Pending | `8lsl2o4p` | TD-MPC2, DreamerV3, SAC, BMPC |
| 09 | Evaluation Head Reduce Ablation | ⏳ Pending | `8dscchoi` | None |
| 10 | Dog Run Results | ⏳ Pending | `xmhg3plq` | TD-MPC2 |
| 11 | Buffer Update Interval Ablation | ⏳ Pending | `69q5r0jo` | None |
| 12 | Encoder Consistency Coefficient | ⏳ Pending | `679dvtrt` | None |
| 13 | DMControl Benchmark | 🔜 Later | TBD | TD-MPC2, DreamerV3, SAC, EfficientZero-V2 |

---

## 07 — Optimistic Exploration Ablation ✅

**Sweep ID:** `mpjo8uje`  
**Task:** quadruped-walk  
**Steps:** 100k | **Seeds:** 10

### Purpose
Demonstrate that optimistic (max) head selection during planning provides better exploration than pessimistic (mean) with additive action noise.

### Design Choice Being Validated
- `planner_head_reduce=max` with `planner_action_noise_std=0` (our method)
- vs `planner_head_reduce=mean` with varying noise levels (0, 0.03, 0.1, 0.3)

### Key Message
Taking the maximum over ensemble heads provides directed, optimistic exploration that outperforms random noise injection.

### Figure
- Sample efficiency curve
- 5 lines: Maximum(σ=0) solid, Mean(σ=0/0.03/0.1/0.3) dashed
- Legend at bottom in 3+2 grid

**Status:** Complete

---

## 08 — Humanoid Bench Baseline ⏳

**Sweep ID:** `8lsl2o4p`  
**Tasks:** humanoid_h1-balance_simple-v0, humanoid_h1-walk-v0, humanoid_h1-slide-v0  
**Steps:** 1M | **Seeds:** 2 per task

### Purpose
Show competitive performance on HumanoidBench tasks using the H1 humanoid robot.

### Baselines Needed
- [ ] TD-MPC2 (from BMPC paper)
- [ ] DreamerV3 (from BMPC paper)
- [ ] SAC (from BMPC paper)
- [ ] BMPC (primary comparison)

### Key Message
Our method achieves strong performance on high-dimensional humanoid control.

### Figure Options
1. **Multi-panel:** 3 subplots (one per task) with all methods
2. **Table:** Final performance comparison at 1M steps
3. **Combined:** Single figure with task as panel, methods as lines

### Data to Add
Baselines from BMPC paper — need to extract CSVs or digitize figures.

**Status:** Waiting for baseline data

---

## 09 — Evaluation Head Reduce Ablation ⏳

**Sweep ID:** `8dscchoi`  
**Task:** quadruped-walk  
**Steps:** 100k | **Seeds:** 10

### Purpose
Show that conservative (min) head reduction during evaluation is more robust than neutral (mean) or optimistic (max).

### Design Choice Being Validated
- Training: always `planner_head_reduce=max` (optimistic)
- Evaluation: `planner_head_reduce_eval` ∈ {min, mean, max}

### Key Message
While training benefits from optimism, evaluation benefits from conservatism. Taking the minimum filters out overconfident predictions from individual ensemble members.

### Figure
- Sample efficiency curve
- 3 lines: Eval=min (solid), Eval=mean (dashed), Eval=max (dotted)
- Same color palette

### Expected Result
min > mean > max (conservative evaluation wins)

**Status:** Ready to create

---

## 10 — Dog Run Results ⏳

**Sweep ID:** `xmhg3plq`  
**Task:** dog-run  
**Steps:** 500k | **Seeds:** 4

### Purpose
Demonstrate strong performance on high-dimensional DMC locomotion (38-dim observation, 12-dim action).

### Baselines Needed
- [ ] TD-MPC2 baseline from `results/tdmpc2/dog-run.csv`

### Key Message
Our method scales to complex, high-dimensional control tasks.

### Figure
- Sample efficiency curve
- Our method vs TD-MPC2 baseline
- Include std bands if available

**Status:** Ready to create

---

## 11 — Buffer Update Interval Ablation ⏳

**Sweep ID:** `69q5r0jo`  
**Task:** quadruped-walk  
**Steps:** 100k | **Seeds:** 5

### Purpose
Show that less frequent priority buffer updates (`buffer_update_interval=500`) improve performance over frequent updates (`buffer_update_interval=1`).

### Design Choice Being Validated
- `buffer_update_interval=1` (update priorities every step)
- `buffer_update_interval=500` (update priorities every 500 steps)

### Key Message
Less frequent buffer updates lead to more stable training. High-frequency updates may cause the agent to over-focus on recently-seen transitions.

### Figure
- Sample efficiency curve
- 2 lines: interval=500 vs interval=1
- Different colors

### Expected Result
interval=500 > interval=1

**Status:** Ready to create

---

## 12 — Encoder Consistency Coefficient ⏳

**Sweep ID:** `679dvtrt`  
**Task:** quadruped-walk  
**Steps:** 100k | **Seeds:** 5

### Purpose
Show that encoder consistency regularization (`encoder_consistency_coef=1`) improves both performance and representation quality.

### Design Choice Being Validated
- `encoder_consistency_coef=0` (no consistency loss)
- `encoder_consistency_coef=1` (with consistency loss)

### Key Message
Encouraging consistent encoder representations across time leads to:
1. Better final performance
2. Lower consistency loss (more stable representations)

### Figures
**Figure 1: Sample Efficiency**
- eval/episode_reward over steps
- 2 lines: coef=0 vs coef=1

**Figure 2: Consistency Loss**
- consistency loss metric over steps
- 2 lines: coef=0 vs coef=1
- Note: coef=0 will still have consistency loss computed but not used for optimization

### Metrics to Fetch
- `eval/episode_reward`
- `train/consistency_loss` (or similar — need to verify metric name)

**Status:** Ready to create (need to verify consistency loss metric name)

---

## 13 — DMControl Benchmark 🔜

**Status:** Later

### Purpose
Full benchmark comparison on DMControl Suite tasks.

### Baselines Needed
- [ ] TD-MPC2
- [ ] DreamerV3
- [ ] SAC
- [ ] EfficientZero-V2 (proprioceptive)

### Figure Options
1. Aggregate performance curve
2. Per-task table
3. IQM/performance profiles

---

## Baseline Data Sources

### Currently Available (`results/` folder)
- `tdmpc2/` — TD-MPC2 state-based (102 tasks)
- `tdmpc2-pixels/` — TD-MPC2 pixel-based (12 tasks)
- `dreamerv3/` — DreamerV3 (102 tasks)
- `sac/` — SAC (102 tasks)

### To Add
- [ ] `efficientzero-v2/` — EfficientZero-V2 proprioceptive results
- [ ] `humanoid-bench/` — Baselines for HumanoidBench tasks
  - TD-MPC2 (from BMPC)
  - DreamerV3 (from BMPC)
  - SAC (from BMPC)
  - BMPC

---

## Execution Order

1. ✅ **07 — Optimistic Exploration Ablation** (done)
2. ⏳ **09 — Evaluation Head Reduce Ablation** (no external baselines needed)
3. ⏳ **11 — Buffer Update Interval Ablation** (no external baselines needed)
4. ⏳ **12 — Encoder Consistency Coefficient** (no external baselines needed)
5. ⏳ **10 — Dog Run Results** (TD-MPC2 baseline already available)
6. ⏳ **08 — Humanoid Bench Baseline** (waiting for BMPC baselines)
7. 🔜 **13 — DMControl Benchmark** (later, needs EfficientZero-V2)

---

## Notes

- All figures use ICML 2026 double-column format styling
- Font sizes: title=22, subtitle=18, axis=22, ticks=20, legend=16
- Line width: 3.5pt
- Figure dimensions: 550×620 (narrow/tall with legend at bottom)
