# Pending Planner Experiments

1. **Value Scaling via `RunningScale`**
   - Wire planner scoring (`compute_values_head0`) to use the existing running scale so trajectory values stay comparable to disagreement bonuses.
   - Track a "max attainable value" statistic to modulate the disagreement weight based on how saturated the current state value is.
   - Log scaled vs unscaled values during planning to verify stability and tune the trade-off.

2. **Particle MPPI Planning**
   - Run multiple MPPI/CEM planners in parallel ("particles") with different random seeds or disagreement weights to escape local minima introduced by the disagreement term.
   - Aggregate candidate actions by selecting the best overall score or using a robustness criterion (e.g., highest value with low disagreement).
   - Measure compute overhead vs. performance gains to size the particle count appropriately.

3. **Advanced Score Estimation**
   - Explore richer scoring objectives that combine value, disagreement, progress-to-goal (e.g., relative to max attainable value), horizon length, and planner iteration index.
   - Prototype adaptive schedules: iteration-dependent temperatures, disagreement weights that shrink when a high-value candidate is absent, or boosted exploration when planning stalls.
   - Ablate alternative heuristics (e.g., per-iteration score annealing, saturation-aware bonuses) to understand which planning objectives best balance exploitation vs. model improvement.

4. **Planner Policy-Seed Efficiency**
   - Cache value estimates for policy rollouts once per planning call instead of recomputing every CEM iteration.
   - Log how often policy-seeded trajectories appear among elites to quantify their influence on optimization.

5. **Warm-Start Mean Shift**
   - When seeding the next planning step, shift the stored mean left by one timestep so previously executed actions drop out and a zero action initializes the tail.
   - Confirm the shifted warm start stabilizes optimization and adjust logging to observe convergence speed.
