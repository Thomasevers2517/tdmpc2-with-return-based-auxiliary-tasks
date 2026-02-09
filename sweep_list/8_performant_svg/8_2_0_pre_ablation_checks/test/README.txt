Testing Ground for Code Changes
================================

This folder is for manually testing code-level improvements before
running the full ablation suite.

Planned tests:
1. Fix .item() graph break on discount in TD target computation
   - Store discount as float at init instead of calling .item() at runtime
   - Verify no performance regression on a short run

2. Parallelize dynamics heads via Ensemble (vmap)
   - Replace nn.ModuleList + for-loop with Ensemble pattern (like reward/value heads)
   - Verify correctness (same eval scores) and measure wall-clock speedup
