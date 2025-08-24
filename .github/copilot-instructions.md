<!-- Base (branch-agnostic) Copilot Instructions -->

# TD-MPC2 Repository – Base AI Assistant Guidance

Purpose: Provide stable, branch-independent rules for automated coding assistance.

Core principles:
0. Look at the branch-specific instructions for the current branch first; they may override or extend these base rules and give more detailed guidance.
1. Preserve planner & policy behavior unless explicitly requested.
2. Maintain torch.compile friendliness (avoid data-dependent Python control flow on hot paths).
3. Keep diffs minimal; no broad reformatting.
4. New features must be gated behind explicit config flags (default = baseline behavior).
5. Respect shape invariants for latent sequences & ensembles.
6. Use existing utility functions (math/two-hot, symlog transforms) unless a feature explicitly replaces them.
7. Always add parameters by registering Modules so checkpoints stay seamless.
8. Provide concise logging via existing Logger; avoid large per-step blobs.

When adding a feature:
* Steps: config parse → model modules → forward path → loss wiring → metrics/logging → checkpoint sanity → minimal tests.
* Add validation for config invariants (clear error on violation).
* Baseline equivalence: with feature disabled, training metrics should statistically match original runs.

Testing checklist (lightweight):
* Import + model instantiation.
* One synthetic batch through `_update` (loss finite, no shape errors).
* (Optional) small rollout to ensure planner unaffected.

Logging recommendations:
* Scalar losses, returns, value stats, planner timing.
* Rare snapshots (e.g., per `eval_freq`) for distributions or auxiliary diagnostics.

Performance: Target <15% overhead for optional research features; surface param deltas & ms/iter if exceeded.

Compatibility: Old checkpoints must load under new code when added feature flags remain off.

Security / Safety: Never include secrets; avoid external network calls inside training loops.

End of base instructions.