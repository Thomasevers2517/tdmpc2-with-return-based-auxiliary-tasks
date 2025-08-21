# 11. TD-MPC2 AI Coding Agent Instructions

Concise reference for AI agents working on this repository. Three parts:
1. Codebase map (what each folder/file does)
2. Behavior guidelines (how to act / interact)
3. Active research plan (multi-discount auxiliary value heads)

---
## 1. Codebase Map (Fast Recall)
Top-level logic: implicit latent world model + distributional Q ensemble + policy prior + MPPI/CEM-style planner. Two modes: online single-task (`OnlineTrainer`), offline multi-task (`OfflineTrainer`).

Directory / File  | Role (essentials)
------------------|-------------------------------------------------------------
`tdmpc2/train.py` | Hydra entry, picks trainer (online vs offline), wires env/agent/buffer/logger.
`tdmpc2/evaluate.py` | Loads checkpoint, runs evaluation (single or multi-task).
`tdmpc2/tdmpc2.py` | `TDMPC2` agent: planning (`_plan`), updates (`_update`), policy update, TD targets.
`common/world_model.py` | Modules: encoder(s), dynamics, reward head, optional termination head, policy prior head, Q ensemble + target/detach copies.
`common/layers.py` | Building blocks: MLP (`mlp`), conv encoder, SimNorm, vectorized ensemble (`Ensemble`), checkpoint format conversion.
`common/math.py` | Two-hot distributional regression (reward/Q), symlog transforms, Gaussian utilities, termination metrics.
`common/buffer.py` | TorchRL replay (episode slices). Samples (horizon+1)-length subsequences for training.
`common/parser.py` | Hydra cfg → frozen dataclass (torch.compile safe), model size presets, discount heuristic.
`common/scale.py` | Running scale estimator for policy Q normalization.
`common/init.py` | Weight init + zeroing for reward/Q final layers.
`trainer/online_trainer.py` | Environment interaction, seeding phase, logging, periodic eval.
`trainer/offline_trainer.py` | Loads large multi-task dataset, pure update loop + periodic eval.
`envs/` | Environment factories & wrappers: multi-task assembly, tensor conversion, action padding/masking.
`envs/wrappers/multitask.py` | Pads obs/actions, tracks per-task dims & lengths.
`envs/wrappers/tensor.py` | Numpy→Torch, standardizes info dict (success, terminated).
`common/__init__.py` | Task sets + possibly model size constants (task embeddings rely here).
`results/` | Example CSV logs for benchmarking (reference format).
`docker/` | Environment reproducibility (base dependencies for DMControl etc.).

Key runtime tensors inside update (`T=horizon`, `B=batch`):
`obs` (T+1,B,*), `action` (T,B,A), `reward` (T,B,1), `terminated` (T,B,1), latent rollout `zs` (T+1,B,latent). Q logits shape: (num_q,T,B,num_bins).

Planning (`_plan`): warm-start shifted mean action sequence; inject policy trajectories + sampled Gaussians; rollout via dynamics + reward; value = discounted reward sum + bootstrap (`avg` Q); elite filtering → updated mean/std; first action (with exploration noise unless eval) returned.

Losses: consistency (latent), reward CE, termination BCE (episodic), value CE (distributional Q), policy (entropy + value scaled) with geometric weight `rho^t`. Target Qs updated via Polyak `tau`.

---
## 2. Behavior Guidelines for AI Agent
- Ask clarifying questions if any requirement, config intent, or shape assumption is unclear (never silently guess critical semantics like discounting, horizon, or distributional bins).
- Preserve existing APIs & logging signatures unless explicitly asked to change; add new config keys under descriptive namespace (e.g., `multi_gamma.*`).
- Keep modifications torch.compile friendly: avoid adding dynamic Python control flow dependent on tensor data inside `_update` / `_plan`.
- Maintain shape invariants (see section 1); if adding heads, co-locate logic in `WorldModel` and wire optimizer param groups in `TDMPC2.__init__`.
- For multi-task-unrelated experiments, guard new single-task features with config flags so multi-task paths remain unaffected.
- Use existing patterns for losses: time weighting `rho**t`, divide by horizon (and by `num_q` where appropriate) for consistent magnitudes.
- Reuse two-hot regression utilities (`math.soft_ce`, `math.two_hot`, `math.two_hot_inv`) for any scalar-to-distribution supervision to stay numerically consistent.
- When adding logging: integrate via `Logger.log` with category `train` / `eval`; avoid large per-step payloads—use periodic snapshot intervals for heavy artifacts.
- Checkpointing: ensure new params appear in model `state_dict` automatically by registering modules inside `WorldModel`; no custom save unless special handling required.
- Validate new config semantics in `common/parser.py` (or separate light validator) before training, raising clear errors (e.g., discount mismatches, bin count constraints).
- Keep planner behavior identical unless a task explicitly requests planner changes; planning stability is sensitive to temperature, std bounds, iteration count.
- Provide minimal, targeted diffs; do not reformat unrelated files.
- Prefer incremental integration (config parse → model changes → loss integration → logging) with brief rationale in comments where non-obvious.
- If uncertainty about statistical impact (e.g., auxiliary head weighting), surface as TODO comment referencing config knob.
- Optimize readability: short descriptive variable names; comment any nontrivial tensor reshaping (especially for joint multi-head outputs).
- Before committing major changes, outline (1) expected param delta, (2) added compute %, (3) possible failure modes (e.g., destabilized Q scale) and mitigation.

---
## 3. Research Plan: Multi-Discount Auxiliary Value Heads
The following plan is the authoritative specification for the current research extension. Implementations must honor listed scope & invariants. (Verbatim user-provided content.)

> **Core goal.** Improve value-equivalent world-model learning in **TD-MPC2** by adding **auxiliary value targets** at **multiple discounts** (multi-γ), **without changing** planning or policy training.
> **Key rule.** The **primary value head** continues to drive decision-making (policy prior & planner). Auxiliary heads are **training-only** signals that shape the latent dynamics/representation.

---

## 0) Scope & Invariants

* **Supervision, not control.** Keep the planner and policy-prior objectives unchanged. Auxiliary heads never influence the planner or the policy loss; they only backprop into the world model.
* **Single-task only.** Ignore any multitask codepaths/configs.
* **Targets & loss.** Use the same **discrete regression** mechanism as TD-MPC2 for reward/value: transform → **two-hot** onto fixed bins → **soft cross-entropy**.
* **TD window.** Use the **existing n-step** TD target length `n` (identical to the primary value head) for all auxiliary heads.
* **Discount set.** `gammas[0]` **must equal** the task’s **primary γ** used by baseline TD-MPC2. You may add up to **5** additional auxiliary γ’s.
* **Planner hyperparameters.** Do **not** change population size, number of elites, policy-prior samples, etc.

---

## 1) Baseline Reproduction

1. **Tasks & seeds.** Choose 3–4 DMControl tasks (e.g., `cheetah-run`, `walker-walk`, `quadruped-run`, `dog-run`) × **3 seeds**, using stock single-task configs.
2. **Run stock TD-MPC2.** Log episodic return, reward/value CE losses, wall-clock per env-step, GPU memory. If available, also log planner stats (population, elites, prior samples).
3. **Freeze baselines.** Save checkpoints, configs, and commit SHAs as “do-not-regress” references.

---

## 2) Method: Multi-γ Auxiliary Value Heads

### 2.1 Targets (scalar TD → two-hot projection)

For each training unroll state $s_{t+k}$ and each $\gamma_i \in \texttt{gammas}$:

$$
y^{(\gamma_i)}_{t+k} = \sum_{j=0}^{n-1}\gamma_i^{j} r_{t+k+j}
+ \gamma_i^{n}\,\underbrace{\hat V_{\text{primary}}(s_{t+k+n})}_{\text{stop-grad}}.
$$

* Transform $y^{(\gamma_i)}$ with the **same** transform used by the primary head (e.g., symlog).
* **Two-hot** project onto the fixed support (same #bins and range as the primary head).
* Train with **soft cross-entropy** (CE) against that two-hot target.
	*This is distributional regression of a scalar (two-hot), not a full return-distribution backup—identical mechanics to TD-MPC2’s value training.*

### 2.2 Head structure

* **Default: `head: "joint"`** — one projection from latent → `(m × B)` logits, reshape to `m` heads of size `B` (`m = len(gammas)`, `B = #bins`).
* **Option: `head: "separate"`** — independent linear layers, one per auxiliary γ (plus the existing primary head).
* **Decision rule unchanged.** Planner & policy still consume **only the primary** value (mean of categorical in inverse-transform space).

### 2.3 Loss integration

$$
\mathcal{L}_{\text{world}} =
\mathcal{L}_{\text{existing}}
+ \sum_{i=1}^{m} \lambda_{\gamma_i}\,\text{CE}(p_{\gamma_i}, q_{\gamma_i}),\quad \lambda_{\gamma_i}=\texttt{loss_weight}.
$$

---

## 3) Config (first-class)

```yaml
multi_gamma:
	enabled: true
	gammas: [0.99, 0.97, 0.995]
	head: "joint"
	loss_weight: 0.5
	log_interval_steps: 10000
	log_num_examples: 8
```

---

## 4) Implementation Checklist (high-level)
1. Config parse + validation (primary gamma match; ≤6 discounts)
2. Model heads (joint or separate)
3. Per-γ TD targets (stop-grad bootstrap)
4. Loss accumulation (world model only)
5. Checkpoint inclusion
6. Logging (summaries + periodic snapshots)
7. Ablation configs (A0–A4)
8. Baseline reproduction toggle (`enabled=false`)

---
## 5) Logging Plan
Primary curve metrics + periodic snapshot (distributions, decoded scalars, errors, planner stats read-only). Limit volume with `log_num_examples`.

---
## 6) Experiments & Ablations
Baseline vs multiple `gammas` sets; metrics: AUC return, final return, aux MAE/RMSE, overhead (params, VRAM, ms/iter), stability (seed variance). Success: +3–5% AUC on ≥2 tasks with ≤15% overhead.

---
## 7) Budget
≈52k params per aux γ (latent 512 × 101 bins). 5 aux ≈260k params (<1% of larger models). Expected runtime overhead <15% for up to 5 aux.

---
## 8) Future Option
Multi-prefix heads (skip for now).

---
## 9) Implementation Defaults
`head="joint"`, discounts `[γ_primary, 0.97, 0.995]`, same TD window & bins, periodic logging every 10k steps.

---
## 10) Minimal To-Do (Agent)
Add config → model head(s) → targets → loss → logging → ablations → sanity checks.

---
## Appendix: Math Reference
Two-hot CE over symlog bins; decode via expectation then inverse symlog.

---
Questions or ambiguities? Ask before implementing; do not modify planner or policy pathways without explicit instruction.

---
## 4. Clarified Constraints & Phased Implementation Plan (Updated)

This section refines the research plan with explicit engineering constraints agreed upon after discussion.

### 4.1 Invariants (Do NOT change unless explicitly requested)
- Primary discount remains the existing heuristic from `_get_discount`; no manual override. `gammas[0]` must numerically equal the computed primary discount.
- Single-task only; all multi-task branches left untouched. No task embedding changes.
- One-step TD only (exactly the current target logic). No n-step expansion.
- Planner, policy loss, MPPI iteration loop, and target Q update procedure remain identical.
- Auxiliary heads never influence action selection or policy update logic; training-only supervision.
- When `multi_gamma.enabled=false` OR `gammas` length == 1, the training trajectory (loss scales, metrics) must match baseline (within normal stochastic noise). Achieve this by only building extra head(s) if `len(gammas) > 1`.
- Gradients from auxiliary CE losses flow through encoder, dynamics, reward/value shared trunk, Q ensemble modules—but NOT through policy prior parameters (`_pi`).
- Torch compile: temporarily disabled for modified update path until feature validated; later re-enable with guard.

### 4.2 Logging Requirements
- Two logging tiers controlled by new config flag `multi_gamma.debug_logging` (bool):
	- Standard (false): per-γ CE scalar, decoded scalar prediction mean/std, param counts (once at init), optional snapshot at existing `eval_freq` (minimal: distributions + decoded vs target for first few batch elements, e.g., 4).
	- Debug (true): richer snapshot (up to `log_num_examples` examples), per-γ per-step CE breakdown (averaged), decoded scalars list, bootstrap scalars, raw two-hot bin indices.
- Overhead soft cap ~20% (not instrumented; rely on user adjusting `eval_freq` or disabling `debug_logging`).
- Always log parameter counts per major component once: encoder, dynamics, reward, termination (if any), policy prior, Q ensemble, auxiliary head(s).
- Include decoded scalar predictions (inverse symlog expectation) for each γ in snapshot logs.

### 4.3 Config Additions (Refined)
```yaml
multi_gamma:
	enabled: false            # default off preserves pure baseline
	gammas: []                # empty or omitted => baseline
	head: joint               # joint | separate
	loss_weight: 0.5          # scalar applied to EACH auxiliary γ (excluding primary)
	debug_logging: false      # richer diagnostics
	log_num_examples: 8       # max examples in debug snapshots
```
Rules:
- If `enabled=true` and `gammas` empty -> error.
- If `enabled=true` and `len(gammas)==1` -> treat as baseline (no extra heads); warn.
- Validate `gammas[0] == primary_discount` (tolerance 1e-6) else raise.
- Enforce `len(gammas) <= 6`.

### 4.4 Model Integration Steps
Phase 1 (Config & Stubs):
1. Add config parsing & validation (parser or a small helper in `tdmpc2.py`).
2. Add placeholder attributes on `WorldModel` (e.g., `_multi_gamma_head=None`). No functional changes.

Phase 2 (Head Implementation):
3. Implement joint head (Linear: latent_dim → bins * (#gammas)) created only if `len(gammas)>1`.
4. Add separate-head option (list of Linear layers) behind `head: separate`.
5. Record parameter counts at init; log once.

Phase 3 (Forward & Target Logic):
6. In `_update`, after existing TD target computation, reuse encoded next_z & existing TD target path to get primary bootstrap; compute scalar TD for each γ (one-step). Stop-grad bootstrap.
7. Two-hot encode each auxiliary scalar with existing bin support (`cfg.num_bins`, `cfg.vmin`, `cfg.vmax`).
8. Forward pass aux head(s) on each time step latent `zs[:-1]` (mirror reward/value usage) producing logits per γ.

Phase 4 (Loss Integration):
9. For each auxiliary γ (excluding γ[0]), compute CE vs two-hot target across time steps with weight `rho**t`, average by horizon, multiply by `loss_weight`.
10. Accumulate into `total_loss` BEFORE backward; ensure no change to policy loss path.
11. Track metrics: `aux_value_ce/g{γ}` and aggregate `aux_value_ce/mean`.

Phase 5 (Logging & Snapshots):
12. Standard logging: attach per-γ CE scalars & decoded means.
13. Debug logging (at `eval_freq`): collect small replay sample, log (a) per-γ target scalar, (b) predicted scalar, (c) absolute error, (d) top-2 bin indices probabilities.
14. Keep snapshot size bounded by `log_num_examples`.

Phase 6 (Baseline Equivalence Check):
15. Add assertion path: if feature disabled or single γ, assert auxiliary structures are `None` and no aux metrics emitted.

Phase 7 (Compile Re-Enable):
16. After basic validation run (manual), wrap aux computations so they are shape-static; re-enable compile if `cfg.compile` true.

### 4.5 Testing & Validation (Internal)
- Smoke test: `cheetah-run`, 100k steps, `enabled=false` vs baseline run diff should show negligible divergence in primary losses.
- Functionality test: same with `gammas=[primary, 0.97, 0.995]`, confirm aux heads produce nontrivial CE decreasing over time.
- Parameter delta printout: show additional param count for added heads.

### 4.6 Checkpoint Considerations
- No special migration: old checkpoints load (aux head absent) when disabled.
- New checkpoints automatically contain aux head params when present.
- (Optional) Embed `gammas` list in a small metadata dict inside checkpoint under key `multi_gamma` for traceability (non-blocking; skip if not needed).

### 4.7 Potential Future Hooks (Not Implemented Now)
- Adaptive per-γ loss normalization.
- Curriculum enabling of auxiliary heads mid-training.
- Per-γ gradient norm monitoring to auto-scale `loss_weight`.

### 4.8 Abort / Rollback Strategy
- If instability (e.g., value loss explosion) occurs, first set `loss_weight=0.25`; if persists, disable feature; verify baseline unaffected.

---
End of detailed plan. Keep this section synchronized with actual implementation status; update phase numbers as tasks complete.
