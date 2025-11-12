# Project Overview

## High-Level Summary
Research reinforcement learning codebase implementing a model-based agent (TD-MPC2 variant). A learned world model encodes observations into latents, rolls forward latent dynamics, and predicts rewards (and optional termination) along imagined trajectories. The agent plans in the real environment using this latent model: real transitions are collected, improving the world model and yielding value estimates for (state, action) pairs. These value estimates bootstrap planning and policy optimization; auxiliary multi-gamma heads provide richer training signals without altering planner targets.

## Core Components
- World Model: Encodes observations -> latent state; predicts next latent, reward distribution, termination, and Q ensembles (primary + optional auxiliary discount heads). Provides imagined rollouts for planning and training targets.
- Planning (MPPI/CEM): Samples candidate action sequences using a mixture of policy prior trajectories and stochastic perturbations; scores them via latent rollouts + value estimates; refines distribution (mean/std) and outputs first action.
- Policy & Value Heads: Policy prior guides planner search; Q ensembles (distributional over value bins) enable value estimation; auxiliary heads learn returns under alternative discounts to enrich representation.
- Training Loop (Online/Offline): Collects environment transitions; performs multiple updates per environment step (UTD ratio). Separate pathways for policy-only vs full world-model+value updates; periodic agent resets heuristically.
- Losses: Consistency (latent prediction), reward CE (distributional), value CE (distributional), auxiliary value losses, termination BCE; combined via weighted sum with rho-based temporal weighting.
- Replay Buffer: Stores sequences (TensorDict episodes) for sampling training batches; supports validation and recent-evaluation buffers.
- Logging & Scaling: Running scale for reward/value normalization; structured logging of metrics, gradients (optionally), validation performance, and episodic stats.
- Reset/EMA Utilities: Supports shrink/perturb or full resets of encoder/policy components; optional EMA targets for policy and encoder stabilization.

## Key Flows
1. Collect step: env -> obs -> act (planner or policy) -> transition stored.
2. Update: sample batch -> compute latent rollout + losses -> optimize model & policy.
3. Plan: encode current obs -> sample candidate action sequences -> evaluate via imagined rollout + Q -> select and execute.
4. Auxiliary: additional discounts produce targets only for training-time enrichment.

## Notation (Typical Shapes)
- obs: (T+1, B, *obs_shape)
- action: (T, B, A)
- latent z: (T+1, B, L); rollout uses subsets
- reward logits: (T, B, K); TD targets: (T, B, K)
- Q logits: (Qe, T, B, K); aux Q: (T, B, G_aux, K)

## Design Emphasis
Vectorized latent rollouts, distributional value modeling, hybrid planning (model + Q), modular resets, multi-gamma auxiliary supervision, and efficient batch-first tensor operations.
