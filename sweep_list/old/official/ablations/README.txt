Ablations: Individual Contribution Studies
==========================================

Purpose:
Isolate and demonstrate the benefit of each individual contribution/modification.

Subfolders:
- ensemble/                  Effect of dynamics/reward ensemble (without optimism/pessimism)
- optimistic_exploration/    Optimistic exploration during training
- pessimistic_evaluation/    Pessimistic evaluation for stability
- td_targets/                Local vs global bootstrapping, mean vs pessimistic TD targets
- model_utd_buffer_update/   Higher model UTD with fast buffer updates
- policy_entropy/            Jacobian compensation and true entropy
- encoder_freezing/          Freezing encoder for dynamics accuracy
- encoder_dynamics_loss/     Encoder consistency loss and latent variance

Each ablation isolates one factor to clearly show its contribution.
