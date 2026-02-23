"""Test MLPWithPrior in distributional mode: inspect init-time outputs.

Shows what the prior actually does to V-head logits and decoded values
at initialization (before any training).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tdmpc2'))

import torch
import torch.nn as nn
from types import SimpleNamespace
from common.layers import MLPWithPrior, mlp
from common.math import two_hot_inv, shift_scale_distribution
from common import init as init_module

torch.manual_seed(42)

# ---- Config matching real training defaults ----
cfg = SimpleNamespace(
    latent_dim=512,
    mlp_dim=512,
    value_dim_div=1,
    num_value_layers=2,
    num_bins=101,
    vmin=-10,
    vmax=10,
    bin_size=(10 - (-10)) / (101 - 1),  # 0.2
    dropout=0.01,
    prior_hidden_div=16,
    value_prior_scale=0.1,
    num_q=8,
)

v_mlp_dim = cfg.mlp_dim // cfg.value_dim_div  # 512
print(f"=== V-head config ===")
print(f"  in_dim={cfg.latent_dim}, hidden_dims={cfg.num_value_layers}x[{v_mlp_dim}], "
      f"out_dim={cfg.num_bins}")
print(f"  prior_hidden_div={cfg.prior_hidden_div} → prior_hidden={v_mlp_dim // cfg.prior_hidden_div}")
print(f"  prior_scale={cfg.value_prior_scale}, distributional=True")
print(f"  num_bins={cfg.num_bins}, vmin={cfg.vmin}, vmax={cfg.vmax}, bin_size={cfg.bin_size:.4f}")
print()

# ---- Build 8 heads like the real code ----
heads = []
for i in range(cfg.num_q):
    h = MLPWithPrior(
        in_dim=cfg.latent_dim,
        hidden_dims=cfg.num_value_layers * [v_mlp_dim],
        out_dim=cfg.num_bins,
        prior_hidden_div=cfg.prior_hidden_div,
        prior_scale=cfg.value_prior_scale,
        dropout=cfg.dropout,
        distributional=True,
        cfg=cfg,
    )
    heads.append(h)

# Apply weight_init (trunc_normal_ std=0.02, skips prior)
container = nn.ModuleList(heads)
container.apply(init_module.weight_init)

# Zero-init main MLP output weight (like world_model does)
for h in heads:
    output_layer = h.main_mlp[cfg.num_value_layers]  # last layer (plain nn.Linear)
    assert isinstance(output_layer, nn.Linear), f"Expected nn.Linear, got {type(output_layer)}"
    output_layer.weight.data.zero_()

print("=== Weight stats after init ===")
h0 = heads[0]
# Main MLP output layer
main_out = h0.main_mlp[cfg.num_value_layers]
print(f"  main_mlp output weight: mean={main_out.weight.data.mean():.6f}, "
      f"std={main_out.weight.data.std():.6f}, "
      f"max={main_out.weight.data.abs().max():.6f}")
print(f"  main_mlp output bias:   mean={main_out.bias.data.mean():.6f}, "
      f"std={main_out.bias.data.std():.6f}")
# Prior MLP
if h0.prior_mlp is not None:
    for i, m in enumerate(h0.prior_mlp):
        if isinstance(m, nn.Linear):
            print(f"  prior_mlp layer {i} weight: mean={m.weight.data.mean():.6f}, "
                  f"std={m.weight.data.std():.6f}, "
                  f"max={m.weight.data.abs().max():.6f}")
            print(f"  prior_mlp layer {i} bias:   mean={m.bias.data.mean():.6f}")
print()

# ---- Forward pass with random latent-like inputs ----
B = 32
# Simulate SimNorm-like latent: L2-normalize chunks then scale
z = torch.randn(B, cfg.latent_dim)
z = z / z.norm(dim=-1, keepdim=True) * (cfg.latent_dim ** 0.5)  # ~unit-variance per dim
print(f"=== Input z stats: mean={z.mean():.4f}, std={z.std():.4f}, "
      f"norm={z.norm(dim=-1).mean():.1f} ===\n")

with torch.no_grad():
    for hi, h in enumerate(heads):
        # Main MLP only (no prior)
        main_logits = h.main_mlp(z)           # [B, 101]
        main_value = two_hot_inv(main_logits, cfg)  # [B, 1]

        # Prior raw output
        prior_raw = h.prior_mlp(z)            # [B, 1]
        prior_shift = prior_raw * cfg.value_prior_scale  # [B, 1]

        # Full forward (main + prior via shift_scale_distribution)
        full_logits = h(z)                    # [B, 101]
        full_value = two_hot_inv(full_logits, cfg)   # [B, 1]

        # Delta in decoded value
        delta = full_value - main_value       # [B, 1]

        if hi < 4 or hi == cfg.num_q - 1:  # Print first 4 and last
            print(f"--- Head {hi} ---")
            print(f"  main_logits: mean={main_logits.mean():.6f}, std={main_logits.std():.6f}")
            print(f"  main_value:  mean={main_value.mean():.6f}, std={main_value.std():.6f}")
            print(f"  prior_raw:   mean={prior_raw.mean():.4f}, std={prior_raw.std():.4f}, "
                  f"min={prior_raw.min():.4f}, max={prior_raw.max():.4f}")
            print(f"  prior_shift (raw*{cfg.value_prior_scale}): "
                  f"mean={prior_shift.mean():.4f}, std={prior_shift.std():.4f}, "
                  f"min={prior_shift.min():.4f}, max={prior_shift.max():.4f}")
            print(f"  full_value:  mean={full_value.mean():.6f}, std={full_value.std():.6f}")
            print(f"  value delta (full - main): mean={delta.mean():.6f}, std={delta.std():.6f}, "
                  f"min={delta.min():.6f}, max={delta.max():.6f}")
            print()

# ---- Cross-head disagreement ----
print("=== Cross-head disagreement (init, same inputs) ===")
all_values = []
all_main_values = []
with torch.no_grad():
    for h in heads:
        full_val = two_hot_inv(h(z), cfg)         # [B, 1]
        main_val = two_hot_inv(h.main_mlp(z), cfg)  # [B, 1]
        all_values.append(full_val)
        all_main_values.append(main_val)

all_values = torch.cat(all_values, dim=-1)           # [B, num_q]
all_main_values = torch.cat(all_main_values, dim=-1)  # [B, num_q]

print(f"  Main-only values per sample: mean_of_means={all_main_values.mean():.6f}, "
      f"mean_of_stds={all_main_values.std(dim=1).mean():.6f}")
print(f"  Full values per sample:      mean_of_means={all_values.mean():.6f}, "
      f"mean_of_stds={all_values.std(dim=1).mean():.6f}")
print(f"  Batch std of full values (across B): {all_values.mean(dim=1).std():.6f}")
print()

# ---- What if values were larger (simulating later training)? ----
print("=== Effect of prior shift at different value scales ===")
for target_value in [0.0, 0.5, 5.0, 50.0]:
    # Create logits that decode to ~target_value
    # two_hot target in symlog space
    import torch.nn.functional as F
    symlog_target = torch.sign(torch.tensor(target_value)) * torch.log(torch.tensor(1.0 + abs(target_value)))
    bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins)
    # Create peaked logits at the target bin
    target_idx = ((symlog_target - cfg.vmin) / cfg.bin_size).clamp(0, cfg.num_bins - 1)
    lo = int(target_idx.floor().item())
    frac = float(target_idx.item() - lo)
    fake_logits = torch.full((1, cfg.num_bins), -10.0)  # very low
    fake_logits[0, lo] = 5.0 * (1 - frac)
    fake_logits[0, min(lo + 1, cfg.num_bins - 1)] = 5.0 * frac

    decoded_no_shift = two_hot_inv(fake_logits, cfg).item()

    # Apply shift from each head's prior (use mean prior output magnitude)
    shifts = []
    with torch.no_grad():
        for h in heads:
            # Use the z we already have, take the mean prior output as representative
            pr = h.prior_mlp(z).mean().item() * cfg.value_prior_scale
            shifts.append(pr)
            shifted_logits = shift_scale_distribution(fake_logits, cfg,
                                                       shift=torch.tensor([[pr]]))
            decoded_shifted = two_hot_inv(shifted_logits, cfg).item()

    shift_arr = torch.tensor(shifts)
    print(f"  target≈{target_value:6.1f} | decoded_no_shift={decoded_no_shift:8.4f} | "
          f"prior shifts: mean={shift_arr.mean():.4f} std={shift_arr.std():.4f} "
          f"range=[{shift_arr.min():.4f}, {shift_arr.max():.4f}]")
