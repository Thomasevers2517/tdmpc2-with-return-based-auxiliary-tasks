"""Test: verify MLPWithPrior distributional effect at initialization.

Replicates the exact V-head construction from world_model.py to show:
1. Main MLP logits → decoded scalar value (via two_hot_inv)
2. Main MLP + prior logits → decoded scalar value
3. The magnitude and direction of the prior's shift
"""
import sys
sys.path.insert(0, '/projects/prjs0951/Thomas/Thesis/RL_weather/tdmpc2-with-return-based-auxiliary-tasks/tdmpc2')

import torch
import torch.nn as nn
from types import SimpleNamespace
from common.layers import MLPWithPrior, mlp
from common import init as init_module
from common.math import two_hot_inv

torch.manual_seed(42)

# ---- Config matching default codebase settings ----
cfg = SimpleNamespace(
    latent_dim=512,
    mlp_dim=512,
    value_dim_div=1,
    num_value_layers=2,
    num_bins=101,
    vmin=-10,
    vmax=10,
    bin_size=(10 - (-10)) / (101 - 1),  # 0.2
    prior_hidden_div=16,
    dropout=0.01,
    simnorm_dim=8,
)

v_mlp_dim = cfg.mlp_dim // cfg.value_dim_div  # 512
num_value_layers = cfg.num_value_layers        # 2
num_q = 8
prior_scale = 0.1

print(f"=== MLPWithPrior Distributional Test ===")
print(f"latent_dim={cfg.latent_dim}, v_mlp_dim={v_mlp_dim}, num_value_layers={num_value_layers}")
print(f"num_bins={cfg.num_bins}, vmin={cfg.vmin}, vmax={cfg.vmax}, bin_size={cfg.bin_size:.4f}")
print(f"prior_hidden_div={cfg.prior_hidden_div}, prior_scale={prior_scale}")
print(f"num_q={num_q}")
print()

# ---- Build ensemble of V-heads exactly like world_model.py ----
heads = []
for i in range(num_q):
    h = MLPWithPrior(
        in_dim=cfg.latent_dim,
        hidden_dims=num_value_layers * [v_mlp_dim],
        out_dim=cfg.num_bins,
        prior_hidden_div=cfg.prior_hidden_div,
        prior_scale=prior_scale,
        dropout=cfg.dropout,
        distributional=True,
        cfg=cfg,
    )
    heads.append(h)

# Apply global weight_init (trunc_normal_ std=0.02) — same as self.apply(init.weight_init)
for h in heads:
    h.apply(init_module.weight_init)

# Zero-init output layer weights — same as init.zero_([...params...])
v_output_layer_key = str(num_value_layers)  # "2"
for h in heads:
    # Main MLP output layer is the last Linear in h.main_mlp
    # In an mlp() with 2 hidden layers: layers are 0=Linear, 1=LN, 2=Mish, 3=Linear, 4=LN, 5=Mish, 6=Linear(output)
    # The output layer index = num_value_layers * 3 (each hidden = Linear+LN+Mish)
    output_layer = None
    idx = 0
    for m in h.main_mlp:
        if isinstance(m, nn.Linear):
            if idx == num_value_layers:  # 3rd Linear = output
                output_layer = m
                break
            idx += 1
    assert output_layer is not None, "Could not find output Linear in main_mlp"
    output_layer.weight.data.zero_()

print("Initialization complete (trunc_normal + zero output weights).\n")

# ---- Generate random latent inputs (mimicking SimNorm output) ----
B = 16  # batch
z = torch.randn(B, cfg.latent_dim)
# SimNorm makes groups of simnorm_dim sum-of-squares = 1, approximate:
z = z / (z.norm(dim=-1, keepdim=True) / (cfg.latent_dim ** 0.5))

print(f"Input z stats: mean={z.mean():.4f}, std={z.std():.4f}, norm={z.norm(dim=-1).mean():.2f}")
print()

# ---- Evaluate each head: main-only vs main+prior ----
print("=" * 90)
print(f"{'Head':>4} | {'Main logits':>20} | {'Main value':>12} | {'Full logits':>20} | {'Full value':>12} | {'Shift':>10}")
print(f"{'':>4} | {'mean / std':>20} | {'mean±std':>12} | {'mean / std':>20} | {'mean±std':>12} | {'mean±std':>10}")
print("-" * 90)

all_main_values = []
all_full_values = []

for i, h in enumerate(heads):
    h.eval()
    with torch.no_grad():
        # Main MLP only (no prior)
        main_logits = h.main_mlp(z)    # [B, 101]
        main_value = two_hot_inv(main_logits, cfg).squeeze(-1)  # [B]
        
        # Full forward (main + prior)
        full_logits = h(z)             # [B, 101]
        full_value = two_hot_inv(full_logits, cfg).squeeze(-1)  # [B]
        
        shift = full_value - main_value  # [B]
        
        all_main_values.append(main_value)
        all_full_values.append(full_value)
        
        print(f"{i:>4} | {main_logits.mean():>9.4f} / {main_logits.std():>7.4f} | "
              f"{main_value.mean():>5.3f}±{main_value.std():>5.3f} | "
              f"{full_logits.mean():>9.4f} / {full_logits.std():>7.4f} | "
              f"{full_value.mean():>5.3f}±{full_value.std():>5.3f} | "
              f"{shift.mean():>4.3f}±{shift.std():>4.3f}")

print("-" * 90)

# Stack: [num_q, B]
main_vals = torch.stack(all_main_values)  # [8, B]
full_vals = torch.stack(all_full_values)  # [8, B]

print()
print("=== Cross-head statistics (per sample, then averaged over batch) ===")
main_std_per_sample = main_vals.std(dim=0)   # [B]
full_std_per_sample = full_vals.std(dim=0)   # [B]
print(f"Main-only:  cross-head std = {main_std_per_sample.mean():.6f} (mean over batch)")
print(f"Full:       cross-head std = {full_std_per_sample.mean():.6f} (mean over batch)")
print(f"Ratio (full/main): {(full_std_per_sample.mean() / main_std_per_sample.mean().clamp(min=1e-10)):.2f}x")

print()
print("=== Prior MLP raw output statistics ===")
for i, h in enumerate(heads):
    if h.prior_mlp is not None:
        with torch.no_grad():
            raw = h.prior_mlp(z)  # [B, 1]
            scaled = raw * prior_scale
            print(f"Head {i}: prior_raw mean={raw.mean():>7.4f} std={raw.std():>7.4f} | "
                  f"scaled (×{prior_scale}) mean={scaled.mean():>7.4f} std={scaled.std():>7.4f}")

print()
print("=== What does a shift of X in real space mean for decoded values? ===")
# Create uniform logits (max entropy → expected value ≈ 0 in symlog → 0 in real)
uniform_logits = torch.zeros(1, cfg.num_bins)
base_val = two_hot_inv(uniform_logits, cfg).item()
print(f"Uniform logits → value = {base_val:.4f}")
for shift_val in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    from common.math import shift_scale_distribution
    shifted_logits = shift_scale_distribution(
        uniform_logits, cfg, shift=torch.tensor([[shift_val]])
    )
    shifted_val = two_hot_inv(shifted_logits, cfg).item()
    print(f"  shift={shift_val:>5.2f} → value = {shifted_val:>8.4f}  (delta = {shifted_val - base_val:>8.4f})")
