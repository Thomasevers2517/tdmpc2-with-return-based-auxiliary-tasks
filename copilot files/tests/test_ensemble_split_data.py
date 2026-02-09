"""Tests for Ensemble split_data functionality.

Verifies:
1. Broadcast mode: same input → different outputs (different weights per head)
2. Split mode: per-head input slicing works correctly
3. Equivalence: split_data with duplicated input == broadcast
4. Gradient flow through both modes
5. DynamicsHeadWithPrior integration (the actual use case)
6. Shape correctness throughout
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tdmpc2'))

import torch
import torch.nn as nn
from types import SimpleNamespace
from common.layers import Ensemble, DynamicsHeadWithPrior, MLPWithPrior, mlp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PASSED = 0
FAILED = 0
T_START = time.time()

print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")
print()


def check(name: str, condition: bool, detail: str = ""):
    """Log a single test assertion with PASS/FAIL status."""
    global PASSED, FAILED
    elapsed = time.time() - T_START
    if condition:
        PASSED += 1
        print(f"  [{elapsed:6.2f}s] PASS: {name}")
    else:
        FAILED += 1
        print(f"  [{elapsed:6.2f}s] FAIL: {name} — {detail}")


def log_tensor(label: str, t: torch.Tensor):
    """Print shape, dtype, device, and basic stats for a tensor."""
    print(f"    {label}: shape={list(t.shape)}, dtype={t.dtype}, device={t.device}, "
          f"min={t.min().item():.4f}, max={t.max().item():.4f}, mean={t.mean().item():.4f}")


def log_ensemble_info(name: str, ens: Ensemble):
    """Print summary of an Ensemble: num heads, param count, devices."""
    n_params = sum(p.numel() for p in ens.parameters() if not p.is_meta)
    n_meta = sum(p.numel() for p in ens.parameters() if p.is_meta)
    devices = set(str(p.device) for p in ens.parameters() if not p.is_meta)
    print(f"    {name}: H={ens.params.shape[0] if hasattr(ens.params, 'shape') else '?'}, "
          f"trainable_params={n_params:,}, meta_params={n_meta:,}, devices={devices}")


# ---------------------------------------------------------------------------
# 1. Basic Ensemble with simple MLPs
# ---------------------------------------------------------------------------
print("\n=== Test 1: Broadcast mode (split_data=False) ===")
H, B, in_dim, out_dim = 4, 8, 16, 10
print(f"  Config: H={H}, B={B}, in_dim={in_dim}, out_dim={out_dim}")
modules = [nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, out_dim)) for _ in range(H)]
# Give each head distinct weights so outputs differ
for i, m in enumerate(modules):
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(0.1 * (i + 1))
ens = Ensemble(modules).to(DEVICE)
log_ensemble_info("ens", ens)

x = torch.randn(B, in_dim, device=DEVICE)
out = ens(x)  # float32[H, B, out_dim]
log_tensor("input x", x)
log_tensor("output", out)

check("output shape", out.shape == (H, B, out_dim), f"expected {(H, B, out_dim)}, got {tuple(out.shape)}")
head_diff = (out[0] - out[1]).abs().max().item()
check("heads differ", not torch.allclose(out[0], out[1], atol=1e-5),
      f"head 0 vs 1 max diff = {head_diff:.2e}, should be > 1e-5")
check("output finite", torch.isfinite(out).all().item(), "NaN/Inf in output")

# ---------------------------------------------------------------------------
# 2. Split mode (split_data=True) — per-head input
# ---------------------------------------------------------------------------
print("\n=== Test 2: Split mode (split_data=True) ===")
print(f"  Config: H={H}, B={B}, in_dim={in_dim} — each head gets a different input slice")
x_split = torch.randn(H, B, in_dim, device=DEVICE)  # different input per head
out_split = ens(x_split, split_data=True)  # float32[H, B, out_dim]
log_tensor("input x_split", x_split)
log_tensor("output", out_split)

check("split output shape", out_split.shape == (H, B, out_dim), f"expected {(H, B, out_dim)}, got {tuple(out_split.shape)}")
check("split output finite", torch.isfinite(out_split).all().item(), "NaN/Inf detected")

# ---------------------------------------------------------------------------
# 3. Equivalence: split_data with same input per head == broadcast
# ---------------------------------------------------------------------------
print("\n=== Test 3: Broadcast vs split equivalence ===")
print("  Strategy: broadcast(x) should equal split(x.expand(H,...)) for identical input")
x_single = torch.randn(B, in_dim, device=DEVICE)
out_broadcast = ens(x_single)  # float32[H, B, out_dim]

# Expand same input to all heads: [B, in_dim] -> [H, B, in_dim]
x_expanded = x_single.unsqueeze(0).expand(H, -1, -1).contiguous()
out_split_same = ens(x_expanded, split_data=True)  # float32[H, B, out_dim]

max_diff = (out_broadcast - out_split_same).abs().max().item()
mean_diff = (out_broadcast - out_split_same).abs().mean().item()
print(f"    max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} (atol=1e-5)")
check("broadcast == split (same input)", torch.allclose(out_broadcast, out_split_same, atol=1e-5),
      f"max diff = {max_diff:.2e}")

# ---------------------------------------------------------------------------
# 4. Split mode: each head gets genuinely different data
# ---------------------------------------------------------------------------
print("\n=== Test 4: Split data independence ===")
print("  Strategy: vmap split_data output should match serial functional_call per head")
# Create input where head i gets x_i, verify output matches serial execution
x_per_head = torch.randn(H, B, in_dim, device=DEVICE)
out_split = ens(x_per_head, split_data=True)

# Serial: manually run each head's params on its input via functional_call
from torch.func import functional_call
serial_outs = []
for i in range(H):
    params_i = ens.params[i]
    params_dict = dict(params_i.flatten_keys(".").items())
    out_i = functional_call(ens.module, params_dict, (x_per_head[i],))
    serial_outs.append(out_i)
out_serial = torch.stack(serial_outs, dim=0)  # float32[H, B, out_dim]

max_diff = (out_split - out_serial).abs().max().item()
per_head_diffs = [(out_split[i] - out_serial[i]).abs().max().item() for i in range(H)]
print(f"    Per-head max diffs: {['%.2e' % d for d in per_head_diffs]}")
check("split == serial per-head",
      torch.allclose(out_split, out_serial, atol=1e-5),
      f"max diff = {max_diff:.2e}")

# ---------------------------------------------------------------------------
# 5. Gradient flow
# ---------------------------------------------------------------------------
print("\n=== Test 5: Gradient flow ===")

# Broadcast mode gradients
print("  Testing broadcast mode gradients...")
x_grad = torch.randn(B, in_dim, device=DEVICE, requires_grad=True)
out_grad = ens(x_grad)
loss = out_grad.sum()
loss.backward()
check("broadcast: input grad exists", x_grad.grad is not None)
if x_grad.grad is not None:
    print(f"    input grad norm={x_grad.grad.norm().item():.4f}")
check("broadcast: input grad nonzero", x_grad.grad.abs().sum().item() > 0)

params_with_grad = sum(1 for p in ens.parameters() if not p.is_meta and p.grad is not None)
params_total = sum(1 for p in ens.parameters() if not p.is_meta)
print(f"    Params with grad: {params_with_grad}/{params_total}")
check("broadcast: param grads exist",
      params_with_grad == params_total,
      f"{params_total - params_with_grad} non-meta params missing gradient")

# Zero grads and test split mode
print("  Testing split mode gradients...")
ens.zero_grad()
x_split_grad = torch.randn(H, B, in_dim, device=DEVICE, requires_grad=True)
out_split_grad = ens(x_split_grad, split_data=True)
loss_split = out_split_grad.sum()
loss_split.backward()
check("split: input grad exists", x_split_grad.grad is not None)
if x_split_grad.grad is not None:
    print(f"    input grad norm={x_split_grad.grad.norm().item():.4f}, shape={list(x_split_grad.grad.shape)}")
check("split: input grad nonzero", x_split_grad.grad.abs().sum().item() > 0)

params_with_grad_split = sum(1 for p in ens.parameters() if not p.is_meta and p.grad is not None)
print(f"    Params with grad: {params_with_grad_split}/{params_total}")
check("split: param grads exist",
      params_with_grad_split == params_total,
      f"{params_total - params_with_grad_split} non-meta params missing gradient")

# ---------------------------------------------------------------------------
# 6. DynamicsHeadWithPrior integration
# ---------------------------------------------------------------------------
print("\n=== Test 6: DynamicsHeadWithPrior in Ensemble ===")
H_dyn = 3
latent_dim, action_dim = 64, 12
cfg = SimpleNamespace(
    simnorm_dim=8,
    latent_dim=latent_dim,
    action_dim=action_dim,
    task_dim=0,
    num_bins=101,
    vmin=-10,
    vmax=10,
    bin_size=20.0 / 100,  # (vmax - vmin) / (num_bins - 1)
)
in_dim_dyn = latent_dim + action_dim
print(f"  Config: H_dyn={H_dyn}, latent_dim={latent_dim}, action_dim={action_dim}, in_dim={in_dim_dyn}")
print(f"  SimNorm dim={cfg.simnorm_dim}, num_bins={cfg.num_bins}")
dyn_heads = [
    DynamicsHeadWithPrior(
        in_dim=in_dim_dyn,
        mlp_dims=[128, 128],
        out_dim=latent_dim,
        cfg=cfg,
        prior_hidden_div=4,
        prior_scale=0.1,
        dropout=0.0,
    )
    for _ in range(H_dyn)
]
dyn_ens = Ensemble(dyn_heads).to(DEVICE)
log_ensemble_info("dyn_ens", dyn_ens)

# Broadcast: same za to all heads
print("  Broadcast mode...")
za = torch.randn(32, in_dim_dyn, device=DEVICE)
out_dyn = dyn_ens(za)  # float32[H_dyn, 32, latent_dim]
log_tensor("broadcast output", out_dyn)
check("dyn broadcast shape", out_dyn.shape == (H_dyn, 32, latent_dim), f"expected {(H_dyn, 32, latent_dim)}, got {tuple(out_dyn.shape)}")
check("dyn broadcast finite", torch.isfinite(out_dyn).all().item())

# Split: per-head za
print("  Split mode...")
za_split = torch.randn(H_dyn, 32, in_dim_dyn, device=DEVICE)
out_dyn_split = dyn_ens(za_split, split_data=True)
log_tensor("split output", out_dyn_split)
check("dyn split shape", out_dyn_split.shape == (H_dyn, 32, latent_dim), f"expected {(H_dyn, 32, latent_dim)}, got {tuple(out_dyn_split.shape)}")
check("dyn split finite", torch.isfinite(out_dyn_split).all().item())

# Equivalence check for dynamics
print("  Equivalence check (broadcast vs split with same input)...")
za_single = torch.randn(32, in_dim_dyn, device=DEVICE)
out_dyn_bc = dyn_ens(za_single)
za_expanded = za_single.unsqueeze(0).expand(H_dyn, -1, -1).contiguous()
out_dyn_split_same = dyn_ens(za_expanded, split_data=True)
dyn_max_diff = (out_dyn_bc - out_dyn_split_same).abs().max().item()
dyn_mean_diff = (out_dyn_bc - out_dyn_split_same).abs().mean().item()
print(f"    max_diff={dyn_max_diff:.2e}, mean_diff={dyn_mean_diff:.2e}")
check("dyn broadcast == split (same input)",
      torch.allclose(out_dyn_bc, out_dyn_split_same, atol=1e-5),
      f"max diff = {dyn_max_diff:.2e}")

# Gradient flow through dynamics ensemble
# Note: prior_mlp params have requires_grad=True (for vmap/compile compat)
# but their output is .detach()'ed in forward(), so they correctly receive no grads.
print("  Gradient flow (split mode)...")
dyn_ens.zero_grad()
za_g = torch.randn(H_dyn, 32, in_dim_dyn, device=DEVICE, requires_grad=True)
out_g = dyn_ens(za_g, split_data=True)
out_g.sum().backward()
check("dyn split: input grad exists", za_g.grad is not None)
if za_g.grad is not None:
    print(f"    input grad norm={za_g.grad.norm().item():.4f}, shape={list(za_g.grad.shape)}")

dyn_params_total = sum(1 for p in dyn_ens.parameters() if not p.is_meta)
dyn_params_with_grad = sum(1 for p in dyn_ens.parameters() if not p.is_meta and p.grad is not None)

# Separate main params (should have grads) from prior params (no grads by design)
main_no_grad = [(n, list(p.shape)) for n, p in dyn_ens.named_parameters()
                if not p.is_meta and p.grad is None and "prior_mlp" not in n]
prior_no_grad = [(n, list(p.shape)) for n, p in dyn_ens.named_parameters()
                 if not p.is_meta and p.grad is None and "prior_mlp" in n]
print(f"    Params with grad: {dyn_params_with_grad}/{dyn_params_total}")
if prior_no_grad:
    print(f"    Prior params without grad (expected, detached by design): {len(prior_no_grad)}")
    for name, shape in prior_no_grad:
        print(f"      - {name}: shape={shape}")
if main_no_grad:
    print(f"    UNEXPECTED main params without grad ({len(main_no_grad)}):")
    for name, shape in main_no_grad:
        print(f"      - {name}: shape={shape}")
check("dyn split: main param grads exist",
      len(main_no_grad) == 0, f"{len(main_no_grad)} main (non-prior) params missing grads")
check("dyn split: prior params correctly have no grads",
      len(prior_no_grad) > 0, "expected prior_mlp params to have no grads (detached)")

# ---------------------------------------------------------------------------
# 7. SimNorm output properties (dynamics heads output should be simplex-normalized)
# ---------------------------------------------------------------------------
print("\n=== Test 7: SimNorm properties ===")
simnorm_dim = cfg.simnorm_dim
print(f"  simnorm_dim={simnorm_dim}, latent_dim={latent_dim}, groups={latent_dim // simnorm_dim}")
# Reshape output to simplex groups and check they sum to 1
out_reshaped = out_dyn_split.view(H_dyn, 32, -1, simnorm_dim)  # [H, B, groups, simnorm_dim]
print(f"    reshaped: {list(out_reshaped.shape)} = [H, B, groups, simnorm_dim]")
sums = out_reshaped.sum(dim=-1)  # should be ~1.0 for each group
print(f"    group sums: min={sums.min().item():.6f}, max={sums.max().item():.6f}, mean={sums.mean().item():.6f}")
check("simnorm sums ≈ 1.0",
      torch.allclose(sums, torch.ones_like(sums), atol=1e-4),
      f"min={sums.min().item():.6f} max={sums.max().item():.6f} (expected 1.0)")
min_val = out_dyn_split.min().item()
print(f"    output min={min_val:.6f} (should be ≥ 0)")
check("simnorm outputs ≥ 0",
      (out_dyn_split >= -1e-6).all().item(),
      f"min value = {min_val:.6f}, softmax outputs should be non-negative")

# ---------------------------------------------------------------------------
# 8. Assertion checks
# ---------------------------------------------------------------------------
print("\n=== Test 8: Error handling ===")
print(f"  Testing wrong H: passing H={H + 1} to ensemble with H={H}")
try:
    bad_input = torch.randn(H + 1, B, in_dim, device=DEVICE)  # wrong H
    ens(bad_input, split_data=True)
    check("wrong H assertion", False, "should have raised AssertionError")
except AssertionError as e:
    print(f"    Caught expected AssertionError: {e}")
    check("wrong H assertion", True)

print("  Testing no args with split_data=True")
try:
    ens(split_data=True)  # no args
    check("no args assertion", False, "should have raised AssertionError")
except AssertionError as e:
    print(f"    Caught expected AssertionError: {e}")
    check("no args assertion", True)

# ---------------------------------------------------------------------------
# 9. MLPWithPrior in Ensemble (reward/value heads pattern)
# ---------------------------------------------------------------------------
print("\n=== Test 9: MLPWithPrior in Ensemble (reward pattern) ===")
H_r = 2
print(f"  Config: H_r={H_r}, in_dim={in_dim_dyn}, out_dim=101 (distributional)")
mlps = [
    MLPWithPrior(
        in_dim=in_dim_dyn,
        hidden_dims=[128, 128],
        out_dim=101,
        prior_hidden_div=4,
        prior_scale=0.1,
        prior_logit_scale=5.0,
        dropout=0.0,
        distributional=True,
        cfg=cfg,
    )
    for _ in range(H_r)
]
r_ens = Ensemble(mlps).to(DEVICE)
log_ensemble_info("r_ens", r_ens)
za_r = torch.randn(32, in_dim_dyn, device=DEVICE)
out_r = r_ens(za_r)  # float32[H_r, 32, 101]
log_tensor("reward output", out_r)
check("reward broadcast shape", out_r.shape == (H_r, 32, 101), f"expected {(H_r, 32, 101)}, got {tuple(out_r.shape)}")
check("reward broadcast finite", torch.isfinite(out_r).all().item())

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
elapsed_total = time.time() - T_START
print(f"\n{'='*60}")
print(f"Results: {PASSED} passed, {FAILED} failed out of {PASSED + FAILED} tests")
print(f"Total time: {elapsed_total:.2f}s")
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    mem_alloc = torch.cuda.memory_allocated() / 1024**2
    mem_reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"GPU memory: {mem_alloc:.1f} MB allocated, {mem_reserved:.1f} MB reserved")
print(f"{'='*60}")
if FAILED > 0:
    print("SOME TESTS FAILED!")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
    sys.exit(0)
