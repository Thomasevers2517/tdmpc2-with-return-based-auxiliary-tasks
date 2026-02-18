"""Debug script to trace NaN source in the GLOBAL td_target path with H=1.

Reproduces the CUDA assertion crash from sweep 18 baseline:
  - planner_num_dynamics_heads=1 (H=1) → crashes
  - planner_num_dynamics_heads=4 (H=4) → works

Run: CUDA_VISIBLE_DEVICES=1 python "copilot files/tests/debug_nan_crash.py"
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tdmpc2'))

import torch
import torch.nn.functional as F
from common.math import two_hot, two_hot_inv, symlog, symexp, soft_ce


class FakeCfg:
    num_bins = 101
    vmin = -10.0
    vmax = 10.0
    bin_size = 0.2


def test_two_hot_edge_cases():
    """Test two_hot with various edge cases to find NaN triggers."""
    cfg = FakeCfg()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test 1: Normal values
    x = torch.randn(1024, 1, device=device) * 100
    result = two_hot(x, cfg)
    print(f"Test 1 (randn*100): NaN={result.isnan().any()}, Inf={result.isinf().any()}")

    # Test 2: Extreme values
    x = torch.tensor([[1e10], [-1e10], [0.0], [float('inf')], [float('-inf')]], device=device)
    for i, val in enumerate(x):
        sl = symlog(val)
        cl = torch.clamp(sl, cfg.vmin, cfg.vmax)
        bi = torch.floor((cl - cfg.vmin) / cfg.bin_size)
        print(f"  val={val.item():.2e} → symlog={sl.item():.6f} → clamp={cl.item():.6f} → bin_idx={bi.item():.0f} "
              f"(NaN={sl.isnan().any().item()}, clamp_NaN={cl.isnan().any().item()})")

    # Test 3: NaN input
    x_nan = torch.tensor([[float('nan')]], device=device)
    sl_nan = symlog(x_nan)
    cl_nan = torch.clamp(sl_nan, cfg.vmin, cfg.vmax)
    bi_nan = torch.floor((cl_nan - cfg.vmin) / cfg.bin_size)
    print(f"  val=NaN → symlog={sl_nan.item()} → clamp={cl_nan.item()} → bin_idx={bi_nan.item()}")
    print(f"  *** NaN passes through clamp unchanged! bin_idx is NaN → .long() gives garbage ***")

    # Test 4: two_hot_inv always produces bounded values
    random_logits = torch.randn(5, 1, 1024, 101, device=device) * 10
    v_values = two_hot_inv(random_logits, cfg)
    print(f"\nTest 4 (two_hot_inv on random logits*10):")
    print(f"  Range: [{v_values.min():.2f}, {v_values.max():.2f}]")
    print(f"  NaN: {v_values.isnan().any()}, Inf: {v_values.isinf().any()}")

    # Test 5: Simulate full td_target computation
    Ve, T, H, B = 5, 1, 1, 1024
    R = 1
    discount = 0.99

    v_logits = torch.randn(Ve, T, H * B, 101, device=device)
    v_values = two_hot_inv(v_logits, cfg)  # [Ve, T, H*B, 1]
    v_values_reshaped = v_values.view(Ve, T, H, B, 1)

    r_logits = torch.randn(R, T, H * B, 101, device=device)
    r_decoded = two_hot_inv(r_logits, cfg).view(T, R, H, B, 1)

    r_mean = r_decoded.mean(dim=1)
    v_mean = v_values_reshaped.mean(dim=0)

    terminated = torch.zeros(T, H, B, 1, device=device)
    td = r_mean + discount * (1 - terminated) * v_mean
    td_flat = td.mean(dim=1)  # reduce H=1
    td_targets = td_flat.unsqueeze(0).expand(Ve, T, B, 1)
    td_ce = td_targets.contiguous().view(Ve * T * B, 1)

    print(f"\nTest 5 (full td_target sim H={H}):")
    print(f"  v_values range: [{v_values.min():.2f}, {v_values.max():.2f}]")
    print(f"  r_decoded range: [{r_decoded.min():.2f}, {r_decoded.max():.2f}]")
    print(f"  td range: [{td.min():.2f}, {td.max():.2f}]")
    print(f"  td NaN: {td.isnan().any()}, Inf: {td.isinf().any()}")
    print(f"  td_ce NaN: {td_ce.isnan().any()}")

    # Now call two_hot on td_targets — this is where the crash would happen
    try:
        result = two_hot(td_ce, cfg)
        print(f"  two_hot(td_ce): NaN={result.isnan().any()}, OK!")
    except Exception as e:
        print(f"  two_hot(td_ce) CRASHED: {e}")

    # Test 6: Same but with H=4
    H4 = 4
    v_logits_h4 = torch.randn(Ve, T, H4 * B, 101, device=device)
    v_values_h4 = two_hot_inv(v_logits_h4, cfg).view(Ve, T, H4, B, 1)
    r_logits_h4 = torch.randn(R, T, H4 * B, 101, device=device)
    r_decoded_h4 = two_hot_inv(r_logits_h4, cfg).view(T, R, H4, B, 1)
    r_mean_h4 = r_decoded_h4.mean(dim=1)
    v_mean_h4 = v_values_h4.mean(dim=0)
    terminated_h4 = torch.zeros(T, H4, B, 1, device=device)
    td_h4 = r_mean_h4 + discount * (1 - terminated_h4) * v_mean_h4
    td_flat_h4 = td_h4.mean(dim=1)
    td_targets_h4 = td_flat_h4.unsqueeze(0).expand(Ve, T, B, 1)
    td_ce_h4 = td_targets_h4.contiguous().view(Ve * T * B, 1)

    print(f"\nTest 6 (full td_target sim H={H4}):")
    print(f"  td range: [{td_h4.min():.2f}, {td_h4.max():.2f}]")
    print(f"  td NaN: {td_h4.isnan().any()}")
    try:
        result_h4 = two_hot(td_ce_h4, cfg)
        print(f"  two_hot(td_ce_h4): NaN={result_h4.isnan().any()}, OK!")
    except Exception as e:
        print(f"  two_hot(td_ce_h4) CRASHED: {e}")


def test_with_trained_model():
    """Test with actual model weights after simulated WM-only training.
    
    This simulates the scenario: 2499 WM updates train the encoder+dynamics
    but V target stays random. Then on update 2500, calculate_value_loss
    runs for the first time.
    """
    print("\n" + "="*60)
    print("Test with model (simulating 2499 WM-only updates)")
    print("="*60)
    
    try:
        import hydra
        from omegaconf import OmegaConf
        from tdmpc2 import TDMPC2
        
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tdmpc2', 'config.yaml')
        cfg = OmegaConf.load(config_path)
        
        # Override with baseline sweep params
        cfg.task = 'quadruped-walk'
        cfg.planner_num_dynamics_heads = 1
        cfg.num_q = 5
        cfg.local_td_bootstrap = False
        cfg.value_per_dynamics = False
        cfg.td_target_std_coef = 0
        cfg.imagination_horizon = 1
        cfg.compile = False  # Disable compile for debugging
        cfg.num_rollouts = 1
        cfg.seed = 1
        cfg.steps = 100000
        
        print(f"H={cfg.planner_num_dynamics_heads}, Ve={cfg.num_q}")
        print(f"local_td_bootstrap={cfg.local_td_bootstrap}")
        print(f"compile={cfg.compile}")
        
        # This would need full env setup, so skip for now
        print("(Full model test requires env setup - skipping)")
        
    except Exception as e:
        print(f"Model test skipped: {e}")


if __name__ == '__main__':
    test_two_hot_edge_cases()
    test_with_trained_model()
