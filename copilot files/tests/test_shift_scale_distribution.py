"""Tests for shift_scale_distribution helper in common/math.py."""
import sys
import types
import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs0951/Thomas/Thesis/RL_weather/tdmpc2-with-return-based-auxiliary-tasks/tdmpc2")
from common.math import shift_scale_distribution, two_hot, two_hot_inv, symlog, symexp


def make_cfg(num_bins=101, vmin=-10.0, vmax=10.0):
    """Create a minimal config namespace matching real config."""
    cfg = types.SimpleNamespace()
    cfg.num_bins = num_bins
    cfg.vmin = vmin
    cfg.vmax = vmax
    cfg.bin_size = (vmax - vmin) / (num_bins - 1)
    return cfg


def probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
    """Convert probability targets (e.g. from two_hot) to logits via log."""
    return torch.log(probs + 1e-8)


def delta_logits(value_symlog: torch.Tensor, cfg) -> torch.Tensor:
    """Create near-delta logits at a given symlog-space value.

    Args:
        value_symlog: float32[B, 1] - scalar values in symlog space.
        cfg: config namespace.

    Returns:
        float32[B, K] - logits representing a near-delta at the given value.
    """
    # two_hot(x, cfg) calls symlog(x) internally, so pass symexp(value) to get
    # a delta at the desired symlog-space location.
    raw_value = symexp(value_symlog)
    return probs_to_logits(two_hot(raw_value, cfg))


def expected_value_symlog(logits, cfg):
    """Compute E[x] in symlog space from logits over bins."""
    bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=logits.device, dtype=logits.dtype)
    probs = F.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)  # float32[...]


def test_identity():
    """shift=0, scale=1 should return identical distribution."""
    cfg = make_cfg()
    B, K = 8, cfg.num_bins
    logits = torch.randn(B, K)

    shift = torch.zeros(B, 1)
    scale = torch.ones(B, 1)
    out = shift_scale_distribution(logits, cfg, shift=shift, scale=scale)

    probs_in = F.softmax(logits, dim=-1)
    probs_out = F.softmax(out, dim=-1)

    assert torch.allclose(probs_in, probs_out, atol=1e-5), (
        f"Identity transform should preserve distribution.\n"
        f"Max diff: {(probs_in - probs_out).abs().max().item():.6e}"
    )
    print("PASS test_identity")


def test_shift_moves_mean():
    """Positive shift should increase the expected value."""
    cfg = make_cfg()
    B, K = 16, cfg.num_bins

    # Create a delta distribution centered around symlog=0 (middle of bins)
    logits = delta_logits(torch.zeros(B, 1), cfg)  # float32[B, K], peaked at center

    shift_val = 2.0  # shift by +2 in symlog space
    shift = torch.full((B, 1), shift_val)
    out = shift_scale_distribution(logits, cfg, shift=shift)

    mean_before = expected_value_symlog(logits, cfg)  # float32[B]
    mean_after = expected_value_symlog(out, cfg)  # float32[B]

    # Expected: mean_after ≈ mean_before + shift_val
    expected_mean = mean_before + shift_val
    assert torch.allclose(mean_after, expected_mean, atol=0.05), (
        f"Shifted mean should be ~{expected_mean[0].item():.3f}, "
        f"got {mean_after[0].item():.3f}"
    )
    print(f"PASS test_shift_moves_mean (before={mean_before[0]:.3f}, after={mean_after[0]:.3f}, expected={expected_mean[0]:.3f})")


def test_negative_shift():
    """Negative shift should decrease the expected value."""
    cfg = make_cfg()
    B, K = 8, cfg.num_bins

    logits = delta_logits(torch.full((B, 1), 3.0), cfg)  # delta at symlog=3.0

    shift_val = -4.0
    shift = torch.full((B, 1), shift_val)
    out = shift_scale_distribution(logits, cfg, shift=shift)

    mean_before = expected_value_symlog(logits, cfg)
    mean_after = expected_value_symlog(out, cfg)

    expected_mean = mean_before + shift_val
    assert torch.allclose(mean_after, expected_mean, atol=0.05), (
        f"Expected mean ~{expected_mean[0].item():.3f}, got {mean_after[0].item():.3f}"
    )
    print(f"PASS test_negative_shift (before={mean_before[0]:.3f}, after={mean_after[0]:.3f})")


def test_scale_stretches():
    """Scale > 1 should stretch the mean away from 0."""
    cfg = make_cfg()
    B, K = 16, cfg.num_bins

    logits = delta_logits(torch.full((B, 1), 2.0), cfg)  # delta at symlog=2.0

    scale_val = 1.5
    scale = torch.full((B, 1), scale_val)
    out = shift_scale_distribution(logits, cfg, scale=scale)

    mean_before = expected_value_symlog(logits, cfg)
    mean_after = expected_value_symlog(out, cfg)

    # For a delta-like distribution at c, scale * c should be the new mean
    expected_mean = mean_before * scale_val
    assert torch.allclose(mean_after, expected_mean, atol=0.05), (
        f"Scaled mean should be ~{expected_mean[0].item():.3f}, got {mean_after[0].item():.3f}"
    )
    print(f"PASS test_scale_stretches (before={mean_before[0]:.3f}, after={mean_after[0]:.3f})")


def test_shift_and_scale_combined():
    """Combined shift + scale: E[scale*x + shift] = scale*E[x] + shift."""
    cfg = make_cfg()
    B, K = 16, cfg.num_bins

    logits = delta_logits(torch.full((B, 1), 1.5), cfg)  # delta at symlog=1.5

    shift_val, scale_val = 1.0, 2.0
    shift = torch.full((B, 1), shift_val)
    scale = torch.full((B, 1), scale_val)
    out = shift_scale_distribution(logits, cfg, shift=shift, scale=scale)

    mean_before = expected_value_symlog(logits, cfg)
    mean_after = expected_value_symlog(out, cfg)
    expected_mean = scale_val * mean_before + shift_val

    assert torch.allclose(mean_after, expected_mean, atol=0.1), (
        f"Combined: expected ~{expected_mean[0].item():.3f}, got {mean_after[0].item():.3f}"
    )
    print(f"PASS test_shift_and_scale_combined (expected={expected_mean[0]:.3f}, got={mean_after[0]:.3f})")


def test_clamping_at_boundaries():
    """Shifting beyond vmax should clamp — probability concentrates at boundary."""
    cfg = make_cfg()
    B, K = 4, cfg.num_bins

    # Distribution near vmax
    logits = delta_logits(torch.full((B, 1), 9.0), cfg)  # delta at symlog=9.0

    # Shift by +5, pushing way past vmax=10
    shift = torch.full((B, 1), 5.0)
    out = shift_scale_distribution(logits, cfg, shift=shift)

    mean_after = expected_value_symlog(out, cfg)
    # Should be clamped to vmax
    assert (mean_after >= cfg.vmax - 0.5).all(), (
        f"Expected mean near vmax={cfg.vmax}, got {mean_after[0].item():.3f}"
    )
    print(f"PASS test_clamping_at_boundaries (mean_after={mean_after[0]:.3f}, vmax={cfg.vmax})")


def test_probabilities_sum_to_one():
    """Output probabilities should sum to 1 (valid distribution)."""
    cfg = make_cfg()
    B, K = 32, cfg.num_bins
    logits = torch.randn(B, K)

    shift = torch.randn(B, 1) * 2.0
    scale = torch.ones(B, 1) + torch.randn(B, 1) * 0.3  # scale near 1

    out = shift_scale_distribution(logits, cfg, shift=shift, scale=scale)
    probs = F.softmax(out, dim=-1)
    sums = probs.sum(dim=-1)

    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"Probabilities should sum to 1, got range [{sums.min():.6f}, {sums.max():.6f}]"
    )
    print("PASS test_probabilities_sum_to_one")


def test_batch_broadcasting():
    """Shift/scale should broadcast with arbitrary leading dimensions."""
    cfg = make_cfg()
    E, B, K = 5, 8, cfg.num_bins

    logits = torch.randn(E, B, K)
    shift = torch.randn(E, B, 1)
    scale = torch.ones(E, B, 1)

    out = shift_scale_distribution(logits, cfg, shift=shift, scale=scale)
    assert out.shape == (E, B, K), f"Expected shape {(E, B, K)}, got {out.shape}"
    print(f"PASS test_batch_broadcasting (shape={out.shape})")


def test_broadcast_scalar_shift():
    """A single scalar shift applied to a batch."""
    cfg = make_cfg()
    B, K = 8, cfg.num_bins

    logits = torch.randn(B, K)
    shift = torch.tensor([[1.5]])  # float32[1, 1], should broadcast to [B, 1]

    out = shift_scale_distribution(logits, cfg, shift=shift)
    assert out.shape == (B, K), f"Expected shape {(B, K)}, got {out.shape}"
    print("PASS test_broadcast_scalar_shift")


def test_two_hot_inv_roundtrip():
    """Shifting a delta distribution and inverting should give shifted scalar."""
    cfg = make_cfg()
    B = 16

    original_val = torch.full((B, 1), 3.0)  # symlog space value
    logits = delta_logits(original_val, cfg)

    shift_val = -2.0
    shift = torch.full((B, 1), shift_val)
    shifted_logits = shift_scale_distribution(logits, cfg, shift=shift)

    # two_hot_inv returns symexp(expected_symlog_value)
    recovered = two_hot_inv(shifted_logits, cfg)  # float32[B, 1]
    # Expected: symexp(3.0 - 2.0) = symexp(1.0)
    expected = symexp(original_val + shift_val)

    assert torch.allclose(recovered, expected, atol=0.1), (
        f"Roundtrip: expected {expected[0].item():.3f}, got {recovered[0].item():.3f}"
    )
    print(f"PASS test_two_hot_inv_roundtrip (expected={expected[0].item():.3f}, got={recovered[0].item():.3f})")


def test_no_shift_no_scale_is_identity():
    """Passing shift=None, scale=None should return unchanged logits."""
    cfg = make_cfg()
    logits = torch.randn(4, cfg.num_bins)

    out = shift_scale_distribution(logits, cfg)

    probs_in = F.softmax(logits, dim=-1)
    probs_out = F.softmax(out, dim=-1)
    assert torch.allclose(probs_in, probs_out, atol=1e-5), "No-op should be identity"
    print("PASS test_no_shift_no_scale_is_identity")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_identity()
    test_shift_moves_mean()
    test_negative_shift()
    test_scale_stretches()
    test_shift_and_scale_combined()
    test_clamping_at_boundaries()
    test_probabilities_sum_to_one()
    test_batch_broadcasting()
    test_broadcast_scalar_shift()
    test_two_hot_inv_roundtrip()
    test_no_shift_no_scale_is_identity()
    print("\n=== ALL TESTS PASSED ===")
