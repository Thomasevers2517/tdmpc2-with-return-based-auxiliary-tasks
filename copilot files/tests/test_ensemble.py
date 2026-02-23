"""Comprehensive tests for the Ensemble class in common/layers.py.

Tests cover:
  - Correctness: N=1 and N=4 ensembles match manually-called modules
  - Modes: broadcast (split_data=False) and split_data=True
  - Compilation: same results with and without torch.compile
  - Gradient flow: grads propagate through the ensemble
  - EMA (.lerp_): params update correctly
  - Edge cases: assertions on wrong split_data shapes

Run:
  python "copilot files/tests/test_ensemble.py"
  # or with srun for GPU:
  srun --gres=gpu:1 python "copilot files/tests/test_ensemble.py"
"""

import sys
import os
import time

# Allow imports from tdmpc2/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tdmpc2'))

import torch
import torch.nn as nn
from copy import deepcopy
from common.layers import Ensemble, NormedLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mlp(in_dim: int, hidden: int, out_dim: int, seed: int = 0) -> nn.Sequential:
    """Build a small deterministic MLP for testing."""
    torch.manual_seed(seed)
    return nn.Sequential(
        NormedLinear(in_dim, hidden),
        nn.Linear(hidden, out_dim),
    )


def make_ensemble(n: int, in_dim: int = 8, hidden: int = 16,
                  out_dim: int = 4, device: str = 'cuda') -> Ensemble:
    """Create an Ensemble of n small MLPs on `device`."""
    modules = [make_mlp(in_dim, hidden, out_dim, seed=i).to(device) for i in range(n)]
    return Ensemble(modules).to(device)


def reference_outputs(modules: list[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """Manually call each module and stack results → [N, *batch, out]."""
    return torch.stack([m(x) for m in modules], dim=0)


def reference_outputs_split(modules: list[nn.Module], xs: torch.Tensor) -> torch.Tensor:
    """Split xs[i] → module[i] and stack results → [N, *batch, out]."""
    return torch.stack([m(xs[i]) for i, m in enumerate(modules)], dim=0)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_n1_broadcast_correctness(device: str = 'cuda'):
    """N=1 ensemble broadcast should match a direct module call."""
    torch.manual_seed(42)
    m = make_mlp(8, 16, 4, seed=42).to(device)
    ens = Ensemble([m]).to(device)
    x = torch.randn(32, 8, device=device)

    ref = m(x).unsqueeze(0)                     # float32[1, 32, 4]
    out = ens(x)                                 # float32[1, 32, 4]

    assert out.shape == (1, 32, 4), f"Shape mismatch: {out.shape}"
    assert torch.allclose(ref, out, atol=1e-6), (
        f"N=1 broadcast output mismatch: max diff {(ref - out).abs().max():.2e}"
    )
    print("  PASS test_n1_broadcast_correctness")


def test_n1_split_data_correctness(device: str = 'cuda'):
    """N=1 ensemble split_data should match a direct module call on squeezed input."""
    torch.manual_seed(42)
    m = make_mlp(8, 16, 4, seed=42).to(device)
    ens = Ensemble([m]).to(device)
    x = torch.randn(1, 32, 8, device=device)    # float32[H=1, 32, 8]

    ref = m(x[0]).unsqueeze(0)                   # float32[1, 32, 4]
    out = ens(x, split_data=True)                # float32[1, 32, 4]

    assert out.shape == (1, 32, 4), f"Shape mismatch: {out.shape}"
    assert torch.allclose(ref, out, atol=1e-6), (
        f"N=1 split_data mismatch: max diff {(ref - out).abs().max():.2e}"
    )
    print("  PASS test_n1_split_data_correctness")


def test_n4_broadcast_correctness(device: str = 'cuda'):
    """N=4 broadcast should match manually stacked module calls."""
    modules = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(4)]
    ens = Ensemble(deepcopy(modules)).to(device)
    x = torch.randn(32, 8, device=device)

    ref = reference_outputs(modules, x)          # float32[4, 32, 4]
    out = ens(x)                                  # float32[4, 32, 4]

    assert out.shape == (4, 32, 4), f"Shape mismatch: {out.shape}"
    assert torch.allclose(ref, out, atol=1e-5), (
        f"N=4 broadcast mismatch: max diff {(ref - out).abs().max():.2e}"
    )
    print("  PASS test_n4_broadcast_correctness")


def test_n4_split_data_correctness(device: str = 'cuda'):
    """N=4 split_data should match each module called with its own slice."""
    modules = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(4)]
    ens = Ensemble(deepcopy(modules)).to(device)
    xs = torch.randn(4, 32, 8, device=device)    # float32[H=4, 32, 8]

    ref = reference_outputs_split(modules, xs)    # float32[4, 32, 4]
    out = ens(xs, split_data=True)                # float32[4, 32, 4]

    assert out.shape == (4, 32, 4), f"Shape mismatch: {out.shape}"
    assert torch.allclose(ref, out, atol=1e-5), (
        f"N=4 split_data mismatch: max diff {(ref - out).abs().max():.2e}"
    )
    print("  PASS test_n4_split_data_correctness")


def test_n1_is_direct_module_call(device: str = 'cuda'):
    """Verify N=1 doesn't use functional_call — _single_module IS the module."""
    m = make_mlp(8, 16, 4, seed=0).to(device)
    ens = Ensemble([m]).to(device)

    assert hasattr(ens, '_single_module'), "N=1 ensemble should have _single_module"
    assert not hasattr(ens, '_module') or '_module' not in ens.__dict__, (
        "N=1 ensemble should NOT have _module (meta template)"
    )
    # _single_module should share the same parameter tensors
    for (name_e, p_e), (name_m, p_m) in zip(
        ens._single_module.named_parameters(), m.named_parameters()
    ):
        assert p_e.data_ptr() == p_m.data_ptr(), (
            f"N=1 _single_module param {name_e} should share storage with original module"
        )
    print("  PASS test_n1_is_direct_module_call")


def test_n4_has_meta_template(device: str = 'cuda'):
    """Verify N>1 uses a meta-device template, not a real module."""
    modules = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(4)]
    ens = Ensemble(modules).to(device)

    assert '_module' in ens.__dict__, "N>1 ensemble should have _module in __dict__"
    assert not hasattr(ens, '_single_module'), "N>1 ensemble should NOT have _single_module"
    # Meta device check
    for p in ens._module.parameters():
        assert p.device.type == 'meta', f"Template param on {p.device}, expected meta"
    print("  PASS test_n4_has_meta_template")


def test_gradient_flow_n1(device: str = 'cuda'):
    """Gradients should propagate through N=1 ensemble."""
    m = make_mlp(8, 16, 4, seed=0).to(device)
    ens = Ensemble([m]).to(device)
    x = torch.randn(16, 8, device=device)

    out = ens(x)                                  # float32[1, 16, 4]
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in ens.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed for N=1 ensemble"
    assert all(g.abs().sum() > 0 for g in grads), "Some gradients are all-zero"
    print("  PASS test_gradient_flow_n1")


def test_gradient_flow_n4(device: str = 'cuda'):
    """Gradients should propagate through N=4 ensemble."""
    modules = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(4)]
    ens = Ensemble(modules).to(device)
    x = torch.randn(16, 8, device=device)

    out = ens(x)                                  # float32[4, 16, 4]
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in ens.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed for N=4 ensemble"
    assert all(g.abs().sum() > 0 for g in grads), "Some gradients are all-zero"
    print("  PASS test_gradient_flow_n4")


def test_ema_lerp(device: str = 'cuda'):
    """EMA .lerp_() should work identically for N=1 and N=4."""
    for n in (1, 4):
        modules = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(n)]
        ens = Ensemble(deepcopy(modules)).to(device)
        target_modules = [make_mlp(8, 16, 4, seed=i + 100).to(device) for i in range(n)]
        target_ens = Ensemble(target_modules).to(device)

        tau = 0.005
        before = target_ens.params.data.clone()
        target_ens.params.data.lerp_(ens.params.data, tau)
        after = target_ens.params.data

        # Params should have moved toward ens params
        diff = (after - before).abs().sum()
        assert diff > 0, f"N={n}: lerp_ did not change target params"
    print("  PASS test_ema_lerp")


def test_params_indexing(device: str = 'cuda'):
    """self.params[i] should work for N=1 and N=4."""
    for n in (1, 4):
        modules = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(n)]
        ens = Ensemble(modules).to(device)
        for i in range(n):
            p = ens.params[i]
            assert p is not None, f"N={n}: params[{i}] returned None"
    print("  PASS test_params_indexing")


def test_split_data_wrong_shape_asserts(device: str = 'cuda'):
    """split_data with wrong dim-0 should raise AssertionError."""
    for n in (1, 4):
        modules = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(n)]
        ens = Ensemble(modules).to(device)
        bad_x = torch.randn(n + 1, 16, 8, device=device)
        try:
            ens(bad_x, split_data=True)
            assert False, f"N={n}: should have raised AssertionError for wrong dim-0"
        except AssertionError:
            pass
    print("  PASS test_split_data_wrong_shape_asserts")


def test_compile_n1_broadcast(device: str = 'cuda'):
    """torch.compile should produce identical results for N=1 broadcast."""
    torch.manual_seed(42)
    m = make_mlp(8, 16, 4, seed=42).to(device)
    ens = Ensemble([m]).to(device)
    x = torch.randn(32, 8, device=device)

    out_eager = ens(x).clone()

    ens_compiled = torch.compile(ens, mode='reduce-overhead')
    # Warm up compile
    for _ in range(3):
        _ = ens_compiled(torch.randn(32, 8, device=device))
    out_compiled = ens_compiled(x)

    assert torch.allclose(out_eager, out_compiled, atol=1e-5), (
        f"N=1 compile broadcast mismatch: max diff {(out_eager - out_compiled).abs().max():.2e}"
    )
    print("  PASS test_compile_n1_broadcast")


def test_compile_n1_split_data(device: str = 'cuda'):
    """torch.compile should produce identical results for N=1 split_data."""
    torch.manual_seed(42)
    m = make_mlp(8, 16, 4, seed=42).to(device)
    ens = Ensemble([m]).to(device)
    x = torch.randn(1, 32, 8, device=device)

    out_eager = ens(x, split_data=True).clone()

    ens_compiled = torch.compile(ens, mode='reduce-overhead')
    for _ in range(3):
        _ = ens_compiled(torch.randn(1, 32, 8, device=device), split_data=True)
    out_compiled = ens_compiled(x, split_data=True)

    assert torch.allclose(out_eager, out_compiled, atol=1e-5), (
        f"N=1 compile split_data mismatch: max diff {(out_eager - out_compiled).abs().max():.2e}"
    )
    print("  PASS test_compile_n1_split_data")


def test_compile_n4_broadcast(device: str = 'cuda'):
    """torch.compile should produce identical results for N=4 broadcast."""
    modules = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(4)]
    ens = Ensemble(deepcopy(modules)).to(device)
    x = torch.randn(32, 8, device=device)

    out_eager = ens(x).clone()

    ens_compiled = torch.compile(ens, mode='reduce-overhead')
    for _ in range(3):
        _ = ens_compiled(torch.randn(32, 8, device=device))
    out_compiled = ens_compiled(x)

    assert torch.allclose(out_eager, out_compiled, atol=1e-4), (
        f"N=4 compile broadcast mismatch: max diff {(out_eager - out_compiled).abs().max():.2e}"
    )
    print("  PASS test_compile_n4_broadcast")


def test_compile_n4_split_data(device: str = 'cuda'):
    """torch.compile should produce identical results for N=4 split_data."""
    modules = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(4)]
    ens = Ensemble(deepcopy(modules)).to(device)
    xs = torch.randn(4, 32, 8, device=device)

    out_eager = ens(xs, split_data=True).clone()

    ens_compiled = torch.compile(ens, mode='reduce-overhead')
    for _ in range(3):
        _ = ens_compiled(torch.randn(4, 32, 8, device=device), split_data=True)
    out_compiled = ens_compiled(xs, split_data=True)

    assert torch.allclose(out_eager, out_compiled, atol=1e-4), (
        f"N=4 compile split_data mismatch: max diff {(out_eager - out_compiled).abs().max():.2e}"
    )
    print("  PASS test_compile_n4_split_data")


def test_compile_repeated_calls_n1(device: str = 'cuda'):
    """Repeated compiled calls with N=1 should stay consistent (cudagraph stability)."""
    torch.manual_seed(42)
    m = make_mlp(8, 16, 4, seed=42).to(device)
    ens = Ensemble([m]).to(device)

    ens_compiled = torch.compile(ens, mode='reduce-overhead')

    # Warm up
    for _ in range(5):
        _ = ens_compiled(torch.randn(64, 8, device=device))

    # Run 20 times with same input — all outputs should be identical
    x = torch.randn(64, 8, device=device)
    outputs = [ens_compiled(x).clone() for _ in range(20)]
    for i, o in enumerate(outputs[1:], 1):
        assert torch.allclose(outputs[0], o, atol=1e-6), (
            f"N=1 compiled call {i} differs from call 0: "
            f"max diff {(outputs[0] - o).abs().max():.2e}"
        )
    print("  PASS test_compile_repeated_calls_n1")


def test_compile_gradient_n1(device: str = 'cuda'):
    """Gradients should work through compiled N=1 ensemble."""
    m = make_mlp(8, 16, 4, seed=0).to(device)
    ens = Ensemble([m]).to(device)
    ens_compiled = torch.compile(ens)
    x = torch.randn(16, 8, device=device)

    out = ens_compiled(x)
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in ens.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients via compiled N=1 ensemble"
    print("  PASS test_compile_gradient_n1")


def test_compile_gradient_n4(device: str = 'cuda'):
    """Gradients should work through compiled N=4 ensemble."""
    modules = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(4)]
    ens = Ensemble(modules).to(device)
    ens_compiled = torch.compile(ens)
    x = torch.randn(16, 8, device=device)

    out = ens_compiled(x)
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in ens.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients via compiled N=4 ensemble"
    print("  PASS test_compile_gradient_n4")


def test_to_device(device: str = 'cuda'):
    """Ensemble.to(device) should move all params correctly for N=1 and N=4."""
    for n in (1, 4):
        modules = [make_mlp(8, 16, 4, seed=i) for i in range(n)]  # CPU
        ens = Ensemble(modules)
        ens = ens.to(device)
        x = torch.randn(16, 8, device=device)
        out = ens(x)
        assert out.device.type == device.split(':')[0], (
            f"N={n}: Output on {out.device}, expected {device}"
        )
    print("  PASS test_to_device")


def test_known_linear_output(device: str = 'cuda'):
    """N=1 ensemble around a single Linear should produce W @ x + b exactly."""
    torch.manual_seed(0)
    linear = nn.Linear(4, 3).to(device)
    ens = Ensemble([linear]).to(device)

    x = torch.randn(8, 4, device=device)
    ref = linear(x)                               # float32[8, 3]
    out = ens(x)                                   # float32[1, 8, 3]

    assert torch.allclose(ref, out[0], atol=1e-6), (
        f"Known linear mismatch: max diff {(ref - out[0]).abs().max():.2e}"
    )
    print("  PASS test_known_linear_output")


def test_n1_vs_n4_first_member_matches(device: str = 'cuda'):
    """N=1 ens with seed=0 should match the first member of N=4 ens with seeds 0-3."""
    m0 = make_mlp(8, 16, 4, seed=0).to(device)
    ens1 = Ensemble([deepcopy(m0)]).to(device)

    modules4 = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(4)]
    ens4 = Ensemble(modules4).to(device)

    x = torch.randn(32, 8, device=device)
    out1 = ens1(x)                                # float32[1, 32, 4]
    out4 = ens4(x)                                # float32[4, 32, 4]

    assert torch.allclose(out1[0], out4[0], atol=1e-5), (
        f"N=1 vs N=4[0] mismatch: max diff {(out1[0] - out4[0]).abs().max():.2e}"
    )
    print("  PASS test_n1_vs_n4_first_member_matches")


def test_repr(device: str = 'cuda'):
    """String representation should show correct ensemble size."""
    m1 = Ensemble([make_mlp(8, 16, 4, seed=0).to(device)])
    m4 = Ensemble([make_mlp(8, 16, 4, seed=i).to(device) for i in range(4)])
    assert '1x' in repr(m1), f"N=1 repr should contain '1x': {repr(m1)}"
    assert '4x' in repr(m4), f"N=4 repr should contain '4x': {repr(m4)}"
    print("  PASS test_repr")


def test_len(device: str = 'cuda'):
    """len(ensemble) should return ensemble size."""
    for n in (1, 4):
        modules = [make_mlp(8, 16, 4, seed=i).to(device) for i in range(n)]
        ens = Ensemble(modules)
        assert len(ens) == n, f"len() returned {len(ens)}, expected {n}"
    print("  PASS test_len")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    # Correctness
    test_n1_broadcast_correctness,
    test_n1_split_data_correctness,
    test_n4_broadcast_correctness,
    test_n4_split_data_correctness,
    test_known_linear_output,
    test_n1_vs_n4_first_member_matches,
    # Structural
    test_n1_is_direct_module_call,
    test_n4_has_meta_template,
    test_repr,
    test_len,
    # Gradients
    test_gradient_flow_n1,
    test_gradient_flow_n4,
    # EMA / params
    test_ema_lerp,
    test_params_indexing,
    # Assertions
    test_split_data_wrong_shape_asserts,
    # Device
    test_to_device,
    # Compile
    test_compile_n1_broadcast,
    test_compile_n1_split_data,
    test_compile_n4_broadcast,
    test_compile_n4_split_data,
    test_compile_repeated_calls_n1,
    test_compile_gradient_n1,
    test_compile_gradient_n4,
]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Ensemble tests on device={device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("-" * 60)

    passed = 0
    failed = 0
    errors = []

    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            test_fn(device=device)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  FAIL {name}: {e}")
        # Clear GPU cache between compile tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch._dynamo.reset()

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
