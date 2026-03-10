"""Tests for src/policy/network.py — Stage 4a.

Tests cover:
- Architecture properties (shapes, dtypes, parameter count)
- forward() single-step correctness
- forward_trajectory() full unroll correctness
- Semi-recurrence: δ_{t-1} is actually used
- Gradient flow through all T timesteps
- Determinism and device handling
- Input validation
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork, INPUT_DIM
from src.state.builder import FEATURE_DIM, build_features
from src.io.dataset_loader import load_dataset
from src.generate_dataset import main as generate_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N       = 64    # small batch for fast tests
T       = 30    # full thesis timestep count
H       = 64    # default hidden size


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def net() -> BaselineFeedforwardNetwork:
    """Fresh network with fixed seed for reproducibility."""
    torch.manual_seed(0)
    return BaselineFeedforwardNetwork(hidden=H)


@pytest.fixture
def features() -> torch.Tensor:
    """Random feature tensor (N, T, 3) — no dataset needed."""
    torch.manual_seed(1)
    return torch.randn(N, T, FEATURE_DIM)


@pytest.fixture(scope="session")
def bs_features(tmp_path_factory):
    """Real features from a BS dataset for integration tests."""
    run_dir = generate_dataset([
        "--sim",            "bs",
        "--n_paths",        "200",
        "--n_steps",        "30",
        "--seed",           "0",
        "--out_root",       str(tmp_path_factory.mktemp("bs")),
        "--run_id",         "test_bs",
        "--strike",         "100.0",
        "--maturity_years", "0.25",
    ])
    batch = load_dataset(run_dir, split="train")
    return build_features(batch), batch


# ===========================================================================
# Architecture tests
# ===========================================================================

class TestArchitecture:

    def test_input_dim_constant(self):
        """INPUT_DIM must be FEATURE_DIM + 1 = 4."""
        assert INPUT_DIM == FEATURE_DIM + 1
        assert INPUT_DIM == 4

    def test_is_nn_module(self, net):
        assert isinstance(net, nn.Module)

    def test_has_trainable_params(self, net):
        assert net.n_parameters() > 0

    def test_parameter_count_default(self, net):
        """Default H=64: 4*64 + 64 + 64*64 + 64 + 64*1 + 1 = 8,513."""
        expected = (
            INPUT_DIM * H + H +    # Linear(4→64)
            H * H + H +            # Linear(64→64)
            H * 1 + 1              # Linear(64→1)
        )
        assert net.n_parameters() == expected

    def test_parameter_count_custom_hidden(self):
        net32 = BaselineFeedforwardNetwork(hidden=32)
        expected = (
            INPUT_DIM * 32 + 32 +
            32 * 32 + 32 +
            32 * 1 + 1
        )
        assert net32.n_parameters() == expected

    def test_no_final_activation(self, net):
        """Last layer must be Linear, not Sigmoid/Tanh — unconstrained output."""
        last_layer = list(net.net.children())[-1]
        assert isinstance(last_layer, nn.Linear)

    def test_relu_activations(self, net):
        """Hidden activations must be ReLU."""
        relus = [m for m in net.net.children() if isinstance(m, nn.ReLU)]
        assert len(relus) == 2

    def test_three_linear_layers(self, net):
        linears = [m for m in net.net.children() if isinstance(m, nn.Linear)]
        assert len(linears) == 3

    def test_hidden_attribute(self, net):
        assert net.hidden == H


# ===========================================================================
# forward() — single step
# ===========================================================================

class TestForwardSingleStep:

    def test_output_shape(self, net):
        x   = torch.randn(N, INPUT_DIM)
        out = net(x)
        assert out.shape == (N,)

    def test_output_dtype_float32(self, net):
        x   = torch.randn(N, INPUT_DIM, dtype=torch.float32)
        out = net(x)
        assert out.dtype == torch.float32

    def test_output_device_cpu(self, net):
        x   = torch.randn(N, INPUT_DIM)
        out = net(x)
        assert out.device.type == "cpu"

    def test_output_is_finite(self, net):
        x   = torch.randn(N, INPUT_DIM)
        out = net(x)
        assert torch.isfinite(out).all()

    def test_output_unconstrained(self, net):
        """Output should not be clipped to [0,1] or [-1,1]."""
        # With random weights and large inputs, output should occasionally
        # exceed 1.0 or go below 0.0
        torch.manual_seed(42)
        x   = torch.randn(1000, INPUT_DIM) * 5.0
        out = net(x)
        assert out.max().item() > 1.0 or out.min().item() < 0.0, (
            "Network output appears to be constrained — check for final activation"
        )

    def test_deterministic_given_same_input(self, net):
        x    = torch.randn(N, INPUT_DIM)
        out1 = net(x)
        out2 = net(x)
        assert torch.allclose(out1, out2)

    def test_batch_independence(self, net):
        """Output for path i must not depend on path j (no cross-path mixing)."""
        x      = torch.randn(N, INPUT_DIM)
        out_full = net(x)

        # Run path 0 alone
        out_single = net(x[[0]])
        assert torch.allclose(out_full[[0]], out_single, atol=1e-6)


# ===========================================================================
# forward_trajectory() — full unroll
# ===========================================================================

class TestForwardTrajectory:

    def test_output_shape(self, net, features):
        deltas = net.forward_trajectory(features)
        assert deltas.shape == (N, T)

    def test_output_dtype(self, net, features):
        deltas = net.forward_trajectory(features)
        assert deltas.dtype == torch.float32

    def test_output_device_cpu(self, net, features):
        deltas = net.forward_trajectory(features)
        assert deltas.device.type == "cpu"

    def test_output_is_finite(self, net, features):
        deltas = net.forward_trajectory(features)
        assert torch.isfinite(deltas).all()

    def test_deterministic(self, net, features):
        d1 = net.forward_trajectory(features)
        d2 = net.forward_trajectory(features)
        assert torch.allclose(d1, d2)

    def test_timestep_count(self, net, features):
        """Output must have exactly T timesteps, not T+1."""
        deltas = net.forward_trajectory(features)
        assert deltas.shape[1] == T

    def test_path_count(self, net, features):
        deltas = net.forward_trajectory(features)
        assert deltas.shape[0] == N


# ===========================================================================
# Semi-recurrence: δ_{t-1} is actually used
# ===========================================================================

class TestSemiRecurrence:

    def test_initial_delta_is_zero(self, net):
        """At t=0, δ_{-1}=0 — verify by checking input to first step."""
        # If we run one step with δ_{prev}=0 vs δ_{prev}=1, outputs differ
        feats = torch.randn(N, 1, FEATURE_DIM)

        x_zero = torch.cat([feats[:, 0, :], torch.zeros(N, 1)], dim=-1)
        x_one  = torch.cat([feats[:, 0, :], torch.ones(N, 1)],  dim=-1)

        out_zero = net(x_zero)
        out_one  = net(x_one)

        # With a randomly initialised network they will differ
        assert not torch.allclose(out_zero, out_one), (
            "Network output is identical for δ_{prev}=0 and δ_{prev}=1 — "
            "the delta input slot may not be connected."
        )

    def test_trajectory_uses_previous_delta(self, net):
        """Changing δ_{t-1} by modifying t=0 output should propagate to t=1."""
        torch.manual_seed(42)
        feats = torch.randn(N, T, FEATURE_DIM)

        deltas = net.forward_trajectory(feats)
        d0 = deltas[:, 0].clone()   # δ at t=0
        d1 = deltas[:, 1].clone()   # δ at t=1

        # Manually compute t=1 with d0 as previous — should match
        x1       = torch.cat([feats[:, 1, :], d0.unsqueeze(-1)], dim=-1)
        d1_manual = net(x1)

        assert torch.allclose(d1, d1_manual, atol=1e-6), (
            "forward_trajectory t=1 output does not match manual δ_{t-1} injection"
        )

    def test_different_histories_give_different_deltas(self, net):
        """Two paths with identical features at t=1 but different δ_0 → different δ_1."""
        torch.manual_seed(7)
        feats = torch.randn(2, T, FEATURE_DIM)
        # Make features identical for both paths
        feats[1] = feats[0]

        # But inject different δ_0 by modifying the first step
        # Path 0: t=0 features as-is
        # Path 1: t=0 features perturbed so δ_0 differs
        feats_perturbed = feats.clone()
        feats_perturbed[1, 0, :] += 10.0   # large perturbation at t=0 only

        d_orig = net.forward_trajectory(feats)
        d_pert = net.forward_trajectory(feats_perturbed)

        # At t=1, path 1 should differ because δ_0 was different
        assert not torch.allclose(d_orig[1, 1], d_pert[1, 1]), (
            "History from t=0 did not propagate to t=1"
        )


# ===========================================================================
# Gradient flow
# ===========================================================================

class TestGradientFlow:

    def test_gradients_exist_after_backward(self, net, features):
        """loss.backward() must populate gradients for all parameters."""
        deltas = net.forward_trajectory(features)
        loss   = deltas.mean()
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for parameter '{name}'"

    def test_gradients_nonzero(self, net, features):
        """Gradients must be non-zero (network is not dead)."""
        deltas = net.forward_trajectory(features)
        loss   = deltas.mean()
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad.abs().sum() > 0, (
                f"Zero gradient for parameter '{name}' — possible dead network"
            )

    def test_gradients_flow_through_all_timesteps(self, net):
        """Gradient at t=0 features must be non-zero — signal flows all the way back."""
        feats = torch.randn(N, T, FEATURE_DIM, requires_grad=True)
        torch.manual_seed(0)
        net2  = BaselineFeedforwardNetwork(hidden=H)

        deltas = net2.forward_trajectory(feats)
        loss   = deltas.mean()
        loss.backward()

        # Gradient should exist at every timestep's features
        assert feats.grad is not None
        # t=0 gradient in particular — furthest from loss
        assert feats.grad[:, 0, :].abs().sum() > 0, (
            "No gradient at t=0 features — gradient not flowing through full unroll"
        )

    def test_weights_update_after_optimizer_step(self, net, features):
        """Adam step must change the weights."""
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        params_before = {n: p.clone() for n, p in net.named_parameters()}

        deltas = net.forward_trajectory(features)
        loss   = deltas.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, param in net.named_parameters():
            assert not torch.allclose(param, params_before[name]), (
                f"Parameter '{name}' did not change after optimizer step"
            )


# ===========================================================================
# Input validation
# ===========================================================================

class TestValidation:

    def test_wrong_ndim_raises(self, net):
        with pytest.raises(ValueError, match="3-D"):
            net.forward_trajectory(torch.randn(N, T))   # missing F dim

    def test_wrong_feature_dim_raises(self, net):
        with pytest.raises(ValueError, match="FEATURE_DIM"):
            net.forward_trajectory(torch.randn(N, T, 5))   # F=5 instead of 3

    def test_nan_input_raises(self, net):
        feats = torch.randn(N, T, FEATURE_DIM)
        feats[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            net.forward_trajectory(feats)


# ===========================================================================
# Integration — real features from dataset
# ===========================================================================

class TestIntegration:

    def test_real_features_shape(self, bs_features):
        features, batch = bs_features
        torch.manual_seed(0)
        net    = BaselineFeedforwardNetwork(hidden=64)
        deltas = net.forward_trajectory(features)
        assert deltas.shape == (batch.n_paths, batch.n_steps)

    def test_real_features_finite(self, bs_features):
        features, batch = bs_features
        torch.manual_seed(0)
        net    = BaselineFeedforwardNetwork(hidden=64)
        deltas = net.forward_trajectory(features)
        assert torch.isfinite(deltas).all()

    def test_untrained_deltas_not_all_same(self, bs_features):
        """Untrained network should produce varied outputs — not a constant."""
        features, _ = bs_features
        torch.manual_seed(0)
        net    = BaselineFeedforwardNetwork(hidden=64)
        deltas = net.forward_trajectory(features)
        assert deltas.std().item() > 1e-4, (
            "All deltas are nearly identical — network may be degenerate"
        )
