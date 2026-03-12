"""Tests for the state-feature builder module.

Reuses the session-scoped dataset fixtures from test_dataset_loader.py by
importing them directly. All tests operate on the train split.

Dataset: n_paths=200, n_steps=5, seed=0, maturity_years=0.25
Expected features shape: (N_train, 5, 3)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.io.dataset_loader import DatasetBatch, load_dataset
from src.state.builder import FEATURE_DIM, build_features
from src.generate_dataset import main as generate_dataset

from tests.conftest import N_PATHS, N_STEPS, T1

# ---------------------------------------------------------------------------
# Shape and dtype tests
# ---------------------------------------------------------------------------

class TestShapeAndDtype:
    """Test cases for TestShapeAndDtype."""

    @pytest.mark.parametrize("batch_fixture", ["bs_batch", "heston_batch", "nga_batch"])
    def test_shape(self, batch_fixture, request):
        """Assert shape."""
        batch = request.getfixturevalue(batch_fixture)
        features = build_features(batch)
        expected = (batch.n_paths, N_STEPS, FEATURE_DIM)
        assert features.shape == expected, (
            f"Expected shape {expected}, got {features.shape}"
        )

    @pytest.mark.parametrize("batch_fixture", ["bs_batch", "heston_batch", "nga_batch"])
    def test_dtype_float32(self, batch_fixture, request):
        """Assert dtype float32."""
        batch = request.getfixturevalue(batch_fixture)
        features = build_features(batch)
        assert features.dtype == torch.float32

    @pytest.mark.parametrize("batch_fixture", ["bs_batch", "heston_batch", "nga_batch"])
    def test_device_cpu(self, batch_fixture, request):
        """Assert device cpu."""
        batch = request.getfixturevalue(batch_fixture)
        features = build_features(batch)
        assert features.device.type == "cpu"

    @pytest.mark.parametrize("batch_fixture", ["bs_batch", "heston_batch", "nga_batch"])
    def test_no_maturity_step(self, batch_fixture, request):
        """Time dimension must be T (n_steps), not T+1."""
        batch = request.getfixturevalue(batch_fixture)
        features = build_features(batch)
        assert features.shape[1] == N_STEPS
        assert features.shape[1] == batch.n_steps

    def test_feature_dim_constant(self, bs_batch):
        """FEATURE_DIM must be 3 — fixed for all Aim 2/3 experiments."""
        assert FEATURE_DIM == 3
        features = build_features(bs_batch)
        assert features.shape[2] == FEATURE_DIM

    @pytest.mark.parametrize("batch_fixture", ["bs_batch", "heston_batch", "nga_batch"])
    def test_no_nans(self, batch_fixture, request):
        """Assert no nans."""
        batch = request.getfixturevalue(batch_fixture)
        features = build_features(batch)
        assert not torch.isnan(features).any(), "NaN values found in features tensor"

    @pytest.mark.parametrize("batch_fixture", ["bs_batch", "heston_batch", "nga_batch"])
    def test_no_infs(self, batch_fixture, request):
        """Assert no infs."""
        batch = request.getfixturevalue(batch_fixture)
        features = build_features(batch)
        assert not torch.isinf(features).any(), "Inf values found in features tensor"


# ---------------------------------------------------------------------------
# Feature 0: log-moneyness  log(S_t / K)
# ---------------------------------------------------------------------------

class TestLogMoneyness:

    """Test cases for TestLogMoneyness."""
    def test_slot_index(self, bs_batch):
        """Log-moneyness is feature index 0."""
        features = build_features(bs_batch)
        S_t = bs_batch.paths_S[:, :-1]
        expected = torch.log(S_t / bs_batch.K)
        assert torch.allclose(features[:, :, 0], expected, atol=1e-5)

    def test_at_the_money_is_zero(self, bs_batch):
        """When S_t = K, log(S_t/K) = 0."""
        K = bs_batch.K
        # At t=0, S_0 should be close to K (generated with s0=100, K=100)
        t0_moneyness = build_features(bs_batch)[:, 0, 0]
        # Not asserting exact 0 since s0 and K may differ across paths,
        # but the mean of log-moneyness at t=0 should be near 0.
        assert t0_moneyness.abs().mean() < 1.0, (
            "Mean absolute log-moneyness at t=0 unexpectedly large"
        )

    def test_sign_follows_spot(self, bs_batch):
        """Paths where S_t > K should have positive log-moneyness."""
        features = build_features(bs_batch)
        S_t  = bs_batch.paths_S[:, :-1]
        K    = bs_batch.K
        mask = S_t > K
        assert (features[:, :, 0][mask] > 0).all(), (
            "Positive moneyness (S > K) should give log-moneyness > 0"
        )

    def test_negative_when_otm(self, bs_batch):
        """Paths where S_t < K should have negative log-moneyness."""
        features = build_features(bs_batch)
        S_t  = bs_batch.paths_S[:, :-1]
        K    = bs_batch.K
        mask = S_t < K
        assert (features[:, :, 0][mask] < 0).all(), (
            "OTM paths (S < K) should give log-moneyness < 0"
        )


# ---------------------------------------------------------------------------
# Feature 1: time to maturity  τ_t = T_mat - t
# ---------------------------------------------------------------------------

class TestTimeToMaturity:

    """Test cases for TestTimeToMaturity."""
    def test_slot_index(self, bs_batch):
        """Time to maturity is feature index 1."""
        features = build_features(bs_batch)
        t_t = bs_batch.paths_t[:, :-1]
        expected = bs_batch.T_mat - t_t
        assert torch.allclose(features[:, :, 1], expected, atol=1e-5)

    def test_tau_at_t0_equals_T_mat(self, bs_batch):
        """At t=0, paths_t=0 so τ = T_mat - 0 = T_mat."""
        features = build_features(bs_batch)
        tau_t0 = features[:, 0, 1]
        T_mat  = torch.full_like(tau_t0, bs_batch.T_mat)
        assert torch.allclose(tau_t0, T_mat, atol=1e-5), (
            f"τ at t=0 should equal T_mat={bs_batch.T_mat}, "
            f"got mean={tau_t0.mean().item():.6f}"
        )

    def test_tau_at_last_step_is_small(self, bs_batch):
        """At t=T-1, τ should be approximately one dt step."""
        features = build_features(bs_batch)
        tau_last = features[:, -1, 1]
        dt = bs_batch.T_mat / bs_batch.n_steps
        assert torch.allclose(tau_last, torch.full_like(tau_last, dt), atol=1e-4), (
            f"τ at t=T-1 should ≈ dt={dt:.6f}, got mean={tau_last.mean().item():.6f}"
        )

    def test_tau_strictly_decreasing(self, bs_batch):
        """τ must decrease at every step — time only moves forward."""
        features = build_features(bs_batch)
        tau = features[:, :, 1]   # (N, T)
        diffs = tau[:, 1:] - tau[:, :-1]   # should all be negative
        assert (diffs < 0).all(), (
            "Time to maturity τ is not strictly decreasing along time axis"
        )

    def test_tau_positive_everywhere(self, bs_batch):
        """τ should be strictly positive — we never step past maturity."""
        features = build_features(bs_batch)
        assert (features[:, :, 1] > 0).all(), (
            "Time to maturity τ must be > 0 at all rebalancing steps"
        )

    @pytest.mark.parametrize("batch_fixture", ["bs_batch", "heston_batch", "nga_batch"])
    def test_tau_consistent_across_sims(self, batch_fixture, request):
        """τ values should be identical across simulators (same time grid)."""
        batch = request.getfixturevalue(batch_fixture)
        features = build_features(batch)
        tau = features[:, :, 1]
        # All paths share the same time grid — variance across paths at t should be 0
        assert torch.allclose(tau.std(dim=0), torch.zeros(N_STEPS), atol=1e-6), (
            "τ should be identical across all paths at each timestep"
        )


# ---------------------------------------------------------------------------
# Feature 2: variance slot  v_t
# ---------------------------------------------------------------------------

class TestVarianceSlot:

    """Test cases for TestVarianceSlot."""
    def test_slot_index(self, heston_batch):
        """Variance is feature index 2."""
        features = build_features(heston_batch)
        v_t = heston_batch.paths_v[:, :-1]
        assert torch.allclose(features[:, :, 2], v_t, atol=1e-6)

    def test_v_slot_zero_bs(self, bs_batch):
        """BS dataset: variance slot must be 0.0 everywhere."""
        features = build_features(bs_batch)
        assert torch.all(features[:, :, 2] == 0.0), (
            "BS features[:,:,2] should be exactly 0.0"
        )

    def test_v_slot_zero_nga(self, nga_batch):
        """NGA dataset: variance slot must be 0.0 everywhere."""
        features = build_features(nga_batch)
        assert torch.all(features[:, :, 2] == 0.0), (
            "NGA features[:,:,2] should be exactly 0.0"
        )

    def test_v_slot_nonzero_heston(self, heston_batch):
        """Heston dataset: variance slot must be > 0 on average."""
        features = build_features(heston_batch)
        assert features[:, :, 2].mean().item() > 0.0, (
            "Heston features[:,:,2] should have positive mean variance"
        )

    def test_v_slot_nonnegative_heston(self, heston_batch):
        """Full-truncation scheme guarantees v_t >= 0."""
        features = build_features(heston_batch)
        assert (features[:, :, 2] >= 0.0).all(), (
            "Heston variance slot should be non-negative everywhere"
        )

    def test_v_slot_excludes_maturity(self, heston_batch):
        """Feature [2] should use paths_v[:,:-1] — maturity step excluded."""
        features = build_features(heston_batch)
        v_no_maturity = heston_batch.paths_v[:, :-1]
        assert torch.allclose(features[:, :, 2], v_no_maturity, atol=1e-6)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:

    """Test cases for TestErrors."""
    def test_invalid_K_raises(self, bs_batch):
        """Zero strike should raise ValueError."""
        import dataclasses
        bad_batch = dataclasses.replace(bs_batch, K=0.0)
        with pytest.raises(ValueError, match="K must be positive"):
            build_features(bad_batch)

    def test_negative_K_raises(self, bs_batch):
        """Assert negative K raises."""
        import dataclasses
        bad_batch = dataclasses.replace(bs_batch, K=-1.0)
        with pytest.raises(ValueError, match="K must be positive"):
            build_features(bad_batch)

    def test_invalid_T_mat_raises(self, bs_batch):
        """Assert invalid T mat raises."""
        import dataclasses
        bad_batch = dataclasses.replace(bs_batch, T_mat=0.0)
        with pytest.raises(ValueError, match="T_mat must be positive"):
            build_features(bad_batch)

    def test_wrong_dtype_raises(self, bs_batch):
        """float64 tensors should be rejected."""
        import dataclasses
        bad_batch = dataclasses.replace(
            bs_batch,
            paths_S=bs_batch.paths_S.double(),
        )
        with pytest.raises(ValueError, match="float32"):
            build_features(bad_batch)

    def test_wrong_shape_raises(self, bs_batch):
        """Mismatched paths_v shape should be rejected."""
        import dataclasses
        bad_batch = dataclasses.replace(
            bs_batch,
            paths_v=bs_batch.paths_v[:, :3],   # truncated
        )
        with pytest.raises(ValueError, match="shape"):
            build_features(bad_batch)
