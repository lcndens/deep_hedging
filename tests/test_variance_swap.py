"""Tests for the variance swap price path computation.

All unit tests construct controlled numpy arrays directly; no dataset fixtures
are needed.  Integration tests go through the conftest session fixtures to
exercise the full generate → load → S2 round-trip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.simulators.variance_swap import compute_variance_swap_paths
from src.simulators.heston import HestonParams, simulate_observations

from tests.conftest import N_PATHS, N_STEPS


# ---------------------------------------------------------------------------
# Unit tests — controlled inputs
# ---------------------------------------------------------------------------

class TestShape:

    def test_output_shape_matches_input(self):
        """Output shape must equal input shape (N, T+1)."""
        N, T1 = 50, 11
        v = np.ones((N, T1)) * 0.04
        S2 = compute_variance_swap_paths(v, kappa=1.5, theta=0.04, maturity_years=1.0)
        assert S2.shape == (N, T1)

    def test_single_path(self):
        """N=1 edge case returns shape (1, T+1)."""
        v = np.ones((1, 6)) * 0.04
        S2 = compute_variance_swap_paths(v, kappa=1.5, theta=0.04, maturity_years=0.5)
        assert S2.shape == (1, 6)

    def test_large_batch(self):
        """N=10_000 returns the correct shape without error."""
        v = np.random.default_rng(0).standard_normal((10_000, 31)) * 0.01 + 0.04
        S2 = compute_variance_swap_paths(v, kappa=1.5, theta=0.04, maturity_years=1.0)
        assert S2.shape == (10_000, 31)


class TestKnownT0Value:

    def test_t0_analytic_default_params(self):
        """At t=0 with default Heston params (v0=theta=0.04): S2[:,0] = theta*T = 0.04."""
        N, T, kappa, theta, v0, T_mat = 100, 30, 1.5, 0.04, 0.04, 1.0
        v = np.full((N, T + 1), v0)
        S2 = compute_variance_swap_paths(v, kappa=kappa, theta=theta, maturity_years=T_mat)
        expected_t0 = theta * T_mat  # = 0.04, since v0 == theta
        np.testing.assert_allclose(S2[:, 0], expected_t0, rtol=1e-6)

    def test_t0_analytic_general(self):
        """S2[:,0] = (v0-theta)/kappa*(1-exp(-kappa*T)) + theta*T for any v0."""
        N, T, kappa, theta, v0, T_mat = 200, 30, 2.0, 0.04, 0.09, 1.0
        v = np.full((N, T + 1), v0)
        S2 = compute_variance_swap_paths(v, kappa=kappa, theta=theta, maturity_years=T_mat)
        expected_t0 = (v0 - theta) / kappa * (1.0 - math.exp(-kappa * T_mat)) + theta * T_mat
        np.testing.assert_allclose(S2[:, 0], expected_t0, rtol=1e-6)

    def test_t0_trapezoidal_term_is_zero(self):
        """The trapezoidal (realized) component must be exactly 0 at t=0 for all paths."""
        rng = np.random.default_rng(7)
        v = rng.standard_normal((50, 21)) * 0.02 + 0.04
        S2 = compute_variance_swap_paths(v, kappa=1.5, theta=0.04, maturity_years=1.0)
        # The realized integral at t=0 is always 0; this holds regardless of v values.
        expected_t0 = (v[:, 0] - 0.04) / 1.5 * (1.0 - np.exp(-1.5)) + 0.04
        np.testing.assert_allclose(S2[:, 0], expected_t0, rtol=1e-6)


class TestMaturityFutureTermVanishes:

    def test_future_term_zero_at_maturity(self):
        """At t=T the analytic future-variance term is 0: S2[:,-1] = cum_trap[:,-1] only."""
        N, T, kappa, theta, T_mat = 80, 10, 1.5, 0.04, 0.5
        dt = T_mat / T
        rng = np.random.default_rng(3)
        v = np.abs(rng.standard_normal((N, T + 1))) * 0.02 + 0.02  # all positive

        S2 = compute_variance_swap_paths(v, kappa=kappa, theta=theta, maturity_years=T_mat)

        # Manually compute cum_trap at T.
        v_pos = np.maximum(v, 0.0)
        steps = 0.5 * (v_pos[:, :-1] + v_pos[:, 1:]) * dt
        realized = steps.sum(axis=1)

        np.testing.assert_allclose(S2[:, -1], realized, rtol=1e-5)


class TestConstantVariance:

    def test_constant_v_equals_theta_gives_theta_T_everywhere(self):
        """When v=theta everywhere, expected future variance = theta*(T-t) and
        realized = theta*t, so S2[:,k] = theta*T for all k."""
        N, T, theta, T_mat = 50, 20, 0.04, 1.0
        v = np.full((N, T + 1), theta)
        kappa = 2.0  # arbitrary; cancels out when v==theta

        S2 = compute_variance_swap_paths(v, kappa=kappa, theta=theta, maturity_years=T_mat)

        expected = theta * T_mat
        np.testing.assert_allclose(S2, expected, rtol=1e-5)

    def test_constant_v_terminal_equals_v0_times_T(self):
        """With constant v=v0, S2[:,-1] must equal v0*T_mat (no future term at maturity)."""
        N, T, v0, theta, kappa, T_mat = 30, 30, 0.09, 0.04, 1.5, 1.0
        v = np.full((N, T + 1), v0)
        S2 = compute_variance_swap_paths(v, kappa=kappa, theta=theta, maturity_years=T_mat)
        # At t=T the future term vanishes; realized integral = v0 * T_mat exactly.
        np.testing.assert_allclose(S2[:, -1], v0 * T_mat, rtol=1e-5)


class TestPositivity:

    def test_nonnegative_with_simulated_heston_paths(self):
        """S2 must be >= 0 for every path and timestep under default Heston params."""
        cfg = HestonParams(n_paths=500, n_steps=30, seed=99)
        _, _, variance_swap_df, _ = simulate_observations(cfg)
        assert (variance_swap_df["S2"].to_numpy() >= 0.0).all(), (
            "Variance swap price S2 must be non-negative"
        )

    def test_nonnegative_numpy_directly(self):
        """Direct call with typical Heston draws also produces non-negative S2."""
        rng = np.random.default_rng(42)
        N, T1 = 1000, 31
        # Simulate plausible variance paths (mostly positive, occasional tiny negatives).
        v = rng.standard_normal((N, T1)) * 0.01 + 0.04
        S2 = compute_variance_swap_paths(v, kappa=1.5, theta=0.04, maturity_years=1.0)
        assert (S2 >= 0.0).all(), (
            f"S2 has {(S2 < 0).sum()} negative values"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_1d_v_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_variance_swap_paths(np.ones(10), kappa=1.5, theta=0.04, maturity_years=1.0)

    def test_single_timestep_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_variance_swap_paths(np.ones((5, 1)), kappa=1.5, theta=0.04, maturity_years=1.0)

    def test_nonpositive_maturity_raises(self):
        with pytest.raises(ValueError, match="maturity_years"):
            compute_variance_swap_paths(np.ones((5, 6)), kappa=1.5, theta=0.04, maturity_years=0.0)


# ---------------------------------------------------------------------------
# Integration: round-trip through dataset_loader
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_heston_paths_S2_nonzero(self, heston_batch):
        """paths_S2 loaded from a Heston dataset must have positive mean."""
        assert heston_batch.paths_S2.mean().item() > 0.0

    def test_heston_paths_S2_shape(self, heston_batch):
        """paths_S2 shape must equal paths_S shape."""
        assert heston_batch.paths_S2.shape == heston_batch.paths_S.shape

    def test_heston_paths_S2_nonnegative(self, heston_batch):
        """All loaded S2 values must be >= 0."""
        assert (heston_batch.paths_S2 >= 0.0).all()

    def test_heston_paths_S2_dtype(self, heston_batch):
        """paths_S2 must be float32."""
        assert heston_batch.paths_S2.dtype == torch.float32

    def test_bs_paths_S2_all_zeros(self, bs_batch):
        """BS dataset has no variance_swap/ directory; paths_S2 must be all zeros."""
        assert torch.all(bs_batch.paths_S2 == 0.0)

    def test_nga_paths_S2_all_zeros(self, nga_batch):
        """NGA dataset has no variance_swap/ directory; paths_S2 must be all zeros."""
        assert torch.all(nga_batch.paths_S2 == 0.0)

    def test_heston_t0_matches_analytic(self, heston_run):
        """S2[:,0] loaded from disk matches the closed-form initial fair value."""
        from src.io.dataset_loader import load_dataset
        batch = load_dataset(heston_run, split="train")
        cfg_v0    = batch.metadata["simulator_params"]["v0"]
        cfg_kappa = batch.metadata["simulator_params"]["kappa"]
        cfg_theta = batch.metadata["simulator_params"]["theta"]
        T_mat     = batch.T_mat

        expected_t0 = (
            (cfg_v0 - cfg_theta) / cfg_kappa * (1.0 - math.exp(-cfg_kappa * T_mat))
            + cfg_theta * T_mat
        )
        s2_t0 = batch.paths_S2[:, 0].float().numpy()
        np.testing.assert_allclose(s2_t0, expected_t0, rtol=1e-4)
