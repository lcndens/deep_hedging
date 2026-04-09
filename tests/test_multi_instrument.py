"""Tests for two-instrument (n_instruments=2) support.

Covers:
  - builder.py  — 4-feature construction and error guards
  - baseline_feedforward_network.py  — dims and trajectory shape
  - pnl/compute.py  — 3-D gains accumulation
  - frictions/proportional.py  — 3-D cost accumulation
  - trainer.py  — --instrument multi guard
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from src.state.builder import FEATURE_DIM, build_features
from src.policy.baseline_feedforward_network import (
    BaselineFeedforwardNetwork,
    INPUT_DIM,
)
from src.pnl.compute import compute_pnl
from src.frictions.proportional import proportional_cost
from src.train.trainer import TrainConfig, train

from tests.conftest import N_PATHS, N_STEPS


# ===========================================================================
# builder — two-instrument features
# ===========================================================================

class TestBuilderMulti:

    def test_shape_n_instruments_2(self, heston_batch):
        """Four features returned for n_instruments=2."""
        features = build_features(heston_batch, n_instruments=2)
        assert features.shape == (heston_batch.n_paths, heston_batch.n_steps, 4)

    def test_first_three_features_unchanged(self, heston_batch):
        """Features 0-2 must be identical to single-instrument output."""
        f1 = build_features(heston_batch, n_instruments=1)
        f2 = build_features(heston_batch, n_instruments=2)
        assert torch.allclose(f1, f2[:, :, :3])

    def test_fourth_feature_is_log_vs_return(self, heston_batch):
        """Feature index 3 must equal log(S2_t / S2_0) per path."""
        features = build_features(heston_batch, n_instruments=2)
        S2_t = heston_batch.paths_S2[:, :-1]           # (N, T)
        S2_0 = heston_batch.paths_S2[:, :1]            # (N, 1)
        expected = torch.log(S2_t / S2_0)
        assert torch.allclose(features[:, :, 3], expected, atol=1e-5)

    def test_fourth_feature_zero_at_t0(self, heston_batch):
        """log(S2_0 / S2_0) = 0 — the first timestep feature is exactly 0."""
        features = build_features(heston_batch, n_instruments=2)
        assert torch.allclose(
            features[:, 0, 3], torch.zeros(heston_batch.n_paths), atol=1e-5
        )

    def test_dtype_float32(self, heston_batch):
        """Output dtype must be float32."""
        features = build_features(heston_batch, n_instruments=2)
        assert features.dtype == torch.float32

    def test_no_nans(self, heston_batch):
        """No NaN values in the 4-feature tensor."""
        features = build_features(heston_batch, n_instruments=2)
        assert not torch.isnan(features).any()

    def test_n_instruments_1_unchanged(self, heston_batch):
        """Default n_instruments=1 on Heston gives (N, T, 3)."""
        features = build_features(heston_batch, n_instruments=1)
        assert features.shape == (heston_batch.n_paths, heston_batch.n_steps, 3)

    def test_raises_on_bs(self, bs_batch):
        """n_instruments=2 on a BS dataset (paths_S2=0) must raise ValueError."""
        with pytest.raises(ValueError, match="n_instruments=2 requires a Heston"):
            build_features(bs_batch, n_instruments=2)

    def test_raises_on_nga(self, nga_batch):
        """n_instruments=2 on an NGA dataset (paths_S2=0) must raise ValueError."""
        with pytest.raises(ValueError, match="n_instruments=2 requires a Heston"):
            build_features(nga_batch, n_instruments=2)

    def test_raises_on_invalid_n_instruments(self, heston_batch):
        """n_instruments values other than 1 or 2 must raise ValueError."""
        with pytest.raises(ValueError, match="n_instruments must be 1 or 2"):
            build_features(heston_batch, n_instruments=3)


# ===========================================================================
# network — two-instrument dims and trajectory shape
# ===========================================================================

class TestNetworkMulti:

    N = 32
    T = 10
    H = 16   # small hidden for fast tests

    @pytest.fixture
    def net2(self):
        torch.manual_seed(0)
        return BaselineFeedforwardNetwork(hidden=self.H, n_instruments=2)

    @pytest.fixture
    def features2(self):
        torch.manual_seed(1)
        return torch.randn(self.N, self.T, FEATURE_DIM + 1)   # 4 features

    def test_input_dim(self, net2):
        """input_dim == (FEATURE_DIM+1) + 2 == 6 for n_instruments=2."""
        assert net2.input_dim == (FEATURE_DIM + 1) + 2
        assert net2.input_dim == 6

    def test_output_dim(self, net2):
        """output_dim == 2 for n_instruments=2."""
        assert net2.output_dim == 2

    def test_n_instruments_stored(self, net2):
        """n_instruments attribute is set correctly."""
        assert net2.n_instruments == 2

    def test_forward_trajectory_shape(self, net2, features2):
        """forward_trajectory returns (N, T, 2) for n_instruments=2."""
        deltas = net2.forward_trajectory(features2)
        assert deltas.shape == (self.N, self.T, 2)

    def test_forward_trajectory_dtype(self, net2, features2):
        """Output dtype is float32."""
        deltas = net2.forward_trajectory(features2)
        assert deltas.dtype == torch.float32

    def test_forward_trajectory_finite(self, net2, features2):
        """Output must be finite everywhere."""
        deltas = net2.forward_trajectory(features2)
        assert torch.isfinite(deltas).all()

    def test_parameter_count(self, net2):
        """Exact parameter count for H=16, input_dim=6, output_dim=2."""
        H = self.H
        expected = (
            6 * H + H +   # Linear(6→H)
            H * H + H +   # Linear(H→H)
            H * 2 + 2     # Linear(H→2)
        )
        assert net2.n_parameters() == expected

    def test_single_instrument_backward_compat(self):
        """Default n_instruments=1 still gives INPUT_DIM=4 and (N,T) output."""
        net1 = BaselineFeedforwardNetwork(hidden=self.H, n_instruments=1)
        assert net1.input_dim == INPUT_DIM   # 4
        assert net1.output_dim == 1
        feats = torch.randn(self.N, self.T, FEATURE_DIM)
        deltas = net1.forward_trajectory(feats)
        assert deltas.shape == (self.N, self.T)

    def test_wrong_feature_dim_raises(self, net2):
        """Network with n_instruments=2 rejects 3-feature input (expects 4)."""
        with pytest.raises(ValueError, match="features last dim"):
            net2.forward_trajectory(torch.randn(self.N, self.T, FEATURE_DIM))

    def test_recurrence_uses_both_prev_deltas(self, net2):
        """Changing one of the two previous deltas changes the next output."""
        torch.manual_seed(5)
        feats = torch.randn(4, 2, FEATURE_DIM + 1)

        # Manually run first step with zero prev-delta.
        delta_prev_zero = torch.zeros(4, 2)
        x0_zero = torch.cat([feats[:, 0, :], delta_prev_zero], dim=-1)
        out0_zero = net2.forward(x0_zero)   # (4, 2)

        # Run with non-zero prev-delta.
        delta_prev_one = torch.ones(4, 2)
        x0_one  = torch.cat([feats[:, 0, :], delta_prev_one],  dim=-1)
        out0_one  = net2.forward(x0_one)

        assert not torch.allclose(out0_zero, out0_one), (
            "Network output is identical for zero and non-zero prev deltas — "
            "the delta input slots may not be connected."
        )

    def test_invalid_n_instruments_raises(self):
        """n_instruments=3 must raise ValueError."""
        with pytest.raises(ValueError, match="n_instruments must be 1 or 2"):
            BaselineFeedforwardNetwork(n_instruments=3)


# ===========================================================================
# pnl/compute — 3-D gains
# ===========================================================================

class TestPnLMulti:

    def test_analytic_two_instrument_gains(self):
        """Verify gains sum correctly over both instruments for a known path."""
        # Instrument 0: price moves 100→110 (+10)
        # Instrument 1: price moves  50→ 60 (+10)
        # delta=[1,1] for both steps — gains = 1*10 + 1*10 = 20
        prices = torch.tensor([[[100.0, 50.0], [110.0, 60.0]]])   # (1, 2, 2)
        deltas = torch.tensor([[[1.0, 1.0]]])                       # (1, 1, 2)
        payoff = torch.tensor([0.0])
        cost   = torch.tensor([0.0])
        pnl = compute_pnl(prices, deltas, payoff, cost)
        assert torch.allclose(pnl, torch.tensor([20.0]))

    def test_analytic_mixed_sign_instruments(self):
        """Instrument 1 falls while instrument 0 rises — gains partially cancel."""
        # Instrument 0: 100→110, delta=1  → +10
        # Instrument 1:  50→ 40, delta=1  → -10
        # Net gains = 0
        prices = torch.tensor([[[100.0, 50.0], [110.0, 40.0]]])
        deltas = torch.tensor([[[1.0, 1.0]]])
        payoff = torch.tensor([0.0])
        cost   = torch.tensor([0.0])
        pnl = compute_pnl(prices, deltas, payoff, cost)
        assert torch.allclose(pnl, torch.tensor([0.0]))

    def test_output_shape(self):
        """Output must be (N,) regardless of instrument count."""
        N, T, I = 50, 10, 2
        prices = torch.randn(N, T + 1, I)
        deltas = torch.randn(N, T, I)
        payoff = torch.zeros(N)
        cost   = torch.zeros(N)
        pnl = compute_pnl(prices, deltas, payoff, cost)
        assert pnl.shape == (N,)

    def test_frictionless_finite(self):
        """PnL must be finite for random prices and deltas."""
        N, T, I = 100, 5, 2
        prices = torch.rand(N, T + 1, I) * 100 + 50
        deltas = torch.randn(N, T, I) * 0.5
        payoff = torch.zeros(N)
        cost   = torch.zeros(N)
        pnl    = compute_pnl(prices, deltas, payoff, cost)
        assert torch.isfinite(pnl).all()

    def test_ndim_mismatch_raises(self):
        """Mixing 2-D paths_S with 3-D deltas must raise ValueError."""
        with pytest.raises(ValueError, match="same number of dimensions"):
            compute_pnl(
                torch.ones(5, 4),        # 2-D
                torch.ones(5, 3, 2),     # 3-D
                torch.zeros(5),
                torch.zeros(5),
            )

    def test_instrument_dim_mismatch_raises(self):
        """Instrument dim of paths and deltas must agree."""
        with pytest.raises(ValueError, match="instrument dim"):
            compute_pnl(
                torch.ones(5, 4, 2),   # I=2
                torch.ones(5, 3, 3),   # I=3
                torch.zeros(5),
                torch.zeros(5),
            )


# ===========================================================================
# frictions/proportional — 3-D costs
# ===========================================================================

class TestFrictionsMulti:

    def test_analytic_two_instrument_cost(self):
        """Buy 1 unit of each instrument from zero — each costs epsilon*S."""
        # Instrument 0: S=100, epsilon=0.01 → cost = 0.01*100*1 = 1.0
        # Instrument 1: S= 50, epsilon=0.01 → cost = 0.01* 50*1 = 0.5
        # Total = 1.5
        S_obs  = torch.tensor([[[100.0, 50.0]]])   # (1, 1, 2)
        deltas = torch.tensor([[[1.0,   1.0]]])    # (1, 1, 2)
        cost = proportional_cost(S_obs, deltas, epsilon=0.01)
        assert torch.allclose(cost, torch.tensor([1.5]), atol=1e-5)

    def test_zero_epsilon_returns_zeros(self):
        """Frictionless case always returns zeros regardless of instrument count."""
        S_obs  = torch.ones(30, 5, 2) * 100.0
        deltas = torch.rand(30, 5, 2)
        cost   = proportional_cost(S_obs, deltas, epsilon=0.0)
        assert torch.all(cost == 0.0)
        assert cost.shape == (30,)

    def test_output_shape(self):
        """Output shape is (N,) for 3-D inputs."""
        N, T, I = 50, 10, 2
        S_obs  = torch.rand(N, T, I) * 100 + 50
        deltas = torch.randn(N, T, I)
        cost   = proportional_cost(S_obs, deltas, epsilon=0.005)
        assert cost.shape == (N,)

    def test_cost_nonnegative(self):
        """Transaction costs are always non-negative."""
        S_obs  = torch.rand(100, 8, 2) * 100 + 50
        deltas = torch.randn(100, 8, 2)
        cost   = proportional_cost(S_obs, deltas, epsilon=0.01)
        assert (cost >= 0).all()

    def test_cost_scales_with_epsilon(self):
        """Doubling epsilon doubles total cost."""
        S_obs  = torch.ones(10, 5, 2) * 100.0
        deltas = torch.ones(10, 5, 2)
        c1 = proportional_cost(S_obs, deltas, epsilon=0.01)
        c2 = proportional_cost(S_obs, deltas, epsilon=0.02)
        assert torch.allclose(c2, 2 * c1)

    def test_shape_mismatch_raises(self):
        """S_obs and deltas instrument dims must match."""
        with pytest.raises(ValueError, match="shape"):
            proportional_cost(
                torch.ones(5, 3, 2),
                torch.ones(5, 3, 3),
                epsilon=0.01,
            )


# ===========================================================================
# trainer — instrument guard
# ===========================================================================

class TestTrainerGuard:

    def _make_cfg(self, sim: str, instrument: str, run_dir) -> TrainConfig:
        from src.generate_dataset import main as generate_dataset
        run = generate_dataset([
            "--sim",            sim,
            "--n_paths",        str(N_PATHS),
            "--n_steps",        str(N_STEPS),
            "--seed",           "0",
            "--out_root",       str(run_dir),
            "--run_id",         f"guard_{sim}",
            "--strike",         "100.0",
            "--maturity_years", "0.25",
        ])
        return TrainConfig(
            dataset_dir = str(run),
            sim         = sim,
            instrument  = instrument,
            n_epochs    = 1,
            batch_size  = N_PATHS,
            run_id      = f"guard_test_{sim}",
            out_root    = str(run_dir / "results"),
        )

    def test_multi_raises_on_bs(self, tmp_path):
        """instrument='multi' with sim='bs' must raise ValueError."""
        cfg = self._make_cfg("bs", "multi", tmp_path)
        with pytest.raises(ValueError, match="--instrument multi requires --sim heston"):
            train(cfg)

    def test_multi_raises_on_nga(self, tmp_path):
        """instrument='multi' with sim='nga' must raise ValueError."""
        cfg = self._make_cfg("nga", "multi", tmp_path)
        with pytest.raises(ValueError, match="--instrument multi requires --sim heston"):
            train(cfg)

    def test_multi_does_not_raise_on_heston(self, tmp_path):
        """instrument='multi' with sim='heston' must train without error."""
        cfg = self._make_cfg("heston", "multi", tmp_path)
        result = train(cfg)
        assert result.n_epochs_trained >= 1

    def test_single_on_heston_unchanged(self, tmp_path):
        """instrument='single' with sim='heston' still works (default)."""
        cfg = self._make_cfg("heston", "single", tmp_path)
        result = train(cfg)
        assert result.n_epochs_trained >= 1
