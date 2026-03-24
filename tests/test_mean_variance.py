"""Tests for src/objective_functions/mean_variance.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.objective_functions.mean_variance import MeanVarianceLoss
from src.objective_functions.cvar import CVaRLoss
from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork
from src.state.builder import build_features, FEATURE_DIM
from src.derivatives.european import call_payoff
from src.frictions.proportional import proportional_cost
from src.pnl.compute import compute_pnl
from src.io.dataset_loader import load_dataset
from src.generate_dataset import main as generate_dataset


@pytest.fixture(scope="module")
def bs_batch(tmp_path_factory):
    run_dir = generate_dataset([
        "--sim",            "bs",
        "--n_paths",        "200",
        "--n_steps",        "5",
        "--seed",           "7",
        "--out_root",       str(tmp_path_factory.mktemp("mv_bs")),
        "--run_id",         "test_bs_mv",
        "--strike",         "100.0",
        "--maturity_years", "0.25",
    ])
    return load_dataset(run_dir, split="train")


# ===========================================================================
# Construction and properties
# ===========================================================================

class TestConstruction:

    def test_default_lam(self):
        mv = MeanVarianceLoss()
        assert mv.lam == 1.0

    def test_custom_lam(self):
        mv = MeanVarianceLoss(lam=2.0)
        assert mv.lam == 2.0

    def test_lam_zero_valid(self):
        mv = MeanVarianceLoss(lam=0.0)
        assert mv.lam == 0.0

    def test_negative_lam_raises(self):
        with pytest.raises(ValueError, match="lam"):
            MeanVarianceLoss(lam=-1.0)

    def test_is_nn_module(self):
        assert isinstance(MeanVarianceLoss(), nn.Module)

    def test_omega_is_none(self):
        """omega property returns None — no trainable threshold."""
        mv = MeanVarianceLoss()
        assert mv.omega is None

    def test_omega_not_nn_parameter(self):
        mv = MeanVarianceLoss()
        assert not isinstance(mv.omega, nn.Parameter)

    def test_extra_repr(self):
        mv = MeanVarianceLoss(lam=2.5)
        assert "2.5" in repr(mv)


# ===========================================================================
# Output properties
# ===========================================================================

class TestOutputProperties:

    def test_output_is_scalar(self):
        mv  = MeanVarianceLoss()
        pnl = torch.randn(1000)
        assert mv(pnl).shape == ()

    def test_output_is_finite(self):
        mv  = MeanVarianceLoss()
        pnl = torch.randn(1000)
        assert torch.isfinite(mv(pnl))

    def test_output_has_grad(self):
        net      = BaselineFeedforwardNetwork(hidden=16)
        mv       = MeanVarianceLoss()
        features = torch.randn(100, 5, FEATURE_DIM)
        deltas   = net.forward_trajectory(features)
        pnl      = deltas.sum(dim=1)
        loss     = mv(pnl)
        assert loss.requires_grad

    def test_output_dtype_float32(self):
        mv  = MeanVarianceLoss()
        pnl = torch.randn(1000, dtype=torch.float32)
        assert mv(pnl).dtype == torch.float32

    def test_deterministic(self):
        mv  = MeanVarianceLoss()
        pnl = torch.randn(1000)
        assert torch.allclose(mv(pnl), mv(pnl))


# ===========================================================================
# Analytic correctness
# ===========================================================================

class TestAnalyticCorrectness:

    def test_lam_zero_equals_negative_mean(self):
        """lam=0: loss = -E[PnL] only, variance term dropped."""
        mv  = MeanVarianceLoss(lam=0.0)
        pnl = torch.randn(10000)
        loss = mv(pnl)
        expected = -pnl.mean()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_constant_pnl_variance_is_zero(self):
        """Constant PnL has zero variance — loss = -mean only regardless of lam."""
        mv  = MeanVarianceLoss(lam=5.0)
        pnl = torch.ones(1000) * 3.0
        loss = mv(pnl)
        expected = torch.tensor(-3.0)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_known_values(self):
        """Hand-computed loss for a simple distribution.

        pnl = [-1, 1]  →  mean=0, var=1
        loss = -0 + 1.0 * 1 = 1.0
        """
        mv   = MeanVarianceLoss(lam=1.0)
        pnl  = torch.tensor([-1.0, 1.0])
        loss = mv(pnl)
        assert torch.allclose(loss, torch.tensor(1.0), atol=1e-5)

    def test_higher_lam_increases_loss_for_variable_pnl(self):
        """Higher lam → higher penalty on variance → higher loss."""
        torch.manual_seed(0)
        pnl = torch.randn(10000)   # non-zero variance

        loss_low  = MeanVarianceLoss(lam=0.5)(pnl)
        loss_mid  = MeanVarianceLoss(lam=1.0)(pnl)
        loss_high = MeanVarianceLoss(lam=2.0)(pnl)

        assert loss_low < loss_mid < loss_high

    def test_formula_decomposition(self):
        """Loss = -mean(PnL) + lam * var(PnL) exactly."""
        torch.manual_seed(42)
        lam = 1.5
        mv  = MeanVarianceLoss(lam=lam)
        pnl = torch.randn(5000)

        loss     = mv(pnl)
        expected = -pnl.mean() + lam * pnl.var(unbiased=False)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_loss_with_all_negative_pnl(self):
        """All-loss paths: mean is negative, -mean is positive (bad outcome)."""
        mv  = MeanVarianceLoss(lam=1.0)
        pnl = -torch.abs(torch.randn(1000)) - 1.0   # all negative
        loss = mv(pnl)
        assert loss > 0.0, "Loss should be positive when all paths are losing"


# ===========================================================================
# Gradient flow
# ===========================================================================

class TestGradientFlow:

    def test_grad_flows_to_network_weights(self):
        torch.manual_seed(0)
        net  = BaselineFeedforwardNetwork(hidden=32)
        mv   = MeanVarianceLoss(lam=1.0)

        features = torch.randn(200, 10, FEATURE_DIM)
        deltas   = net.forward_trajectory(features)
        pnl      = deltas.sum(dim=1)
        loss     = mv(pnl)
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No grad for '{name}'"
            assert param.grad.abs().sum() > 0, f"Zero grad for '{name}'"

    def test_all_paths_contribute_grad(self):
        """Unlike CVaR, mean-variance distributes gradient across all paths."""
        mv  = MeanVarianceLoss(lam=1.0)
        pnl = torch.randn(100, requires_grad=True)
        loss = mv(pnl)
        loss.backward()
        # Every path should have non-zero gradient
        assert (pnl.grad.abs() > 0).all(), (
            "All paths should contribute gradient to mean-variance loss"
        )

    def test_optimizer_step_changes_weights(self):
        torch.manual_seed(0)
        net       = BaselineFeedforwardNetwork(hidden=16)
        mv        = MeanVarianceLoss(lam=1.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        params_before = {n: p.clone() for n, p in net.named_parameters()}

        features = torch.randn(100, 5, FEATURE_DIM)
        deltas   = net.forward_trajectory(features)
        pnl      = deltas.sum(dim=1)
        loss     = mv(pnl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, param in net.named_parameters():
            assert not torch.allclose(param, params_before[name]), (
                f"Parameter '{name}' did not change after optimizer step"
            )


# ===========================================================================
# Trainer compatibility
# ===========================================================================

class TestTrainerCompatibility:

    def test_omega_not_parameter_so_not_added_to_optimizer(self):
        """Trainer checks isinstance(objective.omega, nn.Parameter) before
        adding to optimizer. MeanVarianceLoss.omega is None so it is skipped."""
        mv = MeanVarianceLoss()
        assert not isinstance(mv.omega, nn.Parameter)

    def test_cvar_omega_is_parameter(self):
        """Confirm CVaRLoss.omega IS a Parameter — this is the baseline."""
        cvar = CVaRLoss()
        assert isinstance(cvar.omega, nn.Parameter)

    def test_no_trainable_params_in_mv(self):
        """MeanVarianceLoss has no trainable parameters of its own."""
        mv = MeanVarianceLoss()
        assert list(mv.parameters()) == []

    def test_loss_estimate_returns_float(self):
        mv  = MeanVarianceLoss()
        pnl = torch.randn(1000)
        assert isinstance(mv.loss_estimate(pnl), float)

    def test_loss_estimate_no_grad(self):
        net      = BaselineFeedforwardNetwork(hidden=16)
        mv       = MeanVarianceLoss()
        features = torch.randn(100, 5, FEATURE_DIM)
        deltas   = net.forward_trajectory(features)
        pnl      = deltas.sum(dim=1)
        mv.loss_estimate(pnl)
        # No backward called — network weights should have no grad
        for param in net.parameters():
            assert param.grad is None


# ===========================================================================
# Input validation
# ===========================================================================

class TestValidation:

    def test_wrong_ndim_raises(self):
        mv = MeanVarianceLoss()
        with pytest.raises(ValueError, match="1-D"):
            mv(torch.randn(10, 10))

    def test_empty_tensor_raises(self):
        mv = MeanVarianceLoss()
        with pytest.raises(ValueError, match="empty"):
            mv(torch.tensor([]))

    def test_nan_input_raises(self):
        mv  = MeanVarianceLoss()
        pnl = torch.randn(100)
        pnl[5] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            mv(pnl)

    def test_single_element_raises(self):
        """Variance requires at least 2 elements."""
        mv = MeanVarianceLoss()
        with pytest.raises(ValueError, match="2 elements"):
            mv(torch.tensor([1.0]))


# ===========================================================================
# Integration — full pipeline
# ===========================================================================

class TestIntegration:

    def test_full_pipeline_loss_is_scalar(self, bs_batch):
        torch.manual_seed(0)
        net = BaselineFeedforwardNetwork(hidden=64)
        mv  = MeanVarianceLoss(lam=1.0)

        features   = build_features(bs_batch)
        deltas     = net.forward_trajectory(features)
        payoff     = call_payoff(bs_batch.paths_S[:, -1], bs_batch.K)
        total_cost = proportional_cost(bs_batch.paths_S[:, :-1], deltas, epsilon=0.0)
        pnl        = compute_pnl(bs_batch.paths_S, deltas, payoff, total_cost)
        loss       = mv(pnl)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_full_pipeline_backward(self, bs_batch):
        torch.manual_seed(0)
        net = BaselineFeedforwardNetwork(hidden=64)
        mv  = MeanVarianceLoss(lam=1.0)

        features   = build_features(bs_batch)
        deltas     = net.forward_trajectory(features)
        payoff     = call_payoff(bs_batch.paths_S[:, -1], bs_batch.K)
        total_cost = proportional_cost(bs_batch.paths_S[:, :-1], deltas, epsilon=0.0)
        pnl        = compute_pnl(bs_batch.paths_S, deltas, payoff, total_cost)
        loss       = mv(pnl)
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for '{name}'"
            assert param.grad.abs().sum() > 0, f"Zero gradient for '{name}'"

    def test_optimizer_step_no_omega(self, bs_batch):
        """Trainer builds optimizer from net.parameters() only for mean-variance."""
        torch.manual_seed(0)
        net       = BaselineFeedforwardNetwork(hidden=64)
        mv        = MeanVarianceLoss(lam=1.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        params_before = {n: p.clone() for n, p in net.named_parameters()}

        features   = build_features(bs_batch)
        deltas     = net.forward_trajectory(features)
        payoff     = call_payoff(bs_batch.paths_S[:, -1], bs_batch.K)
        total_cost = proportional_cost(bs_batch.paths_S[:, :-1], deltas, epsilon=0.0)
        pnl        = compute_pnl(bs_batch.paths_S, deltas, payoff, total_cost)
        loss       = mv(pnl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, param in net.named_parameters():
            assert not torch.allclose(param, params_before[name]), (
                f"Parameter '{name}' did not change after optimizer step"
            )
