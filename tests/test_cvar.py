"""Tests for the CVaR objective module."""

from __future__ import annotations
from pathlib import Path
import pytest
import torch
import torch.nn as nn

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
    """Helper for bs batch."""
    run_dir = generate_dataset([
        "--sim",            "bs",
        "--n_paths",        "200",
        "--n_steps",        "5",
        "--seed",           "0",
        "--out_root",       str(tmp_path_factory.mktemp("cvar_bs")),
        "--run_id",         "test_bs_cvar",
        "--strike",         "100.0",
        "--maturity_years", "0.25",
    ])
    return load_dataset(run_dir, split="train")


# ===========================================================================
# Construction and properties
# ===========================================================================

class TestConstruction:

    """Test cases for TestConstruction."""
    def test_default_alpha(self):
        """Assert default alpha."""
        cvar = CVaRLoss()
        assert cvar.alpha == 0.95

    def test_custom_alpha(self):
        """Assert custom alpha."""
        cvar = CVaRLoss(alpha=0.90)
        assert cvar.alpha == 0.90

    def test_omega_is_parameter(self):
        """Assert omega is parameter."""
        cvar = CVaRLoss()
        assert isinstance(cvar.omega, nn.Parameter)

    def test_omega_init_zero(self):
        """Assert omega init zero."""
        cvar = CVaRLoss()
        assert cvar.omega.item() == 0.0

    def test_omega_requires_grad(self):
        """Assert omega requires grad."""
        cvar = CVaRLoss()
        assert cvar.omega.requires_grad

    def test_is_nn_module(self):
        """Assert is nn module."""
        assert isinstance(CVaRLoss(), nn.Module)

    def test_invalid_alpha_one_raises(self):
        """Assert invalid alpha one raises."""
        with pytest.raises(ValueError, match="alpha"):
            CVaRLoss(alpha=1.0)

    def test_invalid_alpha_negative_raises(self):
        """Assert invalid alpha negative raises."""
        with pytest.raises(ValueError, match="alpha"):
            CVaRLoss(alpha=-0.1)

    def test_alpha_zero_valid(self):
        """alpha=0 is valid — reduces to mean loss."""
        cvar = CVaRLoss(alpha=0.0)
        assert cvar.alpha == 0.0

    def test_extra_repr(self):
        """Assert extra repr."""
        cvar = CVaRLoss(alpha=0.95)
        assert "0.95" in repr(cvar)


# ===========================================================================
# Output properties
# ===========================================================================

class TestOutputProperties:

    """Test cases for TestOutputProperties."""
    def test_output_is_scalar(self):
        """Assert output is scalar."""
        cvar = CVaRLoss()
        pnl  = torch.randn(1000)
        loss = cvar(pnl)
        assert loss.shape == ()

    def test_output_is_finite(self):
        """Assert output is finite."""
        cvar = CVaRLoss()
        pnl  = torch.randn(1000)
        loss = cvar(pnl)
        assert torch.isfinite(loss)

    def test_output_has_grad(self):
        """Loss must be part of computation graph."""
        cvar = CVaRLoss()
        pnl  = torch.randn(1000)
        loss = cvar(pnl)
        assert loss.requires_grad

    def test_output_dtype_float32(self):
        """Assert output dtype float32."""
        cvar = CVaRLoss()
        pnl  = torch.randn(1000, dtype=torch.float32)
        loss = cvar(pnl)
        assert loss.dtype == torch.float32

    def test_deterministic(self):
        """Assert deterministic."""
        cvar = CVaRLoss()
        pnl  = torch.randn(1000)
        assert torch.allclose(cvar(pnl), cvar(pnl))


# ===========================================================================
# Analytic correctness
# ===========================================================================

class TestAnalyticCorrectness:

    """Test cases for TestAnalyticCorrectness."""
    def test_alpha_zero_equals_mean_loss(self):
        """Assert alpha=0 reduces to mean loss for all-negative PnL samples."""
        cvar = CVaRLoss(alpha=0.0)
        # All paths are losses (negative PnL) → all exceed omega=0
        pnl  = -torch.abs(torch.randn(10000)) - 1.0   # all negative
        loss = cvar(pnl)
        # With omega=0 and all -pnl > 0:
        # L = 0 + mean(-pnl - 0) / 1.0 = mean(-pnl)
        expected = (-pnl).mean()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_all_paths_profitable_loss_is_omega(self):
        """When all PnL > -omega, excess=0 and loss=omega."""
        cvar = CVaRLoss(alpha=0.95)
        # Large positive PnL — no path falls below omega=0
        pnl  = torch.ones(1000) * 100.0
        loss = cvar(pnl)
        # excess = max(-100 - 0, 0) = 0 for all paths
        # loss = omega + 0 = 0.0
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_analytic_simple_distribution(self):
        """Assert CVaR matches a hand-computed value for a simple distribution."""
        N    = 100
        pnl  = torch.ones(N)
        pnl[-1] = -9.0

        cvar = CVaRLoss(alpha=0.0)
        # omega starts at 0
        loss = cvar(pnl)
        # excess = [max(-1,0)]*99 + [max(9,0)] = 0*99 + 9
        # loss = 0 + 9/100 = 0.09
        expected = torch.tensor(9.0 / N)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_higher_alpha_increases_loss(self):
        """Higher alpha → more focus on tail → higher CVaR loss."""
        torch.manual_seed(0)
        pnl   = torch.randn(10000)   # mixed gains and losses

        loss_low  = CVaRLoss(alpha=0.50)(pnl)
        loss_mid  = CVaRLoss(alpha=0.90)(pnl)
        loss_high = CVaRLoss(alpha=0.95)(pnl)

        assert loss_low < loss_mid < loss_high, (
            "CVaR should increase with alpha (higher alpha = more tail-focused)"
        )

    def test_loss_worse_than_mean(self):
        """CVaR_0.95 >= mean(-PnL) — tail risk is always at least mean risk."""
        torch.manual_seed(42)
        pnl      = torch.randn(10000)
        cvar     = CVaRLoss(alpha=0.95)
        loss     = cvar(pnl)
        mean_loss = (-pnl).mean()
        assert loss >= mean_loss - 1e-5, (
            "CVaR should be >= mean loss"
        )

    def test_scale_invariance(self):
        """Scaling PnL by c should scale CVaR by c."""
        torch.manual_seed(0)
        pnl = torch.randn(1000)
        c   = 2.0

        cvar1 = CVaRLoss(alpha=0.95)
        cvar2 = CVaRLoss(alpha=0.95)

        loss1 = cvar1(pnl)
        loss2 = cvar2(pnl * c)

        # loss2 ≈ c * loss1 when omega=0 for both (both start at 0)
        assert torch.allclose(loss2, c * loss1, atol=1e-5)


# ===========================================================================
# omega optimisation
# ===========================================================================

class TestOmegaOptimisation:

    """Test cases for TestOmegaOptimisation."""
    def test_omega_updates_after_step(self):
        """omega must change after a backward + optimizer step."""
        cvar      = CVaRLoss(alpha=0.95)
        optimizer = torch.optim.Adam([cvar.omega], lr=0.1)

        pnl  = torch.randn(1000)
        loss = cvar(pnl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert cvar.omega.item() != 0.0, "omega did not update after optimizer step"

    def test_omega_converges_toward_var(self):
        """Assert omega converges toward empirical VaR for a Gaussian sample."""
        torch.manual_seed(0)
        cvar      = CVaRLoss(alpha=0.95)
        optimizer = torch.optim.Adam([cvar.omega], lr=0.05)

        # Fixed PnL sample — N(0,1), so loss = -PnL ~ N(0,1)
        pnl = torch.randn(100000)

        for _ in range(500):
            optimizer.zero_grad()
            loss = cvar(pnl)
            loss.backward()
            optimizer.step()

        # VaR_0.95 of N(0,1) ≈ 1.645
        assert abs(cvar.omega.item() - 1.645) < 0.05, (
            f"omega={cvar.omega.item():.4f} did not converge to VaR_0.95≈1.645"
        )

    def test_joint_optimisation_includes_omega(self):
        """optimizer built with [*net.parameters(), cvar.omega] updates omega."""
        torch.manual_seed(0)
        net       = BaselineFeedforwardNetwork(hidden=16)
        cvar      = CVaRLoss(alpha=0.95)
        optimizer = torch.optim.Adam([*net.parameters(), cvar.omega], lr=1e-3)

        omega_before = cvar.omega.item()

        features = torch.randn(100, 5, FEATURE_DIM)
        deltas   = net.forward_trajectory(features)
        pnl      = deltas.sum(dim=1)   # dummy pnl
        loss     = cvar(pnl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert cvar.omega.item() != omega_before


# ===========================================================================
# Gradient flow
# ===========================================================================

class TestGradientFlow:

    """Test cases for TestGradientFlow."""
    def test_grad_flows_to_omega(self):
        """Assert grad flows to omega."""
        cvar = CVaRLoss(alpha=0.95)
        pnl  = torch.randn(1000)
        loss = cvar(pnl)
        loss.backward()
        assert cvar.omega.grad is not None
        assert cvar.omega.grad.abs() > 0

    def test_grad_flows_to_network_weights(self):
        """Assert CVaR backpropagates non-zero gradients to network parameters."""
        torch.manual_seed(0)
        net  = BaselineFeedforwardNetwork(hidden=32)
        # alpha=0.0: loss = omega + mean(max(-pnl - omega, 0)) / 1.0
        # with omega=0: loss = mean(max(-pnl, 0)) — still needs negative pnl
        # Use alpha=0.0 and force omega negative so all paths exceed it
        cvar = CVaRLoss(alpha=0.0)
        with torch.no_grad():
            cvar.omega.fill_(-100.0)   # all paths will exceed this threshold

        features = torch.randn(200, 10, FEATURE_DIM)
        deltas   = net.forward_trajectory(features)
        pnl      = deltas.sum(dim=1)
        loss     = cvar(pnl)
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No grad for '{name}'"
            assert param.grad.abs().sum() > 0, f"Zero grad for '{name}'"

    def test_only_tail_paths_contribute_grad(self):
        """Assert only tail-loss paths contribute gradient when omega is fixed."""
        cvar = CVaRLoss(alpha=0.95)
        # Fix omega at 0 to control which paths contribute
        with torch.no_grad():
            cvar.omega.fill_(0.0)

        pnl = torch.tensor([1.0, 2.0, -1.0, -2.0], requires_grad=True)
        loss = cvar(pnl)
        loss.backward()

        # Paths 0,1 (positive PnL) should have zero grad
        # Paths 2,3 (negative PnL) should have non-zero grad
        assert pnl.grad[0].item() == pytest.approx(0.0), "Profitable path should have zero grad"
        assert pnl.grad[1].item() == pytest.approx(0.0), "Profitable path should have zero grad"
        assert pnl.grad[2].item() != 0.0, "Loss path should have non-zero grad"
        assert pnl.grad[3].item() != 0.0, "Loss path should have non-zero grad"


# ===========================================================================
# Helper methods
# ===========================================================================

class TestHelperMethods:

    """Test cases for TestHelperMethods."""
    def test_cvar_estimate_returns_float(self):
        """Assert cvar estimate returns float."""
        cvar = CVaRLoss()
        pnl  = torch.randn(1000)
        est  = cvar.cvar_estimate(pnl)
        assert isinstance(est, float)

    def test_cvar_estimate_no_grad(self):
        """cvar_estimate should not affect the computation graph."""
        cvar = CVaRLoss()
        pnl  = torch.randn(1000)
        cvar.cvar_estimate(pnl)
        # omega should have no grad after cvar_estimate
        assert cvar.omega.grad is None

    def test_cvar_estimate_matches_forward(self):
        """Assert cvar estimate matches forward."""
        cvar = CVaRLoss()
        pnl  = torch.randn(1000)
        assert abs(cvar.cvar_estimate(pnl) - cvar(pnl).item()) < 1e-6

    def test_var_estimate_returns_float(self):
        """Assert var estimate returns float."""
        cvar = CVaRLoss()
        assert isinstance(cvar.var_estimate(), float)

    def test_var_estimate_matches_omega(self):
        """Assert var estimate matches omega."""
        cvar = CVaRLoss()
        assert cvar.var_estimate() == cvar.omega.item()


# ===========================================================================
# Input validation
# ===========================================================================

class TestValidation:

    """Test cases for TestValidation."""
    def test_wrong_ndim_raises(self):
        """Assert wrong ndim raises."""
        cvar = CVaRLoss()
        with pytest.raises(ValueError, match="1-D"):
            cvar(torch.randn(10, 10))

    def test_empty_tensor_raises(self):
        """Assert empty tensor raises."""
        cvar = CVaRLoss()
        with pytest.raises(ValueError, match="empty"):
            cvar(torch.tensor([]))

    def test_nan_input_raises(self):
        """Assert nan input raises."""
        cvar = CVaRLoss()
        pnl  = torch.randn(100)
        pnl[5] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            cvar(pnl)


# ===========================================================================
# Integration tests across data loading, policy inference, and CVaR loss.
# ===========================================================================

class TestIntegration:

    """Test cases for TestIntegration."""
    def test_full_pipeline_loss_is_scalar(self, bs_batch):
        """End-to-end: features → deltas → pnl → CVaR loss is a scalar."""
        torch.manual_seed(0)
        net  = BaselineFeedforwardNetwork(hidden=64)
        cvar = CVaRLoss(alpha=0.95)

        features   = build_features(bs_batch)
        deltas     = net.forward_trajectory(features)
        payoff     = call_payoff(bs_batch.paths_S[:, -1], bs_batch.K)
        total_cost = proportional_cost(bs_batch.paths_S[:, :-1], deltas, epsilon=0.0)
        pnl        = compute_pnl(bs_batch.paths_S, deltas, payoff, total_cost)
        loss       = cvar(pnl)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_full_pipeline_backward(self, bs_batch):
        """Gradient must flow from CVaR loss all the way back to network weights."""
        torch.manual_seed(0)
        net  = BaselineFeedforwardNetwork(hidden=64)
        cvar = CVaRLoss(alpha=0.95)

        features   = build_features(bs_batch)
        deltas     = net.forward_trajectory(features)
        payoff     = call_payoff(bs_batch.paths_S[:, -1], bs_batch.K)
        total_cost = proportional_cost(bs_batch.paths_S[:, :-1], deltas, epsilon=0.0)
        pnl        = compute_pnl(bs_batch.paths_S, deltas, payoff, total_cost)
        loss       = cvar(pnl)
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for '{name}'"
            assert param.grad.abs().sum() > 0, f"Zero gradient for '{name}'"

    def test_optimizer_step_changes_weights(self, bs_batch):
        """One full forward+backward+step cycle must update network weights."""
        torch.manual_seed(0)
        net       = BaselineFeedforwardNetwork(hidden=64)
        cvar      = CVaRLoss(alpha=0.95)
        optimizer = torch.optim.Adam([*net.parameters(), cvar.omega], lr=1e-3)

        params_before = {n: p.clone() for n, p in net.named_parameters()}

        features   = build_features(bs_batch)
        deltas     = net.forward_trajectory(features)
        payoff     = call_payoff(bs_batch.paths_S[:, -1], bs_batch.K)
        total_cost = proportional_cost(bs_batch.paths_S[:, :-1], deltas, epsilon=0.0)
        pnl        = compute_pnl(bs_batch.paths_S, deltas, payoff, total_cost)
        loss       = cvar(pnl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, param in net.named_parameters():
            assert not torch.allclose(param, params_before[name]), (
                f"Parameter '{name}' did not change after optimizer step"
            )
