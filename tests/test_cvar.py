"""Tests for src/objective_functions/cvar.py — Stage 4b.

Tests cover:
- Output is a differentiable scalar
- Analytic correctness against hand-computed CVaR values
- omega is trainable and updates with optimizer
- Special cases: alpha=0 reduces to mean loss, high alpha focuses on tail
- Gradient flow to network weights through CVaR loss
- Input validation
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.objective_functions.cvar import CVaRLoss
from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork
from src.state.builder import build_features, FEATURE_DIM
from src.derivatives.european import call_payoff
from src.frictions.proportional import proportional_cost
from src.pnl.compute import compute_pnl


# ===========================================================================
# Construction and properties
# ===========================================================================

class TestConstruction:

    def test_default_alpha(self):
        cvar = CVaRLoss()
        assert cvar.alpha == 0.95

    def test_custom_alpha(self):
        cvar = CVaRLoss(alpha=0.90)
        assert cvar.alpha == 0.90

    def test_omega_is_parameter(self):
        cvar = CVaRLoss()
        assert isinstance(cvar.omega, nn.Parameter)

    def test_omega_init_zero(self):
        cvar = CVaRLoss()
        assert cvar.omega.item() == 0.0

    def test_omega_requires_grad(self):
        cvar = CVaRLoss()
        assert cvar.omega.requires_grad

    def test_is_nn_module(self):
        assert isinstance(CVaRLoss(), nn.Module)

    def test_invalid_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            CVaRLoss(alpha=1.0)

    def test_invalid_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            CVaRLoss(alpha=-0.1)

    def test_alpha_zero_valid(self):
        """alpha=0 is valid — reduces to mean loss."""
        cvar = CVaRLoss(alpha=0.0)
        assert cvar.alpha == 0.0

    def test_extra_repr(self):
        cvar = CVaRLoss(alpha=0.95)
        assert "0.95" in repr(cvar)


# ===========================================================================
# Output properties
# ===========================================================================

class TestOutputProperties:

    def test_output_is_scalar(self):
        cvar = CVaRLoss()
        pnl  = torch.randn(1000)
        loss = cvar(pnl)
        assert loss.shape == ()

    def test_output_is_finite(self):
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
        cvar = CVaRLoss()
        pnl  = torch.randn(1000, dtype=torch.float32)
        loss = cvar(pnl)
        assert loss.dtype == torch.float32

    def test_deterministic(self):
        cvar = CVaRLoss()
        pnl  = torch.randn(1000)
        assert torch.allclose(cvar(pnl), cvar(pnl))


# ===========================================================================
# Analytic correctness
# ===========================================================================

class TestAnalyticCorrectness:

    def test_alpha_zero_equals_mean_loss(self):
        """alpha=0: CVaR = mean(-PnL) = -mean(PnL).

        With alpha=0 and omega converged, L = omega + mean(max(-PnL-omega,0)).
        At the optimum omega=VaR_0 = min(-PnL), so all paths exceed omega
        and L reduces to mean(-PnL).

        Simpler check: fix omega=0, then L = mean(max(-PnL, 0)).
        But with alpha=0 and omega fixed at 0:
            L = 0 + mean(max(-PnL, 0)) / 1.0
        This equals mean of positive losses only — not exactly mean(-PnL).

        Instead verify by checking that the optimal omega=0 gives mean(-PnL)
        when all PnL values are negative (all losses exceed omega=0).
        """
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
        """Hand-computed CVaR for a known distribution.

        PnL = [-10, -10, ..., -10, 0, 0, ..., 0]
              first 5% are -10, rest are 0
        CVaR_0.95 = mean of worst 5% = 10 (loss of 10).

        With omega optimised to VaR=10 (threshold between -10 and 0):
        excess_i = max(-pnl_i - 10, 0)
               = max(10-10, 0) = 0  for pnl=-10 paths
               = max(0-10,  0) = 0  for pnl=0  paths
        Hmm — need omega at the right level.

        Simpler: use known pnl, fix omega manually, verify formula.
        pnl = [1, 1, 1, ..., 1, -9]  (N-1 paths gain 1, 1 path loses 9)
        alpha=0.0, omega=0:
            excess = [max(-1,0), ..., max(9,0)] = [0,...,0, 9]
            loss = 0 + 9/N
        """
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
        """After many steps, omega should converge toward the empirical VaR_alpha.

        For standard normal PnL and alpha=0.95, VaR_0.95 of the loss
        distribution (-PnL ~ N(0,1)) is approximately 1.645.
        So omega should converge toward ~1.645.
        """
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

    def test_grad_flows_to_omega(self):
        cvar = CVaRLoss(alpha=0.95)
        pnl  = torch.randn(1000)
        loss = cvar(pnl)
        loss.backward()
        assert cvar.omega.grad is not None
        assert cvar.omega.grad.abs() > 0

    def test_grad_flows_to_network_weights(self):
        """Gradient must flow from CVaR loss back to network weights."""
        torch.manual_seed(0)
        net  = BaselineFeedforwardNetwork(hidden=32)
        cvar = CVaRLoss(alpha=0.95)

        features   = torch.randn(200, 10, FEATURE_DIM)
        deltas     = net.forward_trajectory(features)
        pnl        = deltas.sum(dim=1)   # dummy pnl — just needs grad path
        loss       = cvar(pnl)
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No grad for '{name}'"
            assert param.grad.abs().sum() > 0, f"Zero grad for '{name}'"

    def test_only_tail_paths_contribute_grad(self):
        """Paths with PnL > -omega should have zero contribution to grad.

        With omega=0, only paths where -PnL > 0 (i.e. PnL < 0) contribute.
        """
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

    def test_cvar_estimate_returns_float(self):
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
        cvar = CVaRLoss()
        pnl  = torch.randn(1000)
        assert abs(cvar.cvar_estimate(pnl) - cvar(pnl).item()) < 1e-6

    def test_var_estimate_returns_float(self):
        cvar = CVaRLoss()
        assert isinstance(cvar.var_estimate(), float)

    def test_var_estimate_matches_omega(self):
        cvar = CVaRLoss()
        assert cvar.var_estimate() == cvar.omega.item()


# ===========================================================================
# Input validation
# ===========================================================================

class TestValidation:

    def test_wrong_ndim_raises(self):
        cvar = CVaRLoss()
        with pytest.raises(ValueError, match="1-D"):
            cvar(torch.randn(10, 10))

    def test_empty_tensor_raises(self):
        cvar = CVaRLoss()
        with pytest.raises(ValueError, match="empty"):
            cvar(torch.tensor([]))

    def test_nan_input_raises(self):
        cvar = CVaRLoss()
        pnl  = torch.randn(100)
        pnl[5] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            cvar(pnl)


# ===========================================================================
# Integration — full Stage 1→4 chain
# ===========================================================================

class TestIntegration:

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
