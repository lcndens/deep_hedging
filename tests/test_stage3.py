"""Tests for derivatives, frictions, and terminal PnL computations.

All tests use analytically verifiable inputs so correctness can be checked
exactly, not just by shape. No dataset fixtures needed for most tests —
we construct controlled tensors directly.

A small set of integration tests load a real DatasetBatch to verify that the
data-to-PnL pipeline produces sensible terminal distributions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.derivatives.european import call_payoff, put_payoff
from src.frictions.proportional import proportional_cost
from src.pnl.compute import compute_pnl
from src.io.dataset_loader import load_dataset
from src.generate_dataset import main as generate_dataset

from tests.conftest import N_PATHS, N_STEPS, T1, SEED

def _gen(tmp_path: Path, sim: str) -> Path:
    """Helper for gen."""
    return generate_dataset([
        "--sim",            sim,
        "--n_paths",        str(N_PATHS),
        "--n_steps",        str(N_STEPS),
        "--seed",           str(SEED),
        "--out_root",       str(tmp_path),
        "--run_id",         f"test_{sim}",
        "--strike",         "100.0",
        "--maturity_years", "0.25",
    ])


@pytest.fixture(scope="session")
def bs_batch(tmp_path_factory):
    """Helper for bs batch."""
    run_dir = _gen(tmp_path_factory.mktemp("bs"), "bs")
    return load_dataset(run_dir, split="train")


# ===========================================================================
# DERIVATIVES
# ===========================================================================

class TestCallPayoff:

    """Test cases for TestCallPayoff."""
    def test_itm_call(self):
        """In-the-money call: payoff = S_T - K."""
        S_T = torch.tensor([110.0, 120.0, 150.0])
        K   = 100.0
        out = call_payoff(S_T, K)
        expected = torch.tensor([10.0, 20.0, 50.0])
        assert torch.allclose(out, expected)

    def test_otm_call_is_zero(self):
        """Out-of-the-money call: payoff = 0."""
        S_T = torch.tensor([80.0, 90.0, 99.9])
        K   = 100.0
        out = call_payoff(S_T, K)
        assert torch.all(out == 0.0)

    def test_atm_call_is_zero(self):
        """At-the-money call: S_T = K → payoff = 0."""
        S_T = torch.tensor([100.0, 100.0])
        out = call_payoff(S_T, K=100.0)
        assert torch.all(out == 0.0)

    def test_call_nonnegative(self):
        """Call payoff must always be >= 0."""
        S_T = torch.linspace(50.0, 200.0, 100)
        out = call_payoff(S_T, K=100.0)
        assert (out >= 0).all()

    def test_call_shape(self):
        """Assert call shape."""
        S_T = torch.rand(1000) * 200
        out = call_payoff(S_T, K=100.0)
        assert out.shape == (1000,)

    def test_call_dtype_preserved(self):
        """Assert call dtype preserved."""
        S_T = torch.tensor([110.0], dtype=torch.float32)
        out = call_payoff(S_T, K=100.0)
        assert out.dtype == torch.float32

    def test_call_device_preserved(self):
        """Assert call device preserved."""
        S_T = torch.tensor([110.0])
        out = call_payoff(S_T, K=100.0)
        assert out.device == S_T.device

    def test_call_invalid_K_raises(self):
        """Assert call invalid K raises."""
        with pytest.raises(ValueError, match="K must be positive"):
            call_payoff(torch.tensor([110.0]), K=0.0)

    def test_call_wrong_shape_raises(self):
        """Assert call wrong shape raises."""
        with pytest.raises(ValueError, match="1-D"):
            call_payoff(torch.ones(3, 3), K=100.0)


class TestPutPayoff:

    """Test cases for TestPutPayoff."""
    def test_itm_put(self):
        """In-the-money put: payoff = K - S_T."""
        S_T = torch.tensor([80.0, 90.0, 50.0])
        K   = 100.0
        out = put_payoff(S_T, K)
        expected = torch.tensor([20.0, 10.0, 50.0])
        assert torch.allclose(out, expected)

    def test_otm_put_is_zero(self):
        """Out-of-the-money put: payoff = 0."""
        S_T = torch.tensor([110.0, 120.0, 100.1])
        out = put_payoff(S_T, K=100.0)
        assert torch.all(out == 0.0)

    def test_atm_put_is_zero(self):
        """Assert atm put is zero."""
        S_T = torch.tensor([100.0, 100.0])
        out = put_payoff(S_T, K=100.0)
        assert torch.all(out == 0.0)

    def test_put_nonnegative(self):
        """Assert put nonnegative."""
        S_T = torch.linspace(50.0, 200.0, 100)
        out = put_payoff(S_T, K=100.0)
        assert (out >= 0).all()

    def test_put_shape(self):
        """Assert put shape."""
        S_T = torch.rand(500) * 200
        out = put_payoff(S_T, K=100.0)
        assert out.shape == (500,)

    def test_put_call_parity_structure(self):
        """call(S,K) - put(S,K) = S_T - K for all S_T (payoff parity)."""
        S_T = torch.linspace(50.0, 200.0, 50)
        K   = 100.0
        call = call_payoff(S_T, K)
        put  = put_payoff(S_T, K)
        diff = call - put
        expected = S_T - K
        assert torch.allclose(diff, expected, atol=1e-5)

    def test_put_invalid_K_raises(self):
        """Assert put invalid K raises."""
        with pytest.raises(ValueError, match="K must be positive"):
            put_payoff(torch.tensor([90.0]), K=-5.0)

    def test_put_wrong_shape_raises(self):
        """Assert put wrong shape raises."""
        with pytest.raises(ValueError, match="1-D"):
            put_payoff(torch.ones(2, 5), K=100.0)


# ===========================================================================
# FRICTIONS
# ===========================================================================

class TestProportionalCost:

    """Test cases for TestProportionalCost."""
    def test_zero_epsilon_returns_zeros(self):
        """Frictionless case: epsilon=0 → all zeros regardless of deltas."""
        N, T = 50, 10
        S_obs  = torch.ones(N, T) * 100.0
        deltas = torch.rand(N, T)
        out    = proportional_cost(S_obs, deltas, epsilon=0.0)
        assert torch.all(out == 0.0)
        assert out.shape == (N,)

    def test_zero_epsilon_shape(self):
        """Assert zero epsilon shape."""
        out = proportional_cost(torch.ones(30, 5), torch.ones(30, 5), epsilon=0.0)
        assert out.shape == (30,)

    def test_analytic_single_path_single_step(self):
        """One path, one step: buy 1 unit at S=100, epsilon=0.01 → cost=1.0."""
        # delta_{-1}=0, delta_0=1, S_0=100, epsilon=0.01
        # cost = 0.01 * 100 * |1 - 0| = 1.0
        S_obs  = torch.tensor([[100.0]])
        deltas = torch.tensor([[1.0]])
        out    = proportional_cost(S_obs, deltas, epsilon=0.01)
        assert torch.allclose(out, torch.tensor([1.0]))

    def test_analytic_single_path_two_steps(self):
        """One path, two steps with known costs."""
        # Step 0: delta_{-1}=0 → delta_0=1, S=100 → cost = 0.01*100*1 = 1.0
        # Step 1: delta_0=1   → delta_1=2, S=110 → cost = 0.01*110*1 = 1.1
        # total = 2.1
        S_obs  = torch.tensor([[100.0, 110.0]])
        deltas = torch.tensor([[1.0, 2.0]])
        out    = proportional_cost(S_obs, deltas, epsilon=0.01)
        assert torch.allclose(out, torch.tensor([2.1]), atol=1e-5)

    def test_no_rebalancing_zero_cost(self):
        """Constant delta → no rebalancing → zero cost (except first step)."""
        # delta_{-1}=0, delta_0=1 → cost at step 0 = epsilon*S*1
        # delta_0=1, delta_1=1 → cost at step 1 = 0
        S_obs  = torch.tensor([[100.0, 100.0]])
        deltas = torch.tensor([[1.0, 1.0]])
        out    = proportional_cost(S_obs, deltas, epsilon=0.01)
        # Only step 0 incurs cost: 0.01 * 100 * 1 = 1.0
        assert torch.allclose(out, torch.tensor([1.0]))

    def test_symmetry_buy_vs_sell(self):
        """Cost is symmetric — selling 1 unit costs same as buying 1 unit."""
        S_obs  = torch.tensor([[100.0]])
        buy    = proportional_cost(S_obs, torch.tensor([[1.0]]),  epsilon=0.01)
        sell   = proportional_cost(S_obs, torch.tensor([[-1.0]]), epsilon=0.01)
        assert torch.allclose(buy, sell)

    def test_cost_nonnegative(self):
        """Transaction costs must always be >= 0."""
        N, T   = 100, 10
        S_obs  = torch.rand(N, T) * 100 + 50
        deltas = torch.randn(N, T)
        out    = proportional_cost(S_obs, deltas, epsilon=0.005)
        assert (out >= 0).all()

    def test_cost_scales_with_epsilon(self):
        """Doubling epsilon should double total cost."""
        S_obs  = torch.ones(10, 5) * 100.0
        deltas = torch.ones(10, 5)
        cost1  = proportional_cost(S_obs, deltas, epsilon=0.01)
        cost2  = proportional_cost(S_obs, deltas, epsilon=0.02)
        assert torch.allclose(cost2, 2 * cost1)

    def test_output_shape(self):
        """Assert output shape."""
        N, T = 137, 30
        out  = proportional_cost(torch.ones(N, T), torch.ones(N, T), epsilon=0.01)
        assert out.shape == (N,)

    def test_dtype_preserved(self):
        """Assert dtype preserved."""
        S   = torch.ones(5, 3, dtype=torch.float32)
        d   = torch.ones(5, 3, dtype=torch.float32)
        out = proportional_cost(S, d, epsilon=0.01)
        assert out.dtype == torch.float32

    def test_shape_mismatch_raises(self):
        """Assert shape mismatch raises."""
        with pytest.raises(ValueError, match="shape"):
            proportional_cost(torch.ones(10, 5), torch.ones(10, 4), epsilon=0.01)

    def test_wrong_ndim_raises(self):
        """Assert wrong ndim raises."""
        with pytest.raises(ValueError, match="2-D"):
            proportional_cost(torch.ones(10), torch.ones(10), epsilon=0.01)

    def test_negative_epsilon_raises(self):
        """Assert negative epsilon raises."""
        with pytest.raises(ValueError, match="epsilon"):
            proportional_cost(torch.ones(5, 3), torch.ones(5, 3), epsilon=-0.01)


# ===========================================================================
# PNL
# ===========================================================================

class TestComputePnL:

    """Test cases for TestComputePnL."""
    def test_analytic_perfect_hedge(self):
        """Assert a one-step perfect hedge yields zero terminal PnL."""
        paths_S    = torch.tensor([[100.0, 110.0]])   # (1, 2) → T=1
        deltas     = torch.tensor([[1.0]])             # (1, 1)
        payoff     = torch.tensor([10.0])              # (1,)
        total_cost = torch.tensor([0.0])               # (1,)
        pnl        = compute_pnl(paths_S, deltas, payoff, total_cost, p0=0.0)
        assert torch.allclose(pnl, torch.tensor([0.0]))

    def test_analytic_unhedged(self):
        """Assert an unhedged short call incurs the expected payoff loss."""
        paths_S    = torch.tensor([[100.0, 110.0]])
        deltas     = torch.tensor([[0.0]])
        payoff     = torch.tensor([10.0])
        total_cost = torch.tensor([0.0])
        pnl        = compute_pnl(paths_S, deltas, payoff, total_cost)
        assert torch.allclose(pnl, torch.tensor([-10.0]))

    def test_gains_computation(self):
        """Multi-step gains: delta=1 throughout, prices rise uniformly."""
        # prices: 100, 105, 110, 115  → increments: 5, 5, 5
        # gains = 1*5 + 1*5 + 1*5 = 15
        paths_S    = torch.tensor([[100.0, 105.0, 110.0, 115.0]])  # (1, 4)
        deltas     = torch.ones(1, 3)
        payoff     = torch.tensor([0.0])
        total_cost = torch.tensor([0.0])
        pnl        = compute_pnl(paths_S, deltas, payoff, total_cost)
        assert torch.allclose(pnl, torch.tensor([15.0]))

    def test_p0_shifts_pnl(self):
        """p0 shifts every PnL value by exactly p0."""
        paths_S    = torch.tensor([[100.0, 100.0]])
        deltas     = torch.tensor([[0.5]])
        payoff     = torch.tensor([0.0])
        total_cost = torch.tensor([0.0])
        pnl0  = compute_pnl(paths_S, deltas, payoff, total_cost, p0=0.0)
        pnl5  = compute_pnl(paths_S, deltas, payoff, total_cost, p0=5.0)
        assert torch.allclose(pnl5 - pnl0, torch.tensor([5.0]))

    def test_cost_reduces_pnl(self):
        """Higher transaction cost reduces PnL by exactly that amount."""
        paths_S    = torch.tensor([[100.0, 110.0]])
        deltas     = torch.tensor([[1.0]])
        payoff     = torch.tensor([0.0])
        cost_low   = torch.tensor([1.0])
        cost_high  = torch.tensor([3.0])
        pnl_low  = compute_pnl(paths_S, deltas, payoff, cost_low)
        pnl_high = compute_pnl(paths_S, deltas, payoff, cost_high)
        assert torch.allclose(pnl_low - pnl_high, torch.tensor([2.0]))

    def test_payoff_reduces_pnl(self):
        """Higher payoff reduces PnL by exactly that amount."""
        paths_S    = torch.tensor([[100.0, 100.0]])
        deltas     = torch.tensor([[0.0]])
        total_cost = torch.tensor([0.0])
        pay_low  = torch.tensor([5.0])
        pay_high = torch.tensor([10.0])
        pnl_low  = compute_pnl(paths_S, deltas, pay_low,  total_cost)
        pnl_high = compute_pnl(paths_S, deltas, pay_high, total_cost)
        assert torch.allclose(pnl_low - pnl_high, torch.tensor([5.0]))

    def test_output_shape(self):
        """Assert output shape."""
        N, T = 137, 30
        pnl  = compute_pnl(
            paths_S    = torch.ones(N, T + 1),
            deltas     = torch.ones(N, T),
            payoff     = torch.zeros(N),
            total_cost = torch.zeros(N),
        )
        assert pnl.shape == (N,)

    def test_dtype_preserved(self):
        """Assert dtype preserved."""
        pnl = compute_pnl(
            torch.ones(5, 4, dtype=torch.float32),
            torch.ones(5, 3, dtype=torch.float32),
            torch.zeros(5),
            torch.zeros(5),
        )
        assert pnl.dtype == torch.float32

    def test_multiple_paths(self):
        """Vectorised computation works correctly across N paths."""
        # Path 0: price stays flat → gains=0, payoff=0 → pnl=0
        # Path 1: price rises 10   → gains=10, payoff=10 → pnl=0 (perfect hedge)
        # Path 2: price falls 10   → gains=-10, payoff=0 → pnl=-10
        paths_S = torch.tensor([
            [100.0, 100.0],
            [100.0, 110.0],
            [100.0,  90.0],
        ])
        deltas     = torch.tensor([[1.0], [1.0], [1.0]])
        payoff     = torch.tensor([0.0, 10.0, 0.0])
        total_cost = torch.zeros(3)
        pnl        = compute_pnl(paths_S, deltas, payoff, total_cost)
        expected   = torch.tensor([0.0, 0.0, -10.0])
        assert torch.allclose(pnl, expected)

    # --- Error handling ---

    def test_paths_S_wrong_ndim_raises(self):
        """Assert paths S wrong ndim raises."""
        with pytest.raises(ValueError, match="2-D"):
            compute_pnl(torch.ones(10), torch.ones(10, 1), torch.zeros(10), torch.zeros(10))

    def test_deltas_wrong_ndim_raises(self):
        """Assert deltas wrong ndim raises."""
        with pytest.raises(ValueError, match="2-D"):
            compute_pnl(torch.ones(5, 4), torch.ones(15), torch.zeros(5), torch.zeros(5))

    def test_timestep_mismatch_raises(self):
        """paths_S.shape[1] must equal deltas.shape[1] + 1."""
        with pytest.raises(ValueError, match="paths_S.shape\\[1\\]"):
            compute_pnl(
                torch.ones(5, 10),   # T+1=10
                torch.ones(5, 5),    # T=5 → mismatch
                torch.zeros(5),
                torch.zeros(5),
            )

    def test_N_mismatch_raises(self):
        """Assert N mismatch raises."""
        with pytest.raises(ValueError, match="N="):
            compute_pnl(
                torch.ones(5, 4),
                torch.ones(10, 3),
                torch.zeros(5),
                torch.zeros(5),
            )

    def test_payoff_wrong_shape_raises(self):
        """Assert payoff wrong shape raises."""
        with pytest.raises(ValueError, match="payoff"):
            compute_pnl(
                torch.ones(5, 4),
                torch.ones(5, 3),
                torch.zeros(3),   # wrong N
                torch.zeros(5),
            )

    def test_total_cost_wrong_shape_raises(self):
        """Assert total cost wrong shape raises."""
        with pytest.raises(ValueError, match="total_cost"):
            compute_pnl(
                torch.ones(5, 4),
                torch.ones(5, 3),
                torch.zeros(5),
                torch.zeros(3),   # wrong N
            )


# ===========================================================================
# Integration tests for the full data-to-PnL path.
# ===========================================================================

class TestIntegration:

    """Test cases for TestIntegration."""
    def test_frictionless_pnl_finite(self, bs_batch):
        """End-to-end with random deltas: PnL should be finite everywhere."""
        N = bs_batch.n_paths
        T = bs_batch.n_steps

        deltas     = torch.randn(N, T) * 0.5
        payoff     = call_payoff(bs_batch.paths_S[:, -1], bs_batch.K)
        total_cost = proportional_cost(bs_batch.paths_S[:, :-1], deltas, epsilon=0.0)
        pnl        = compute_pnl(bs_batch.paths_S, deltas, payoff, total_cost)

        assert pnl.shape == (N,)
        assert torch.isfinite(pnl).all(), "PnL contains non-finite values"

    def test_friction_reduces_mean_pnl(self, bs_batch):
        """Adding transaction costs should reduce mean PnL."""
        N = bs_batch.n_paths
        T = bs_batch.n_steps

        deltas  = torch.randn(N, T) * 0.5
        payoff  = call_payoff(bs_batch.paths_S[:, -1], bs_batch.K)
        S_obs   = bs_batch.paths_S[:, :-1]

        cost_free = proportional_cost(S_obs, deltas, epsilon=0.0)
        cost_fric = proportional_cost(S_obs, deltas, epsilon=0.01)

        pnl_free = compute_pnl(bs_batch.paths_S, deltas, payoff, cost_free)
        pnl_fric = compute_pnl(bs_batch.paths_S, deltas, payoff, cost_fric)

        assert pnl_fric.mean() < pnl_free.mean(), (
            "Adding friction should reduce mean PnL"
        )

    def test_delta_one_pnl_structure(self, bs_batch):
        """Assert delta=1 PnL equals cumulative gains minus terminal payoff."""
        N      = bs_batch.n_paths
        T      = bs_batch.n_steps
        S_0    = bs_batch.paths_S[:, 0]
        S_T    = bs_batch.paths_S[:, -1]
        K      = bs_batch.K

        deltas     = torch.ones(N, T)
        payoff     = call_payoff(S_T, K)
        total_cost = proportional_cost(bs_batch.paths_S[:, :-1], deltas, epsilon=0.0)
        pnl        = compute_pnl(bs_batch.paths_S, deltas, payoff, total_cost)

        gains    = (bs_batch.paths_S[:, 1:] - bs_batch.paths_S[:, :-1]).sum(dim=1)
        expected = gains - payoff
        assert torch.allclose(pnl, expected, atol=1e-4)
