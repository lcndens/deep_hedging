"""Tests for src/derivatives/barrier.py — up-and-out call barrier payoff."""

from __future__ import annotations

import pytest
import torch

from src.derivatives.barrier import compute_barrier_payoff, UpAndOutBarrierPayoff
from src.derivatives.european import call_payoff
from src.pnl.compute import compute_pnl
from src.frictions.proportional import proportional_cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paths(rows: list[list[float]]) -> torch.Tensor:
    """Build a (N, T+1) float32 path tensor from a list of lists."""
    return torch.tensor(rows, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

class TestBarrierPayoffKnockOut:
    def test_single_path_breaches_at_intermediate_step_zero_payoff(self):
        """Path that touches barrier at an intermediate timestep → 0."""
        # S_t: 95, 100, 115(breach), 110, 108 — terminal 108 > K=100 but knocked out
        paths_S = _make_paths([[95.0, 100.0, 115.0, 110.0, 108.0]])
        out = compute_barrier_payoff(paths_S, K=100.0, B=110.0)
        assert out.shape == (1,)
        assert out.item() == pytest.approx(0.0)

    def test_single_path_breaches_at_terminal_zero_payoff(self):
        """Path that reaches barrier exactly at maturity → 0."""
        paths_S = _make_paths([[95.0, 98.0, 101.0, 104.0, 110.0]])
        out = compute_barrier_payoff(paths_S, K=100.0, B=110.0)
        assert out.item() == pytest.approx(0.0)

    def test_single_path_breaches_at_t0_zero_payoff(self):
        """Path that starts at or above barrier → 0."""
        paths_S = _make_paths([[110.0, 105.0, 108.0, 103.0, 106.0]])
        out = compute_barrier_payoff(paths_S, K=100.0, B=110.0)
        assert out.item() == pytest.approx(0.0)

    def test_multiple_paths_mixed(self):
        """Only paths that never breach barrier get nonzero payoff."""
        # Path 0: breaches (max=115) → 0
        # Path 1: never breaches (max=108), terminal=108 > K=100 → 8
        # Path 2: never breaches (max=105), terminal=95 < K=100 → 0
        paths_S = _make_paths([
            [95.0, 100.0, 115.0, 110.0, 108.0],
            [95.0, 100.0, 105.0, 107.0, 108.0],
            [95.0, 100.0, 103.0, 105.0,  95.0],
        ])
        out = compute_barrier_payoff(paths_S, K=100.0, B=110.0)
        assert out.shape == (3,)
        assert out[0].item() == pytest.approx(0.0)
        assert out[1].item() == pytest.approx(8.0)
        assert out[2].item() == pytest.approx(0.0)


class TestBarrierEqualsEuropean:
    def test_no_breach_equals_call_payoff(self):
        """When no path ever breaches the barrier, payoff == European call payoff."""
        # All paths stay well below barrier=200
        paths_S = _make_paths([
            [100.0, 102.0, 105.0, 103.0, 108.0],
            [100.0,  98.0,  95.0,  92.0,  90.0],
            [100.0, 101.0, 103.0, 106.0, 112.0],
        ])
        K, B = 100.0, 200.0
        barrier_out  = compute_barrier_payoff(paths_S, K=K, B=B)
        european_out = call_payoff(paths_S[:, -1], K=K)
        assert torch.allclose(barrier_out, european_out)

    def test_otm_no_breach_zero(self):
        """Below strike and never at barrier → payoff is 0 for both."""
        paths_S = _make_paths([
            [100.0, 98.0, 95.0, 93.0, 90.0],
        ])
        out = compute_barrier_payoff(paths_S, K=100.0, B=150.0)
        assert out.item() == pytest.approx(0.0)


class TestBarrierShape:
    def test_output_shape_is_N(self):
        """Output tensor has shape (N,) regardless of path length."""
        N, T = 50, 10
        paths_S = torch.rand(N, T + 1) * 20 + 90   # values in [90, 110]
        out = compute_barrier_payoff(paths_S, K=100.0, B=200.0)
        assert out.shape == (N,)

    def test_output_dtype_float32(self):
        paths_S = _make_paths([[100.0, 102.0, 104.0]])
        out = compute_barrier_payoff(paths_S, K=95.0, B=150.0)
        assert out.dtype == torch.float32


class TestBarrierValidation:
    def test_barrier_must_exceed_strike(self):
        """B <= K should raise ValueError."""
        paths_S = _make_paths([[100.0, 102.0, 104.0]])
        with pytest.raises(ValueError, match="Barrier B"):
            compute_barrier_payoff(paths_S, K=110.0, B=105.0)

    def test_barrier_equal_to_strike_raises(self):
        paths_S = _make_paths([[100.0, 102.0, 104.0]])
        with pytest.raises(ValueError, match="Barrier B"):
            compute_barrier_payoff(paths_S, K=100.0, B=100.0)

    def test_non_positive_strike_raises(self):
        paths_S = _make_paths([[100.0, 102.0, 104.0]])
        with pytest.raises(ValueError, match="Strike K"):
            compute_barrier_payoff(paths_S, K=0.0, B=150.0)

    def test_wrong_ndim_raises(self):
        """1-D or 3-D paths_S should raise ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            compute_barrier_payoff(torch.tensor([100.0, 102.0]), K=95.0, B=150.0)
        with pytest.raises(ValueError, match="2-D"):
            compute_barrier_payoff(torch.ones(5, 6, 2), K=95.0, B=150.0)


# ---------------------------------------------------------------------------
# UpAndOutBarrierPayoff class
# ---------------------------------------------------------------------------

class TestUpAndOutBarrierPayoffClass:
    def test_callable_matches_function(self):
        payoff_obj = UpAndOutBarrierPayoff(K=100.0, B=130.0)
        paths_S = torch.rand(20, 6) * 30 + 90
        out_class = payoff_obj(paths_S)
        out_fn    = compute_barrier_payoff(paths_S, K=100.0, B=130.0)
        assert torch.allclose(out_class, out_fn)

    def test_class_validates_b_gt_k(self):
        with pytest.raises(ValueError, match="Barrier B"):
            UpAndOutBarrierPayoff(K=110.0, B=100.0)

    def test_class_validates_positive_strike(self):
        with pytest.raises(ValueError, match="Strike K"):
            UpAndOutBarrierPayoff(K=-5.0, B=100.0)

    def test_repr(self):
        obj = UpAndOutBarrierPayoff(K=100.0, B=130.0)
        assert "K=100.0" in repr(obj)
        assert "B=130.0" in repr(obj)


# ---------------------------------------------------------------------------
# Round-trip through compute_pnl
# ---------------------------------------------------------------------------

class TestBarrierRoundTrip:
    def test_finite_pnl_through_compute_pnl(self):
        """Barrier payoff produces finite PnL values through the full pipeline."""
        N, T = 100, 5
        torch.manual_seed(0)
        # Paths staying well below barrier
        paths_S = torch.rand(N, T + 1) * 15 + 95   # values in [95, 110]
        K, B = 100.0, 200.0

        # Constant delta of 0.5 across all timesteps
        deltas     = torch.full((N, T), 0.5)
        payoff     = compute_barrier_payoff(paths_S, K=K, B=B)
        total_cost = proportional_cost(paths_S[:, :-1], deltas, epsilon=0.0)
        pnl        = compute_pnl(paths_S, deltas, payoff, total_cost)

        assert pnl.shape == (N,)
        assert torch.isfinite(pnl).all(), "PnL contains non-finite values"

    def test_knocked_out_paths_pnl_shifted(self):
        """Paths that are knocked out pay 0 payoff — PnL is gains only."""
        # Single path that always breaches — payoff should be 0
        paths_S = _make_paths([[100.0, 105.0, 115.0, 112.0, 108.0]])
        K, B    = 100.0, 110.0
        deltas  = torch.zeros(1, 4)   # no position
        payoff  = compute_barrier_payoff(paths_S, K=K, B=B)
        assert payoff.item() == pytest.approx(0.0)

        total_cost = proportional_cost(paths_S[:, :-1], deltas, epsilon=0.0)
        pnl        = compute_pnl(paths_S, deltas, payoff, total_cost)
        # gains=0, payoff=0, cost=0 → pnl = 0
        assert pnl.item() == pytest.approx(0.0)
