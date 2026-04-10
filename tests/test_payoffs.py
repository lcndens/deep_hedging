"""Tests for path-dependent payoff functions.

Covers:
  - asian_call_payoff: equals European call payoff on flat paths; shape (N,)
  - lookback_call_payoff: >= European call payoff for all paths; shape (N,)
  - binary_call_payoff: output is exactly 0 or cash; shape (N,)
  - Input validation: ndim and K>0 checks for all three functions
"""

from __future__ import annotations

import pytest
import torch

from src.derivatives.asian import asian_call_payoff
from src.derivatives.lookback import lookback_call_payoff
from src.derivatives.binary import binary_call_payoff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _european_call(S_T: torch.Tensor, K: float) -> torch.Tensor:
    """Vanilla European call payoff max(S_T - K, 0) for reference."""
    return torch.clamp(S_T - K, min=0.0)


def _flat_paths(n_paths: int, n_steps: int, price: float) -> torch.Tensor:
    """Return constant-price paths of shape (N, T+1)."""
    return torch.full((n_paths, n_steps + 1), price, dtype=torch.float32)


def _random_paths(n_paths: int, n_steps: int, seed: int = 0) -> torch.Tensor:
    """Return random GBM-like paths centred near 100."""
    g = torch.Generator()
    g.manual_seed(seed)
    z = torch.randn(n_paths, n_steps, generator=g)
    log_increments = -0.02 * z  # tiny drift, sigma≈0.2 normalised
    logS = torch.cat([torch.zeros(n_paths, 1), log_increments.cumsum(dim=1)], dim=1)
    return 100.0 * logS.exp()


N = 200
T = 10
K = 100.0


# ---------------------------------------------------------------------------
# Asian call payoff
# ---------------------------------------------------------------------------

class TestAsianCallPayoff:

    def test_shape(self):
        paths = _random_paths(N, T)
        out = asian_call_payoff(paths, K)
        assert out.shape == (N,), f"Expected shape ({N},), got {out.shape}"

    def test_non_negative(self):
        paths = _random_paths(N, T)
        out = asian_call_payoff(paths, K)
        assert (out >= 0.0).all(), "Asian payoff must be non-negative"

    def test_flat_path_equals_european(self):
        """On a flat path (constant price P), Asian mean == P, so Asian == European."""
        P = 110.0
        paths = _flat_paths(N, T, P)
        asian = asian_call_payoff(paths, K)
        european = _european_call(torch.full((N,), P), K)
        torch.testing.assert_close(asian, european, rtol=1e-5, atol=1e-5)

    def test_flat_otm_path_zero(self):
        """OTM flat path should produce zero payoff."""
        paths = _flat_paths(N, T, 90.0)
        out = asian_call_payoff(paths, K)
        assert (out == 0.0).all(), "OTM flat path: Asian payoff should be 0"

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            asian_call_payoff(torch.randn(N), K)

    def test_nonpositive_K_raises(self):
        with pytest.raises(ValueError, match="positive"):
            asian_call_payoff(_random_paths(N, T), 0.0)

    def test_negative_K_raises(self):
        with pytest.raises(ValueError, match="positive"):
            asian_call_payoff(_random_paths(N, T), -5.0)


# ---------------------------------------------------------------------------
# Lookback call payoff
# ---------------------------------------------------------------------------

class TestLookbackCallPayoff:

    def test_shape(self):
        paths = _random_paths(N, T)
        out = lookback_call_payoff(paths, K)
        assert out.shape == (N,), f"Expected shape ({N},), got {out.shape}"

    def test_non_negative(self):
        paths = _random_paths(N, T)
        out = lookback_call_payoff(paths, K)
        assert (out >= 0.0).all(), "Lookback payoff must be non-negative"

    def test_dominates_european(self):
        """Lookback payoff >= European call for every path (max >= terminal)."""
        paths = _random_paths(N, T, seed=1)
        S_T = paths[:, -1]
        lookback = lookback_call_payoff(paths, K)
        european = _european_call(S_T, K)
        assert (lookback >= european - 1e-5).all(), (
            "Lookback payoff must dominate European call payoff for all paths"
        )

    def test_flat_path_equals_european(self):
        """On a flat path, running max == S, so lookback == European."""
        P = 105.0
        paths = _flat_paths(N, T, P)
        lookback = lookback_call_payoff(paths, K)
        european = _european_call(torch.full((N,), P), K)
        torch.testing.assert_close(lookback, european, rtol=1e-5, atol=1e-5)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            lookback_call_payoff(torch.randn(N), K)

    def test_nonpositive_K_raises(self):
        with pytest.raises(ValueError, match="positive"):
            lookback_call_payoff(_random_paths(N, T), 0.0)


# ---------------------------------------------------------------------------
# Binary (cash-or-nothing) call payoff
# ---------------------------------------------------------------------------

class TestBinaryCallPayoff:

    def test_shape(self):
        S_T = torch.randn(N).abs() * 10 + 100
        out = binary_call_payoff(S_T, K)
        assert out.shape == (N,), f"Expected shape ({N},), got {out.shape}"

    def test_output_is_zero_or_cash(self):
        """Every output entry must be exactly 0 or exactly cash."""
        cash = 2.5
        S_T = torch.randn(N) * 20 + 100
        out = binary_call_payoff(S_T, K, cash=cash)
        valid = (out == 0.0) | (out == cash)
        assert valid.all(), "Binary payoff must be exactly 0 or cash for every path"

    def test_itm_paths_get_cash(self):
        """All ITM paths must receive the cash amount."""
        cash = 1.0
        S_T = torch.full((N,), 110.0)
        out = binary_call_payoff(S_T, K, cash=cash)
        assert (out == cash).all(), "All ITM paths must get cash payoff"

    def test_otm_paths_get_zero(self):
        """All OTM paths must receive zero."""
        S_T = torch.full((N,), 90.0)
        out = binary_call_payoff(S_T, K)
        assert (out == 0.0).all(), "All OTM paths must get zero payoff"

    def test_atm_paths_get_zero(self):
        """Exactly at-the-money (S_T == K) should return zero (strict >)."""
        S_T = torch.full((N,), K)
        out = binary_call_payoff(S_T, K)
        assert (out == 0.0).all(), "ATM paths (S_T == K) must get zero (strict > condition)"

    def test_custom_cash_amount(self):
        """Cash amount is passed through correctly."""
        cash = 5.0
        S_T = torch.full((N,), 120.0)
        out = binary_call_payoff(S_T, K, cash=cash)
        assert (out == cash).all(), f"ITM paths must get cash={cash}"

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            binary_call_payoff(torch.randn(N, T), K)

    def test_nonpositive_K_raises(self):
        with pytest.raises(ValueError, match="positive"):
            binary_call_payoff(torch.randn(N), 0.0)

    def test_negative_K_raises(self):
        with pytest.raises(ValueError, match="positive"):
            binary_call_payoff(torch.randn(N), -1.0)
