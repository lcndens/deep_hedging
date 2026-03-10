"""Option payoff functions — Stage 3 (Derivatives).

Terminal payoffs for European options. This module is intentionally minimal —
it computes the option liability at maturity that the hedger must cover.

Payoffs are computed once per forward pass from the final spot price S_T.
No parameters, no gradients flow through these functions.

To swap option types per experiment, change which function is called in the
training loop. The PnL module accepts any (N,) payoff tensor.

Usage
-----
    from src.derivatives.european import call_payoff, put_payoff

    payoff = call_payoff(batch.paths_S[:, -1], batch.K)   # (N,)
"""

from __future__ import annotations

import torch


def call_payoff(S_T: torch.Tensor, K: float) -> torch.Tensor:
    """European call payoff at maturity: max(S_T - K, 0).

    Parameters
    ----------
    S_T : torch.Tensor, shape (N,)
        Spot price at maturity for each path.
    K : float
        Strike price. Must be positive.

    Returns
    -------
    payoff : torch.Tensor, shape (N,), same dtype and device as S_T.
    """
    _validate_inputs(S_T, K)
    return torch.clamp(S_T - K, min=0.0)


def put_payoff(S_T: torch.Tensor, K: float) -> torch.Tensor:
    """European put payoff at maturity: max(K - S_T, 0).

    Parameters
    ----------
    S_T : torch.Tensor, shape (N,)
        Spot price at maturity for each path.
    K : float
        Strike price. Must be positive.

    Returns
    -------
    payoff : torch.Tensor, shape (N,), same dtype and device as S_T.
    """
    _validate_inputs(S_T, K)
    return torch.clamp(K - S_T, min=0.0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(S_T: torch.Tensor, K: float) -> None:
    if S_T.ndim != 1:
        raise ValueError(
            f"S_T must be a 1-D tensor of shape (N,), got shape {tuple(S_T.shape)}."
        )
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}.")
