"""European option payoff functions used in terminal liability calculation.

The module maps maturity prices ``S_T`` to per-path liabilities for calls or
puts, which are then subtracted in the terminal PnL identity.
"""

from __future__ import annotations

import torch


def call_payoff(S_T: torch.Tensor, K: float) -> torch.Tensor:
    """European call payoff at maturity: max(S_T - K, 0).

    Parameters
    ----------
    S_T : torch.Tensor
        Spot prices at maturity with shape ``(N,)``, where ``N`` is the number
        of simulated paths.
    K : float
        Strike price. Must be positive.

    Returns
    -------
    torch.Tensor
        Call payoff tensor ``max(S_T - K, 0)`` with shape ``(N,)``.

    Raises
    ------
    ValueError
        If ``S_T`` is not one-dimensional or ``K <= 0``.
    """
    _validate_inputs(S_T, K)
    return torch.clamp(S_T - K, min=0.0)


def put_payoff(S_T: torch.Tensor, K: float) -> torch.Tensor:
    """European put payoff at maturity: max(K - S_T, 0).

    Parameters
    ----------
    S_T : torch.Tensor
        Spot prices at maturity with shape ``(N,)``, where ``N`` is the number
        of simulated paths.
    K : float
        Strike price. Must be positive.

    Returns
    -------
    torch.Tensor
        Put payoff tensor ``max(K - S_T, 0)`` with shape ``(N,)``.

    Raises
    ------
    ValueError
        If ``S_T`` is not one-dimensional or ``K <= 0``.
    """
    _validate_inputs(S_T, K)
    return torch.clamp(K - S_T, min=0.0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(S_T: torch.Tensor, K: float) -> None:
    """Validate payoff input rank and strike positivity.

    Parameters
    ----------
    S_T : torch.Tensor
        Maturity spot tensor with expected shape ``(N,)``.
    K : float
        Strike value.

    Raises
    ------
    ValueError
        If ``S_T`` is not one-dimensional or ``K <= 0``.
    """
    if S_T.ndim != 1:
        raise ValueError(
            f"S_T must be a 1-D tensor of shape (N,), got shape {tuple(S_T.shape)}."
        )
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}.")
