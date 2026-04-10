"""Binary (digital) option payoff functions.

Implements a cash-or-nothing binary call that pays a fixed cash amount
if the terminal spot price exceeds the strike, and zero otherwise.
"""

from __future__ import annotations

import torch


def binary_call_payoff(
    S_T:  torch.Tensor,
    K:    float,
    cash: float = 1.0,
) -> torch.Tensor:
    """Cash-or-nothing binary call payoff: cash if S_T > K, else 0.

    Parameters
    ----------
    S_T : torch.Tensor
        Spot prices at maturity with shape ``(N,)``, where ``N`` is the number
        of simulated paths.
    K : float
        Strike price. Must be positive.
    cash : float, optional
        Fixed payment amount when the option finishes in the money.
        Default: 1.0.

    Returns
    -------
    torch.Tensor
        Payoff tensor of shape ``(N,)``. Each entry is ``cash`` when
        ``S_T > K`` and ``0.0`` otherwise.

    Raises
    ------
    ValueError
        If ``S_T`` is not one-dimensional or ``K <= 0``.
    """
    _validate_inputs(S_T, K)
    return torch.where(S_T > K, torch.full_like(S_T, cash), torch.zeros_like(S_T))


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
