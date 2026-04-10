"""Asian option payoff functions.

Implements the fixed-strike arithmetic Asian call payoff based on the
path average of the underlying over all T+1 observed prices.
"""

from __future__ import annotations

import torch


def asian_call_payoff(paths_S: torch.Tensor, K: float) -> torch.Tensor:
    """Fixed-strike arithmetic Asian call payoff: max(mean(S_0..S_T) - K, 0).

    Parameters
    ----------
    paths_S : torch.Tensor
        Full spot price paths with shape ``(N, T+1)``, including the initial
        price and all intermediate and terminal prices.
    K : float
        Strike price. Must be positive.

    Returns
    -------
    torch.Tensor
        Payoff tensor of shape ``(N,)``. Each entry is
        ``max(arithmetic_mean(S_0,...,S_T) - K, 0)``.

    Raises
    ------
    ValueError
        If ``paths_S`` is not 2-D or ``K <= 0``.
    """
    _validate_inputs(paths_S, K)
    path_mean = paths_S.mean(dim=1)          # (N,)
    return torch.clamp(path_mean - K, min=0.0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(paths_S: torch.Tensor, K: float) -> None:
    if paths_S.ndim != 2:
        raise ValueError(
            f"paths_S must be a 2-D tensor of shape (N, T+1), "
            f"got shape {tuple(paths_S.shape)}."
        )
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}.")
