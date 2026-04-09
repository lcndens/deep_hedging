"""Terminal PnL computation for deep hedging trajectories.

This module implements the per-path terminal profit-and-loss identity used by
the training objective, consistent with Buehler et al. (2019, eq. 2.1):
``PnL_i = p0 + sum_t delta_t (S_{t+1} - S_t) - C_T - payoff(S_T)``.

Single-instrument:
    paths_S  (N, T+1)  — spot prices
    deltas   (N, T)    — hedge ratios

Two-instrument:
    paths_prices  (N, T+1, I)  — stacked instrument prices
    deltas        (N, T, I)    — stacked hedge ratios
    Gains sum over both the time and instrument axes.
"""

from __future__ import annotations

import torch


def compute_pnl(
    paths_S: torch.Tensor,
    deltas: torch.Tensor,
    payoff: torch.Tensor,
    total_cost: torch.Tensor,
    p0: float = 0.0,
) -> torch.Tensor:
    """Compute terminal PnL for each simulated path.

    The function combines trading gains, cumulative transaction costs, and
    terminal option liability into a single ``PnL`` tensor used by CVaR
    optimisation.

    Parameters
    ----------
    paths_S : torch.Tensor
        Single-instrument: spot-price paths ``(N, T+1)``.
        Two-instrument:    stacked prices ``(N, T+1, I)`` where ``I=2``.
    deltas : torch.Tensor
        Single-instrument: hedge ratios ``(N, T)``.
        Two-instrument:    stacked deltas ``(N, T, I)``.
    payoff : torch.Tensor, shape (N,)
        Terminal option payoff ``payoff(S_T)`` per path.
    total_cost : torch.Tensor, shape (N,)
        Cumulative transaction costs ``C_T`` per path.
    p0 : float, optional
        Initial premium term in the PnL identity. During training this is
        typically fixed at ``0.0`` by cash-invariance of CVaR.

    Returns
    -------
    torch.Tensor
        Terminal PnL tensor with shape ``(N,)`` on the same device as
        ``paths_S``.

    Raises
    ------
    ValueError
        If tensor shapes are inconsistent or ndim combinations are invalid.
    """
    _validate_inputs(paths_S, deltas, payoff, total_cost)

    # Price increments aligned with delta_t for t = 0, ..., T-1.
    price_increments = paths_S[:, 1:] - paths_S[:, :-1]   # (N, T) or (N, T, I)

    if deltas.ndim == 2:
        # Single-instrument: sum over T.
        gains = (deltas * price_increments).sum(dim=1)
    else:
        # Two-instrument: sum over T and I.
        gains = (deltas * price_increments).sum(dim=(1, 2))

    pnl = p0 + gains - total_cost - payoff
    return pnl


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    paths_S: torch.Tensor,
    deltas: torch.Tensor,
    payoff: torch.Tensor,
    total_cost: torch.Tensor,
) -> None:
    """Validate tensor ranks and shape consistency for PnL inputs.

    Parameters
    ----------
    paths_S : torch.Tensor
        ``(N, T+1)`` for single-instrument or ``(N, T+1, I)`` for multi.
    deltas : torch.Tensor
        ``(N, T)`` for single-instrument or ``(N, T, I)`` for multi.
    payoff : torch.Tensor
        Terminal payoff vector with expected shape ``(N,)``.
    total_cost : torch.Tensor
        Transaction-cost vector with expected shape ``(N,)``.

    Raises
    ------
    ValueError
        If any tensor has an incompatible rank or shape.
    """
    if paths_S.ndim not in (2, 3):
        raise ValueError(
            f"paths_S must be 2-D (N, T+1) or 3-D (N, T+1, I), "
            f"got shape {tuple(paths_S.shape)}."
        )
    if deltas.ndim not in (2, 3):
        raise ValueError(
            f"deltas must be 2-D (N, T) or 3-D (N, T, I), "
            f"got shape {tuple(deltas.shape)}."
        )
    if paths_S.ndim != deltas.ndim:
        raise ValueError(
            f"paths_S and deltas must have the same number of dimensions, "
            f"got paths_S.ndim={paths_S.ndim} and deltas.ndim={deltas.ndim}."
        )

    N      = paths_S.shape[0]
    T_plus = paths_S.shape[1]
    T      = deltas.shape[1]

    if T_plus != T + 1:
        raise ValueError(
            f"paths_S has {T_plus} timesteps but deltas has {T} steps. "
            f"Expected paths_S.shape[1] == deltas.shape[1] + 1."
        )
    if deltas.shape[0] != N:
        raise ValueError(
            f"paths_S has N={N} paths but deltas has N={deltas.shape[0]}."
        )
    if deltas.ndim == 3 and paths_S.shape[2] != deltas.shape[2]:
        raise ValueError(
            f"paths_S instrument dim {paths_S.shape[2]} does not match "
            f"deltas instrument dim {deltas.shape[2]}."
        )
    if payoff.shape != (N,):
        raise ValueError(
            f"payoff must have shape ({N},), got {tuple(payoff.shape)}."
        )
    if total_cost.shape != (N,):
        raise ValueError(
            f"total_cost must have shape ({N},), got {tuple(total_cost.shape)}."
        )
