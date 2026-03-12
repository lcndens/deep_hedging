"""Terminal PnL computation for deep hedging trajectories.

This module implements the per-path terminal profit-and-loss identity used by
the training objective, consistent with Buehler et al. (2019, eq. 2.1):
``PnL_i = p0 + sum_t delta_t (S_{t+1} - S_t) - C_T - payoff(S_T)``.
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
        Spot-price paths with shape ``(N, T+1)``, where ``N`` is the number of
        simulated paths and ``T`` is the number of hedging timesteps.
    deltas : torch.Tensor
        Hedge ratios ``delta_t`` with shape ``(N, T)``.
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

    Notes
    -----
    The increment term uses ``paths_S[:, 1:] - paths_S[:, :-1]`` so price
    changes are aligned with ``deltas[:, t]`` for ``t = 0, ..., T-1``.

    Raises
    ------
    ValueError
        If tensor shapes are inconsistent.
    """
    _validate_inputs(paths_S, deltas, payoff, total_cost)

    # Aligns delta_t with the corresponding price increment S_{t+1} - S_t.
    price_increments = paths_S[:, 1:] - paths_S[:, :-1]
    gains = (deltas * price_increments).sum(dim=1)

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
        Spot paths with expected shape ``(N, T+1)``.
    deltas : torch.Tensor
        Hedge ratios with expected shape ``(N, T)``.
    payoff : torch.Tensor
        Terminal payoff vector with expected shape ``(N,)``.
    total_cost : torch.Tensor
        Transaction-cost vector with expected shape ``(N,)``.

    Raises
    ------
    ValueError
        If any tensor has an incompatible rank or shape.
    """
    if paths_S.ndim != 2:
        raise ValueError(
            f"paths_S must be 2-D (N, T+1), got shape {tuple(paths_S.shape)}."
        )
    if deltas.ndim != 2:
        raise ValueError(
            f"deltas must be 2-D (N, T), got shape {tuple(deltas.shape)}."
        )

    N = paths_S.shape[0]
    T_plus = paths_S.shape[1]
    T = deltas.shape[1]

    if T_plus != T + 1:
        raise ValueError(
            f"paths_S has {T_plus} timesteps but deltas has {T} steps. "
            f"Expected paths_S.shape[1] == deltas.shape[1] + 1."
        )
    if deltas.shape[0] != N:
        raise ValueError(
            f"paths_S has N={N} paths but deltas has N={deltas.shape[0]}."
        )
    if payoff.shape != (N,):
        raise ValueError(
            f"payoff must have shape ({N},), got {tuple(payoff.shape)}."
        )
    if total_cost.shape != (N,):
        raise ValueError(
            f"total_cost must have shape ({N},), got {tuple(total_cost.shape)}."
        )
