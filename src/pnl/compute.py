"""PnL computation — Stage 3 (PnL).

Computes terminal profit and loss for each simulated hedging path.
Pure tensor arithmetic — no trainable parameters.

Formula (Buehler et al. 2019, eq. 2.1):
    PnL_i = p0
           + sum_{t=0}^{T-1} delta_t * (S_{t+1} - S_t)   [trading gains]
           - total_cost                                     [transaction costs]
           - payoff(S_T)                                    [option liability]

p0 is fixed at 0.0 during training. By the cash-invariance of CVaR, p0
shifts all PnL by a constant and does not affect the optimal hedge.

Shape verification:
    paths_S[:, 1:]   prices at t=1..T     (N, T)
    paths_S[:, :-1]  prices at t=0..T-1   (N, T)
    difference       S_{t+1} - S_t        (N, T)  ← aligns with deltas (N, T) ✓

Usage
-----
    from src.pnl.compute import compute_pnl

    pnl = compute_pnl(
        paths_S=batch.paths_S,   # (N, T+1)
        deltas=deltas,            # (N, T)
        payoff=payoff,            # (N,)
        total_cost=total_cost,    # (N,)
        p0=0.0,
    )   # → (N,)
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
    """Compute terminal PnL per path.

    Parameters
    ----------
    paths_S : torch.Tensor, shape (N, T+1)
        Full spot price paths including the maturity step.
        Typically ``batch.paths_S``.
    deltas : torch.Tensor, shape (N, T)
        Hedge ratios output by the policy network.
    payoff : torch.Tensor, shape (N,)
        Option payoff at maturity. Output of ``call_payoff`` or ``put_payoff``.
    total_cost : torch.Tensor, shape (N,)
        Total transaction costs. Output of ``proportional_cost``.
    p0 : float, optional
        Initial option premium received. Fixed at 0.0 during training
        by cash-invariance of CVaR. Default: 0.0.

    Returns
    -------
    pnl : torch.Tensor, shape (N,), same dtype and device as paths_S.
        Terminal PnL per path. Positive = profit, negative = loss.

    Raises
    ------
    ValueError
        If tensor shapes are inconsistent.
    """
    _validate_inputs(paths_S, deltas, payoff, total_cost)

    # Trading gains: sum_t delta_t * (S_{t+1} - S_t)
    # paths_S[:,1:]  → S_{t+1} for t=0..T-1   shape (N, T)
    # paths_S[:,:-1] → S_t     for t=0..T-1   shape (N, T)
    price_increments = paths_S[:, 1:] - paths_S[:, :-1]          # (N, T)
    gains            = (deltas * price_increments).sum(dim=1)     # (N,)

    pnl = p0 + gains - total_cost - payoff                        # (N,)
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
    if paths_S.ndim != 2:
        raise ValueError(
            f"paths_S must be 2-D (N, T+1), got shape {tuple(paths_S.shape)}."
        )
    if deltas.ndim != 2:
        raise ValueError(
            f"deltas must be 2-D (N, T), got shape {tuple(deltas.shape)}."
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
    if payoff.shape != (N,):
        raise ValueError(
            f"payoff must have shape ({N},), got {tuple(payoff.shape)}."
        )
    if total_cost.shape != (N,):
        raise ValueError(
            f"total_cost must have shape ({N},), got {tuple(total_cost.shape)}."
        )
