"""Proportional transaction-cost model for deep hedging.

This module computes cumulative trading costs along hedging paths using
``cost_t = epsilon * S_t * |delta_t - delta_{t-1}|`` with ``delta_{-1} = 0``.

Single-instrument:
    S_obs   (N, T)   — spot prices at each hedging step
    deltas  (N, T)   — hedge ratios

Two-instrument:
    S_obs   (N, T, I)  — stacked instrument prices at each hedging step
    deltas  (N, T, I)  — stacked hedge ratios
    Cost sums over both the time and instrument axes.
"""

from __future__ import annotations

import torch


def proportional_cost(
    S_obs: torch.Tensor,
    deltas: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Compute total proportional transaction costs per path.

    Parameters
    ----------
    S_obs : torch.Tensor
        Single-instrument: spot prices ``(N, T)``.
        Two-instrument:    stacked prices ``(N, T, I)`` where ``I=2``.
    deltas : torch.Tensor
        Single-instrument: hedge ratios ``(N, T)``.
        Two-instrument:    stacked deltas ``(N, T, I)``.
    epsilon : float
        Proportional cost rate. 0.0 → frictionless (returns zero tensor).
        Typical values: 0.001 (10 bps), 0.01 (100 bps).

    Returns
    -------
    torch.Tensor
        Total transaction cost per path with shape ``(N,)``.

    Notes
    -----
    A zero column (or plane for multi-instrument) is prepended so that
    ``delta_prev[:, 0] = 0``, matching the convention that initial inventory
    before trading is zero.

    Raises
    ------
    ValueError
        If S_obs and deltas shapes do not match, tensors are not 2-D or 3-D,
        or ``epsilon < 0``.
    """
    _validate_inputs(S_obs, deltas, epsilon)

    if epsilon == 0.0:
        return torch.zeros(S_obs.shape[0], dtype=S_obs.dtype, device=S_obs.device)

    N = S_obs.shape[0]

    if deltas.ndim == 2:
        # Single-instrument: delta_prev prepended as a zero column (N, 1).
        zeros      = torch.zeros(N, 1, dtype=deltas.dtype, device=deltas.device)
        delta_prev = torch.cat([zeros, deltas[:, :-1]], dim=1)       # (N, T)
        cost_per_step = epsilon * S_obs * torch.abs(deltas - delta_prev)
        return cost_per_step.sum(dim=1)                               # (N,)

    else:
        # Two-instrument: delta_prev prepended as a zero plane (N, 1, I).
        zeros      = torch.zeros(N, 1, deltas.shape[2], dtype=deltas.dtype, device=deltas.device)
        delta_prev = torch.cat([zeros, deltas[:, :-1, :]], dim=1)    # (N, T, I)
        cost_per_step = epsilon * S_obs * torch.abs(deltas - delta_prev)
        return cost_per_step.sum(dim=(1, 2))                          # (N,)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    S_obs: torch.Tensor,
    deltas: torch.Tensor,
    epsilon: float,
) -> None:
    """Validate shapes and constraints for transaction-cost inputs.

    Parameters
    ----------
    S_obs : torch.Tensor
        ``(N, T)`` or ``(N, T, I)`` instrument prices.
    deltas : torch.Tensor
        ``(N, T)`` or ``(N, T, I)`` hedge ratios.
    epsilon : float
        Proportional transaction-cost rate.

    Raises
    ------
    ValueError
        If tensor ranks or shapes are invalid, or ``epsilon < 0``.
    """
    if S_obs.ndim not in (2, 3):
        raise ValueError(
            f"S_obs must be 2-D (N, T) or 3-D (N, T, I), "
            f"got shape {tuple(S_obs.shape)}."
        )
    if deltas.ndim not in (2, 3):
        raise ValueError(
            f"deltas must be 2-D (N, T) or 3-D (N, T, I), "
            f"got shape {tuple(deltas.shape)}."
        )
    if S_obs.shape != deltas.shape:
        raise ValueError(
            f"S_obs shape {tuple(S_obs.shape)} does not match "
            f"deltas shape {tuple(deltas.shape)}."
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}.")
