"""Proportional transaction-cost model for deep hedging.

This module computes cumulative trading costs along hedging paths using
``cost_t = epsilon * S_t * |delta_t - delta_{t-1}|`` with ``delta_{-1} = 0``.
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
        Spot prices with shape ``(N, T)``, where ``N`` is the number of paths
        and ``T`` is the number of hedging timesteps.
    deltas : torch.Tensor
        Hedge ratios ``delta_t`` with shape ``(N, T)``.
    epsilon : float
        Proportional cost rate. 0.0 → frictionless (returns zero tensor).
        Typical values: 0.001 (10 bps), 0.01 (100 bps).

    Returns
    -------
    torch.Tensor
        Total transaction cost per path with shape ``(N,)``.

    Notes
    -----
    A zero column is prepended conceptually so that ``delta_prev[:, 0] = 0``,
    matching the convention that the initial inventory before trading is zero.

    Raises
    ------
    ValueError
        If S_obs and deltas shapes do not match or are not 2-D.
    """
    _validate_inputs(S_obs, deltas, epsilon)

    if epsilon == 0.0:
        return torch.zeros(S_obs.shape[0], dtype=S_obs.dtype, device=S_obs.device)

    # This constructs delta_{t-1} for each hedging step with delta_{-1}=0.
    N = S_obs.shape[0]
    zeros = torch.zeros(N, 1, dtype=deltas.dtype, device=deltas.device)
    delta_prev = torch.cat([zeros, deltas[:, :-1]], dim=1)   # (N, T)

    cost_per_step = epsilon * S_obs * torch.abs(deltas - delta_prev)
    total_cost = cost_per_step.sum(dim=1)

    return total_cost


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
        Spot tensor expected to have shape ``(N, T)``.
    deltas : torch.Tensor
        Hedge-ratio tensor expected to have shape ``(N, T)``.
    epsilon : float
        Proportional transaction-cost rate.

    Raises
    ------
    ValueError
        If tensor ranks or shapes are invalid, or ``epsilon < 0``.
    """
    if S_obs.ndim != 2:
        raise ValueError(
            f"S_obs must be 2-D (N, T), got shape {tuple(S_obs.shape)}."
        )
    if deltas.ndim != 2:
        raise ValueError(
            f"deltas must be 2-D (N, T), got shape {tuple(deltas.shape)}."
        )
    if S_obs.shape != deltas.shape:
        raise ValueError(
            f"S_obs shape {tuple(S_obs.shape)} does not match "
            f"deltas shape {tuple(deltas.shape)}."
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}.")
