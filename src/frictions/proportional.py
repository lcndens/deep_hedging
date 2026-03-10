"""Proportional transaction costs — Stage 3 (Frictions).

Computes the total transaction cost incurred along a hedging trajectory.
Isolated so cost structures can be swapped cleanly per experiment.

Cost formula (Buehler et al. 2019, eq. 2.2):
    cost_t = epsilon * S_t * |delta_t - delta_{t-1}|
    total_cost = sum over t=0..T-1

Convention:
    delta_{-1} = 0  (agent starts with no position before t=0)

With epsilon=0.0 this returns a zero tensor — the frictionless baseline
used for all Aim 2 experiments.

Usage
-----
    from src.frictions.proportional import proportional_cost

    total_cost = proportional_cost(
        S_obs=batch.paths_S[:, :-1],   # (N, T)
        deltas=deltas,                  # (N, T)
        epsilon=0.0,
    )   # → (N,)
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
    S_obs : torch.Tensor, shape (N, T)
        Spot prices at rebalancing steps t=0..T-1.
        Typically ``batch.paths_S[:, :-1]``.
    deltas : torch.Tensor, shape (N, T)
        Hedge ratios output by the policy network.
    epsilon : float
        Proportional cost rate. 0.0 → frictionless (returns zero tensor).
        Typical values: 0.001 (10 bps), 0.01 (100 bps).

    Returns
    -------
    total_cost : torch.Tensor, shape (N,), same dtype and device as S_obs.
        Total transaction cost per path, summed over all T rebalancing steps.

    Raises
    ------
    ValueError
        If S_obs and deltas shapes do not match or are not 2-D.
    """
    _validate_inputs(S_obs, deltas, epsilon)

    if epsilon == 0.0:
        return torch.zeros(S_obs.shape[0], dtype=S_obs.dtype, device=S_obs.device)

    # delta_{-1} = 0: prepend a column of zeros so delta_prev[:,0] = 0
    # delta_prev[:,t] = delta_{t-1} for t=0..T-1
    N = S_obs.shape[0]
    zeros = torch.zeros(N, 1, dtype=deltas.dtype, device=deltas.device)
    delta_prev = torch.cat([zeros, deltas[:, :-1]], dim=1)   # (N, T)

    # cost_t = epsilon * S_t * |delta_t - delta_{t-1}|
    cost_per_step = epsilon * S_obs * torch.abs(deltas - delta_prev)  # (N, T)
    total_cost    = cost_per_step.sum(dim=1)                           # (N,)

    return total_cost


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    S_obs: torch.Tensor,
    deltas: torch.Tensor,
    epsilon: float,
) -> None:
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
