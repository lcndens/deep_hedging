"""CVaR objective function for deep hedging — Stage 4b.

Implements the Conditional Value at Risk (CVaR) loss via the Optimized
Certainty Equivalent (OCE) formulation (He et al. 2025, eq. 2.5):

    L(θ, ω) = ω + mean( max(-PnL - ω, 0) ) / (1 - α)

where:
    α     = confidence level (default 0.95 → penalise worst 5% of paths)
    ω     = trainable scalar that converges to VaR_α at optimum
    -PnL  = loss per path (positive = bad outcome)

ω is an nn.Parameter optimised jointly with network weights in the same
Adam step. At convergence ω ≈ VaR_α(loss distribution).

The clamp focuses gradients only on paths where the hedge failed badly
(loss exceeded ω). Well-hedged paths contribute zero gradient.

Special cases:
    α = 0.0  →  reduces to mean(-PnL)  (risk-neutral)
    α → 1.0  →  approaches worst-case loss

Usage
-----
    from src.objective_functions.cvar import CVaRLoss

    cvar      = CVaRLoss(alpha=0.95)
    optimizer = torch.optim.Adam(
        [*network.parameters(), cvar.omega], lr=1e-3
    )

    loss = cvar(pnl)      # scalar
    loss.backward()
    optimizer.step()
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CVaRLoss(nn.Module):
    """Differentiable CVaR loss via the OCE formulation.

    Parameters
    ----------
    alpha : float
        CVaR confidence level in (0, 1). Default: 0.95.
        Higher alpha → more focus on tail losses.
    """

    def __init__(self, alpha: float = 0.95) -> None:
        super().__init__()

        if not 0.0 <= alpha < 1.0:
            raise ValueError(
                f"alpha must be in [0, 1), got {alpha}."
            )

        self.alpha = alpha
        # Trainable threshold — converges to VaR_α at optimum.
        # Initialised at 0.0; optimised jointly with network weights.
        self.omega = nn.Parameter(torch.tensor(0.0))

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """Compute CVaR loss.

        Parameters
        ----------
        pnl : torch.Tensor, shape (N,)
            Terminal PnL per path. Output of ``compute_pnl()``.
            Positive = profit, negative = loss.

        Returns
        -------
        loss : torch.Tensor, scalar
            Differentiable CVaR estimate. Call ``loss.backward()`` to
            compute gradients for both network weights and ``self.omega``.
        """
        _validate_pnl(pnl)

        # excess_i = max(-PnL_i - ω, 0)
        # Non-zero only for paths where loss exceeded the threshold ω.
        excess = torch.clamp(-pnl - self.omega, min=0.0)

        # OCE form: ω + E[excess] / (1 - α)
        loss = self.omega + excess.mean() / (1.0 - self.alpha)
        return loss

    def cvar_estimate(self, pnl: torch.Tensor) -> float:
        """Return the current CVaR estimate as a Python float (no grad).

        Useful for logging during training without affecting the computation
        graph.

        Parameters
        ----------
        pnl : torch.Tensor, shape (N,)

        Returns
        -------
        float
            CVaR estimate at level alpha.
        """
        with torch.no_grad():
            return self.forward(pnl).item()

    def var_estimate(self) -> float:
        """Return the current VaR_α estimate (value of ω) as a Python float.

        At convergence, ω ≈ VaR_α of the loss distribution.

        Returns
        -------
        float
        """
        return self.omega.item()

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_pnl(pnl: torch.Tensor) -> None:
    if pnl.ndim != 1:
        raise ValueError(
            f"pnl must be a 1-D tensor of shape (N,), got shape {tuple(pnl.shape)}."
        )
    if pnl.numel() == 0:
        raise ValueError("pnl tensor is empty.")
    if torch.isnan(pnl).any():
        raise ValueError("pnl contains NaN values.")
