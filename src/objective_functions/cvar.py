"""CVaR loss via optimized certainty equivalent for deep hedging.

The loss follows He et al. (2025, eq. 2.5):
``L(theta, omega) = omega + E[max(-PnL - omega, 0)] / (1 - alpha)``,
where ``omega`` is a trainable scalar converging to ``VaR_alpha``.
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
        """Initialize the CVaR objective module.

        Parameters
        ----------
        alpha : float, default=0.95
            CVaR confidence level in ``[0, 1)``.

        Raises
        ------
        ValueError
            If ``alpha`` is outside ``[0, 1)``.
        """
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
        pnl : torch.Tensor
            Terminal PnL with shape ``(N,)``, where ``N`` is the number of
            simulated paths.

        Returns
        -------
        torch.Tensor
            Scalar differentiable CVaR loss.

        Raises
        ------
        ValueError
            If ``pnl`` is not a valid one-dimensional finite tensor.
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
        pnl : torch.Tensor
            Terminal PnL vector with shape ``(N,)``.

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
            Current value of the trainable scalar ``omega``.
        """
        return self.omega.item()

    def extra_repr(self) -> str:
        """Return module representation metadata.

        Returns
        -------
        str
            Text representation containing the configured ``alpha`` value.
        """
        return f"alpha={self.alpha}"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_pnl(pnl: torch.Tensor) -> None:
    """Validate PnL tensor before CVaR computation.

    Parameters
    ----------
    pnl : torch.Tensor
        PnL tensor expected to have shape ``(N,)`` and finite values.

    Raises
    ------
    ValueError
        If rank is wrong, tensor is empty, or values contain NaNs.
    """
    if pnl.ndim != 1:
        raise ValueError(
            f"pnl must be a 1-D tensor of shape (N,), got shape {tuple(pnl.shape)}."
        )
    if pnl.numel() == 0:
        raise ValueError("pnl tensor is empty.")
    if torch.isnan(pnl).any():
        raise ValueError("pnl contains NaN values.")
