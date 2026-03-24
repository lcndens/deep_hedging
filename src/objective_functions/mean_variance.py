"""Mean-variance objective function for deep hedging.

Implements the mean-variance risk objective from Cao et al. (2021):

    L(θ) = -E[PnL] + λ · Var[PnL]

where:
    E[PnL]   = expected terminal PnL across all paths
    Var[PnL] = variance of terminal PnL across all paths
    λ        = risk-aversion coefficient (default: 1.0)

Minimising this loss simultaneously pushes mean PnL higher (towards zero,
since p0=0 and the hedge is short the option) and penalises dispersion in
outcomes. Higher λ → more variance aversion → tighter PnL distribution at
the cost of lower mean.

Comparison with CVaR
--------------------
CVaR focuses gradient signal exclusively on the worst (1-α) fraction of
paths, ignoring well-hedged outcomes. Mean-variance distributes gradient
across all paths in proportion to their deviation from the mean — it is
sensitive to the full distribution, not just the tail.

This makes mean-variance faster to optimise (no tail-only gradient sparsity)
but less appropriate when tail risk is the primary concern.

Interface compatibility
-----------------------
MeanVarianceLoss exposes the same ``forward(pnl)`` interface as CVaRLoss
and a dummy ``omega`` property (always zero, not a trainable Parameter) so
that the trainer can build its optimizer parameter list identically for both
objectives by checking ``isinstance(obj.omega, nn.Parameter)``.

Usage
-----
    from src.objective_functions.mean_variance import MeanVarianceLoss

    mv        = MeanVarianceLoss(lam=1.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    loss = mv(pnl)      # scalar
    loss.backward()
    optimizer.step()
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MeanVarianceLoss(nn.Module):
    """Mean-variance risk objective.

    Parameters
    ----------
    lam : float
        Risk-aversion coefficient (λ ≥ 0). Controls the trade-off between
        expected PnL and variance. λ=0 reduces to minimising -E[PnL] only
        (risk-neutral). Default: 1.0.
    """

    def __init__(self, lam: float = 1.0) -> None:
        super().__init__()

        if lam < 0.0:
            raise ValueError(f"lam must be >= 0, got {lam}.")

        self.lam = lam

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """Compute mean-variance loss.

        Parameters
        ----------
        pnl : torch.Tensor, shape (N,)
            Terminal PnL per path. Output of ``compute_pnl()``.
            Positive = profit, negative = loss.

        Returns
        -------
        loss : torch.Tensor, scalar
            Differentiable mean-variance loss. Call ``loss.backward()`` to
            compute gradients for network weights.
        """
        _validate_pnl(pnl)

        mean_pnl = pnl.mean()
        var_pnl  = pnl.var(unbiased=False)

        # Loss = -E[PnL] + λ · Var[PnL]
        # Minimising -E[PnL] pushes mean PnL upward (toward zero from below).
        # Minimising λ · Var[PnL] tightens the distribution around the mean.
        loss = -mean_pnl + self.lam * var_pnl
        return loss

    @property
    def omega(self) -> None:
        """Dummy omega for trainer compatibility.

        CVaRLoss has a trainable omega (nn.Parameter) that must be included in
        the optimizer. MeanVarianceLoss has no such parameter. Returning None
        here allows the trainer to check ``isinstance(objective.omega, nn.Parameter)``
        and skip adding it to the optimizer parameter list.
        """
        return None

    def loss_estimate(self, pnl: torch.Tensor) -> float:
        """Return the current loss as a Python float (no grad).

        Useful for logging during training without affecting the computation
        graph.

        Parameters
        ----------
        pnl : torch.Tensor, shape (N,)

        Returns
        -------
        float
        """
        with torch.no_grad():
            return self.forward(pnl).item()

    def extra_repr(self) -> str:
        return f"lam={self.lam}"


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
    if pnl.numel() < 2:
        raise ValueError("pnl must have at least 2 elements to compute variance.")
