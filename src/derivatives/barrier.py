"""Barrier option payoff functions.

Implements knock-out barrier options where the option is cancelled (payoff = 0)
if the underlying spot price breaches the barrier at any observed timestep.

The knock-out check is applied to ALL T+1 columns of the path tensor, i.e. it
includes both intermediate observations and the terminal price.
"""

from __future__ import annotations

import torch


def compute_barrier_payoff(
    paths_S:  torch.Tensor,
    K:        float,
    B:        float,
    epsilon:  float = 0.0,
) -> torch.Tensor:
    """Up-and-out call payoff: (S_T - K)+ if max_t(S_t) < B, else 0.

    Parameters
    ----------
    paths_S : torch.Tensor
        Full spot price paths with shape ``(N, T+1)``, including all
        intermediate and terminal prices.
    K : float
        Strike price. Must be positive.
    B : float
        Barrier level. Must satisfy ``B > K``.
    epsilon : float
        Reserved for future soft-barrier extensions. Currently unused;
        the barrier check is always the exact condition ``max_t(S_t) >= B``.

    Returns
    -------
    torch.Tensor
        Payoff tensor of shape ``(N,)``. Each entry is ``max(S_T - K, 0)``
        for paths that never touched or crossed the barrier, and ``0.0`` for
        paths that did.

    Raises
    ------
    ValueError
        If ``paths_S`` is not 2-D, ``K <= 0``, or ``B <= K``.
    """
    _validate_inputs(paths_S, K, B)

    # Knock-out: any timestep (including t=0 and t=T) at or above barrier
    knocked_out = paths_S.max(dim=1).values >= B   # (N,) bool

    S_T    = paths_S[:, -1]                         # (N,)
    payoff = torch.clamp(S_T - K, min=0.0)          # (N,)
    payoff = payoff.clone()
    payoff[knocked_out] = 0.0

    return payoff


class UpAndOutBarrierPayoff:
    """Callable wrapper for the up-and-out call barrier payoff.

    Stores ``K`` and ``B`` so the payoff can be called with just the
    path tensor — matching the interface expected by the trainer's
    ``payoff_fn`` protocol.

    Parameters
    ----------
    K : float
        Strike price. Must be positive.
    B : float
        Barrier level. Must satisfy ``B > K``.
    """

    def __init__(self, K: float, B: float) -> None:
        if B <= K:
            raise ValueError(
                f"Barrier B={B} must be strictly greater than strike K={K}."
            )
        if K <= 0:
            raise ValueError(f"Strike K must be positive, got {K}.")
        self.K = K
        self.B = B

    def __call__(self, paths_S: torch.Tensor) -> torch.Tensor:
        """Compute payoff for a batch of paths.

        Parameters
        ----------
        paths_S : torch.Tensor
            Shape ``(N, T+1)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)``.
        """
        return compute_barrier_payoff(paths_S, self.K, self.B)

    def __repr__(self) -> str:
        return f"UpAndOutBarrierPayoff(K={self.K}, B={self.B})"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(paths_S: torch.Tensor, K: float, B: float) -> None:
    if paths_S.ndim != 2:
        raise ValueError(
            f"paths_S must be a 2-D tensor of shape (N, T+1), "
            f"got shape {tuple(paths_S.shape)}."
        )
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}.")
    if B <= K:
        raise ValueError(
            f"Barrier B={B} must be strictly greater than strike K={K}."
        )
