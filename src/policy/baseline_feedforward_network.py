"""Baseline feedforward policy network for deep hedging.

The network parameterizes the hedge rule ``delta_t = f_theta(I_t, delta_{t-1})``
using shared feedforward layers applied across timesteps.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.state.builder import FEATURE_DIM

# Input dim = features (F=3) + previous delta (1)
INPUT_DIM = FEATURE_DIM + 1   # 4


class BaselineFeedforwardNetwork(nn.Module):
    """Baseline shared feedforward network for hedge ratio prediction.

    The architecture uses explicit recurrence by concatenating the previous
    hedge ``delta_{t-1}`` to current state features.

    Parameters
    ----------
    hidden : int
        Number of neurons in each hidden layer. Default: 64.
    """

    def __init__(self, hidden: int = 64) -> None:
        """Initialize the baseline hedge policy network.

        Parameters
        ----------
        hidden : int, default=64
            Width of both hidden layers.
        """
        super().__init__()
        self.hidden = hidden

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(N, FEATURE_DIM + 1)``.

        Returns
        -------
        torch.Tensor
            Predicted hedge ratio with shape ``(N,)``.
        """
        return self.net(x).squeeze(-1)   # (N, 1) → (N,)

    def forward_trajectory(self, features: torch.Tensor) -> torch.Tensor:
        """Full trajectory forward pass across all T timesteps.

        Unrolls the network T times, feeding δ_{t-1} back at each step.
        Gradient flows back through all T unrolled steps via autograd.

        Parameters
        ----------
        features : torch.Tensor
            Feature tensor with shape ``(N, T, F)`` where ``N`` is the number
            of paths, ``T`` is the number of hedging timesteps, and ``F`` must
            equal ``FEATURE_DIM``.

        Returns
        -------
        torch.Tensor
            Hedge-ratio tensor with shape ``(N, T)``.

        Raises
        ------
        ValueError
            If the feature tensor rank, feature dimension, or values are
            invalid.
        """
        _validate_features(features)

        N, T, F = features.shape
        device  = features.device
        dtype   = features.dtype

        # delta_{-1}=0 encodes zero initial inventory before the first trade.
        delta_prev = torch.zeros(N, dtype=dtype, device=device)

        deltas = []
        for t in range(T):
            x_t = torch.cat([features[:, t, :], delta_prev.unsqueeze(-1)], dim=-1)
            delta_t = self.forward(x_t)
            deltas.append(delta_t)
            delta_prev = delta_t

        return torch.stack(deltas, dim=1)

    def n_parameters(self) -> int:
        """Return the number of trainable parameters.

        Returns
        -------
        int
            Count of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_features(features: torch.Tensor) -> None:
    """Validate trajectory feature tensor shape and values.

    Parameters
    ----------
    features : torch.Tensor
        Feature tensor expected to have shape ``(N, T, FEATURE_DIM)``.

    Raises
    ------
    ValueError
        If rank or feature dimension is invalid, or if NaNs are present.
    """
    if features.ndim != 3:
        raise ValueError(
            f"features must be 3-D (N, T, F), got shape {tuple(features.shape)}."
        )
    if features.shape[2] != FEATURE_DIM:
        raise ValueError(
            f"features last dim must be FEATURE_DIM={FEATURE_DIM}, "
            f"got {features.shape[2]}."
        )
    if torch.isnan(features).any():
        raise ValueError("features contains NaN values.")
