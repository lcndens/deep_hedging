"""Baseline feedforward policy network for deep hedging.

The network parameterizes the hedge rule ``delta_t = f_theta(I_t, delta_{t-1})``
using shared feedforward layers applied across timesteps.

Single-instrument (n_instruments=1):
    input_dim  = FEATURE_DIM + 1 = 4   (3 features + 1 prev delta)
    output_dim = 1
    forward_trajectory returns (N, T)

Two-instrument (n_instruments=2):
    input_dim  = (FEATURE_DIM + 1) + 2 = 6   (4 features + 2 prev deltas)
    output_dim = 2
    forward_trajectory returns (N, T, 2)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.state.builder import FEATURE_DIM

# Module-level constant kept for backward compatibility: single-instrument
# input dim = FEATURE_DIM + 1 prev-delta = 4.
INPUT_DIM = FEATURE_DIM + 1   # 4


class BaselineFeedforwardNetwork(nn.Module):
    """Baseline shared feedforward network for hedge ratio prediction.

    The architecture uses explicit recurrence by concatenating the previous
    hedge ``delta_{t-1}`` to current state features.

    Parameters
    ----------
    hidden : int
        Number of neurons in each hidden layer. Default: 64.
    n_instruments : int
        Number of hedging instruments.  Must be 1 or 2.
        ``1`` (default): scalar delta per step, trajectory shape ``(N, T)``.
        ``2``: two deltas per step, trajectory shape ``(N, T, 2)``.
    """

    def __init__(self, hidden: int = 64, n_instruments: int = 1) -> None:
        """Initialise the baseline hedge policy network.

        Parameters
        ----------
        hidden : int, default=64
            Width of both hidden layers.
        n_instruments : int, default=1
            Number of hedging instruments (1 or 2).
        """
        if n_instruments not in (1, 2):
            raise ValueError(
                f"n_instruments must be 1 or 2, got {n_instruments}."
            )
        super().__init__()
        self.hidden       = hidden
        self.n_instruments = n_instruments

        # Number of state features fed in by build_features.
        n_features = FEATURE_DIM + (n_instruments - 1)   # 3 or 4

        # input  = state features + one prev-delta per instrument
        # output = one delta per instrument
        self.input_dim  = n_features + n_instruments      # 4 or 6
        self.output_dim = n_instruments                   # 1 or 2

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(N, input_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted hedge ratio(s).
            Shape ``(N,)`` for n_instruments=1; ``(N, 2)`` for n_instruments=2.
        """
        out = self.net(x)               # (N, output_dim)
        return out.squeeze(-1)          # (N,) when output_dim=1; no-op for 2

    def forward_trajectory(self, features: torch.Tensor) -> torch.Tensor:
        """Full trajectory forward pass across all T timesteps.

        Unrolls the network T times, feeding ``delta_{t-1}`` back at each step.
        Gradient flows back through all T unrolled steps via autograd.

        Parameters
        ----------
        features : torch.Tensor
            Feature tensor with shape ``(N, T, F)`` where ``F`` must equal
            ``FEATURE_DIM + (n_instruments - 1)``.

        Returns
        -------
        torch.Tensor
            Hedge-ratio tensor.
            Shape ``(N, T)`` for n_instruments=1;
            ``(N, T, 2)`` for n_instruments=2.

        Raises
        ------
        ValueError
            If the feature tensor rank, feature dimension, or values are
            invalid.
        """
        expected_feat_dim = FEATURE_DIM + (self.n_instruments - 1)
        _validate_features(features, expected_feat_dim)

        N, T, F = features.shape
        device  = features.device
        dtype   = features.dtype

        if self.n_instruments == 1:
            # delta_{-1} = 0 scalar per path.
            delta_prev = torch.zeros(N, dtype=dtype, device=device)   # (N,)

            deltas = []
            for t in range(T):
                x_t     = torch.cat([features[:, t, :], delta_prev.unsqueeze(-1)], dim=-1)
                delta_t = self.forward(x_t)           # (N,)
                deltas.append(delta_t)
                delta_prev = delta_t

            return torch.stack(deltas, dim=1)         # (N, T)

        else:
            # delta_{-1} = zeros for both instruments.
            delta_prev = torch.zeros(N, 2, dtype=dtype, device=device)   # (N, 2)

            deltas = []
            for t in range(T):
                x_t     = torch.cat([features[:, t, :], delta_prev], dim=-1)
                delta_t = self.forward(x_t)           # (N, 2)
                deltas.append(delta_t)
                delta_prev = delta_t

            return torch.stack(deltas, dim=1)         # (N, T, 2)

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

def _validate_features(features: torch.Tensor, expected_feat_dim: int) -> None:
    """Validate trajectory feature tensor shape and values.

    Parameters
    ----------
    features : torch.Tensor
        Feature tensor expected to have shape ``(N, T, expected_feat_dim)``.
    expected_feat_dim : int
        Expected size of the feature (last) dimension.

    Raises
    ------
    ValueError
        If rank or feature dimension is invalid, or if NaNs are present.
    """
    if features.ndim != 3:
        raise ValueError(
            f"features must be 3-D (N, T, F), got shape {tuple(features.shape)}."
        )
    if features.shape[2] != expected_feat_dim:
        raise ValueError(
            f"features last dim must be {expected_feat_dim}, "
            f"got {features.shape[2]}."
        )
    if torch.isnan(features).any():
        raise ValueError("features contains NaN values.")
