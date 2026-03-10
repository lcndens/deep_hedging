"""Baseline feedforward policy network for deep hedging — Stage 4a.

Approximates the optimal hedging strategy:
    δ_t = f_θ(I_t, δ_{t-1})

A single shared feedforward network applied at every rebalancing timestep.
Recurrence is achieved by feeding the previous hedge ratio δ_{t-1} back as
an input — not through hidden state. This is valid under the Markov assumption
for European options on Markov price processes (Buehler et al. 2019, Remark 4.6).

This is the primary baseline for all Aim 2 and Aim 3 experiments. All future
network variants (NTB, LSTM, etc.) are modifications of this architecture.

Architecture:
    Input  (N, F+1) = (N, 4)    [log-moneyness, τ, v, δ_{t-1}]
    Linear(4 → H) + ReLU
    Linear(H → H) + ReLU
    Linear(H → 1)               [no final activation — unconstrained output]
    Output (N,)

Default H=64 gives ~8,500 trainable parameters.

Usage
-----
    from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork

    net    = BaselineFeedforwardNetwork(hidden=64)
    deltas = net.forward_trajectory(features)   # (N, T)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.state.builder import FEATURE_DIM

# Input dim = features (F=3) + previous delta (1)
INPUT_DIM = FEATURE_DIM + 1   # 4


class BaselineFeedforwardNetwork(nn.Module):
    """Baseline shared feedforward network for hedge ratio prediction.

    The primary baseline for all deep hedging experiments. All Aim 3 network
    variants are modifications of this architecture.

    Parameters
    ----------
    hidden : int
        Number of neurons in each hidden layer. Default: 64.
    """

    def __init__(self, hidden: int = 64) -> None:
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
        x : torch.Tensor, shape (N, 4)
            Concatenation of [features_t, delta_{t-1}] for one timestep.

        Returns
        -------
        delta : torch.Tensor, shape (N,)
            Hedge ratio for this timestep.
        """
        return self.net(x).squeeze(-1)   # (N, 1) → (N,)

    def forward_trajectory(self, features: torch.Tensor) -> torch.Tensor:
        """Full trajectory forward pass across all T timesteps.

        Unrolls the network T times, feeding δ_{t-1} back at each step.
        Gradient flows back through all T unrolled steps via autograd.

        Parameters
        ----------
        features : torch.Tensor, shape (N, T, F)
            Feature tensor from the state builder. F must equal FEATURE_DIM=3.

        Returns
        -------
        deltas : torch.Tensor, shape (N, T)
            Hedge ratio at each of the T rebalancing steps.
        """
        _validate_features(features)

        N, T, F = features.shape
        device  = features.device
        dtype   = features.dtype

        # δ_{-1} = 0: agent starts with no position
        delta_prev = torch.zeros(N, dtype=dtype, device=device)

        deltas = []
        for t in range(T):
            # Concatenate current features with previous delta → (N, 4)
            x_t     = torch.cat([features[:, t, :], delta_prev.unsqueeze(-1)], dim=-1)
            delta_t = self.forward(x_t)      # (N,)
            deltas.append(delta_t)
            delta_prev = delta_t

        return torch.stack(deltas, dim=1)    # (N, T)

    def n_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_features(features: torch.Tensor) -> None:
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
