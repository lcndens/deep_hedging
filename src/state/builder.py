"""State construction for deep hedging pipeline — Stage 2.

Transforms raw DatasetBatch tensors into the feature tensor I_t that the
policy network observes at each rebalancing timestep.

This is the only module where financial feature engineering occurs. Adding,
removing, or changing features happens here and only here.

Feature vector (F=3, fixed across all simulators):
    [0]  log(S_t / K)       log-moneyness
    [1]  T_mat - t_years_t  time to maturity τ
    [2]  v_t                variance slot (real for Heston; 0.0 for BS/NGA)

The output shape is (N, T, 3) — NOT (N, T+1, 3). The maturity step t=T is
excluded because the agent takes no hedging action at maturity.

Usage
-----
    from src.state.builder import build_features

    features = build_features(batch)   # (N, T, 3), float32, CPU
"""

from __future__ import annotations

import torch

from src.io.dataset_loader import DatasetBatch

# Feature dimension — fixed for all simulators and all Aim 2/3 experiments.
# The policy network input dim is FEATURE_DIM + 1 (for δ_{t-1}).
FEATURE_DIM = 3


def build_features(batch: DatasetBatch) -> torch.Tensor:
    """Construct the information process I_t for all paths and timesteps.

    Under the Markov assumption (Buehler et al. 2019, Remark 4.6), the
    optimal hedge at time t depends only on the current state. The three
    features below — moneyness, time to maturity, and variance — fully
    characterise the state for BS, Heston, and NGA.

    The previous hedge ratio δ_{t-1} is NOT included here. It is concatenated
    by the policy network at Stage 5, keeping state construction decoupled
    from network architecture.

    Parameters
    ----------
    batch : DatasetBatch
        Output of ``load_dataset()``. All tensors must be CPU float32.

    Returns
    -------
    features : torch.Tensor
        Shape ``(N, T, 3)``, dtype float32, on CPU.

        features[:, t, 0]  log(S_t / K)       log-moneyness
        features[:, t, 1]  T_mat - t_years_t  time to maturity τ
        features[:, t, 2]  v_t                variance (0.0 for BS/NGA)

    Raises
    ------
    ValueError
        If batch tensors have unexpected shapes or dtypes.
    """
    _validate_inputs(batch)

    # Slice off the maturity step — shape goes from (N, T+1) → (N, T)
    # Index :-1 keeps t=0 through t=T-1 (the T rebalancing steps).
    S_t = batch.paths_S[:, :-1]   # (N, T)
    v_t = batch.paths_v[:, :-1]   # (N, T)
    t_t = batch.paths_t[:, :-1]   # (N, T)

    # Feature 0: log-moneyness  log(S_t / K)
    # Positive → in-the-money (call), negative → out-of-the-money.
    log_moneyness = torch.log(S_t / batch.K)          # (N, T)

    # Feature 1: time to maturity  τ_t = T_mat - t
    # Starts near T_mat at t=0, approaches 0 as t→T.
    tau = batch.T_mat - t_t                            # (N, T)

    # Feature 2: variance slot
    # Real variance process v_t for Heston; 0.0 everywhere for BS/NGA
    # (already zero in batch.paths_v — no branching needed here).
    v_slot = v_t                                       # (N, T)

    # Stack along new last dimension → (N, T, 3)
    features = torch.stack([log_moneyness, tau, v_slot], dim=-1)

    _validate_outputs(features, batch)
    return features


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(batch: DatasetBatch) -> None:
    """Fast pre-flight checks on the DatasetBatch before any computation."""
    expected_S_shape = (batch.n_paths, batch.n_steps + 1)

    if batch.paths_S.shape != expected_S_shape:
        raise ValueError(
            f"batch.paths_S has shape {tuple(batch.paths_S.shape)}, "
            f"expected {expected_S_shape}."
        )
    if batch.paths_S.dtype != torch.float32:
        raise ValueError(
            f"batch.paths_S dtype is {batch.paths_S.dtype}, expected torch.float32."
        )
    if batch.paths_v.shape != batch.paths_S.shape:
        raise ValueError(
            f"batch.paths_v shape {tuple(batch.paths_v.shape)} does not match "
            f"batch.paths_S shape {tuple(batch.paths_S.shape)}."
        )
    if batch.paths_t.shape != batch.paths_S.shape:
        raise ValueError(
            f"batch.paths_t shape {tuple(batch.paths_t.shape)} does not match "
            f"batch.paths_S shape {tuple(batch.paths_S.shape)}."
        )
    if batch.K <= 0:
        raise ValueError(f"batch.K must be positive, got {batch.K}.")
    if batch.T_mat <= 0:
        raise ValueError(f"batch.T_mat must be positive, got {batch.T_mat}.")


def _validate_outputs(features: torch.Tensor, batch: DatasetBatch) -> None:
    """Post-computation shape and NaN checks."""
    expected = (batch.n_paths, batch.n_steps, FEATURE_DIM)
    if features.shape != expected:
        raise ValueError(
            f"features has shape {tuple(features.shape)}, expected {expected}."
        )
    if torch.isnan(features).any():
        n_nan = int(torch.isnan(features).sum())
        raise ValueError(
            f"{n_nan} NaN value(s) in features tensor. "
            "Check for zero or negative spot prices in paths_S."
        )
