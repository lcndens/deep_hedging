"""Feature construction for the deep-hedging information process.

This module maps dataset tensors to the state tensor ``I_t`` used by the
policy network, with features ``[log(S_t/K), tau_t, v_t]`` over hedging times
``t = 0, ..., T-1``.
"""

from __future__ import annotations

import torch

from src.io.dataset_loader import DatasetBatch

# The policy input is FEATURE_DIM plus the previous hedge ratio delta_{t-1}.
FEATURE_DIM = 3


def build_features(batch: DatasetBatch) -> torch.Tensor:
    """Construct the information process I_t for all paths and timesteps.

    Under the Markov assumption (Buehler et al. 2019, Remark 4.6), the
    optimal hedge at time t depends only on the current state. The three
    features below — moneyness, time to maturity, and variance — fully
    characterise the state for BS, Heston, and NGA.

    The previous hedge ratio ``delta_{t-1}`` is intentionally excluded and
    concatenated later by the policy network.

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

    # Exclude maturity because no hedge is placed at t = T.
    S_t = batch.paths_S[:, :-1]
    v_t = batch.paths_v[:, :-1]
    t_t = batch.paths_t[:, :-1]

    log_moneyness = torch.log(S_t / batch.K)

    tau = batch.T_mat - t_t

    v_slot = v_t

    features = torch.stack([log_moneyness, tau, v_slot], dim=-1)

    _validate_outputs(features, batch)
    return features


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(batch: DatasetBatch) -> None:
    """Validate dataset tensors before feature construction.

    Parameters
    ----------
    batch : DatasetBatch
        Input batch expected to contain ``(N, T+1)`` float32 tensors.

    Raises
    ------
    ValueError
        If shapes, dtypes, or scalar metadata are invalid.
    """
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
    """Validate feature tensor shape and finite values.

    Parameters
    ----------
    features : torch.Tensor
        Constructed feature tensor with expected shape ``(N, T, 3)``.
    batch : DatasetBatch
        Source batch used to infer expected dimensions.

    Raises
    ------
    ValueError
        If shape mismatches occur or NaN values are present.
    """
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
