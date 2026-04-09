"""Feature construction for the deep-hedging information process.

This module maps dataset tensors to the state tensor ``I_t`` used by the
policy network.

Single-instrument (n_instruments=1):
    features  ``[log(S_t/K), tau_t, v_t]`` — shape ``(N, T, 3)``

Two-instrument (n_instruments=2):
    features  ``[log(S_t/K), tau_t, v_t, log(S2_t/S2_0)]`` — shape ``(N, T, 4)``
    where ``S2_t`` is the variance swap price path and ``S2_0`` is its t=0 value
    per path.
"""

from __future__ import annotations

import torch

from src.io.dataset_loader import DatasetBatch

# Base feature dimension for a single hedging instrument.
# Two-instrument mode adds one additional feature (log-VS-return), giving 4.
FEATURE_DIM = 3


def build_features(batch: DatasetBatch, n_instruments: int = 1) -> torch.Tensor:
    """Construct the information process I_t for all paths and timesteps.

    Parameters
    ----------
    batch : DatasetBatch
        Output of ``load_dataset()``. All tensors must be CPU float32.
    n_instruments : int, optional
        Number of hedging instruments.  Must be 1 or 2.
        ``1`` (default): return the three canonical features ``(N, T, 3)``.
        ``2``: append ``log(S2_t / S2_0)`` as a fourth feature, giving
        ``(N, T, 4)``.  Requires a Heston dataset (``paths_S2`` non-zero).

    Returns
    -------
    features : torch.Tensor
        Shape ``(N, T, 3)`` for ``n_instruments=1`` or ``(N, T, 4)`` for
        ``n_instruments=2``, dtype float32, on CPU.

        features[:, t, 0]  log(S_t / K)           log-moneyness
        features[:, t, 1]  T_mat - t_years_t       time to maturity τ
        features[:, t, 2]  v_t                     variance (0.0 for BS/NGA)
        features[:, t, 3]  log(S2_t / S2_0)        log variance-swap return
                                                    (n_instruments=2 only)

    Raises
    ------
    ValueError
        If ``n_instruments`` is not 1 or 2, batch tensors have unexpected
        shapes or dtypes, or ``n_instruments=2`` is requested on a non-Heston
        dataset (``paths_S2`` all zeros).
    """
    if n_instruments not in (1, 2):
        raise ValueError(
            f"n_instruments must be 1 or 2, got {n_instruments}."
        )

    _validate_inputs(batch, n_instruments)

    # Exclude maturity because no hedge is placed at t = T.
    S_t = batch.paths_S[:, :-1]
    v_t = batch.paths_v[:, :-1]
    t_t = batch.paths_t[:, :-1]

    log_moneyness = torch.log(S_t / batch.K)
    tau           = batch.T_mat - t_t
    v_slot        = v_t

    if n_instruments == 1:
        features = torch.stack([log_moneyness, tau, v_slot], dim=-1)
    else:
        S2_t  = batch.paths_S2[:, :-1]              # (N, T)
        S2_0  = batch.paths_S2[:, :1]               # (N, 1) — t=0 value per path
        log_vs_return = torch.log(S2_t / S2_0)      # (N, T)
        features = torch.stack([log_moneyness, tau, v_slot, log_vs_return], dim=-1)

    _validate_outputs(features, batch, n_instruments)
    return features


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(batch: DatasetBatch, n_instruments: int) -> None:
    """Validate dataset tensors before feature construction.

    Parameters
    ----------
    batch : DatasetBatch
        Input batch expected to contain ``(N, T+1)`` float32 tensors.
    n_instruments : int
        Number of hedging instruments (1 or 2).

    Raises
    ------
    ValueError
        If shapes, dtypes, scalar metadata, or instrument/dataset compatibility
        are invalid.
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

    if n_instruments == 2:
        if batch.paths_S2.shape != batch.paths_S.shape:
            raise ValueError(
                f"batch.paths_S2 shape {tuple(batch.paths_S2.shape)} does not "
                f"match batch.paths_S shape {tuple(batch.paths_S.shape)}."
            )
        if not batch.paths_S2.any():
            raise ValueError(
                "n_instruments=2 requires a Heston dataset with non-zero "
                "paths_S2. For BS and NGA datasets paths_S2 is all zeros — "
                "use n_instruments=1 instead."
            )


def _validate_outputs(
    features: torch.Tensor, batch: DatasetBatch, n_instruments: int
) -> None:
    """Validate feature tensor shape and finite values.

    Parameters
    ----------
    features : torch.Tensor
        Constructed feature tensor.
    batch : DatasetBatch
        Source batch used to infer expected dimensions.
    n_instruments : int
        Number of hedging instruments (1 or 2).

    Raises
    ------
    ValueError
        If shape mismatches occur or NaN values are present.
    """
    expected_feat_dim = FEATURE_DIM + (n_instruments - 1)
    expected = (batch.n_paths, batch.n_steps, expected_feat_dim)
    if features.shape != expected:
        raise ValueError(
            f"features has shape {tuple(features.shape)}, expected {expected}."
        )
    if torch.isnan(features).any():
        n_nan = int(torch.isnan(features).sum())
        raise ValueError(
            f"{n_nan} NaN value(s) in features tensor. "
            "Check for zero or negative spot/variance-swap prices."
        )
