"""Feature construction for the deep-hedging information process.

This module maps dataset tensors to the state tensor ``I_t`` used by the
policy network.

Base features (all payoffs):
    ``[log(S_t/K), tau_t, v_t]`` — shape ``(N, T, 3)``

Two-instrument mode adds ``log(S2_t/S2_0)`` after the base features.

Payoff-specific extra features (appended last):
    european / binary : none            → total 3 or 4
    asian             : log(mean_t / K) → total 4 or 5
    lookback          : log(S_t / min_t)→ total 4 or 5
    barrier           : knocked_out, log(S_t / B) → total 5 or 6
"""

from __future__ import annotations

import torch

from src.io.dataset_loader import DatasetBatch

# Base feature dimension for a single hedging instrument.
# Two-instrument mode adds one additional feature (log-VS-return), giving 4.
FEATURE_DIM = 3

# Extra features contributed by each payoff type.
_PAYOFF_EXTRA: dict[str, int] = {
    "european": 0,
    "binary":   0,
    "asian":    1,
    "lookback": 1,
    "barrier":  2,
}

_VALID_PAYOFFS = frozenset(_PAYOFF_EXTRA)


def _get_feature_dim(n_instruments: int, payoff: str) -> int:
    """Return the total feature dimension for the given configuration.

    Parameters
    ----------
    n_instruments : int
        Number of hedging instruments (1 or 2).
    payoff : str
        Payoff type name.

    Returns
    -------
    int
        Expected last dimension of the features tensor.
    """
    return FEATURE_DIM + (n_instruments - 1) + _PAYOFF_EXTRA[payoff]


def build_features(
    batch:         DatasetBatch,
    n_instruments: int = 1,
    payoff:        str = "european",
    barrier_level: float | None = None,
) -> torch.Tensor:
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
    payoff : str, optional
        Payoff type.  One of ``"european"``, ``"binary"``, ``"asian"``,
        ``"lookback"``, or ``"barrier"``.  Default: ``"european"``.
        Payoff-specific features are appended after the base (and optional
        variance-swap) features:

        - ``"asian"``    : ``log(running_mean_t / K)``   — 1 extra feature
        - ``"lookback"`` : ``log(S_t / running_min_t)``  — 1 extra feature
        - ``"barrier"``  : ``knocked_out_t``, ``log(S_t / B)`` — 2 extra features
        - ``"european"`` / ``"binary"`` : no extra features
    barrier_level : float or None, optional
        Barrier level ``B`` required when ``payoff="barrier"``.  Raises
        ``ValueError`` if payoff is ``"barrier"`` and this is ``None``.

    Returns
    -------
    features : torch.Tensor
        Shape ``(N, T, F)`` where ``F = _get_feature_dim(n_instruments, payoff)``,
        dtype float32, on CPU.

    Raises
    ------
    ValueError
        If ``n_instruments`` is not 1 or 2, ``payoff`` is unknown,
        ``payoff="barrier"`` without ``barrier_level``, batch tensors have
        unexpected shapes/dtypes, or ``n_instruments=2`` is requested on a
        non-Heston dataset.
    """
    if n_instruments not in (1, 2):
        raise ValueError(
            f"n_instruments must be 1 or 2, got {n_instruments}."
        )
    if payoff not in _VALID_PAYOFFS:
        raise ValueError(
            f"Unknown payoff '{payoff}'. Must be one of {sorted(_VALID_PAYOFFS)}."
        )
    if payoff == "barrier" and barrier_level is None:
        raise ValueError(
            "payoff='barrier' requires barrier_level to be specified."
        )

    _validate_inputs(batch, n_instruments)

    # Exclude maturity because no hedge is placed at t = T.
    S_t = batch.paths_S[:, :-1]   # (N, T)
    v_t = batch.paths_v[:, :-1]   # (N, T)
    t_t = batch.paths_t[:, :-1]   # (N, T)

    log_moneyness = torch.log(S_t / batch.K)
    tau           = batch.T_mat - t_t
    v_slot        = v_t

    # --- Base feature stack ---
    if n_instruments == 1:
        feature_list = [log_moneyness, tau, v_slot]
    else:
        S2_t  = batch.paths_S2[:, :-1]              # (N, T)
        S2_0  = batch.paths_S2[:, :1]               # (N, 1)
        log_vs_return = torch.log(S2_t / S2_0)      # (N, T)
        feature_list = [log_moneyness, tau, v_slot, log_vs_return]

    # --- Payoff-specific extra features ---
    if payoff == "asian":
        mean_t = batch.paths_running_mean[:, :-1]   # (N, T)
        feature_list.append(torch.log(mean_t / batch.K))

    elif payoff == "lookback":
        min_t = batch.paths_running_min[:, :-1]     # (N, T)
        feature_list.append(torch.log(S_t / min_t))

    elif payoff == "barrier":
        # running_max_t[i, t] = max(S[i, 0], ..., S[i, t])
        # Computed from batch.paths_S directly: barrier_level is a training-time
        # parameter unavailable at simulation time, so this cannot be precomputed.
        running_max_t = batch.paths_S[:, :-1].cummax(dim=1).values  # (N, T)
        knocked_out   = (running_max_t >= barrier_level).float()    # (N, T)
        log_dist      = torch.log(S_t / barrier_level)              # (N, T)
        feature_list.extend([knocked_out, log_dist])

    features = torch.stack(feature_list, dim=-1)

    _validate_outputs(features, batch, n_instruments, payoff)
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
    features:      torch.Tensor,
    batch:         DatasetBatch,
    n_instruments: int,
    payoff:        str,
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
    payoff : str
        Payoff type used to determine the expected feature count.

    Raises
    ------
    ValueError
        If shape mismatches occur or NaN values are present.
    """
    expected_feat_dim = _get_feature_dim(n_instruments, payoff)
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
