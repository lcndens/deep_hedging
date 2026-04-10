"""Dataset loading utilities for pre-generated deep-hedging runs.

The loader reads canonical v1.0 parquet files, validates schema consistency,
and returns dense CPU tensors required by feature construction and training.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from src.schema.v1_0 import (
    OBS_SCHEMA,
    CONTRACTS_SCHEMA,
    LATENT_STATE_SCHEMA,
    VARIANCE_SWAP_SCHEMA,
    PATH_STATISTICS_SCHEMA,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetBatch:
    """Container for one dataset split in tensor form.

    Parameters
    ----------
    paths_S : torch.Tensor
        Spot tensor with shape ``(N, T+1)`` and dtype ``float32``.
    paths_v : torch.Tensor
        Variance tensor with shape ``(N, T+1)`` and dtype ``float32``.
        For BS and NGA, this is identically zero.
    paths_t : torch.Tensor
        Time-grid tensor with shape ``(N, T+1)`` and dtype ``float32``.
    paths_S2 : torch.Tensor
        Variance swap price path with shape ``(N, T+1)`` and dtype ``float32``.
        Non-zero only for Heston datasets; zero elsewhere.
    paths_running_mean : torch.Tensor
        Running mean of spot with shape ``(N, T+1)`` and dtype ``float32``.
        ``paths_running_mean[:, t] = mean(S[:, 0:t+1])``. Zero when
        ``path_statistics/`` directory is absent.
    paths_running_min : torch.Tensor
        Running minimum of spot with shape ``(N, T+1)`` and dtype ``float32``.
        ``paths_running_min[:, t] = min(S[:, 0:t+1])``. Zero when
        ``path_statistics/`` directory is absent.
    K : float
        Option strike price.
    T_mat : float
        Option maturity in years.
    n_paths : int
        Number of paths ``N`` in the split.
    n_steps : int
        Number of hedging timesteps ``T``.
    metadata : dict
        Parsed ``metadata.json`` content for logging and validation.
    """

    paths_S:            torch.Tensor
    paths_v:            torch.Tensor
    paths_t:            torch.Tensor
    paths_S2:           torch.Tensor
    paths_running_mean: torch.Tensor
    paths_running_min:  torch.Tensor
    K:                  float
    T_mat:              float
    n_paths:            int
    n_steps:            int
    metadata:           dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(run_dir: Path | str, split: str) -> DatasetBatch:
    """Load one split of a pre-generated dataset into CPU tensors.

    Parameters
    ----------
    run_dir : Path | str
        Path to the dataset run folder, e.g.
        ``data/datasets/v1.0/heston/<run_id>/``
    split : str
        One of ``'train'``, ``'val'``, or ``'test'``.

    Returns
    -------
    DatasetBatch
        All tensors are on CPU, dtype float32. Tensors share memory with the
        underlying numpy arrays (zero-copy via ``torch.from_numpy``).

    Raises
    ------
    ValueError
        If ``split`` is invalid, columns are missing/extra, or row count is
        inconsistent with the metadata.
    FileNotFoundError
        If ``run_dir`` or the observations folder does not exist.
    """
    run_dir = Path(run_dir)
    _validate_split(split)
    _require_dir(run_dir)

    # Load metadata and contract scalars first to recover T and K.
    metadata = _load_metadata(run_dir)
    n_steps  = metadata["time_grid"]["n_steps"]
    T1       = n_steps + 1
    K, T_mat = _load_contract_scalars(run_dir)

    # Read long-format observations for the requested split.
    obs_path = run_dir / "observations" / split / "part-00000.parquet"
    _require_file(obs_path)
    obs = _read_parquet(obs_path, OBS_SCHEMA)

    # Latent state is optional; fallback returns zeros when files are absent.
    lat_path = run_dir / "latent_state" / split / "part-00000.parquet"
    lat = _load_latent_state(lat_path, obs)

    # Variance swap is only written for Heston; fallback returns zeros.
    vs_path = run_dir / "variance_swap" / split / "part-00000.parquet"
    vs = _load_variance_swap(vs_path, obs)

    # Path statistics written for all simulators; fallback returns zeros.
    ps_path = run_dir / "path_statistics" / split / "part-00000.parquet"
    ps = _load_path_statistics(ps_path, obs)

    # Join all columns onto observations by path/time index.
    merged = obs.merge(
        lat[["path_id", "t_idx", "v"]],
        on=["path_id", "t_idx"],
        how="left",
    ).merge(
        vs[["path_id", "t_idx", "S2"]],
        on=["path_id", "t_idx"],
        how="left",
    ).merge(
        ps[["path_id", "t_idx", "running_mean", "running_min"]],
        on=["path_id", "t_idx"],
        how="left",
    )

    # Sorting ensures each path is contiguous before reshape to (N, T+1).
    merged = merged.sort_values(["path_id", "t_idx"]).reset_index(drop=True)

    # Every path must contribute exactly T+1 rows.
    n_rows = len(merged)
    if n_rows % T1 != 0:
        raise ValueError(
            f"Row count {n_rows} is not divisible by T+1={T1}. "
            f"The dataset in '{run_dir}' may be corrupted or truncated."
        )
    N = n_rows // T1

    # Pivot long tables to dense path matrices.
    S_arr    = merged["S"].to_numpy(dtype=np.float32).reshape(N, T1).copy()
    v_arr    = merged["v"].to_numpy(dtype=np.float32).reshape(N, T1).copy()
    t_arr    = merged["t_years"].to_numpy(dtype=np.float32).reshape(N, T1).copy()
    S2_arr   = merged["S2"].to_numpy(dtype=np.float32).reshape(N, T1).copy()
    mean_arr = merged["running_mean"].to_numpy(dtype=np.float32).reshape(N, T1).copy()
    min_arr  = merged["running_min"].to_numpy(dtype=np.float32).reshape(N, T1).copy()

    # Validate basic numerical invariants before tensor conversion.
    _check_tensors(S_arr, v_arr, t_arr, T_mat, metadata, run_dir)

    # torch.from_numpy keeps data on CPU and avoids extra copies.
    return DatasetBatch(
        paths_S            = torch.from_numpy(S_arr),
        paths_v            = torch.from_numpy(v_arr),
        paths_t            = torch.from_numpy(t_arr),
        paths_S2           = torch.from_numpy(S2_arr),
        paths_running_mean = torch.from_numpy(mean_arr),
        paths_running_min  = torch.from_numpy(min_arr),
        K                  = K,
        T_mat              = T_mat,
        n_paths            = N,
        n_steps            = n_steps,
        metadata           = metadata,
    )


def load_split_sizes(run_dir: Path | str) -> dict[str, int]:
    """Return the number of paths in each split without loading tensors.

    Reads only Parquet file footers (no row data). Useful for logging setup
    in the training loop before any tensors are allocated.

    Parameters
    ----------
    run_dir : Path | str
        Path to the dataset run folder.

    Returns
    -------
    dict[str, int]
        e.g. ``{'train': 70000, 'val': 15000, 'test': 15000}``
    """
    run_dir = Path(run_dir)
    _require_dir(run_dir)

    metadata = _load_metadata(run_dir)
    n_steps  = metadata["time_grid"]["n_steps"]
    T1       = n_steps + 1

    sizes: dict[str, int] = {}
    for split in ("train", "val", "test"):
        path = run_dir / "observations" / split / "part-00000.parquet"
        _require_file(path)
        # pq.read_metadata reads only the file footer — no row data loaded
        file_meta = pq.read_metadata(path)
        sizes[split] = file_meta.num_rows // T1

    return sizes


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_split(split: str) -> None:
    """Validate split name against the canonical train/val/test set.

    Parameters
    ----------
    split : str
        Requested split name.

    Raises
    ------
    ValueError
        If ``split`` is not one of ``train``, ``val``, or ``test``.
    """
    valid = {"train", "val", "test"}
    if split not in valid:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of {sorted(valid)}."
        )


def _require_dir(path: Path) -> None:
    """Require that a path exists and is a directory.

    Parameters
    ----------
    path : Path
        Directory path to validate.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or is not a directory.
    """
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: '{path}'")
    if not path.is_dir():
        raise FileNotFoundError(f"Expected a directory at '{path}', found a file.")


def _require_file(path: Path) -> None:
    """Require that a file path exists.

    Parameters
    ----------
    path : Path
        File path to validate.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: '{path}'")


def _load_metadata(run_dir: Path) -> dict:
    """Load and parse dataset metadata.

    Parameters
    ----------
    run_dir : Path
        Dataset run directory.

    Returns
    -------
    dict
        Parsed JSON metadata.
    """
    meta_path = run_dir / "metadata.json"
    _require_file(meta_path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_contract_scalars(run_dir: Path) -> tuple[float, float]:
    """Load strike and maturity scalars from ``contracts.parquet``.

    Parameters
    ----------
    run_dir : Path
        Dataset run directory.

    Returns
    -------
    tuple[float, float]
        ``(K, T_mat)``.
    """
    con_path = run_dir / "contracts.parquet"
    _require_file(con_path)
    df = _read_parquet(con_path, CONTRACTS_SCHEMA)
    K     = float(df["strike"].iloc[0])
    T_mat = float(df["maturity_years"].iloc[0])
    return K, T_mat


def _read_parquet(path: Path, schema) -> pd.DataFrame:
    """Read a parquet table and validate schema columns.

    Parameters
    ----------
    path : Path
        Parquet file path.
    schema : pyarrow.Schema
        Expected Arrow schema.

    Returns
    -------
    pd.DataFrame
        Loaded table as a pandas dataframe.

    Raises
    ------
    ValueError
        If actual columns differ from the expected schema columns.
    """
    expected_cols = [f.name for f in schema]
    table = pq.read_table(path)
    actual_cols   = table.schema.names

    if set(actual_cols) != set(expected_cols):
        extras  = sorted(set(actual_cols) - set(expected_cols))
        missing = sorted(set(expected_cols) - set(actual_cols))
        raise ValueError(
            f"Column mismatch reading '{path}'.\n"
            f"  expected: {expected_cols}\n"
            f"  got:      {actual_cols}\n"
            f"  extras:   {extras}\n"
            f"  missing:  {missing}"
        )

    return table.to_pandas()


def _load_latent_state(lat_path: Path, obs: pd.DataFrame) -> pd.DataFrame:
    """Load latent variance data or construct a zero fallback.

    Parameters
    ----------
    lat_path : Path
        Expected latent-state parquet file path.
    obs : pd.DataFrame
        Observations table used for path/time indices in fallback mode.

    Returns
    -------
    pd.DataFrame
        Latent-state table with columns ``path_id``, ``t_idx``, and ``v``.
    """
    if lat_path.exists():
        return _read_parquet(lat_path, LATENT_STATE_SCHEMA)

    # Zero fallback preserves the shared state layout across simulators.
    return pd.DataFrame({
        "path_id": obs["path_id"].to_numpy(),
        "t_idx":   obs["t_idx"].to_numpy(),
        "v":       np.zeros(len(obs), dtype=np.float32),
    })


def _load_variance_swap(vs_path: Path, obs: pd.DataFrame) -> pd.DataFrame:
    """Load variance swap data or construct a zero fallback.

    Parameters
    ----------
    vs_path : Path
        Expected variance-swap parquet file path
        (``run_dir/variance_swap/<split>/part-00000.parquet``).
    obs : pd.DataFrame
        Observations table used for path/time indices in fallback mode.

    Returns
    -------
    pd.DataFrame
        Variance-swap table with columns ``path_id``, ``t_idx``, and ``S2``.
        Returns zeros when the ``variance_swap/`` directory is absent (i.e.
        for BS and NGA datasets).
    """
    if vs_path.exists():
        return _read_parquet(vs_path, VARIANCE_SWAP_SCHEMA)

    # Zero fallback for simulators that do not produce a variance swap path.
    return pd.DataFrame({
        "path_id": obs["path_id"].to_numpy(),
        "t_idx":   obs["t_idx"].to_numpy(),
        "S2":      np.zeros(len(obs), dtype=np.float32),
    })


def _load_path_statistics(ps_path: Path, obs: pd.DataFrame) -> pd.DataFrame:
    """Load path statistics or construct a zero fallback.

    Parameters
    ----------
    ps_path : Path
        Expected path-statistics parquet file path
        (``run_dir/path_statistics/<split>/part-00000.parquet``).
    obs : pd.DataFrame
        Observations table used for path/time indices in fallback mode.

    Returns
    -------
    pd.DataFrame
        Path-statistics table with columns ``path_id``, ``t_idx``,
        ``running_mean``, and ``running_min``. Returns zeros when the
        ``path_statistics/`` directory is absent.
    """
    if ps_path.exists():
        return _read_parquet(ps_path, PATH_STATISTICS_SCHEMA)

    # Zero fallback for datasets generated before path statistics were added.
    return pd.DataFrame({
        "path_id":      obs["path_id"].to_numpy(),
        "t_idx":        obs["t_idx"].to_numpy(),
        "running_mean": np.zeros(len(obs), dtype=np.float32),
        "running_min":  np.zeros(len(obs), dtype=np.float32),
    })


def _check_tensors(
    S_arr: np.ndarray,
    v_arr: np.ndarray,
    t_arr: np.ndarray,
    T_mat: float,
    metadata: dict,
    run_dir: Path,
) -> None:
    """Run numerical sanity checks on reshaped path arrays.

    Parameters
    ----------
    S_arr : np.ndarray
        Spot matrix with shape ``(N, T+1)``.
    v_arr : np.ndarray
        Variance matrix with shape ``(N, T+1)``.
    t_arr : np.ndarray
        Time-grid matrix with shape ``(N, T+1)``.
    T_mat : float
        Maturity from the contracts table.
    metadata : dict
        Parsed metadata used for simulator-specific checks.
    run_dir : Path
        Dataset path used in error messages.

    Raises
    ------
    ValueError
        If time-grid constraints fail or NaNs are present.

    Warns
    -----
    UserWarning
        If a Heston dataset is loaded with all-zero variance values.
    """

    # t=0 must be 0.0 for every path
    if not np.allclose(t_arr[:, 0], 0.0, atol=1e-6):
        raise ValueError(
            f"paths_t[:,0] is not all zeros in '{run_dir}'. "
            "Check simulator output — all paths should start at t=0."
        )

    # t=T must match T_mat within float32 tolerance
    if not np.allclose(t_arr[:, -1], T_mat, atol=1e-4):
        actual = float(t_arr[:, -1].mean())
        raise ValueError(
            f"paths_t[:,-1] (mean={actual:.6f}) does not match "
            f"T_mat={T_mat:.6f} in '{run_dir}'."
        )

    # No NaN values anywhere
    for name, arr in [("S", S_arr), ("v", v_arr), ("t_years", t_arr)]:
        if np.isnan(arr).any():
            n_nan = int(np.isnan(arr).sum())
            raise ValueError(
                f"{n_nan} NaN value(s) found in column '{name}' "
                f"after loading '{run_dir}'."
            )

    # All-zero variance for Heston usually means latent-state files are absent.
    simulator = metadata.get("simulator", "")
    if simulator == "Heston" and np.allclose(v_arr, 0.0):
        warnings.warn(
            f"Heston dataset loaded from '{run_dir}' but paths_v is all zeros. "
            "The latent_state/ parquet files are missing — the variance process "
            "will not be available as a feature. Run generate_dataset.py with "
            "the updated Heston simulator to produce latent state files.",
            UserWarning,
            stacklevel=3,
        )
