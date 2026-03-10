"""Dataset loader for deep hedging pipeline — Stage 1.

Reads pre-generated Parquet datasets from disk and returns dense CPU tensors
wrapped in a DatasetBatch dataclass. This is the only module that touches disk
during a training run; it is called once at startup.

Folder layout expected (schema v1.0):
    <run_dir>/
        metadata.json
        contracts.parquet
        observations/<split>/part-00000.parquet
        latent_state/<split>/part-00000.parquet   (optional — zeros if absent)

Usage
-----
    from src.io.dataset_loader import load_dataset, load_split_sizes

    batch = load_dataset("data/datasets/v1.0/heston/<run_id>", split="train")
    # batch.paths_S  : torch.Tensor  (N, T+1)
    # batch.paths_v  : torch.Tensor  (N, T+1)
    # batch.paths_t  : torch.Tensor  (N, T+1)
    # batch.K        : float
    # batch.T_mat    : float
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

from src.schema.v1_0 import OBS_SCHEMA, CONTRACTS_SCHEMA, LATENT_STATE_SCHEMA

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetBatch:
    """All tensors and scalars for one split of a pre-generated dataset.

    Attributes
    ----------
    paths_S : (N, T+1) float32 CPU tensor — spot prices at every timestep.
    paths_v : (N, T+1) float32 CPU tensor — variance process (0.0 for BS/NGA
              or when latent_state/ files are absent).
    paths_t : (N, T+1) float32 CPU tensor — time in years at every timestep.
    K       : strike price (Python float).
    T_mat   : maturity in years (Python float).
    n_paths : number of paths N in this split.
    n_steps : number of rebalancing steps T (not T+1).
    metadata: raw metadata.json dict, kept for logging and config validation.
    """

    paths_S:  torch.Tensor
    paths_v:  torch.Tensor
    paths_t:  torch.Tensor
    K:        float
    T_mat:    float
    n_paths:  int
    n_steps:  int
    metadata: dict


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

    # Step 1 — Load metadata and contracts
    metadata = _load_metadata(run_dir)
    n_steps  = metadata["time_grid"]["n_steps"]
    T1       = n_steps + 1
    K, T_mat = _load_contract_scalars(run_dir)

    # Step 2 — Read observations parquet
    obs_path = run_dir / "observations" / split / "part-00000.parquet"
    _require_file(obs_path)
    obs = _read_parquet(obs_path, OBS_SCHEMA)

    # Step 3 — Read latent state parquet (with fallback to zeros)
    lat_path = run_dir / "latent_state" / split / "part-00000.parquet"
    lat = _load_latent_state(lat_path, obs)

    # Step 4 — Merge on (path_id, t_idx)
    merged = obs.merge(
        lat[["path_id", "t_idx", "v"]],
        on=["path_id", "t_idx"],
        how="left",
    )

    # Step 5 — Sort by (path_id, t_idx) — required for reshape correctness
    merged = merged.sort_values(["path_id", "t_idx"]).reset_index(drop=True)

    # Step 6 — Validate row count
    n_rows = len(merged)
    if n_rows % T1 != 0:
        raise ValueError(
            f"Row count {n_rows} is not divisible by T+1={T1}. "
            f"The dataset in '{run_dir}' may be corrupted or truncated."
        )
    N = n_rows // T1

    # Step 7 — Pivot long → wide via numpy reshape (fast, zero intermediate copy)
    S_arr = merged["S"].to_numpy(dtype=np.float32).reshape(N, T1).copy()
    v_arr = merged["v"].to_numpy(dtype=np.float32).reshape(N, T1).copy()
    t_arr = merged["t_years"].to_numpy(dtype=np.float32).reshape(N, T1).copy()

    # Step 8 — Sanity checks
    _check_tensors(S_arr, v_arr, t_arr, T_mat, metadata, run_dir)

    # Step 9 — Convert to tensors (torch.from_numpy shares memory — no copy)
    return DatasetBatch(
        paths_S  = torch.from_numpy(S_arr),
        paths_v  = torch.from_numpy(v_arr),
        paths_t  = torch.from_numpy(t_arr),
        K        = K,
        T_mat    = T_mat,
        n_paths  = N,
        n_steps  = n_steps,
        metadata = metadata,
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
    run_dir  = Path(run_dir)
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
    valid = {"train", "val", "test"}
    if split not in valid:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of {sorted(valid)}."
        )


def _require_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: '{path}'")
    if not path.is_dir():
        raise FileNotFoundError(f"Expected a directory at '{path}', found a file.")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: '{path}'")


def _load_metadata(run_dir: Path) -> dict:
    meta_path = run_dir / "metadata.json"
    _require_file(meta_path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_contract_scalars(run_dir: Path) -> tuple[float, float]:
    """Read strike K and maturity T_mat from contracts.parquet."""
    con_path = run_dir / "contracts.parquet"
    _require_file(con_path)
    df = _read_parquet(con_path, CONTRACTS_SCHEMA)
    K     = float(df["strike"].iloc[0])
    T_mat = float(df["maturity_years"].iloc[0])
    return K, T_mat


def _read_parquet(path: Path, schema) -> pd.DataFrame:
    """Read a Parquet file and validate columns against an Arrow schema.

    Uses pyarrow for reading (consistent with how the writer writes) then
    converts to pandas. Column validation raises a clear error rather than
    a cryptic downstream KeyError.
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
    """Read latent state parquet, or construct a zero fallback if absent."""
    if lat_path.exists():
        return _read_parquet(lat_path, LATENT_STATE_SCHEMA)

    # Fallback: zero variance — keeps loader functional before Week 1 is done
    # and for BS/NGA datasets generated before the latent state schema existed.
    return pd.DataFrame({
        "path_id": obs["path_id"].to_numpy(),
        "t_idx":   obs["t_idx"].to_numpy(),
        "v":       np.zeros(len(obs), dtype=np.float32),
    })


def _check_tensors(
    S_arr: np.ndarray,
    v_arr: np.ndarray,
    t_arr: np.ndarray,
    T_mat: float,
    metadata: dict,
    run_dir: Path,
) -> None:
    """Run sanity checks on the pivoted arrays before converting to tensors."""

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

    # Heston-specific: warn if variance is all zeros
    # (indicates latent_state/ files are missing and fallback was used)
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
