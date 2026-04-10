"""Black-Scholes path simulation for dataset generation.

This module generates spot trajectories under geometric Brownian motion (GBM)
for the deep-hedging pipeline and emits observations in long-table format plus
a zero-valued latent variance placeholder.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BSParams:
    """Configuration for Black-Scholes path generation.

    Parameters
    ----------
    s0 : float, default=100.0
        Initial spot price ``S_0``.
    sigma : float, default=0.2
        Volatility parameter in ``dS_t = m S_t dt + sigma S_t dW_t``.
    m : float, default=0.0
        Drift parameter in the GBM dynamics.
    maturity_years : float, default=1.0
        Contract maturity ``T_mat`` in years.
    n_steps : int, default=30
        Number of hedging timesteps ``T``. The simulated path has ``T+1``
        points including ``t=0`` and maturity.
    n_paths : int, default=100_000
        Number of Monte Carlo paths ``N``.
    seed : int, default=42
        Random seed for reproducible simulation.
    """

    # Defaults locked by your decision
    s0: float = 100.0
    sigma: float = 0.2
    m: float = 0.0  # drift in dS = m S dt + sigma S dW

    maturity_years: float = 1.0
    n_steps: int = 30  # time grid includes maturity: 0..n_steps (n_steps+1 points)

    n_paths: int = 100_000
    seed: int = 42


def simulate_observations(cfg: BSParams) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate Black-Scholes spot paths and format canonical output tables.

    Parameters
    ----------
    cfg : BSParams
        Black-Scholes simulation settings.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Three long-format tables:
        ``observations`` with columns ``path_id, t_idx, t_years, S``;
        ``latent_state`` with columns ``path_id, t_idx, v``; and
        ``path_statistics`` with columns ``path_id, t_idx, running_mean, running_min``.

    Notes
    -----
    The simulation uses exact log-space discretization:
    ``log S_{t+1} = log S_t + (m - 0.5 sigma^2) dt + sigma sqrt(dt) Z_t``.
    For Black-Scholes, latent variance is stored as zeros to keep a shared
    schema with Heston datasets.

    Raises
    ------
    ValueError
        If configuration values violate positivity or range constraints.
    """
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if cfg.n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if cfg.maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    if cfg.s0 <= 0:
        raise ValueError("s0 must be positive")
    if cfg.sigma < 0:
        raise ValueError("sigma must be nonnegative")

    rng = np.random.default_rng(cfg.seed)

    dt = cfg.maturity_years / cfg.n_steps
    t_grid = np.linspace(0.0, cfg.maturity_years, cfg.n_steps + 1, dtype=np.float64)

    # Exact log-Euler update avoids positivity violations for S_t.
    z = rng.standard_normal(size=(cfg.n_paths, cfg.n_steps), dtype=np.float64)
    incr = (cfg.m - 0.5 * cfg.sigma**2) * dt + cfg.sigma * np.sqrt(dt) * z

    logS0 = np.log(cfg.s0)
    logS = np.empty((cfg.n_paths, cfg.n_steps + 1), dtype=np.float64)
    logS[:, 0] = logS0
    logS[:, 1:] = logS0 + np.cumsum(incr, axis=1)

    S = np.exp(logS).astype(np.float32)

    # Flatten to canonical long format expected by downstream parquet writer.
    path_id = np.repeat(np.arange(cfg.n_paths, dtype=np.int64), cfg.n_steps + 1)
    t_idx = np.tile(np.arange(cfg.n_steps + 1, dtype=np.int32), cfg.n_paths)
    t_years = np.tile(t_grid.astype(np.float32), cfg.n_paths)
    S_long = S.reshape(-1)

    df = pd.DataFrame(
        {"path_id": path_id, "t_idx": t_idx, "t_years": t_years, "S": S_long}
    )

    # Enforce Arrow-compatible dtypes before serialization.
    df["path_id"] = df["path_id"].astype("int64")
    df["t_idx"] = df["t_idx"].astype("int32")
    df["t_years"] = df["t_years"].astype("float32")
    df["S"] = df["S"].astype("float32")

    latent_df = pd.DataFrame(
        {
            "path_id": path_id,
            "t_idx": t_idx,
            "v": np.zeros(path_id.shape[0], dtype=np.float32),
        }
    )
    latent_df["path_id"] = latent_df["path_id"].astype("int64")
    latent_df["t_idx"] = latent_df["t_idx"].astype("int32")
    latent_df["v"] = latent_df["v"].astype("float32")

    # Running path statistics computed over S[:, 0:t+1] at each timestep t.
    counts = np.arange(1, cfg.n_steps + 2, dtype=np.float64)  # [1, 2, ..., T+1]
    running_mean = (np.cumsum(S, axis=1) / counts).astype(np.float32)
    running_min  = np.minimum.accumulate(S, axis=1).astype(np.float32)

    path_stats_df = pd.DataFrame({
        "path_id":      path_id,
        "t_idx":        t_idx,
        "running_mean": running_mean.reshape(-1),
        "running_min":  running_min.reshape(-1),
    })
    path_stats_df["path_id"]      = path_stats_df["path_id"].astype("int64")
    path_stats_df["t_idx"]        = path_stats_df["t_idx"].astype("int32")
    path_stats_df["running_mean"] = path_stats_df["running_mean"].astype("float32")
    path_stats_df["running_min"]  = path_stats_df["running_min"].astype("float32")

    return df, latent_df, path_stats_df
