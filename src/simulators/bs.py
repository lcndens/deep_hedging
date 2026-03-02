"""Black-Scholes simulator for observable-only price paths."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BSParams:
    """Configuration for Black-Scholes path generation.

    Attributes define market parameters, the time grid, and simulation size.
    """

    # Defaults locked by your decision (Option A)
    s0: float = 100.0
    sigma: float = 0.2
    m: float = 0.0  # drift in dS = m S dt + sigma S dW

    maturity_years: float = 1.0
    n_steps: int = 30  # time grid includes maturity: 0..n_steps (n_steps+1 points)

    n_paths: int = 100_000
    seed: int = 42


def simulate_observations(cfg: BSParams) -> pd.DataFrame:
    """
    Simulate GBM paths under Black–Scholes:
        dS_t = m S_t dt + sigma S_t dW_t

    Returns long-format DataFrame with columns:
        path_id (int64), t_idx (int32), t_years (float32), S (float32)

    Note: No 'split' column is ever created.
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

    # Exact GBM discretization in log space
    z = rng.standard_normal(size=(cfg.n_paths, cfg.n_steps), dtype=np.float64)
    incr = (cfg.m - 0.5 * cfg.sigma**2) * dt + cfg.sigma * np.sqrt(dt) * z

    logS0 = np.log(cfg.s0)
    logS = np.empty((cfg.n_paths, cfg.n_steps + 1), dtype=np.float64)
    logS[:, 0] = logS0
    logS[:, 1:] = logS0 + np.cumsum(incr, axis=1)

    S = np.exp(logS).astype(np.float32)

    # Long format
    path_id = np.repeat(np.arange(cfg.n_paths, dtype=np.int64), cfg.n_steps + 1)
    t_idx = np.tile(np.arange(cfg.n_steps + 1, dtype=np.int32), cfg.n_paths)
    t_years = np.tile(t_grid.astype(np.float32), cfg.n_paths)
    S_long = S.reshape(-1)

    df = pd.DataFrame(
        {"path_id": path_id, "t_idx": t_idx, "t_years": t_years, "S": S_long}
    )

    # Explicit dtype enforcement (keeps everything consistent)
    df["path_id"] = df["path_id"].astype("int64")
    df["t_idx"] = df["t_idx"].astype("int32")
    df["t_years"] = df["t_years"].astype("float32")
    df["S"] = df["S"].astype("float32")

    return df
