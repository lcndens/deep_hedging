"""Heston path simulation with latent variance output.

This module simulates coupled spot and variance dynamics for the deep-hedging
dataset pipeline and returns canonical observations plus latent-state tables.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.simulators.variance_swap import compute_variance_swap_paths


@dataclass(frozen=True)
class HestonParams:
    """Configuration for Heston path generation.

    Parameters
    ----------
    s0 : float, default=100.0
        Initial spot price ``S_0``.
    m : float, default=0.0
        Spot drift parameter in ``dS_t = m S_t dt + sqrt(v_t) S_t dW_t^S``.
    v0 : float, default=0.04
        Initial variance level ``v_0``.
    kappa : float, default=1.5
        Mean-reversion speed in the variance process.
    theta : float, default=0.04
        Long-run variance level.
    xi : float, default=0.3
        Volatility-of-volatility coefficient.
    rho : float, default=-0.7
        Correlation between spot and variance Brownian motions.
    maturity_years : float, default=1.0
        Contract maturity ``T_mat`` in years.
    n_steps : int, default=30
        Number of hedging timesteps ``T``. Paths contain ``T+1`` points.
    n_paths : int, default=100_000
        Number of Monte Carlo paths ``N``.
    seed : int, default=42
        Random seed.
    """

    # Market / spot
    s0: float = 100.0
    m: float = 0.0  # drift in dS = m S dt + sqrt(v) S dW

    # Variance process
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7

    # Time grid
    maturity_years: float = 1.0
    n_steps: int = 30  # includes maturity => n_steps+1 points

    # Simulation size
    n_paths: int = 100_000
    seed: int = 42


def simulate_observations(cfg: HestonParams) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate Heston spot/variance paths and return canonical tables.

    Parameters
    ----------
    cfg : HestonParams
        Heston simulation configuration.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Three long-format tables:
        ``observations`` with ``path_id, t_idx, t_years, S``;
        ``latent_state`` with ``path_id, t_idx, v``; and
        ``variance_swap`` with ``path_id, t_idx, S2``.

    Notes
    -----
    The implementation uses full-truncation Euler for variance updates and
    log-Euler for spot updates. The positive part ``v_t^+ = max(v_t, 0)`` is
    used in drift and diffusion terms to stabilize discretization.

    Raises
    ------
    ValueError
        If configuration values are out of supported ranges.
    """
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if cfg.n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if cfg.maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    if cfg.s0 <= 0:
        raise ValueError("s0 must be positive")
    if cfg.v0 < 0:
        raise ValueError("v0 must be nonnegative")
    if not (-1.0 <= cfg.rho <= 1.0):
        raise ValueError("rho must be in [-1, 1]")

    rng = np.random.default_rng(cfg.seed)

    dt = cfg.maturity_years / cfg.n_steps
    sqrt_dt = np.sqrt(dt)
    t_grid = np.linspace(0.0, cfg.maturity_years, cfg.n_steps + 1, dtype=np.float64)

    # Build correlated shocks with corr(dW^S, dW^v) = rho.
    z1 = rng.standard_normal(size=(cfg.n_paths, cfg.n_steps), dtype=np.float64)
    z2 = rng.standard_normal(size=(cfg.n_paths, cfg.n_steps), dtype=np.float64)
    dW_s = z1
    dW_v = cfg.rho * z1 + np.sqrt(max(0.0, 1.0 - cfg.rho**2)) * z2

    # Allocate full path tensors including t=0.
    S = np.empty((cfg.n_paths, cfg.n_steps + 1), dtype=np.float64)
    v = np.empty((cfg.n_paths, cfg.n_steps + 1), dtype=np.float64)
    S[:, 0] = cfg.s0
    v[:, 0] = cfg.v0

    # Step forward across T hedging intervals.
    for t in range(cfg.n_steps):
        v_pos = np.maximum(v[:, t], 0.0)

        # Full truncation keeps sqrt(v_t) well-defined in the update.
        v_next = (
            v[:, t]
            + cfg.kappa * (cfg.theta - v_pos) * dt
            + cfg.xi * np.sqrt(v_pos) * sqrt_dt * dW_v[:, t]
        )
        # v can be negative between steps; next update uses v_pos again.
        v[:, t + 1] = v_next

        # Log-Euler update preserves positivity of S_t.
        vol = np.sqrt(v_pos)
        S[:, t + 1] = S[:, t] * np.exp(
            (cfg.m - 0.5 * v_pos) * dt + vol * sqrt_dt * dW_s[:, t]
        )

    # Flatten to canonical long format for parquet serialization.
    path_id = np.repeat(np.arange(cfg.n_paths, dtype=np.int64), cfg.n_steps + 1)
    t_idx = np.tile(np.arange(cfg.n_steps + 1, dtype=np.int32), cfg.n_paths)
    t_years = np.tile(t_grid.astype(np.float32), cfg.n_paths)
    S_long = S.astype(np.float32).reshape(-1)

    df = pd.DataFrame({"path_id": path_id, "t_idx": t_idx, "t_years": t_years, "S": S_long})

    # Enforce Arrow-compatible dtypes before writing.
    df["path_id"] = df["path_id"].astype("int64")
    df["t_idx"] = df["t_idx"].astype("int32")
    df["t_years"] = df["t_years"].astype("float32")
    df["S"] = df["S"].astype("float32")

    v_floor = np.float64(np.finfo(np.float32).tiny)
    v_long = np.maximum(v, v_floor).astype(np.float32).reshape(-1)
    latent_df = pd.DataFrame({"path_id": path_id, "t_idx": t_idx, "v": v_long})
    latent_df["path_id"] = latent_df["path_id"].astype("int64")
    latent_df["t_idx"] = latent_df["t_idx"].astype("int32")
    latent_df["v"] = latent_df["v"].astype("float32")

    # Variance swap price path: realized integral + analytic expected future variance.
    S2 = compute_variance_swap_paths(v, cfg.kappa, cfg.theta, cfg.maturity_years)
    S2_long = S2.astype(np.float32).reshape(-1)
    variance_swap_df = pd.DataFrame({"path_id": path_id, "t_idx": t_idx, "S2": S2_long})
    variance_swap_df["path_id"] = variance_swap_df["path_id"].astype("int64")
    variance_swap_df["t_idx"] = variance_swap_df["t_idx"].astype("int32")
    variance_swap_df["S2"] = variance_swap_df["S2"].astype("float32")

    return df, latent_df, variance_swap_df
