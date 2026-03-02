"""Heston simulator for observable-only price paths."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HestonParams:
    """Configuration for Heston path generation.

    Includes spot process settings, variance process settings, and simulation
    size/time-grid controls.
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


def simulate_observations(cfg: HestonParams) -> pd.DataFrame:
    """
    Full truncation Euler Heston simulation:
      dS_t = m S_t dt + sqrt(v_t) S_t dW^S_t
      dv_t = kappa(theta - v_t) dt + xi sqrt(v_t) dW^v_t
      corr(dW^S, dW^v) = rho

    We clamp variance in the diffusion and drift using v_pos = max(v, 0).
    Only S is returned (observable-only dataset).
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

    # Correlated normals
    z1 = rng.standard_normal(size=(cfg.n_paths, cfg.n_steps), dtype=np.float64)
    z2 = rng.standard_normal(size=(cfg.n_paths, cfg.n_steps), dtype=np.float64)
    dW_s = z1
    dW_v = cfg.rho * z1 + np.sqrt(max(0.0, 1.0 - cfg.rho**2)) * z2

    # Allocate
    S = np.empty((cfg.n_paths, cfg.n_steps + 1), dtype=np.float64)
    v = np.empty((cfg.n_paths, cfg.n_steps + 1), dtype=np.float64)
    S[:, 0] = cfg.s0
    v[:, 0] = cfg.v0

    # Simulate
    for t in range(cfg.n_steps):
        v_pos = np.maximum(v[:, t], 0.0)

        # Variance: full truncation Euler
        v_next = (
            v[:, t]
            + cfg.kappa * (cfg.theta - v_pos) * dt
            + cfg.xi * np.sqrt(v_pos) * sqrt_dt * dW_v[:, t]
        )
        v[:, t + 1] = v_next  # can go negative; truncation happens via v_pos next step

        # Spot: log-Euler using v_pos (keeps S positive)
        vol = np.sqrt(v_pos)
        S[:, t + 1] = S[:, t] * np.exp(
            (cfg.m - 0.5 * v_pos) * dt + vol * sqrt_dt * dW_s[:, t]
        )

    # Convert to long format
    path_id = np.repeat(np.arange(cfg.n_paths, dtype=np.int64), cfg.n_steps + 1)
    t_idx = np.tile(np.arange(cfg.n_steps + 1, dtype=np.int32), cfg.n_paths)
    t_years = np.tile(t_grid.astype(np.float32), cfg.n_paths)
    S_long = S.astype(np.float32).reshape(-1)

    df = pd.DataFrame({"path_id": path_id, "t_idx": t_idx, "t_years": t_years, "S": S_long})

    # Enforce dtypes explicitly
    df["path_id"] = df["path_id"].astype("int64")
    df["t_idx"] = df["t_idx"].astype("int32")
    df["t_years"] = df["t_years"].astype("float32")
    df["S"] = df["S"].astype("float32")

    return df
