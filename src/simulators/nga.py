"""NGA simulator for observable-only price paths."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NGAParams:
    """
    Nonlinear generalized affine (NGA) simulator used in Robust Deep Hedging.

    SDE form (conceptually):
        dX_t = (b0 + b1 X_t) dt + (a0 + a1 X_t^+)^gamma dW_t

    Discretization:
        Euler–Maruyama with parameters (b0,b1,a0,a1,gamma) resampled
        uniformly from intervals at EACH time step (robust training setting).
    """
    # Initial state
    x0: float = 100.0

    # Parameter intervals
    a0_min: float = 0.3
    a0_max: float = 0.7
    a1_min: float = 0.04
    a1_max: float = 0.06
    b0_min: float = -0.2
    b0_max: float = 0.2
    b1_min: float = -0.1
    b1_max: float = 0.1
    gamma_min: float = 0.5
    gamma_max: float = 1.5

    # Time grid
    maturity_years: float = 1.0
    n_steps: int = 30  # includes maturity => n_steps+1 points

    # Simulation size
    n_paths: int = 100_000
    seed: int = 42


def _check_interval(lo: float, hi: float, name: str) -> None:
    """Validate inclusive interval bounds used for per-step parameter sampling."""
    if hi < lo:
        raise ValueError(f"{name}: max < min ({hi} < {lo})")


def simulate_observations(cfg: NGAParams) -> pd.DataFrame:
    """Simulate NGA paths and return canonical long-format observations."""
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if cfg.n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if cfg.maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    if cfg.x0 <= 0:
        raise ValueError("x0 must be positive")

    _check_interval(cfg.a0_min, cfg.a0_max, "a0")
    _check_interval(cfg.a1_min, cfg.a1_max, "a1")
    _check_interval(cfg.b0_min, cfg.b0_max, "b0")
    _check_interval(cfg.b1_min, cfg.b1_max, "b1")
    _check_interval(cfg.gamma_min, cfg.gamma_max, "gamma")

    rng = np.random.default_rng(cfg.seed)

    dt = cfg.maturity_years / cfg.n_steps
    sqrt_dt = np.sqrt(dt)
    t_grid = np.linspace(0.0, cfg.maturity_years, cfg.n_steps + 1, dtype=np.float64)

    # Allocate X paths
    X = np.empty((cfg.n_paths, cfg.n_steps + 1), dtype=np.float64)
    X[:, 0] = cfg.x0

    # Pre-draw Brownian normals (faster than drawing in-loop)
    Z = rng.standard_normal(size=(cfg.n_paths, cfg.n_steps), dtype=np.float64)

    for i in range(cfg.n_steps):
        # Sample parameters uniformly from bounds EACH time step (Remark 3.2)
        a0 = rng.uniform(cfg.a0_min, cfg.a0_max, size=cfg.n_paths)
        a1 = rng.uniform(cfg.a1_min, cfg.a1_max, size=cfg.n_paths)
        b0 = rng.uniform(cfg.b0_min, cfg.b0_max, size=cfg.n_paths)
        b1 = rng.uniform(cfg.b1_min, cfg.b1_max, size=cfg.n_paths)
        gamma = rng.uniform(cfg.gamma_min, cfg.gamma_max, size=cfg.n_paths)

        x = X[:, i]
        x_pos = np.maximum(x, 0.0)

        drift = (b0 + b1 * x) * dt

        # Vol term corresponds to sqrt(a(x)) where a(x) = (a0 + a1 x^+)^(2 gamma)
        vol = np.power(a0 + a1 * x_pos, gamma)

        X[:, i + 1] = x + drift + vol * sqrt_dt * Z[:, i]

    # Long format to match canonical observations schema: store X as S
    path_id = np.repeat(np.arange(cfg.n_paths, dtype=np.int64), cfg.n_steps + 1)
    t_idx = np.tile(np.arange(cfg.n_steps + 1, dtype=np.int32), cfg.n_paths)
    t_years = np.tile(t_grid.astype(np.float32), cfg.n_paths)
    S_long = X.astype(np.float32).reshape(-1)

    df = pd.DataFrame({"path_id": path_id, "t_idx": t_idx, "t_years": t_years, "S": S_long})

    # enforce dtypes
    df["path_id"] = df["path_id"].astype("int64")
    df["t_idx"] = df["t_idx"].astype("int32")
    df["t_years"] = df["t_years"].astype("float32")
    df["S"] = df["S"].astype("float32")

    return df
