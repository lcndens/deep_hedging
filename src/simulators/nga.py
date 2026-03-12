"""NGA path simulation for robust deep hedging datasets.

This module generates trajectories from a nonlinear generalized affine process
with per-step parameter resampling and returns canonical observations plus a
zero-valued latent variance placeholder.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NGAParams:
    """Configuration for nonlinear generalized affine path simulation.

    Parameters
    ----------
    x0 : float, default=100.0
        Initial process value.
    a0_min, a0_max : float
        Bounds for ``a0`` sampled each timestep.
    a1_min, a1_max : float
        Bounds for ``a1`` sampled each timestep.
    b0_min, b0_max : float
        Bounds for ``b0`` sampled each timestep.
    b1_min, b1_max : float
        Bounds for ``b1`` sampled each timestep.
    gamma_min, gamma_max : float
        Bounds for exponent ``gamma`` sampled each timestep.
    maturity_years : float, default=1.0
        Contract maturity ``T_mat`` in years.
    n_steps : int, default=30
        Number of hedging timesteps ``T``; paths include ``T+1`` points.
    n_paths : int, default=100_000
        Number of Monte Carlo paths ``N``.
    seed : int, default=42
        Random seed.

    Notes
    -----
    The process follows
    ``dX_t = (b0 + b1 X_t) dt + (a0 + a1 X_t^+)^gamma dW_t``
    with parameters sampled uniformly from the configured intervals at each
    timestep, matching the robust setting in He et al. (2025).
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
    """Validate interval bounds used for parameter resampling.

    Parameters
    ----------
    lo : float
        Lower bound.
    hi : float
        Upper bound.
    name : str
        Parameter name for error reporting.

    Raises
    ------
    ValueError
        If ``hi < lo``.
    """
    if hi < lo:
        raise ValueError(f"{name}: max < min ({hi} < {lo})")


def simulate_observations(cfg: NGAParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate NGA paths and return canonical observation tables.

    Parameters
    ----------
    cfg : NGAParams
        NGA simulation configuration.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Two long-format tables:
        ``observations`` with ``path_id, t_idx, t_years, S`` (where ``S`` stores
        the simulated ``X_t`` values) and ``latent_state`` with ``path_id, t_idx,
        v`` set to zero.

    Raises
    ------
    ValueError
        If configuration values are invalid.
    """
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

    # Allocate all paths including t=0 and maturity.
    X = np.empty((cfg.n_paths, cfg.n_steps + 1), dtype=np.float64)
    X[:, 0] = cfg.x0

    # Pre-sampling shocks avoids repeated RNG overhead inside the loop.
    Z = rng.standard_normal(size=(cfg.n_paths, cfg.n_steps), dtype=np.float64)

    for i in range(cfg.n_steps):
        # Robust setting: resample coefficients at every timestep.
        a0 = rng.uniform(cfg.a0_min, cfg.a0_max, size=cfg.n_paths)
        a1 = rng.uniform(cfg.a1_min, cfg.a1_max, size=cfg.n_paths)
        b0 = rng.uniform(cfg.b0_min, cfg.b0_max, size=cfg.n_paths)
        b1 = rng.uniform(cfg.b1_min, cfg.b1_max, size=cfg.n_paths)
        gamma = rng.uniform(cfg.gamma_min, cfg.gamma_max, size=cfg.n_paths)

        x = X[:, i]
        x_pos = np.maximum(x, 0.0)

        drift = (b0 + b1 * x) * dt

        # Diffusion term equals sqrt(a(x)) for a(x)=(a0+a1 x^+)^(2*gamma).
        vol = np.power(a0 + a1 * x_pos, gamma)

        X[:, i + 1] = x + drift + vol * sqrt_dt * Z[:, i]

    # Canonical observations schema uses column name S for the process value.
    path_id = np.repeat(np.arange(cfg.n_paths, dtype=np.int64), cfg.n_steps + 1)
    t_idx = np.tile(np.arange(cfg.n_steps + 1, dtype=np.int32), cfg.n_paths)
    t_years = np.tile(t_grid.astype(np.float32), cfg.n_paths)
    S_long = X.astype(np.float32).reshape(-1)

    df = pd.DataFrame({"path_id": path_id, "t_idx": t_idx, "t_years": t_years, "S": S_long})

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

    return df, latent_df
