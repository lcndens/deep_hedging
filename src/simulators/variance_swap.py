"""Variance swap price path computation for the Heston model.

The variance swap fair value at time t decomposes into a realized component
(trapezoidal approximation of the accumulated variance integral) and an
analytic expected-future-variance component derived from the Heston
mean-reversion formula.
"""

from __future__ import annotations

import numpy as np


def compute_variance_swap_paths(
    v: np.ndarray,
    kappa: float,
    theta: float,
    maturity_years: float,
) -> np.ndarray:
    """Compute the variance swap price path under the Heston model.

    Parameters
    ----------
    v : np.ndarray
        Variance array of shape ``(N, T+1)``. Raw simulation output from the
        Heston full-truncation Euler scheme; values may be slightly negative.
    kappa : float
        Mean-reversion speed of the variance process.
    theta : float
        Long-run variance level.
    maturity_years : float
        Contract maturity ``T`` in years.

    Returns
    -------
    np.ndarray
        Variance swap price path ``S2`` of shape ``(N, T+1)``, dtype float64.

    Notes
    -----
    The formula at each discrete time step ``t_k`` is:

    .. math::

        S^2_{t_k} = \\underbrace{\\int_0^{t_k} v_s^+ \\, ds}_{\\text{trapezoidal}}
                  + \\frac{v_{t_k} - \\theta}{\\kappa}
                    \\bigl(1 - e^{-\\kappa (T - t_k)}\\bigr)
                  + \\theta (T - t_k)

    The trapezoidal integral uses ``v_pos = max(v, 0)`` consistent with the
    full-truncation Euler discretization (which only ever feeds ``v_pos`` into
    drift and diffusion terms).  The analytic future-variance term uses the raw
    ``v`` values so that the conditional expectation is unbiased.

    At ``t=0`` the trapezoidal term vanishes and the expression reduces to the
    known closed-form initial fair value::

        S2[:, 0] = (v0 - theta) / kappa * (1 - exp(-kappa * T)) + theta * T

    At ``t=T`` the future-variance term vanishes (remaining time = 0) and
    ``S2[:, -1]`` equals only the realized integrated variance.

    Raises
    ------
    ValueError
        If ``v`` is not 2-D, has fewer than 2 time steps, or ``maturity_years``
        is non-positive.
    """
    if v.ndim != 2:
        raise ValueError(
            f"v must be 2-D with shape (N, T+1), got shape {v.shape}"
        )
    N, T1 = v.shape
    T = T1 - 1
    if T <= 0:
        raise ValueError(
            f"v must have at least 2 time steps, got T+1={T1}"
        )
    if maturity_years <= 0:
        raise ValueError(
            f"maturity_years must be positive, got {maturity_years}"
        )

    dt = maturity_years / T

    # ------------------------------------------------------------------
    # Realized component: trapezoidal integral of v_pos from 0 to t_k.
    # v_pos floors the raw variance at zero, matching the full-truncation
    # convention used in the simulation loop.
    # ------------------------------------------------------------------
    v_pos = np.maximum(v, 0.0)
    steps = 0.5 * (v_pos[:, :-1] + v_pos[:, 1:]) * dt  # (N, T)
    cum_trap = np.zeros((N, T1), dtype=np.float64)
    cum_trap[:, 1:] = np.cumsum(steps, axis=1)

    # ------------------------------------------------------------------
    # Analytic component: Heston conditional expectation of future
    # integrated variance E_t[int_t^T v_s ds].
    # ------------------------------------------------------------------
    t_grid = np.linspace(0.0, maturity_years, T1)
    remaining = maturity_years - t_grid           # (T+1,) — time to maturity
    exp_decay = np.exp(-kappa * remaining)        # (T+1,)

    # Broadcasting: v is (N, T+1), remaining and exp_decay are (T+1,).
    future_var = (
        (v - theta) / kappa * (1.0 - exp_decay)
        + theta * remaining
    )  # (N, T+1)

    return cum_trap + future_var
