"""Tests for path statistics (running_mean, running_min) across all simulators.

Covers:
  - Shape (N, T+1) returned by all three simulate_observations functions
  - At t=0, running_mean[:, 0] == S[:, 0] and running_min[:, 0] == S[:, 0]
  - running_min is non-increasing (monotone) over time
  - running_mean is bounded between the path min and path max
  - Round-trip: DatasetBatch.paths_running_mean / paths_running_min match
    the raw simulator output after loading from disk
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.simulators.bs import BSParams
from src.simulators.bs import simulate_observations as bs_simulate
from src.simulators.heston import HestonParams
from src.simulators.heston import simulate_observations as heston_simulate
from src.simulators.nga import NGAParams
from src.simulators.nga import simulate_observations as nga_simulate
from src.io.dataset_loader import load_dataset

from tests.conftest import N_PATHS, N_STEPS, T1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pivot_ps(ps_df: pd.DataFrame, n_paths: int, t1: int) -> tuple[np.ndarray, np.ndarray]:
    """Pivot path_statistics long-format DataFrame to (N, T+1) arrays."""
    ps_sorted = ps_df.sort_values(["path_id", "t_idx"])
    mean_arr = ps_sorted["running_mean"].to_numpy(dtype=np.float32).reshape(n_paths, t1)
    min_arr  = ps_sorted["running_min"].to_numpy(dtype=np.float32).reshape(n_paths, t1)
    return mean_arr, min_arr


def _pivot_S(obs_df: pd.DataFrame, n_paths: int, t1: int) -> np.ndarray:
    """Pivot observations long-format DataFrame to (N, T+1) array."""
    obs_sorted = obs_df.sort_values(["path_id", "t_idx"])
    return obs_sorted["S"].to_numpy(dtype=np.float32).reshape(n_paths, t1)


# ---------------------------------------------------------------------------
# Small simulation parameters used by unit tests
# ---------------------------------------------------------------------------

N_SMALL = 50
T_SMALL = 10
T1_SMALL = T_SMALL + 1


# ---------------------------------------------------------------------------
# Return-shape tests (direct simulator calls)
# ---------------------------------------------------------------------------

class TestPathStatisticsShape:

    def test_bs_shape(self):
        cfg = BSParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=1)
        _, _, ps = bs_simulate(cfg)
        mean_arr, min_arr = _pivot_ps(ps, N_SMALL, T1_SMALL)
        assert mean_arr.shape == (N_SMALL, T1_SMALL)
        assert min_arr.shape  == (N_SMALL, T1_SMALL)

    def test_heston_shape(self):
        cfg = HestonParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=1)
        _, _, _, ps = heston_simulate(cfg)
        mean_arr, min_arr = _pivot_ps(ps, N_SMALL, T1_SMALL)
        assert mean_arr.shape == (N_SMALL, T1_SMALL)
        assert min_arr.shape  == (N_SMALL, T1_SMALL)

    def test_nga_shape(self):
        cfg = NGAParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=1)
        _, _, ps = nga_simulate(cfg)
        mean_arr, min_arr = _pivot_ps(ps, N_SMALL, T1_SMALL)
        assert mean_arr.shape == (N_SMALL, T1_SMALL)
        assert min_arr.shape  == (N_SMALL, T1_SMALL)


# ---------------------------------------------------------------------------
# Initial-value tests: at t=0, statistics equal S[:, 0]
# ---------------------------------------------------------------------------

class TestInitialValues:

    def test_bs_mean_at_t0(self):
        cfg = BSParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=2)
        obs, _, ps = bs_simulate(cfg)
        mean_arr, _ = _pivot_ps(ps, N_SMALL, T1_SMALL)
        S = _pivot_S(obs, N_SMALL, T1_SMALL)
        np.testing.assert_allclose(mean_arr[:, 0], S[:, 0], rtol=1e-5,
                                   err_msg="BS: running_mean[:, 0] != S[:, 0]")

    def test_bs_min_at_t0(self):
        cfg = BSParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=2)
        obs, _, ps = bs_simulate(cfg)
        _, min_arr = _pivot_ps(ps, N_SMALL, T1_SMALL)
        S = _pivot_S(obs, N_SMALL, T1_SMALL)
        np.testing.assert_allclose(min_arr[:, 0], S[:, 0], rtol=1e-5,
                                   err_msg="BS: running_min[:, 0] != S[:, 0]")

    def test_heston_mean_at_t0(self):
        cfg = HestonParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=2)
        obs, _, _, ps = heston_simulate(cfg)
        mean_arr, _ = _pivot_ps(ps, N_SMALL, T1_SMALL)
        S = _pivot_S(obs, N_SMALL, T1_SMALL)
        np.testing.assert_allclose(mean_arr[:, 0], S[:, 0], rtol=1e-5,
                                   err_msg="Heston: running_mean[:, 0] != S[:, 0]")

    def test_heston_min_at_t0(self):
        cfg = HestonParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=2)
        obs, _, _, ps = heston_simulate(cfg)
        _, min_arr = _pivot_ps(ps, N_SMALL, T1_SMALL)
        S = _pivot_S(obs, N_SMALL, T1_SMALL)
        np.testing.assert_allclose(min_arr[:, 0], S[:, 0], rtol=1e-5,
                                   err_msg="Heston: running_min[:, 0] != S[:, 0]")

    def test_nga_mean_at_t0(self):
        cfg = NGAParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=2)
        obs, _, ps = nga_simulate(cfg)
        mean_arr, _ = _pivot_ps(ps, N_SMALL, T1_SMALL)
        S = _pivot_S(obs, N_SMALL, T1_SMALL)
        np.testing.assert_allclose(mean_arr[:, 0], S[:, 0], rtol=1e-5,
                                   err_msg="NGA: running_mean[:, 0] != S[:, 0]")

    def test_nga_min_at_t0(self):
        cfg = NGAParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=2)
        obs, _, ps = nga_simulate(cfg)
        _, min_arr = _pivot_ps(ps, N_SMALL, T1_SMALL)
        S = _pivot_S(obs, N_SMALL, T1_SMALL)
        np.testing.assert_allclose(min_arr[:, 0], S[:, 0], rtol=1e-5,
                                   err_msg="NGA: running_min[:, 0] != S[:, 0]")


# ---------------------------------------------------------------------------
# Monotonicity: running_min must be non-increasing
# ---------------------------------------------------------------------------

class TestMonotonicity:

    @pytest.mark.parametrize("sim,sim_fn,unpack", [
        ("bs",     lambda cfg: bs_simulate(cfg),      lambda r: (r[0], r[2])),
        ("heston", lambda cfg: heston_simulate(cfg),  lambda r: (r[0], r[3])),
        ("nga",    lambda cfg: nga_simulate(cfg),     lambda r: (r[0], r[2])),
    ])
    def test_running_min_nonincreasing(self, sim, sim_fn, unpack):
        cfg_map = {
            "bs":     BSParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=3),
            "heston": HestonParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=3),
            "nga":    NGAParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=3),
        }
        result = sim_fn(cfg_map[sim])
        _, ps = unpack(result)
        _, min_arr = _pivot_ps(ps, N_SMALL, T1_SMALL)
        # Consecutive differences must be <= 0 (allow tiny float32 rounding)
        diffs = np.diff(min_arr, axis=1)
        assert (diffs <= 1e-6).all(), (
            f"{sim}: running_min is not non-increasing; max increase = {diffs.max():.2e}"
        )


# ---------------------------------------------------------------------------
# Bounds: running_mean must lie in [path_min, path_max] at every timestep
# ---------------------------------------------------------------------------

class TestMeanBounds:

    def test_bs_running_mean_bounded(self):
        cfg = BSParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=4)
        obs, _, ps = bs_simulate(cfg)
        mean_arr, _ = _pivot_ps(ps, N_SMALL, T1_SMALL)
        S = _pivot_S(obs, N_SMALL, T1_SMALL)
        path_min = S.min(axis=1, keepdims=True)
        path_max = S.max(axis=1, keepdims=True)
        assert (mean_arr >= path_min - 1e-4).all(), "BS: running_mean below path min"
        assert (mean_arr <= path_max + 1e-4).all(), "BS: running_mean above path max"

    def test_heston_running_mean_bounded(self):
        cfg = HestonParams(n_paths=N_SMALL, n_steps=T_SMALL, seed=4)
        obs, _, _, ps = heston_simulate(cfg)
        mean_arr, _ = _pivot_ps(ps, N_SMALL, T1_SMALL)
        S = _pivot_S(obs, N_SMALL, T1_SMALL)
        path_min = S.min(axis=1, keepdims=True)
        path_max = S.max(axis=1, keepdims=True)
        assert (mean_arr >= path_min - 1e-4).all(), "Heston: running_mean below path min"
        assert (mean_arr <= path_max + 1e-4).all(), "Heston: running_mean above path max"


# ---------------------------------------------------------------------------
# Round-trip: DatasetBatch tensors match raw simulator output
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Verify that paths_running_mean and paths_running_min loaded from disk
    match the values computed directly by the simulator."""

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_paths_running_mean_shape(self, sim_fixture, request):
        run_dir = request.getfixturevalue(sim_fixture)
        batch = load_dataset(run_dir, split="train")
        N = batch.n_paths
        assert batch.paths_running_mean.shape == (N, T1)
        assert batch.paths_running_mean.dtype == torch.float32

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_paths_running_min_shape(self, sim_fixture, request):
        run_dir = request.getfixturevalue(sim_fixture)
        batch = load_dataset(run_dir, split="train")
        N = batch.n_paths
        assert batch.paths_running_min.shape == (N, T1)
        assert batch.paths_running_min.dtype == torch.float32

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_running_min_at_t0_equals_S_at_t0(self, sim_fixture, request):
        """After round-trip through disk, running_min[:, 0] == paths_S[:, 0]."""
        run_dir = request.getfixturevalue(sim_fixture)
        batch = load_dataset(run_dir, split="train")
        np.testing.assert_allclose(
            batch.paths_running_min[:, 0].numpy(),
            batch.paths_S[:, 0].numpy(),
            rtol=1e-5,
            err_msg=f"{sim_fixture}: loaded running_min[:, 0] != paths_S[:, 0]",
        )

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_running_mean_at_t0_equals_S_at_t0(self, sim_fixture, request):
        """After round-trip through disk, running_mean[:, 0] == paths_S[:, 0]."""
        run_dir = request.getfixturevalue(sim_fixture)
        batch = load_dataset(run_dir, split="train")
        np.testing.assert_allclose(
            batch.paths_running_mean[:, 0].numpy(),
            batch.paths_S[:, 0].numpy(),
            rtol=1e-5,
            err_msg=f"{sim_fixture}: loaded running_mean[:, 0] != paths_S[:, 0]",
        )
