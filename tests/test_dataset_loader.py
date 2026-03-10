"""Tests for src/io/dataset_loader.py — Stage 1.

All tests use pytest fixtures that generate small datasets into a tmp_path
directory via generate_dataset.main(). This mirrors the real pipeline exactly
without touching pre-existing data on disk.

Dataset sizes used: n_paths=200, n_steps=5, seed=0
Expected split N values come from the deterministic split_path_ids() logic.
We check proportions (roughly 70/15/15) rather than exact counts.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from src.generate_dataset import main as generate_dataset
from src.io.dataset_loader import DatasetBatch, load_dataset, load_split_sizes

from tests.conftest import N_PATHS, N_STEPS, T1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_train(run_dir: Path) -> DatasetBatch:
    return load_dataset(run_dir, split="train")


def _approx_split_n(n_paths: int, frac: float) -> int:
    """Expected approximate path count for a split fraction."""
    return round(n_paths * frac)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes:

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_paths_S_shape(self, sim_fixture, request):
        run_dir = request.getfixturevalue(sim_fixture)
        batch = _load_train(run_dir)
        N = batch.n_paths
        assert batch.paths_S.shape == (N, T1), (
            f"Expected paths_S shape ({N}, {T1}), got {batch.paths_S.shape}"
        )

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_paths_v_shape(self, sim_fixture, request):
        run_dir = request.getfixturevalue(sim_fixture)
        batch = _load_train(run_dir)
        assert batch.paths_v.shape == batch.paths_S.shape

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_paths_t_shape(self, sim_fixture, request):
        run_dir = request.getfixturevalue(sim_fixture)
        batch = _load_train(run_dir)
        assert batch.paths_t.shape == batch.paths_S.shape

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_n_steps_correct(self, sim_fixture, request):
        run_dir = request.getfixturevalue(sim_fixture)
        batch = _load_train(run_dir)
        assert batch.n_steps == N_STEPS

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_n_paths_matches_tensor(self, sim_fixture, request):
        run_dir = request.getfixturevalue(sim_fixture)
        batch = _load_train(run_dir)
        assert batch.paths_S.shape[0] == batch.n_paths

    def test_all_splits_load(self, bs_run):
        for split in ("train", "val", "test"):
            batch = load_dataset(bs_run, split=split)
            assert batch.paths_S.ndim == 2
            assert batch.paths_S.shape[1] == T1

    def test_train_is_largest_split(self, bs_run):
        sizes = load_split_sizes(bs_run)
        assert sizes["train"] > sizes["val"]
        assert sizes["train"] > sizes["test"]

    def test_split_sizes_sum_to_n_paths(self, bs_run):
        sizes = load_split_sizes(bs_run)
        assert sum(sizes.values()) == N_PATHS

    def test_split_proportions_approx(self, bs_run):
        sizes = load_split_sizes(bs_run)
        # Train should be roughly 70% ± 5%
        train_frac = sizes["train"] / N_PATHS
        assert 0.65 <= train_frac <= 0.75, f"Train fraction {train_frac:.2f} outside expected range"


# ---------------------------------------------------------------------------
# Dtype tests
# ---------------------------------------------------------------------------

class TestDtypes:

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_paths_S_float32(self, sim_fixture, request):
        batch = _load_train(request.getfixturevalue(sim_fixture))
        assert batch.paths_S.dtype == torch.float32

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_paths_v_float32(self, sim_fixture, request):
        batch = _load_train(request.getfixturevalue(sim_fixture))
        assert batch.paths_v.dtype == torch.float32

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_paths_t_float32(self, sim_fixture, request):
        batch = _load_train(request.getfixturevalue(sim_fixture))
        assert batch.paths_t.dtype == torch.float32

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_K_is_python_float(self, sim_fixture, request):
        batch = _load_train(request.getfixturevalue(sim_fixture))
        assert isinstance(batch.K, float), f"K should be float, got {type(batch.K)}"

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_T_mat_is_python_float(self, sim_fixture, request):
        batch = _load_train(request.getfixturevalue(sim_fixture))
        assert isinstance(batch.T_mat, float), f"T_mat should be float, got {type(batch.T_mat)}"

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_tensors_on_cpu(self, sim_fixture, request):
        batch = _load_train(request.getfixturevalue(sim_fixture))
        assert batch.paths_S.device.type == "cpu"
        assert batch.paths_v.device.type == "cpu"
        assert batch.paths_t.device.type == "cpu"


# ---------------------------------------------------------------------------
# Time grid tests
# ---------------------------------------------------------------------------

class TestTimeGrid:

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_t_starts_at_zero(self, sim_fixture, request):
        batch = _load_train(request.getfixturevalue(sim_fixture))
        t0 = batch.paths_t[:, 0]
        assert torch.allclose(t0, torch.zeros_like(t0), atol=1e-6), (
            f"paths_t[:,0] should be all 0.0, got max={t0.max().item():.2e}"
        )

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_t_ends_at_T_mat(self, sim_fixture, request):
        batch = _load_train(request.getfixturevalue(sim_fixture))
        tT = batch.paths_t[:, -1]
        T_mat = torch.full_like(tT, batch.T_mat)
        assert torch.allclose(tT, T_mat, atol=1e-4), (
            f"paths_t[:,-1] should be ≈ T_mat={batch.T_mat}, "
            f"got mean={tT.mean().item():.6f}"
        )

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_t_is_monotone_increasing(self, sim_fixture, request):
        batch = _load_train(request.getfixturevalue(sim_fixture))
        diffs = batch.paths_t[:, 1:] - batch.paths_t[:, :-1]
        assert (diffs >= 0).all(), "paths_t is not monotone increasing along time axis"

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_K_matches_metadata(self, sim_fixture, request):
        run_dir = request.getfixturevalue(sim_fixture)
        batch = _load_train(run_dir)
        assert batch.K == pytest.approx(batch.metadata["contract"]["strike"])

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_T_mat_matches_metadata(self, sim_fixture, request):
        run_dir = request.getfixturevalue(sim_fixture)
        batch = _load_train(run_dir)
        assert batch.T_mat == pytest.approx(batch.metadata["time_grid"]["maturity_years"])


# ---------------------------------------------------------------------------
# Variance / latent state tests
# ---------------------------------------------------------------------------

class TestVariance:

    def test_v_all_zeros_bs(self, bs_run):
        batch = _load_train(bs_run)
        assert torch.all(batch.paths_v == 0.0), (
            "BS dataset should have paths_v = 0.0 everywhere"
        )

    def test_v_all_zeros_nga(self, nga_run):
        batch = _load_train(nga_run)
        assert torch.all(batch.paths_v == 0.0), (
            "NGA dataset should have paths_v = 0.0 everywhere"
        )

    def test_v_nonzero_heston(self, heston_run):
        batch = _load_train(heston_run)
        assert batch.paths_v.mean().item() > 0.0, (
            "Heston dataset should have paths_v > 0 on average"
        )

    def test_v_nonnegative_heston(self, heston_run):
        batch = _load_train(heston_run)
        assert torch.all(batch.paths_v >= 0.0), (
            "Heston variance paths_v should be non-negative (full-truncation scheme)"
        )

    def test_v_shape_matches_S_heston(self, heston_run):
        batch = _load_train(heston_run)
        assert batch.paths_v.shape == batch.paths_S.shape


# ---------------------------------------------------------------------------
# Fallback behaviour (no latent_state/ folder)
# ---------------------------------------------------------------------------

class TestLatentStateFallback:

    def test_fallback_no_error(self, tmp_path, bs_run):
        """Loader should not raise when latent_state/ is absent."""
        # Copy the bs run to a temp dir and delete latent_state/
        run_copy = tmp_path / "bs_no_latent"
        shutil.copytree(bs_run, run_copy)
        shutil.rmtree(run_copy / "latent_state", ignore_errors=True)

        # Should load cleanly
        batch = load_dataset(run_copy, split="train")
        assert batch.paths_S.shape[1] == T1

    def test_fallback_v_is_zero(self, tmp_path, bs_run):
        """When latent_state/ is absent, paths_v should be all zeros."""
        run_copy = tmp_path / "bs_no_latent_v"
        shutil.copytree(bs_run, run_copy)
        shutil.rmtree(run_copy / "latent_state", ignore_errors=True)

        batch = load_dataset(run_copy, split="train")
        assert torch.all(batch.paths_v == 0.0)

    def test_fallback_heston_emits_warning(self, tmp_path, heston_run):
        """When latent_state/ is absent on a Heston run, a UserWarning is emitted."""
        run_copy = tmp_path / "heston_no_latent"
        shutil.copytree(heston_run, run_copy)
        shutil.rmtree(run_copy / "latent_state", ignore_errors=True)

        with pytest.warns(UserWarning, match="latent_state"):
            load_dataset(run_copy, split="train")

    def test_fallback_bs_no_warning(self, tmp_path, bs_run):
        """BS fallback (v=0) should NOT emit a warning — zeros are expected."""
        run_copy = tmp_path / "bs_no_latent_warn"
        shutil.copytree(bs_run, run_copy)
        shutil.rmtree(run_copy / "latent_state", ignore_errors=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error")   # any warning → fail
            load_dataset(run_copy, split="train")


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrors:

    def test_invalid_split_raises(self, bs_run):
        with pytest.raises(ValueError, match="Invalid split"):
            load_dataset(bs_run, split="bad")

    def test_missing_run_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "does_not_exist", split="train")

    def test_missing_observations_file_raises(self, tmp_path, bs_run):
        run_copy = tmp_path / "bs_missing_obs"
        shutil.copytree(bs_run, run_copy)
        (run_copy / "observations" / "train" / "part-00000.parquet").unlink()

        with pytest.raises(FileNotFoundError):
            load_dataset(run_copy, split="train")

    def test_invalid_split_sizes_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_split_sizes(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# DatasetBatch immutability
# ---------------------------------------------------------------------------

class TestBatchImmutability:

    def test_frozen_prevents_attribute_set(self, bs_run):
        """DatasetBatch is frozen — direct attribute assignment must raise."""
        from dataclasses import FrozenInstanceError
        batch = _load_train(bs_run)
        with pytest.raises(FrozenInstanceError):
            batch.K = 999.0  # type: ignore[misc]

    def test_metadata_is_dict(self, bs_run):
        batch = _load_train(bs_run)
        assert isinstance(batch.metadata, dict)
        assert "simulator" in batch.metadata

    def test_metadata_contains_n_steps(self, bs_run):
        batch = _load_train(bs_run)
        assert batch.metadata["time_grid"]["n_steps"] == N_STEPS


# ---------------------------------------------------------------------------
# load_split_sizes
# ---------------------------------------------------------------------------

class TestLoadSplitSizes:

    def test_returns_three_keys(self, bs_run):
        sizes = load_split_sizes(bs_run)
        assert set(sizes.keys()) == {"train", "val", "test"}

    def test_all_values_positive(self, bs_run):
        sizes = load_split_sizes(bs_run)
        for k, v in sizes.items():
            assert v > 0, f"Split '{k}' has zero paths"

    def test_sum_equals_n_paths(self, bs_run):
        sizes = load_split_sizes(bs_run)
        assert sum(sizes.values()) == N_PATHS

    def test_consistent_with_load_dataset(self, bs_run):
        """load_split_sizes and load_dataset must agree on N for each split."""
        sizes = load_split_sizes(bs_run)
        for split in ("train", "val", "test"):
            batch = load_dataset(bs_run, split=split)
            assert batch.n_paths == sizes[split], (
                f"Mismatch for split '{split}': "
                f"load_split_sizes={sizes[split]}, "
                f"load_dataset.n_paths={batch.n_paths}"
            )

    @pytest.mark.parametrize("sim_fixture", ["bs_run", "heston_run", "nga_run"])
    def test_works_for_all_sims(self, sim_fixture, request):
        run_dir = request.getfixturevalue(sim_fixture)
        sizes = load_split_sizes(run_dir)
        assert sum(sizes.values()) == N_PATHS
