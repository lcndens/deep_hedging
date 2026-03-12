"""Tests for the latent state values pipeline component."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_generate(sim: str, out_root: Path, run_id: str, n_paths: int = 50, n_steps: int = 5) -> Path:
    """Helper for run generate."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.generate_dataset",
            "--sim",
            sim,
            "--out_root",
            str(out_root),
            "--run_id",
            run_id,
            "--n_paths",
            str(n_paths),
            "--n_steps",
            str(n_steps),
            "--seed",
            "1",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return out_root / sim / run_id


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_heston_latent_variance_is_positive(split: str, tmp_path: Path) -> None:
    """Assert heston latent variance is positive."""
    out_root = tmp_path / "datasets" / "v1.0"
    run_dir = _run_generate(sim="heston", out_root=out_root, run_id="latent-values-heston")

    latent = pq.read_table(run_dir / "latent_state" / split / "part-00000.parquet").to_pandas()
    assert (latent["v"] > 0.0).all()


@pytest.mark.parametrize("sim", ["bs", "nga"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_bs_and_nga_latent_variance_is_zero(sim: str, split: str, tmp_path: Path) -> None:
    """Assert bs and nga latent variance is zero."""
    out_root = tmp_path / "datasets" / "v1.0"
    run_dir = _run_generate(sim=sim, out_root=out_root, run_id=f"latent-values-{sim}")

    latent = pq.read_table(run_dir / "latent_state" / split / "part-00000.parquet").to_pandas()
    assert (latent["v"] == 0.0).all()
