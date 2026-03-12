"""Tests for the plotting outputs pipeline component."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
matplotlib = pytest.importorskip("matplotlib")
assert matplotlib is not None


def _run_generate(sim: str, out_root: Path, run_id: str, n_paths: int = 200, n_steps: int = 6) -> Path:
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


@pytest.mark.parametrize("sim", ["bs", "heston", "nga"])
def test_plotting_outputs(sim: str, tmp_path: Path) -> None:
    """Ensure module-mode plotting succeeds for all simulators and writes PNG under test split."""
    out_root = tmp_path / "datasets" / "v1.0"
    run_dir = _run_generate(sim=sim, out_root=out_root, run_id=f"plot-{sim}")
    out_name = f"sample_paths_{sim}_test.png"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.plot_observations",
            "--run_dir",
            str(run_dir),
            "--split",
            "test",
            "--n_paths_plot",
            "8",
            "--out_name",
            out_name,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    assert (run_dir / "observations" / "test" / out_name).exists()
