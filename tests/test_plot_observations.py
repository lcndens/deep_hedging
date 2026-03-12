"""Tests for the plot observations pipeline component."""

from pathlib import Path

import pytest

from src.generate_dataset import main as generate_dataset_main


matplotlib = pytest.importorskip("matplotlib")
assert matplotlib is not None

from src.plot_observations import main as plot_observations_main


def _latest_run_dir(sim_root: Path) -> Path:
    """Helper for latest run dir."""
    runs = sorted(sim_root.glob("*"))
    assert runs, f"No runs found under {sim_root}"
    return runs[-1]


@pytest.mark.parametrize("sim", ["bs", "heston"])
def test_plot_observations_writes_png(tmp_path: Path, sim: str) -> None:
    """Verify plotting CLI writes a PNG to observations/<split>/ for generated datasets."""
    out_root = tmp_path / "datasets" / "v1.0"
    generate_dataset_main(
        [
            "--sim",
            sim,
            "--out_root",
            str(out_root),
            "--n_paths",
            "24",
            "--n_steps",
            "5",
            "--seed",
            "77",
        ]
    )
    run_dir = _latest_run_dir(out_root / sim)
    out_name = f"sample_paths_{sim}_test.png"

    plot_observations_main(
        [
            "--run_dir",
            str(run_dir),
            "--split",
            "test",
            "--n_paths_plot",
            "5",
            "--out_name",
            out_name,
        ]
    )
    assert (run_dir / "observations" / "test" / out_name).exists()
