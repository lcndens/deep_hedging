"""Tests for the splits and counts pipeline component."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _assert_monotone_per_path(df: pd.DataFrame, col: str) -> None:
    """Helper for assert monotone per path."""
    for _, sub in df.groupby("path_id", sort=False):
        values = sub.sort_values("t_idx")[col].to_numpy()
        assert (values[1:] >= values[:-1]).all()


@pytest.mark.parametrize("sim", ["bs", "heston", "nga"])
def test_splits_and_counts(sim: str, tmp_path: Path) -> None:
    """Validate 70/15/15 split sizes, row counts, and per-path monotone time coordinates."""
    n_paths = 200
    n_steps = 6
    expected_paths = {"train": 140, "val": 30, "test": 30}

    out_root = tmp_path / "datasets" / "v1.0"
    run_dir = _run_generate(
        sim=sim,
        out_root=out_root,
        run_id=f"splits-{sim}",
        n_paths=n_paths,
        n_steps=n_steps,
    )

    all_ids: set[int] = set()
    for split, n_split_paths in expected_paths.items():
        part_path = run_dir / "observations" / split / "part-00000.parquet"
        df = pq.read_table(part_path).to_pandas()

        assert len(df) == n_split_paths * (n_steps + 1)
        split_ids = set(df["path_id"].unique().tolist())
        assert len(split_ids) == n_split_paths
        assert split_ids.isdisjoint(all_ids)
        all_ids |= split_ids

        assert int(df["t_idx"].min()) == 0
        assert int(df["t_idx"].max()) == n_steps
        _assert_monotone_per_path(df, "t_idx")
        _assert_monotone_per_path(df, "t_years")

    assert len(all_ids) == n_paths
