"""Tests for the schema types and columns pipeline component."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pyarrow as pa
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


@pytest.mark.parametrize("sim", ["bs", "heston", "nga"])
def test_schema_types_and_columns(sim: str, tmp_path: Path) -> None:
    """Assert each simulator writes exactly v1.0 observation columns/types and no hive split folders."""
    out_root = tmp_path / "datasets" / "v1.0"
    run_dir = _run_generate(sim=sim, out_root=out_root, run_id=f"schema-{sim}")

    expected_names = ["path_id", "t_idx", "t_years", "S"]
    expected_types = [pa.int64(), pa.int32(), pa.float32(), pa.float32()]

    for split in ("train", "val", "test"):
        part_path = run_dir / "observations" / split / "part-00000.parquet"
        parquet_file = pq.ParquetFile(part_path)
        schema = parquet_file.schema_arrow
        names = schema.names
        types = [field.type for field in schema]

        assert names == expected_names
        assert types == expected_types
        assert "split" not in names

    hive_dirs = list(run_dir.rglob("split=*"))
    assert not hive_dirs
