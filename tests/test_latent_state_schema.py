"""Tests for the latent state schema pipeline component."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.schema.v1_0 import LATENT_STATE_SCHEMA


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


@pytest.mark.parametrize("sim", ["bs", "heston", "nga"])
def test_latent_state_layout_schema_and_alignment(sim: str, tmp_path: Path) -> None:
    """Assert latent state layout schema and alignment."""
    out_root = tmp_path / "datasets" / "v1.0"
    run_dir = _run_generate(sim=sim, out_root=out_root, run_id=f"latent-schema-{sim}")

    expected_names = [field.name for field in LATENT_STATE_SCHEMA]
    expected_types = [field.type for field in LATENT_STATE_SCHEMA]

    for split in ("train", "val", "test"):
        obs_path = run_dir / "observations" / split / "part-00000.parquet"
        latent_path = run_dir / "latent_state" / split / "part-00000.parquet"

        assert latent_path.exists()
        assert obs_path.exists()

        latent_table = pq.read_table(latent_path)
        obs_table = pq.read_table(obs_path)

        assert latent_table.column_names == expected_names
        assert [field.type for field in latent_table.schema] == expected_types

        latent_df = latent_table.to_pandas()
        obs_df = obs_table.to_pandas()

        assert len(latent_df) == len(obs_df)

        pd.testing.assert_frame_equal(
            latent_df.loc[:, ["path_id", "t_idx"]].reset_index(drop=True),
            obs_df.loc[:, ["path_id", "t_idx"]].reset_index(drop=True),
            check_dtype=True,
        )
