from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.generate_dataset import main as generate_dataset_main


def _latest_run_dir(sim_root: Path) -> Path:
    runs = sorted(sim_root.glob("*"))
    assert runs, f"No runs found under {sim_root}"
    return runs[-1]


@pytest.mark.parametrize("sim,simulator_name", [("bs", "BS"), ("heston", "Heston")])
def test_generate_dataset_writes_expected_layout_and_schema(
    tmp_path: Path, sim: str, simulator_name: str
) -> None:
    """Validate CLI generation creates the expected folder layout and v1.0 observations schema."""
    out_root = tmp_path / "datasets" / "v1.0"
    n_paths = 20
    n_steps = 4
    generate_dataset_main(
        [
            "--sim",
            sim,
            "--out_root",
            str(out_root),
            "--n_paths",
            str(n_paths),
            "--n_steps",
            str(n_steps),
            "--seed",
            "123",
        ]
    )

    run_dir = _latest_run_dir(out_root / sim)
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "contracts.parquet").exists()
    assert (run_dir / "observations" / "train").exists()
    assert (run_dir / "observations" / "val").exists()
    assert (run_dir / "observations" / "test").exists()

    metadata = (run_dir / "metadata.json").read_text(encoding="utf-8")
    assert '"schema_version": "v1.0"' in metadata
    assert f'"simulator": "{simulator_name}"' in metadata

    expected_cols = ["path_id", "t_idx", "t_years", "S"]
    expected_types = ["int64", "int32", "float", "float"]
    expected_path_counts = {"train": 14, "val": 3, "test": 3}

    observed_path_ids = set()
    total_rows = 0
    for split in ("train", "val", "test"):
        table = pq.read_table(run_dir / "observations" / split / "part-00000.parquet")
        df = table.to_pandas()

        assert table.column_names == expected_cols
        assert [str(f.type) for f in table.schema] == expected_types
        assert set(df.columns) == set(expected_cols)
        assert "split" not in df.columns

        unique_paths = set(df["path_id"].unique().tolist())
        assert len(unique_paths) == expected_path_counts[split]
        assert observed_path_ids.isdisjoint(unique_paths)
        observed_path_ids |= unique_paths
        total_rows += len(df)

    assert len(observed_path_ids) == n_paths
    assert total_rows == n_paths * (n_steps + 1)
