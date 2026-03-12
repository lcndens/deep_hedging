"""Tests for the parquet writer pipeline component."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.io.parquet_writer import write_parquet_part
from src.schema.v1_0 import OBS_SCHEMA


def test_write_parquet_part_enforces_schema_and_casts_types(tmp_path: Path) -> None:
    """Ensure strict parquet writing preserves canonical columns and Arrow-compatible dtypes."""
    df = pd.DataFrame(
        {
            "path_id": [1, 2],
            "t_idx": [0, 1],
            "t_years": [0.0, 0.1],
            "S": [100.0, 101.5],
        }
    )
    out_path = write_parquet_part(df=df, out_dir=tmp_path, schema=OBS_SCHEMA)
    table = pq.read_table(out_path)

    assert table.column_names == ["path_id", "t_idx", "t_years", "S"]
    assert [str(f.type) for f in table.schema] == ["int64", "int32", "float", "float"]


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"path_id": [1], "t_idx": [0], "t_years": [0.0], "S": [100.0], "v": [0.04]}),
        pd.DataFrame({"path_id": [1], "t_idx": [0], "S": [100.0]}),
    ],
)
def test_write_parquet_part_rejects_extra_or_missing_columns(tmp_path: Path, df: pd.DataFrame) -> None:
    """Ensure parquet writes fail fast when dataframe columns drift from the schema contract."""
    with pytest.raises(ValueError, match="Column mismatch"):
        write_parquet_part(df=df, out_dir=tmp_path, schema=OBS_SCHEMA)
