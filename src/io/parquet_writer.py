"""Strict Parquet writer utilities with schema enforcement."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _schema_columns(schema: pa.Schema) -> list[str]:
    """Return schema field names in declared order."""
    return [field.name for field in schema]


def _cast_df_to_schema(df: pd.DataFrame, schema: pa.Schema) -> pd.DataFrame:
    """
    Cast DataFrame columns to match an Arrow schema exactly.

    This helper only handles primitive numeric/string fields used by the
    project schemas and raises if an unsupported Arrow type is encountered.
    """
    df2 = df.copy()
    for field in schema:
        name = field.name
        t = field.type
        if pa.types.is_int64(t):
            df2[name] = df2[name].astype("int64")
        elif pa.types.is_int32(t):
            df2[name] = df2[name].astype("int32")
        elif pa.types.is_float32(t):
            df2[name] = df2[name].astype("float32")
        elif pa.types.is_float64(t):
            df2[name] = df2[name].astype("float64")
        elif pa.types.is_string(t):
            df2[name] = df2[name].astype("string")
        else:
            raise TypeError(f"Unsupported schema type for '{name}': {t}")
    return df2


def write_parquet_part(
    df: pd.DataFrame,
    out_dir: Path,
    schema: pa.Schema,
    filename: str = "part-00000.parquet",
    compression: str = "zstd",
) -> Path:
    """
    Write one parquet part with strict schema validation and dtype coercion.

    The write fails fast when columns are missing or extra so accidental schema
    drift is caught during dataset generation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = _schema_columns(schema)
    df_cols = list(df.columns)

    # Fail fast if any extra/missing columns exist
    if set(df_cols) != set(cols):
        extras = sorted(set(df_cols) - set(cols))
        missing = sorted(set(cols) - set(df_cols))
        raise ValueError(
            f"Column mismatch for Parquet write.\n"
            f"  expected: {cols}\n"
            f"  got:      {df_cols}\n"
            f"  extras:   {extras}\n"
            f"  missing:  {missing}\n"
        )

    # Reorder to schema order and cast types
    df2 = df.loc[:, cols]
    df2 = _cast_df_to_schema(df2, schema)

    table = pa.Table.from_pandas(df2, schema=schema, preserve_index=False)
    out_path = out_dir / filename
    pq.write_table(table, out_path, compression=compression)
    return out_path
