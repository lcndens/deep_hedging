"""Parquet writing utilities with strict schema enforcement.

This module writes canonical dataset tables to disk while enforcing exact
column sets, ordering, and Arrow-compatible dtypes to prevent silent schema
drift between generation and loading stages.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _schema_columns(schema: pa.Schema) -> list[str]:
    """Return schema field names in declared order.

    Parameters
    ----------
    schema : pa.Schema
        Arrow schema definition.

    Returns
    -------
    list[str]
        Ordered field names.
    """
    return [field.name for field in schema]


def _cast_df_to_schema(df: pd.DataFrame, schema: pa.Schema) -> pd.DataFrame:
    """Cast DataFrame columns to match Arrow field dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing schema columns.
    schema : pa.Schema
        Target Arrow schema.

    Returns
    -------
    pd.DataFrame
        New dataframe with columns cast to schema-compatible pandas dtypes.

    Raises
    ------
    TypeError
        If the schema contains an unsupported Arrow field type.
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
    """Write one parquet part with strict schema checks.

    Parameters
    ----------
    df : pd.DataFrame
        Table to write. Columns must exactly match ``schema``.
    out_dir : Path
        Output directory. It is created if missing.
    schema : pa.Schema
        Arrow schema used for validation and conversion.
    filename : str, default="part-00000.parquet"
        Output file name within ``out_dir``.
    compression : str, default="zstd"
        Parquet compression codec.

    Returns
    -------
    Path
        Path to the written parquet file.

    Raises
    ------
    ValueError
        If dataframe columns differ from schema columns.
    TypeError
        If schema casting encounters an unsupported Arrow type.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = _schema_columns(schema)
    df_cols = list(df.columns)

    # Exact set equality prevents accidental additions or dropped fields.
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

    # Table columns must follow schema order before Arrow conversion.
    df2 = df.loc[:, cols]
    df2 = _cast_df_to_schema(df2, schema)

    table = pa.Table.from_pandas(df2, schema=schema, preserve_index=False)
    out_path = out_dir / filename
    pq.write_table(table, out_path, compression=compression)
    return out_path
