"""Canonical Arrow schemas for deep-hedging dataset format v1.0.

The schemas define the on-disk contract between simulation output and training
input: long-format observations, optional latent variance state, and a single
European contract descriptor.
"""

import pyarrow as pa

SCHEMA_VERSION = "v1.0"

# Canonical observations table with one row per (path_id, t_idx).
OBS_SCHEMA = pa.schema([
    ("path_id", pa.int64()),
    ("t_idx", pa.int32()),
    ("t_years", pa.float32()),
    ("S", pa.float32()),
])

LATENT_STATE_SCHEMA = pa.schema([
    ("path_id", pa.int64()),
    ("t_idx", pa.int32()),
    ("v", pa.float32()),
])

# Current pipeline stores exactly one contract row per dataset run.
CONTRACTS_SCHEMA = pa.schema([
    ("contract_id", pa.int32()),
    ("type", pa.string()),
    ("strike", pa.float32()),
    ("maturity_years", pa.float32()),
])
