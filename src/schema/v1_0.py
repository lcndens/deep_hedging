"""Arrow schemas for dataset format version 1.0."""

import pyarrow as pa

SCHEMA_VERSION = "v1.0"

# Canonical observations schema: market paths only (NO split column)
OBS_SCHEMA = pa.schema([
    ("path_id", pa.int64()),
    ("t_idx", pa.int32()),
    ("t_years", pa.float32()),
    ("S", pa.float32()),
])

# One contract per dataset for now
CONTRACTS_SCHEMA = pa.schema([
    ("contract_id", pa.int32()),
    ("type", pa.string()),
    ("strike", pa.float32()),
    ("maturity_years", pa.float32()),
])
