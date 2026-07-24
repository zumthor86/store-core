"""Schema-of-record for the price store.

This is THE canonical price schema for the workspace. Both the write path
(producer ingest) and the read path (consumers) agree here, so a change to the
on-disk layout happens in exactly one place.
"""
from __future__ import annotations

import pyarrow as pa

# Physical column order of a per-symbol Parquet file. ``symbol`` and ``exchange``
# are carried in-file as well as being the hive partition key on ``exchange``.
PRICE_COLUMNS: tuple[str, ...] = (
    "date", "open", "high", "low", "close", "adjusted_close", "volume",
    "symbol", "exchange",
)

# The OHLCV numeric columns. Enforced to float64 on every write so readers get a
# stable physical type (a mixed int/float ``volume`` across files breaks a lazy
# scan's schema unification).
OHLCV_FLOAT_COLUMNS: tuple[str, ...] = (
    "open", "high", "low", "close", "adjusted_close", "volume",
)

# The vendor EOD payload columns (no symbol/exchange), for reference by writers.
EOD_COLUMNS: tuple[str, ...] = (
    "date", "open", "high", "low", "close", "adjusted_close", "volume",
)

PRICE_PARQUET_SCHEMA: pa.Schema = pa.schema([
    pa.field("date",           pa.date32()),
    pa.field("open",           pa.float64()),
    pa.field("high",           pa.float64()),
    pa.field("low",            pa.float64()),
    pa.field("close",          pa.float64()),
    pa.field("adjusted_close", pa.float64()),
    pa.field("volume",         pa.float64()),
    pa.field("symbol",         pa.string()),
    pa.field("exchange",       pa.string()),
])
