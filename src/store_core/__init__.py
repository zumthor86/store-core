"""store-core: shared market-data storage contracts.

Currently the canonical hive-partitioned Parquet OHLCV price store. Room to
absorb the other cross-consumed read stores (surface features, VX futures) later.
"""
from __future__ import annotations

from .price_store import PriceStore
from .types import (
    EOD_COLUMNS,
    OHLCV_FLOAT_COLUMNS,
    PRICE_COLUMNS,
    PRICE_PARQUET_SCHEMA,
)

__all__ = [
    "PriceStore",
    "PRICE_COLUMNS",
    "OHLCV_FLOAT_COLUMNS",
    "EOD_COLUMNS",
    "PRICE_PARQUET_SCHEMA",
]
