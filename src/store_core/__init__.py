"""store-core: shared market-data storage contracts.

Owns the workspace's cross-consumed on-disk stores — the hive-partitioned Parquet OHLCV price
store, and the versioned point-in-time realized-volatility forecast history. Producers write
through these classes and consumers read through them, so the layout is a real contract rather
than a path each caller reconstructs for itself.

Storage only: no scheduling, no HTTP, and no domain logic.
"""
from __future__ import annotations

from .price_store import PriceStore
from .rv_forecast_store import (
    MethodologyMismatchError,
    RegenerationInProgressError,
    RvForecastStore,
)
from .types import (
    EOD_COLUMNS,
    OHLCV_FLOAT_COLUMNS,
    PRICE_COLUMNS,
    PRICE_PARQUET_SCHEMA,
    RV_FORECAST_COLUMNS,
    RV_FORECAST_HORIZONS,
    RV_FORECAST_PARQUET_SCHEMA,
    RV_FORECAST_VALUE_COLUMNS,
)

__all__ = [
    "PriceStore",
    "PRICE_COLUMNS",
    "OHLCV_FLOAT_COLUMNS",
    "EOD_COLUMNS",
    "PRICE_PARQUET_SCHEMA",
    "RvForecastStore",
    "RegenerationInProgressError",
    "MethodologyMismatchError",
    "RV_FORECAST_COLUMNS",
    "RV_FORECAST_VALUE_COLUMNS",
    "RV_FORECAST_HORIZONS",
    "RV_FORECAST_PARQUET_SCHEMA",
]
