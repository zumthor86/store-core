"""Schemas-of-record for the stores this library owns.

These are THE canonical schemas for the workspace. Both the write path (producer
ingest) and the read path (consumers) agree here, so a change to an on-disk
layout happens in exactly one place.
"""
from __future__ import annotations

import polars as pl
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


# --- realized-volatility forecast store ---------------------------------------------
#
# One record per symbol per trading date. The trading date is carried by the hive
# partition path (``date=YYYY-MM-DD``), not per row, so it is absent from the file
# schema below — same convention as the producer's measures store.

#: Physical column order within a forecast partition file.
RV_FORECAST_COLUMNS: tuple[str, ...] = (
    "symbol", "measures_date", "rv_21d", "predicted_rv_21d", "predicted_rv_63d",
    "methodology_id", "source",
)

#: Horizons stored, in sessions. Pairs with the 30-day and 90-day implied variance
#: rates on the consumer side.
RV_FORECAST_HORIZONS: tuple[int, ...] = (21, 63)

#: Forecast value columns — nullable by design. A null here means "no forecast for this
#: symbol-date" (insufficient history, or the model declined); it is NOT the same as an
#: absent row, which means the symbol had no measures, or an absent partition, which
#: means the date was never processed.
RV_FORECAST_VALUE_COLUMNS: tuple[str, ...] = (
    "rv_21d", "predicted_rv_21d", "predicted_rv_63d",
)

RV_FORECAST_PARQUET_SCHEMA: pa.Schema = pa.schema([
    pa.field("symbol",           pa.string()),
    # Date of the newest measures actually used. Normally the partition date; differs
    # when a symbol did not trade or ingest that session, which is how staleness stays
    # visible per row rather than being inferred.
    pa.field("measures_date",    pa.date32()),
    # Trailing 21-session annualized realized vol. Null below 21 observations — never a
    # partial average, because consumers use this as the read-time fallback when no
    # forecast was produced.
    pa.field("rv_21d",           pa.float64()),
    pa.field("predicted_rv_21d", pa.float64()),
    pa.field("predicted_rv_63d", pa.float64()),
    # Fingerprint of the methodology that produced the row. Enforced on write: a
    # mismatch against the target directory's methodology is rejected, which is what
    # makes configuration drift impossible to persist rather than merely discouraged.
    pa.field("methodology_id",   pa.string()),
    # Which path wrote the row: "screen", "complement", "backfill", or "restate".
    # Diagnostic only — it never affects a value.
    pa.field("source",           pa.string()),
])


# --- intraday bar store --------------------------------------------------------------
#
# One continuous, back-adjusted series per symbol (futures: ratio-spliced across rolls; stocks:
# split/dividend-adjusted). ``factor`` = adjusted / raw, so a raw contract or share price — the one
# costs and fills are charged on — is always ``price / factor``.

#: Physical column order within a bar partition file.
BAR_COLUMNS: tuple[str, ...] = (
    "start", "session", "open", "high", "low", "close", "volume", "factor", "instrument_id", "despiked",
)

BAR_POLARS_SCHEMA: dict = {
    # Bar OPEN time, New York wall clock, no time zone.
    "start": pl.Datetime("us"),
    # Trading session the bar belongs to. Futures: the date of start + 6h (an 18:00 ET open belongs to the
    # next day's session); stocks: the calendar date.
    "session": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "factor": pl.Float64,
    # Futures: the contract the bar came from (a change marks a roll). Null for stocks.
    "instrument_id": pl.UInt32,
    # True where the producer clipped an off-market print. Null where the producer does not de-spike.
    "despiked": pl.Boolean,
}
