"""Hermetic contract tests for PriceStore against a temp Parquet tree."""
from __future__ import annotations

from datetime import date

import pandas as pd
import polars as pl
import pytest

from store_core import PriceStore


def _rows(symbol, exchange, dates, close_start=100.0):
    return pd.DataFrame(
        {
            "date": [date.fromisoformat(d) for d in dates],
            "open": [close_start + i for i in range(len(dates))],
            "high": [close_start + i + 1 for i in range(len(dates))],
            "low": [close_start + i - 1 for i in range(len(dates))],
            "close": [close_start + i for i in range(len(dates))],
            "adjusted_close": [close_start + i for i in range(len(dates))],
            "volume": [1000 + i for i in range(len(dates))],
            "symbol": symbol,
            "exchange": exchange,
        }
    )


@pytest.fixture()
def store(tmp_path):
    return PriceStore(base_dir=tmp_path)


def test_upsert_then_read_roundtrip(store):
    n = store.upsert(_rows("AAPL", "US", ["2025-01-02", "2025-01-03"]), "US", "AAPL")
    assert n == 2
    df = store.read(["AAPL"])
    assert df.height == 2
    assert set(df.columns) >= {"date", "close", "symbol", "exchange"}
    assert df["symbol"].unique().to_list() == ["AAPL"]


def test_upsert_is_idempotent_and_merges_on_date(store):
    store.upsert(_rows("AAPL", "US", ["2025-01-02", "2025-01-03"]), "US", "AAPL")
    # Re-write overlapping + one new date: dedup on date keeps last, net-new = 1.
    net_new = store.upsert(
        _rows("AAPL", "US", ["2025-01-03", "2025-01-06"], close_start=200.0), "US", "AAPL"
    )
    assert net_new == 1
    df = store.read(["AAPL"]).sort("date")
    assert df.height == 3
    # The 2025-01-03 row is the later write (close_start=200.0 → first row close 200.0).
    row = df.filter(pl.col("date") == date(2025, 1, 3))
    assert row["close"].item() == 200.0


def test_date_range_filter_is_inclusive(store):
    store.upsert(_rows("AAPL", "US", ["2025-01-02", "2025-01-03", "2025-01-06"]), "US", "AAPL")
    df = store.read(["AAPL"], start_date=date(2025, 1, 3), end_date=date(2025, 1, 3))
    assert df.height == 1
    assert df["date"].item() == date(2025, 1, 3)


def test_dot_dash_variant_lookup(store):
    # Stored with a dash (EODHD convention); requested with a dot.
    store.upsert(_rows("PBR-A", "US", ["2025-01-02"]), "US", "PBR-A")
    df = store.read(["PBR.A"])
    assert df.height == 1
    # Returned symbol is normalised back to the requested form.
    assert df["symbol"].item() == "PBR.A"


def test_exchange_filter_disambiguates(store):
    store.upsert(_rows("TSEM", "US", ["2025-01-02"]), "US", "TSEM")
    store.upsert(_rows("TSEM", "TA", ["2025-01-02"]), "TA", "TSEM")
    both = store.read(["TSEM"])
    assert set(both["exchange"].unique().to_list()) == {"US", "TA"}
    only_us = store.read(["TSEM"], exchanges=["us"])  # case-insensitive
    assert only_us["exchange"].unique().to_list() == ["US"]


def test_min_data_points_drops_short_symbols(store):
    store.upsert(_rows("AAPL", "US", ["2025-01-02", "2025-01-03"]), "US", "AAPL")
    store.upsert(_rows("MSFT", "US", ["2025-01-02"]), "US", "MSFT")
    df = store.read(["AAPL", "MSFT"], min_data_points=2)
    assert df["symbol"].unique().to_list() == ["AAPL"]


def test_volume_written_as_float64(store):
    store.upsert(_rows("AAPL", "US", ["2025-01-02"]), "US", "AAPL")
    df = store.read(["AAPL"])
    assert df.schema["volume"] == pl.Float64


def test_missing_store_returns_empty_not_error(tmp_path):
    store = PriceStore(base_dir=tmp_path / "does_not_exist")
    df = store.read(["AAPL"])
    assert isinstance(df, pl.DataFrame)
    assert df.is_empty()


def test_empty_upsert_is_noop(store):
    assert store.upsert(pd.DataFrame(), "US", "AAPL") == 0
