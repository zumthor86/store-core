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


def test_exchanges_for_lists_every_partition(store):
    store.upsert(_rows("TSEM", "US", ["2025-01-02"]), "US", "TSEM")
    store.upsert(_rows("TSEM", "TA", ["2025-01-02"]), "TA", "TSEM")
    assert set(store.exchanges_for("TSEM")) == {"US", "TA"}


def test_exchanges_for_single_partition_symbol(store):
    store.upsert(_rows("AAPL", "US", ["2025-01-02"]), "US", "AAPL")
    assert store.exchanges_for("AAPL") == ["US"]


def test_exchanges_for_unknown_symbol_is_empty(store):
    assert store.exchanges_for("NOPE") == []


def test_exchanges_for_on_missing_store_is_empty(tmp_path):
    store = PriceStore(base_dir=tmp_path / "does_not_exist")
    assert store.exchanges_for("AAPL") == []


def test_remove_deletes_the_partition_file(store):
    store.upsert(_rows("TSEM", "TA", ["2025-01-02"]), "TA", "TSEM")
    assert store.path("TA", "TSEM").exists()
    removed = store.remove("TA", "TSEM")
    assert removed is True
    assert not store.path("TA", "TSEM").exists()


def test_remove_leaves_other_partitions_for_the_same_symbol_untouched(store):
    store.upsert(_rows("TSEM", "US", ["2025-01-02"]), "US", "TSEM")
    store.upsert(_rows("TSEM", "TA", ["2025-01-02"]), "TA", "TSEM")
    store.remove("TA", "TSEM")
    assert store.exchanges_for("TSEM") == ["US"]
    df = store.read(["TSEM"])
    assert df["exchange"].unique().to_list() == ["US"]


def test_remove_is_idempotent_when_nothing_to_delete(store):
    assert store.remove("TA", "NOPE") is False
    # Calling it twice in a row (simulating a re-run after partial failure) is also safe.
    store.upsert(_rows("TSEM", "TA", ["2025-01-02"]), "TA", "TSEM")
    assert store.remove("TA", "TSEM") is True
    assert store.remove("TA", "TSEM") is False
