"""Hermetic contract tests for BarStore against a temp Parquet tree."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from store_core import BAR_COLUMNS, BarStore


def _bars(first: datetime, n: int, price: float = 100.0, futures: bool = True) -> pl.DataFrame:
    starts = [first + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame({
        "start": starts,
        "session": [(s + timedelta(hours=6)).date() for s in starts],
        "open": [price + i for i in range(n)],
        "high": [price + i + 1 for i in range(n)],
        "low": [price + i - 1 for i in range(n)],
        "close": [price + i + 0.5 for i in range(n)],
        "volume": [10 * (i + 1) for i in range(n)],          # integers on the way in, float64 in the store
        "factor": [1.25] * n,
        "instrument_id": [42] * n if futures else [None] * n,
        "despiked": [False] * n if futures else [None] * n,
    })


@pytest.fixture()
def store(tmp_path):
    return BarStore(base_dir=tmp_path)


def test_replace_then_read_roundtrip(store):
    df = _bars(datetime(2025, 12, 31, 23, 58), 5)           # spans a year boundary
    assert store.replace(df, "gc", "databento") == 5
    out = store.read("GC", "databento")
    assert out.columns == list(BAR_COLUMNS)
    assert out["volume"].dtype == pl.Float64
    assert out.select("start", "open", "close").equals(df.select("start", "open", "close"))
    years = sorted(p.parent.name for p in store.symbol_dir("GC", "databento").glob("year=*/data.parquet"))
    assert years == ["year=2025", "year=2026"]


def test_replace_drops_years_the_new_series_no_longer_has(store):
    store.replace(_bars(datetime(2024, 6, 3, 9, 30), 3), "ES", "databento")
    store.replace(_bars(datetime(2025, 6, 3, 9, 30), 3, price=200.0), "ES", "databento")
    out = store.read("ES", "databento")
    assert out.height == 3 and out["open"][0] == 200.0
    assert [p.parent.name for p in store.symbol_dir("ES", "databento").glob("year=*/data.parquet")] == ["year=2025"]
    assert not list(store.symbol_dir("ES", "databento").parent.glob("*.staging-*"))
    assert not list(store.symbol_dir("ES", "databento").parent.glob("*.old-*"))


def test_session_filter_is_inclusive(store):
    store.replace(_bars(datetime(2025, 3, 3, 17, 0), 120), "CL", "databento")   # 17:00-18:59: two sessions
    sessions = store.read("CL", "databento")["session"].unique().sort().to_list()
    assert sessions == [date(2025, 3, 3), date(2025, 3, 4)]
    assert store.read("CL", "databento", start=date(2025, 3, 4))["session"].unique().to_list() == [date(2025, 3, 4)]
    assert store.read("CL", "databento", end="2025-03-03")["session"].unique().to_list() == [date(2025, 3, 3)]


def test_stock_series_have_null_contract_fields(store):
    store.replace(_bars(datetime(2025, 1, 2, 9, 30), 3, futures=False), "AAPL", "eodhd")
    out = store.read("AAPL", "eodhd")
    assert out["instrument_id"].null_count() == 3 and out["despiked"].null_count() == 3


def test_missing_symbol_raises_rather_than_reading_empty(store):
    with pytest.raises(LookupError):
        store.read("NOPE", "databento")


def test_symbols_lists_series_and_ignores_a_swap_in_progress(store):
    store.replace(_bars(datetime(2025, 1, 2, 9, 30), 2), "GC", "databento")
    store.replace(_bars(datetime(2025, 1, 2, 9, 30), 2, futures=False), "BRK.B", "eodhd")
    (store.symbol_dir("ES", "databento").with_name("symbol=ES.staging-deadbeef")).mkdir(parents=True)
    assert store.symbols() == [("databento", "GC"), ("eodhd", "BRK.B")]
    assert store.symbols(source="eodhd") == [("eodhd", "BRK.B")]


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.with_columns(pl.lit(1).alias("S_date")), "unexpected"),
    (lambda d: d.drop("factor"), "missing"),
    (lambda d: pl.concat([d, d.head(1)]), "sorted"),
    (lambda d: pl.concat([d.head(2), d.slice(1, 1)]).sort("start"), "duplicate"),
    (lambda d: d.with_columns(pl.when(pl.int_range(pl.len()) == 1).then(None).otherwise(pl.col("close")).alias("close")), "null price"),
])
def test_bad_frames_are_rejected_and_leave_the_old_series(store, mutate, message):
    good = _bars(datetime(2025, 1, 2, 9, 30), 3)
    store.replace(good, "GC", "databento")
    with pytest.raises(ValueError, match=message):
        store.replace(mutate(_bars(datetime(2025, 1, 2, 9, 30), 3, price=500.0)), "GC", "databento")
    assert store.read("GC", "databento")["open"].to_list() == good["open"].cast(pl.Float64).to_list()


def _contracts(*ids: int) -> pl.DataFrame:
    return pl.DataFrame({
        "instrument_id": list(ids), "root": ["CL"] * len(ids), "raw_symbol": [f"CL{i}" for i in ids],
        "expiration": [date(2026, 10, 20)] * len(ids), "micro_raw_symbol": [f"MCL{i}" for i in ids],
        "micro_expiration": [date(2026, 10, 19)] * len(ids), "first_notice": [None] * len(ids),
        "cut_off": [date(2026, 10, 19)] * len(ids),
    })


def _rolls(*times: datetime, old: int = 41, new: int = 42) -> pl.DataFrame:
    return pl.DataFrame({
        "roll_at": list(times), "old_id": [old] * len(times), "new_id": [new] * len(times),
        "ratio": [1.01] * len(times), "matched_at": list(times), "method": ["same_hour"] * len(times),
    }).with_columns(pl.col("roll_at", "matched_at").dt.replace_time_zone("UTC"))


def test_contract_and_roll_tables_round_trip(store):
    store.replace(_bars(datetime(2026, 10, 16, 9, 30), 3), "CL", "databento",
                  contracts=_contracts(41, 42), rolls=_rolls(datetime(2026, 10, 16, 21, 0)))
    assert store.read_contracts("CL", "databento")["raw_symbol"].to_list() == ["CL41", "CL42"]
    assert store.read_rolls("CL", "databento")["new_id"].to_list() == [42]


def test_a_bar_contract_missing_from_the_table_is_refused(store):
    with pytest.raises(ValueError, match="no contract"):
        store.replace(_bars(datetime(2026, 10, 16, 9, 30), 3), "CL", "databento", contracts=_contracts(41))


def test_roll_times_must_increase(store):
    t = datetime(2026, 10, 16, 21, 0)
    with pytest.raises(ValueError, match="strictly increase"):
        store.replace(_bars(t, 3), "CL", "databento", contracts=_contracts(41, 42), rolls=_rolls(t, t))


def test_a_replace_without_tables_leaves_none_and_a_read_says_so(store):
    store.replace(_bars(datetime(2026, 10, 16, 9, 30), 3), "CL", "databento", contracts=_contracts(41, 42))
    store.replace(_bars(datetime(2026, 10, 16, 9, 30), 3), "CL", "databento")
    with pytest.raises(LookupError):
        store.read_contracts("CL", "databento")
    with pytest.raises(LookupError):
        store.read_rolls("CL", "databento")


def test_a_refused_table_keeps_the_previous_series(store):
    store.replace(_bars(datetime(2026, 10, 16, 9, 30), 3), "CL", "databento", contracts=_contracts(41, 42))
    with pytest.raises(ValueError):
        store.replace(_bars(datetime(2026, 10, 16, 9, 30), 5), "CL", "databento", contracts=_contracts(7))
    assert store.read("CL", "databento").height == 3
    assert store.read_contracts("CL", "databento").height == 2
