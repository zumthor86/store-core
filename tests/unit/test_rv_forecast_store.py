"""Invariants for the versioned realized-volatility forecast store.

Hermetic: every test builds its own temporary tree, never the real store.

The load-bearing ones are the silent-corruption guards — a fingerprint mismatch being rejected
rather than written, a reader never seeing two methodologies at once, and a write refused while a
swap stages. Each of those failures produces data that still looks like forecasts.
"""
from datetime import date, timedelta

import polars as pl
import pytest

from store_core import (
    MethodologyMismatchError,
    RegenerationInProgressError,
    RvForecastStore,
)

M1 = "log-har-rs-j-lb1260-mh756-aaaaaaaaaa"
M2 = "log-har-rs-j-lb756-mh756-bbbbbbbbbb"


def _rows(symbols, methodology=M1, source="backfill", rv=0.20, pred=0.21):
    return pl.DataFrame({
        "symbol": list(symbols),
        "measures_date": [date(2024, 1, 2)] * len(symbols),
        "rv_21d": [rv] * len(symbols),
        "predicted_rv_21d": [pred] * len(symbols),
        "predicted_rv_63d": [pred] * len(symbols),
        "methodology_id": [methodology] * len(symbols),
        "source": [source] * len(symbols),
    })


@pytest.fixture()
def store(tmp_path):
    return RvForecastStore(base_dir=tmp_path / "rv_forecast")


# --- basics ----------------------------------------------------------------------------


def test_empty_store_reads_empty_and_reports_no_methodology(store):
    assert store.active_methodology() is None
    assert store.latest_date() is None
    assert store.read().is_empty()


def test_round_trip(store):
    store.write_date_partition(_rows(["AAPL", "MSFT"]), date(2024, 1, 2), M1)
    df = store.read()
    assert df.height == 2
    assert df["symbol"].to_list() == ["AAPL", "MSFT"]
    assert df["date"].to_list() == [date(2024, 1, 2)] * 2
    assert store.active_methodology() == M1
    assert store.latest_date() == date(2024, 1, 2)


def test_first_write_establishes_the_active_methodology(store):
    assert store.active_methodology() is None
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    assert store.active_methodology() == M1


def test_read_filters_symbols_and_dates(store):
    for i, d in enumerate([date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]):
        store.write_date_partition(_rows(["AAPL", "MSFT"]), d, M1)
    df = store.read(symbols=["AAPL"], start_date=date(2024, 1, 3), end_date=date(2024, 1, 3))
    assert df.height == 1 and df["symbol"][0] == "AAPL" and df["date"][0] == date(2024, 1, 3)


# --- null semantics: three distinguishable states (FR-004) ------------------------------


def test_null_forecast_rows_survive_a_round_trip_and_are_not_filtered(store):
    """A null forecast means 'no forecast for this symbol-date'. Dropping it on read would
    collapse it into 'never processed', and would also discard the row a consumer falls back to
    rv_21d on."""
    df = _rows(["SHORTHIST"]).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("predicted_rv_21d"),
        pl.lit(None, dtype=pl.Float64).alias("predicted_rv_63d"),
    )
    store.write_date_partition(df, date(2024, 1, 2), M1)
    out = store.read()
    assert out.height == 1
    assert out["predicted_rv_21d"][0] is None
    assert out["rv_21d"][0] == pytest.approx(0.20)   # the fallback input is still there


def test_absent_partition_versus_present_row_with_null(store):
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    assert date(2024, 1, 2) in store.covered_dates()      # processed
    assert date(2024, 1, 3) not in store.covered_dates()  # never processed


# --- drift cannot be persisted (FR-012) -------------------------------------------------


def test_write_with_a_mismatched_methodology_is_rejected(store):
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    with pytest.raises(MethodologyMismatchError):
        store.write_date_partition(_rows(["MSFT"], methodology=M2), date(2024, 1, 3), M2)


def test_frame_stamped_differently_from_the_target_is_rejected(store):
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    mixed = _rows(["MSFT"], methodology=M2)
    with pytest.raises(MethodologyMismatchError):
        store.write_date_partition(mixed, date(2024, 1, 3), M1)


# --- idempotency and merge-on-symbol (FR-011, FR-024/026) -------------------------------


def test_rewriting_the_same_date_changes_nothing(store):
    store.write_date_partition(_rows(["AAPL", "MSFT"]), date(2024, 1, 2), M1)
    before = store.read()
    store.write_date_partition(_rows(["AAPL", "MSFT"]), date(2024, 1, 2), M1)
    assert store.read().equals(before)


def test_complement_write_merges_without_clobbering_the_first(store):
    """The daily run writes the screen's frame, then the complement of symbols it does not rank.
    The second must add to the first, not replace the partition."""
    store.write_date_partition(_rows(["AAPL"], source="screen"), date(2024, 1, 2), M1)
    store.write_date_partition(_rows(["SPY", "QQQ"], source="complement"), date(2024, 1, 2), M1)
    df = store.read()
    assert sorted(df["symbol"].to_list()) == ["AAPL", "QQQ", "SPY"]
    assert set(df["source"].to_list()) == {"screen", "complement"}


# --- regeneration: readers never see a mixture (FR-015, FR-016) -------------------------


def test_read_during_staging_still_sees_only_the_old_methodology(store):
    store.write_date_partition(_rows(["AAPL"], pred=0.21), date(2024, 1, 2), M1)
    store.begin_regeneration(M2)
    store.write_date_partition(_rows(["AAPL"], methodology=M2, pred=0.99), date(2024, 1, 2), M2)

    df = store.read()
    assert store.active_methodology() == M1
    assert df["methodology_id"].unique().to_list() == [M1]
    assert df["predicted_rv_21d"][0] == pytest.approx(0.21)   # not the staged 0.99


def test_promote_swaps_atomically_and_removes_the_superseded_history(store):
    store.write_date_partition(_rows(["AAPL"], pred=0.21), date(2024, 1, 2), M1)
    store.begin_regeneration(M2)
    store.write_date_partition(_rows(["AAPL"], methodology=M2, pred=0.99), date(2024, 1, 2), M2)
    store.promote(M2)

    df = store.read()
    assert store.active_methodology() == M2
    assert df["methodology_id"].unique().to_list() == [M2]
    assert df["predicted_rv_21d"][0] == pytest.approx(0.99)
    assert not (store.base_dir / M1).exists()
    assert store.regeneration_in_progress() is None


def test_interrupted_regeneration_leaves_the_store_untouched(store):
    store.write_date_partition(_rows(["AAPL"], pred=0.21), date(2024, 1, 2), M1)
    before = store.read()
    store.begin_regeneration(M2)
    store.write_date_partition(_rows(["AAPL"], methodology=M2, pred=0.99), date(2024, 1, 2), M2)
    store.discard_regeneration(M2)   # crash / abort

    assert store.active_methodology() == M1
    assert store.read().equals(before)
    assert not (store.base_dir / M2).exists()
    assert store.regeneration_in_progress() is None


def test_active_methodology_is_determinable_at_every_point(store):
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    assert store.active_methodology() == M1
    store.begin_regeneration(M2)
    assert store.active_methodology() == M1        # still, mid-staging
    store.promote(M2)
    assert store.active_methodology() == M2


def test_cannot_discard_the_active_methodology(store):
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    with pytest.raises(ValueError):
        store.discard_regeneration(M1)


def test_two_concurrent_regenerations_are_refused(store):
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    store.begin_regeneration(M2)
    with pytest.raises(RegenerationInProgressError):
        store.begin_regeneration("some-third-methodology")


# --- the lock refuses live appends (FR-034) ---------------------------------------------


def test_daily_append_is_refused_while_a_regeneration_stages(store):
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    store.begin_regeneration(M2)
    with pytest.raises(RegenerationInProgressError):
        store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 3), M1)


def test_refusal_reports_the_lock_so_the_skip_can_be_explained(store):
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    store.begin_regeneration(M2)
    lock = store.regeneration_in_progress()
    assert lock["target"] == M2 and lock["started_at"]


def test_lock_does_not_self_expire(store):
    """A crashed regeneration must stall the store visibly rather than resume appends into a
    directory it may still be writing. Clearing is deliberate, never time-based."""
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    store.begin_regeneration(M2)
    # Backdate the lock well past any plausible expiry window.
    lock_path = store.base_dir / "REGENERATING"
    lock_path.write_text(
        '{"target": "%s", "started_at": "2020-01-01T00:00:00"}' % M2, encoding="utf-8"
    )
    assert store.regeneration_in_progress() is not None
    with pytest.raises(RegenerationInProgressError):
        store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 3), M1)


def test_appends_resume_after_the_lock_clears(store):
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    store.begin_regeneration(M2)
    store.discard_regeneration(M2)
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 3), M1)
    assert store.latest_date() == date(2024, 1, 3)


# --- restatement: bounded, same-methodology rewrite (FR-032) -----------------------------


def _seed_span(store, symbol, days, pred):
    for i in range(days):
        d = date(2024, 1, 1) + timedelta(days=i)
        store.write_date_partition(_rows([symbol, "OTHER"], pred=pred), d, M1)


def test_rewrite_span_replaces_only_the_given_span(store):
    _seed_span(store, "AAPL", 6, pred=0.21)
    new = pl.concat([
        _rows(["AAPL"], pred=0.55).with_columns(pl.lit(date(2024, 1, 2)).alias("date")),
        _rows(["AAPL"], pred=0.55).with_columns(pl.lit(date(2024, 1, 3)).alias("date")),
    ])
    store.rewrite_span(new, "AAPL", date(2024, 1, 2), date(2024, 1, 3), M1)

    df = store.read(symbols=["AAPL"]).sort("date")
    by_date = dict(zip(df["date"].to_list(), df["predicted_rv_21d"].to_list()))
    assert by_date[date(2024, 1, 2)] == pytest.approx(0.55)
    assert by_date[date(2024, 1, 3)] == pytest.approx(0.55)
    assert by_date[date(2024, 1, 1)] == pytest.approx(0.21)   # before the span
    assert by_date[date(2024, 1, 4)] == pytest.approx(0.21)   # after it


def test_rewrite_span_leaves_other_symbols_untouched(store):
    _seed_span(store, "AAPL", 4, pred=0.21)
    new = _rows(["AAPL"], pred=0.55).with_columns(pl.lit(date(2024, 1, 2)).alias("date"))
    store.rewrite_span(new, "AAPL", date(2024, 1, 2), date(2024, 1, 2), M1)
    other = store.read(symbols=["OTHER"])
    assert other["predicted_rv_21d"].to_list() == pytest.approx([0.21] * other.height)


def test_rewrite_span_rejects_a_different_methodology(store):
    """A data correction must not be a vehicle for slipping in a different model."""
    _seed_span(store, "AAPL", 2, pred=0.21)
    new = _rows(["AAPL"], methodology=M2).with_columns(pl.lit(date(2024, 1, 1)).alias("date"))
    with pytest.raises(MethodologyMismatchError):
        store.rewrite_span(new, "AAPL", date(2024, 1, 1), date(2024, 1, 1), M2)


def test_rewrite_span_refused_during_a_regeneration(store):
    _seed_span(store, "AAPL", 2, pred=0.21)
    store.begin_regeneration(M2)
    new = _rows(["AAPL"], pred=0.55).with_columns(pl.lit(date(2024, 1, 1)).alias("date"))
    with pytest.raises(RegenerationInProgressError):
        store.rewrite_span(new, "AAPL", date(2024, 1, 1), date(2024, 1, 1), M1)


# --- read-side assertions and coverage --------------------------------------------------


def test_methodology_assertion_raises_when_the_history_moved(store):
    """Lets a long-running backtest notice the ground shifted, without putting version selection
    into the read path."""
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    assert not store.read(methodology_id=M1).is_empty()
    store.begin_regeneration(M2)
    store.write_date_partition(_rows(["AAPL"], methodology=M2), date(2024, 1, 2), M2)
    store.promote(M2)
    with pytest.raises(MethodologyMismatchError):
        store.read(methodology_id=M1)


def test_read_staged_reads_a_methodology_that_is_not_active(store):
    """The promotion gate must score a candidate *before* it becomes active. Plain read() asserts
    the opposite and correctly refuses, so staged reads need their own operation."""
    store.write_date_partition(_rows(["AAPL"], pred=0.21), date(2024, 1, 2), M1)
    store.begin_regeneration(M2)
    store.write_date_partition(_rows(["AAPL"], methodology=M2, pred=0.99), date(2024, 1, 2), M2)

    staged = store.read_staged(M2)
    assert staged.height == 1
    assert staged["predicted_rv_21d"][0] == pytest.approx(0.99)
    assert staged["methodology_id"].unique().to_list() == [M2]
    # ...while the live read is entirely unaffected
    assert store.read()["predicted_rv_21d"][0] == pytest.approx(0.21)


def test_read_staged_does_not_weaken_the_read_assertion(store):
    """read(methodology_id=) must keep raising — that is how a long-running study learns the
    history moved underneath it."""
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    store.begin_regeneration(M2)
    store.write_date_partition(_rows(["AAPL"], methodology=M2), date(2024, 1, 2), M2)
    with pytest.raises(MethodologyMismatchError):
        store.read(methodology_id=M2)


def test_read_staged_rejects_an_unknown_methodology(store):
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    with pytest.raises(FileNotFoundError):
        store.read_staged("never-staged")


def test_read_latest_returns_one_row_per_symbol(store):
    store.write_date_partition(_rows(["AAPL", "MSFT"], pred=0.10), date(2024, 1, 2), M1)
    store.write_date_partition(_rows(["AAPL", "MSFT"], pred=0.30), date(2024, 1, 3), M1)
    latest = store.read_latest()
    assert latest.height == 2
    assert set(latest["date"].to_list()) == {date(2024, 1, 3)}
    assert latest["predicted_rv_21d"].to_list() == pytest.approx([0.30, 0.30])


def test_coverage_reports_span_and_forecast_count(store):
    store.write_date_partition(_rows(["AAPL"]), date(2024, 1, 2), M1)
    nulled = _rows(["AAPL"]).with_columns(pl.lit(None, dtype=pl.Float64).alias("predicted_rv_21d"))
    store.write_date_partition(nulled, date(2024, 1, 3), M1)
    cov = store.coverage()
    row = cov.row(0, named=True)
    assert row["records"] == 2
    assert row["forecasts"] == 1          # one date had no forecast
    assert row["first_date"] == date(2024, 1, 2)
    assert row["last_date"] == date(2024, 1, 3)
