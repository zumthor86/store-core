from datetime import date

import polars as pl
import pytest

from store_core import OptionQuoteReader


def _write(root, underlying, day, rows):
    folder = root / f"underlying_symbol={underlying}" / f"trade_date={day}"
    folder.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(folder / "data.parquet")


@pytest.fixture
def reader(tmp_path):
    _write(tmp_path, "SPY", "2026-10-28", {"symbol": ["A", "B"], "delta": [-0.1, -0.2]})
    _write(tmp_path, "SPY", "2026-10-29", {"symbol": ["A"], "delta": [-0.3]})
    (tmp_path / "underlying_symbol=SPY" / "trade_date=2026-10-30").mkdir()      # a day with no quotes has no file
    return OptionQuoteReader(tmp_path)


def test_it_reads_the_named_contracts_of_the_named_day(reader):
    assert reader.read("SPY", date(2026, 10, 28), ["B"], ["symbol", "delta"]).to_dicts() == [{"symbol": "B", "delta": -0.2}]


def test_a_day_or_contract_it_does_not_hold_is_empty(reader):
    assert reader.read("SPY", date(2026, 10, 30), ["A"], ["delta"]).is_empty()
    assert reader.read("QQQ", date(2026, 10, 28), ["A"], ["delta"]).is_empty()
    assert reader.read("SPY", date(2026, 10, 29), ["B"], ["delta"]).is_empty()


def test_dates_are_newest_first_and_leave_out_a_day_with_no_file(reader):
    assert reader.dates("SPY") == [date(2026, 10, 29), date(2026, 10, 28)]
    assert reader.dates("QQQ") == []


def test_an_unreadable_file_raises(tmp_path):
    folder = tmp_path / "underlying_symbol=SPY" / "trade_date=2026-10-29"
    folder.mkdir(parents=True)
    (folder / "data.parquet").write_bytes(b"not parquet")
    with pytest.raises(Exception):
        OptionQuoteReader(tmp_path).read("SPY", date(2026, 10, 29), ["A"], ["delta"])
