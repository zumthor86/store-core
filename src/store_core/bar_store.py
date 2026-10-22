"""The intraday bar store: hive-partitioned Parquet, one continuous adjusted series per symbol.

Layout: ``<base_dir>/freq=<1m|1h>/source=<databento|eodhd>/symbol=<SYM>/year=<YYYY>/data.parquet``. A futures series
may also hold ``contracts.parquet`` (the contract of each ``instrument_id``) and ``rolls.parquet`` (each splice) in the
symbol folder, written in the same swap as the bars.

Every series here is back-adjusted: a futures roll or a stock split rescales all the history before it,
so a producer always rebuilds a symbol's whole series and **replaces** it — there is no append path.
``replace`` swaps the whole symbol in with two directory renames, so a reader sees the old series or
the new one, never a mix of years from both.

``base_dir`` defaults to the ``BARS_DIR`` env var (falling back to ``data/bars``).
"""
from __future__ import annotations

import os
import shutil
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import polars as pl

from .types import BAR_COLUMNS, BAR_POLARS_SCHEMA, CONTRACT_POLARS_SCHEMA, ROLL_POLARS_SCHEMA

_DEFAULT_BASE_DIR = "data/bars"
DateLike = Union[date, datetime, str]


def _default_base_dir() -> str:
    return os.environ.get("BARS_DIR", _DEFAULT_BASE_DIR)


def _as_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _check_columns(frame: pl.DataFrame, allowed, what: str) -> None:
    """Raise unless the frame has exactly the allowed columns (an allow-list: a stray column is never persisted)."""
    missing = [c for c in allowed if c not in frame.columns]
    extra = [c for c in frame.columns if c not in allowed]
    if missing or extra:
        raise ValueError(f"{what} columns: missing {missing}, unexpected {extra}")


class BarStore:
    """Read/replace access to the intraday bar store."""

    def __init__(self, base_dir: Optional[Union[str, os.PathLike]] = None):
        self.base_dir = Path(base_dir) if base_dir is not None else Path(_default_base_dir())

    # ---- paths -----------------------------------------------------------

    def symbol_dir(self, symbol: str, source: str, freq: str = "1m") -> Path:
        return self.base_dir / f"freq={freq}" / f"source={source}" / f"symbol={symbol.upper()}"

    def symbols(self, source: Optional[str] = None, freq: str = "1m") -> List[Tuple[str, str]]:
        """``(source, symbol)`` for every series on disk at ``freq``, optionally one source only."""
        root = self.base_dir / f"freq={freq}"
        if not root.exists():
            return []
        out = []
        for d in sorted(root.glob("source=*/symbol=*")):
            if not d.is_dir() or ".staging-" in d.name or ".old-" in d.name:   # a swap in progress
                continue
            src = d.parent.name.split("=", 1)[1]
            if source is None or src == source:
                out.append((src, d.name.split("=", 1)[1]))
        return out

    # ---- reads -----------------------------------------------------------

    def scan(self, symbol: str, source: str, freq: str = "1m") -> pl.LazyFrame:
        """Lazy frame of one series in :data:`BAR_COLUMNS` order. Raises ``LookupError`` if absent —
        a mistyped symbol must not read as an empty series."""
        d = self.symbol_dir(symbol, source, freq)
        files = sorted(d.glob("year=*/data.parquet"))
        if not files:
            raise LookupError(f"no {freq} bars for {symbol!r} from {source!r} under {self.base_dir}")
        return pl.scan_parquet([str(f) for f in files]).select(BAR_COLUMNS)

    def read(self, symbol: str, source: str, freq: str = "1m", start: Optional[DateLike] = None,
             end: Optional[DateLike] = None, columns: Optional[Sequence[str]] = None) -> pl.DataFrame:
        """One series, sorted by ``start``. ``start``/``end`` are inclusive **session** dates."""
        lf = self.scan(symbol, source, freq)
        if start is not None:
            lf = lf.filter(pl.col("session") >= _as_date(start))
        if end is not None:
            lf = lf.filter(pl.col("session") <= _as_date(end))
        df = lf.collect().sort("start")
        return df if columns is None else df.select(list(columns))

    def read_contracts(self, symbol: str, source: str, freq: str = "1m") -> pl.DataFrame:
        """The series' contract table (:data:`CONTRACT_COLUMNS`). Raises ``LookupError`` when the series has none."""
        return self._read_table(symbol, source, freq, "contracts")

    def read_rolls(self, symbol: str, source: str, freq: str = "1m") -> pl.DataFrame:
        """The series' roll table (:data:`ROLL_COLUMNS`), oldest first. Raises ``LookupError`` when the series has none."""
        return self._read_table(symbol, source, freq, "rolls")

    def _read_table(self, symbol: str, source: str, freq: str, name: str) -> pl.DataFrame:
        f = self.symbol_dir(symbol, source, freq) / f"{name}.parquet"
        if not f.exists():
            raise LookupError(f"no {name} table for {symbol!r} from {source!r} at {freq}")
        return pl.read_parquet(f)

    # ---- writes ----------------------------------------------------------

    @staticmethod
    def validate(df: pl.DataFrame) -> pl.DataFrame:
        """Cast to the schema of record. Raises on a missing or extra column (an allow-list: a stray
        column must never be persisted), on duplicate or unsorted bar times, and on null prices."""
        _check_columns(df, BAR_COLUMNS, "bar frame")
        out = df.select([pl.col(c).cast(t) for c, t in BAR_POLARS_SCHEMA.items()])
        if out.height == 0:
            raise ValueError("empty bar frame")
        if sum(out[c].null_count() for c in ("start", "session")):
            raise ValueError("null start or session")
        if sum(out[c].null_count() for c in ("open", "high", "low", "close")):
            raise ValueError("null price")
        if not out["start"].is_sorted():
            raise ValueError("bars not sorted by start")
        if out["start"].n_unique() != out.height:
            raise ValueError("duplicate bar start times")
        return out

    @staticmethod
    def validate_tables(bars: pl.DataFrame, contracts: Optional[pl.DataFrame],
                        rolls: Optional[pl.DataFrame]) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """Cast the contract and roll tables to their schemas (allow-lists, like the bars) and check they fit the bars:
        every bar's ``instrument_id`` has a contract, roll times strictly increase, and every roll names known contracts."""
        def cast(frame: pl.DataFrame, schema: dict, what: str) -> pl.DataFrame:
            _check_columns(frame, schema, what)
            return frame.select([pl.col(c).cast(t) for c, t in schema.items()])

        c = r = None
        if contracts is not None:
            c = cast(contracts, CONTRACT_POLARS_SCHEMA, "contract table")
            if c["instrument_id"].n_unique() != c.height:
                raise ValueError("contract table: an instrument_id appears twice")
            unknown = set(bars["instrument_id"].drop_nulls().unique().to_list()) - set(c["instrument_id"].to_list())
            if unknown:
                raise ValueError(f"bars hold instrument_id(s) with no contract: {sorted(unknown)[:5]}")
        if rolls is not None:
            r = cast(rolls, ROLL_POLARS_SCHEMA, "roll table")
            if not r["roll_at"].is_sorted() or r["roll_at"].n_unique() != r.height:
                raise ValueError("roll table: roll_at must strictly increase")
            if c is not None:
                ids = set(c["instrument_id"].to_list())
                if not set(r["old_id"].to_list()) | set(r["new_id"].to_list()) <= ids:
                    raise ValueError("roll table: a roll names a contract not in the contract table")
        return c, r

    def replace(self, df: pl.DataFrame, symbol: str, source: str, freq: str = "1m", *,
                contracts: Optional[pl.DataFrame] = None, rolls: Optional[pl.DataFrame] = None) -> int:
        """Replace a symbol's whole series with ``df`` (and its contract and roll tables, when given). Returns the row
        count written. A replace without a table leaves the series with none: the tables always describe these bars."""
        bars = self.validate(df)
        contracts, rolls = self.validate_tables(bars, contracts, rolls)
        final = self.symbol_dir(symbol, source, freq)
        final.parent.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex[:8]
        staging = final.with_name(f"{final.name}.staging-{tag}")
        old = final.with_name(f"{final.name}.old-{tag}")
        try:
            for (yr,), part in bars.group_by(pl.col("start").dt.year(), maintain_order=True):
                ydir = staging / f"year={yr}"
                ydir.mkdir(parents=True)
                part.write_parquet(ydir / "data.parquet", compression="zstd", statistics=True)
            if contracts is not None:
                contracts.write_parquet(staging / "contracts.parquet")
            if rolls is not None:
                rolls.write_parquet(staging / "rolls.parquet")
            if final.exists():
                final.rename(old)
            staging.rename(final)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            if old.exists() and not final.exists():
                old.rename(final)                  # put the previous series back
            raise
        shutil.rmtree(old, ignore_errors=True)
        return bars.height
