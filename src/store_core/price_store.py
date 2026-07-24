"""The canonical price store: hive-partitioned Parquet OHLCV.

One owner of the on-disk layout, the merge semantics, and the read contract.

- Layout: ``<base_dir>/exchange=<XX>/<TICKER>.parquet`` (hive-partitioned on
  ``exchange``), one file per symbol, sorted by date, append-only via merge.
- Reads are Polars (lazy scan + predicate pushdown). Writes are pandas/pyarrow
  (read-merge-dedup + atomic temp-file rename).

``base_dir`` defaults to the ``PRICES_DIR`` env var (falling back to
``data/prices``); a consumer reading another app's store passes the producer's
prices directory explicitly.
"""
from __future__ import annotations

import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .types import OHLCV_FLOAT_COLUMNS

DateLike = Union[date, datetime, str]

_DEFAULT_BASE_DIR = "data/prices"
_MIN_DATE = "1900-01-01"


def _default_base_dir() -> str:
    return os.environ.get("PRICES_DIR", _DEFAULT_BASE_DIR)


def _to_date(value: Optional[DateLike], *, fallback: date) -> date:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _dot_dash_variants(symbols: Sequence[str]) -> tuple[List[str], dict[str, str]]:
    """Expand a symbol list to include dot↔dash variants for Parquet lookup.

    EODHD stores class-share tickers with dashes (PBR-A) while user-facing
    identifiers may use dots (PBR.A). Returns the expanded list plus a mapping
    from every variant back to the originally requested symbol so callers can
    normalise the returned ``symbol`` column.
    """
    expanded: List[str] = []
    variant_to_canonical: dict[str, str] = {}
    for s in symbols:
        expanded.append(s)
        variant_to_canonical[s] = s
        if "." in s:
            alt = s.replace(".", "-")
            expanded.append(alt)
            variant_to_canonical[alt] = s
        elif "-" in s:
            alt = s.replace("-", ".")
            expanded.append(alt)
            variant_to_canonical[alt] = s
    return expanded, variant_to_canonical


class PriceStore:
    """Read/write access to the hive-partitioned Parquet OHLCV store."""

    def __init__(self, base_dir: Optional[Union[str, os.PathLike]] = None):
        self.base_dir = Path(base_dir) if base_dir is not None else Path(_default_base_dir())

    # ---- paths -----------------------------------------------------------

    def path(self, exchange: str, ticker: str) -> Path:
        """Hive path for one symbol's file: ``exchange=<XX>/<TICKER>.parquet``."""
        return self.base_dir / f"exchange={exchange}" / f"{ticker.upper()}.parquet"

    # ---- reads (Polars) --------------------------------------------------

    def read(
        self,
        symbols: Optional[Sequence[str]] = None,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        exchanges: Optional[Sequence[str]] = None,
        min_data_points: int = 0,
    ) -> pl.DataFrame:
        """Read OHLCV as a Polars DataFrame ``[date, open..volume, symbol, exchange]``.

        Filters by symbol (dot/dash-variant aware), inclusive date range, and
        optionally ``exchanges`` (the hive partition key — pass e.g. ``["US"]``
        to disambiguate a ticker dual-listed on a foreign exchange). Drops
        symbols with fewer than ``min_data_points`` rows when > 0.

        Returns an empty DataFrame on any error or when nothing matches — a
        missing store is a normal "no data" case, not an exception.
        """
        start = _to_date(start_date, fallback=_to_date(_MIN_DATE, fallback=date(1900, 1, 1)))
        end = _to_date(end_date, fallback=date.today())

        glob = str(self.base_dir / "**" / "*.parquet")
        start_pl = pl.lit(start)
        end_pl = pl.lit(end)

        variant_to_canonical: dict[str, str] = {}
        try:
            lf = pl.scan_parquet(glob, hive_partitioning=True)
            lf = lf.filter((pl.col("date") >= start_pl) & (pl.col("date") <= end_pl))
            if symbols is not None:
                expanded, variant_to_canonical = _dot_dash_variants(list(symbols))
                lf = lf.filter(pl.col("symbol").is_in(expanded))
            if exchanges:
                lf = lf.filter(pl.col("exchange").is_in([e.strip().upper() for e in exchanges]))
            df = lf.collect()
        except Exception:
            return pl.DataFrame()

        if df.is_empty():
            return pl.DataFrame()

        if variant_to_canonical:
            df = df.with_columns(pl.col("symbol").replace(variant_to_canonical))

        if min_data_points > 0 and "symbol" in df.columns:
            counts = df.group_by("symbol").len().rename({"len": "_cnt"})
            keep = counts.filter(pl.col("_cnt") >= min_data_points).select("symbol")
            df = df.join(keep, on="symbol", how="inner")

        return df

    # ---- writes (pandas / pyarrow) --------------------------------------

    def upsert(self, df_new: pd.DataFrame, exchange: str, ticker: str) -> int:
        """Merge ``df_new`` into the file for ``(exchange, ticker)``. Idempotent
        (dedups on ``date``, keeping the last). Returns the count of net-new rows."""
        return self.upsert_to_path(df_new, self.path(exchange, ticker))

    def upsert_to_path(self, df_new: pd.DataFrame, path: Union[str, os.PathLike]) -> int:
        """Lower-level upsert to an explicit path (for callers that key files
        differently, e.g. the intraday panel). Prefer :meth:`upsert`."""
        path = Path(path)
        if df_new is None or df_new.empty:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            df_existing = pd.read_parquet(path)
            df_existing["date"] = pd.to_datetime(df_existing["date"]).dt.date
            existing_count = len(df_existing)
            df_combined = (
                pd.concat([df_existing, df_new], ignore_index=True)
                .drop_duplicates(subset="date", keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
        else:
            df_combined = df_new.sort_values("date").reset_index(drop=True)
            existing_count = 0
        for col in OHLCV_FLOAT_COLUMNS:
            if col in df_combined.columns:
                df_combined[col] = df_combined[col].astype("float64")
        # Atomic write: a crash or concurrent reader must never see a half-written
        # file. Write to a temp file in the same directory, then rename into place.
        fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=str(path.parent))
        os.close(fd)
        try:
            pq.write_table(
                pa.Table.from_pandas(df_combined, preserve_index=False),
                tmp_path,
                compression="snappy",
            )
            os.replace(tmp_path, path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return len(df_combined) - existing_count
