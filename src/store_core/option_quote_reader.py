"""Read-only access to Hephaestus's option quote store, for a consumer outside Hephaestus.

Hephaestus owns and writes the store (``processing/stores/optionquote_store.py``); its layout is
``underlying_symbol=<SYM>/trade_date=<YYYY-MM-DD>/data.parquet``. A partition with no file is a day with no quotes.
Contract symbols there are OCC with no padding: ``AAPL261030P00332500``.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import List, Optional, Sequence, Union

import polars as pl

_DEFAULT_BASE_DIR = "data/optionquote"
_DATE_PREFIX = "trade_date="


def _default_base_dir() -> str:
    return os.environ.get("OPTIONQUOTE_STORE", _DEFAULT_BASE_DIR)


class OptionQuoteReader:
    """The quotes of named contracts on one day. A store that cannot be read raises: it is not a day with no quotes."""

    def __init__(self, base_dir: Optional[Union[str, os.PathLike]] = None):
        self.base_dir = Path(base_dir) if base_dir is not None else Path(_default_base_dir())

    def _path(self, underlying: str, trade_date: date) -> Path:
        return self.base_dir / f"underlying_symbol={underlying}" / f"{_DATE_PREFIX}{trade_date.isoformat()}" / "data.parquet"

    def read(self, underlying: str, trade_date: date, symbols: Sequence[str], columns: Sequence[str]) -> pl.DataFrame:
        """*columns* of *symbols* on *trade_date*; empty when the day has no quotes for them."""
        path = self._path(underlying, trade_date)
        if not path.exists():
            return pl.DataFrame()
        return pl.scan_parquet(path).filter(pl.col("symbol").is_in(list(symbols))).select(list(columns)).collect()

    def dates(self, underlying: str) -> List[date]:
        """The days the store holds quotes for *underlying*, newest first."""
        root = self.base_dir / f"underlying_symbol={underlying}"
        if not root.exists():
            return []
        return sorted((date.fromisoformat(d.name[len(_DATE_PREFIX):]) for d in root.iterdir()
                       if d.name.startswith(_DATE_PREFIX) and (d / "data.parquet").exists()), reverse=True)
