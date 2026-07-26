"""Point-in-time realized-volatility forecasts: one record per symbol per trading date.

Layout::

    <base_dir>/
      CURRENT                       # names the active methodology directory
      REGENERATING                  # present only while a swap is staging
      <methodology_id>/
        date=2024-01-02/part.parquet
        ...

**Why a versioned directory rather than partitions rewritten in place.** A model change rewrites
every record, and a regeneration over ~1,250 partitions is not atomic — a reader part-way through
would see some dates under the old methodology and some under the new, which is silently wrong
rather than obviously broken. Staging a whole new directory and flipping a one-line pointer makes
the swap a single atomic filesystem operation however long the rebuild took, and a failed rebuild
simply never flips.

The producer (Hermes) writes; consumers (Hephaestus's backtester) point ``base_dir`` here and read.
This library owns the layout and the read/write semantics and nothing else — no scheduling, no
HTTP, and deliberately no domain logic: it cannot know whether a forecast is any good, so the
quality gate that guards promotion lives in the producer, one level above.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import polars as pl

from .types import RV_FORECAST_COLUMNS, RV_FORECAST_VALUE_COLUMNS

DateLike = Union[date, datetime, str]

_DEFAULT_BASE_DIR = "data/rv_forecast"
_CURRENT = "CURRENT"
_LOCK = "REGENERATING"
_PART_FILE = "part.parquet"


class RegenerationInProgressError(RuntimeError):
    """A write was attempted against the live history while a methodology swap is staging."""


class MethodologyMismatchError(ValueError):
    """A write's methodology fingerprint disagrees with the directory it targets."""


def _default_base_dir() -> str:
    return os.environ.get("RV_FORECAST_DIR", _DEFAULT_BASE_DIR)


def _to_date(value: Optional[DateLike], fallback: date) -> date:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via temp file + os.replace so a crash cannot leave a half-written pointer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


class RvForecastStore:
    """Read/write access to the versioned, date-partitioned forecast history."""

    def __init__(self, base_dir: Optional[Union[str, os.PathLike]] = None):
        self.base_dir = Path(base_dir) if base_dir is not None else Path(_default_base_dir())

    # ---- methodology pointer & lock --------------------------------------

    def active_methodology(self) -> Optional[str]:
        """The methodology the store currently serves, or None when empty."""
        pointer = self.base_dir / _CURRENT
        if not pointer.exists():
            return None
        value = pointer.read_text(encoding="utf-8").strip()
        return value or None

    def regeneration_in_progress(self) -> Optional[dict]:
        """The active regeneration lock (``target``, ``started_at``), or None.

        Exposed so the daily path can say *why* it skipped a date, and so a health report can
        surface a lock that has been held implausibly long — a crashed regeneration stalls the
        store until someone clears it, by design (there is no automatic expiry).
        """
        lock = self.base_dir / _LOCK
        if not lock.exists():
            return None
        try:
            return json.loads(lock.read_text(encoding="utf-8"))
        except Exception:
            # A lock we cannot parse still means "something is staging" — the safe reading is
            # that a regeneration is underway, not that it is absent.
            return {"target": None, "started_at": None, "unparseable": True}

    def _methodology_dir(self, methodology_id: str) -> Path:
        return self.base_dir / methodology_id

    def _partition_dir(self, methodology_id: str, trade_date: date) -> Path:
        return self._methodology_dir(methodology_id) / f"date={trade_date.isoformat()}"

    # ---- reads -----------------------------------------------------------

    def read(
        self,
        symbols: Optional[Sequence[str]] = None,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        methodology_id: Optional[str] = None,
    ) -> pl.DataFrame:
        """The point-in-time panel: one row per (symbol, date).

        ``methodology_id`` is an **assertion**, not a version selector — supply it and the call
        raises if the store has since been regenerated under a different methodology. That lets a
        long-running backtest notice the ground moved without introducing version selection into
        the read path.

        Rows whose forecasts are null are returned, never filtered: a null forecast means "no
        forecast for this symbol-date", which is different from an absent row (no measures) and
        from an absent partition (never processed). Dropping them would collapse those states and
        also remove exactly the rows a consumer would fall back to ``rv_21d`` on.
        """
        active = self.active_methodology()
        if active is None:
            return self._empty()
        if methodology_id is not None and methodology_id != active:
            raise MethodologyMismatchError(
                f"store now holds methodology {active!r}, caller asserted {methodology_id!r} — "
                f"the history was regenerated underneath this read"
            )

        glob = str(self._methodology_dir(active) / "**" / _PART_FILE)
        start = _to_date(start_date, date(1900, 1, 1))
        end = _to_date(end_date, date.today())
        try:
            lf = pl.scan_parquet(glob, hive_partitioning=True)
            lf = lf.filter((pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end)))
            if symbols is not None:
                lf = lf.filter(pl.col("symbol").is_in([s.upper() for s in symbols]))
            df = lf.collect()
        except Exception:
            return self._empty()

        if df.is_empty():
            return self._empty()
        ordered = ["symbol", "date"] + [c for c in RV_FORECAST_COLUMNS if c not in ("symbol",)]
        ordered = [c for c in ordered if c in df.columns]
        ordered += [c for c in df.columns if c not in ordered]
        return df.select(ordered).sort(["symbol", "date"])

    def read_staged(
        self,
        methodology_id: str,
        symbols: Optional[Sequence[str]] = None,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
    ) -> pl.DataFrame:
        """Read a methodology that is **not** the active one — a staged rebuild.

        Separate from :meth:`read` on purpose. There, ``methodology_id`` is an *assertion* that
        the store still holds what the caller expects, and it raising is the whole point: it is
        how a long-running study learns the history moved underneath it. Overloading it to also
        mean "read this other directory" would destroy that guarantee.

        The promotion gate needs exactly this: score a candidate before deciding whether it may
        replace the live history. Nothing else should — reading a staged methodology means
        reading forecasts that no consumer is served.
        """
        target = self._methodology_dir(methodology_id)
        if not target.exists():
            raise FileNotFoundError(f"no methodology staged at {target}")

        glob = str(target / "**" / _PART_FILE)
        start = _to_date(start_date, date(1900, 1, 1))
        end = _to_date(end_date, date.today())
        try:
            lf = pl.scan_parquet(glob, hive_partitioning=True)
            lf = lf.filter((pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end)))
            if symbols is not None:
                lf = lf.filter(pl.col("symbol").is_in([s.upper() for s in symbols]))
            df = lf.collect()
        except Exception:
            return self._empty()
        if df.is_empty():
            return self._empty()
        ordered = ["symbol", "date"] + [c for c in RV_FORECAST_COLUMNS if c != "symbol"]
        ordered = [c for c in ordered if c in df.columns]
        ordered += [c for c in df.columns if c not in ordered]
        return df.select(ordered).sort(["symbol", "date"])

    def read_latest(self, symbols: Optional[Sequence[str]] = None) -> pl.DataFrame:
        """Most recent record per symbol, each carrying its own date and ``measures_date`` so a
        caller can see per-symbol staleness rather than only the store-wide newest date."""
        df = self.read(symbols=symbols)
        if df.is_empty():
            return df
        return df.sort(["symbol", "date"]).group_by("symbol", maintain_order=True).last()

    def latest_date(self) -> Optional[date]:
        """Newest partition present, or None for an empty store."""
        active = self.active_methodology()
        if active is None:
            return None
        dates = []
        for p in self._methodology_dir(active).glob("date=*"):
            try:
                dates.append(date.fromisoformat(p.name.split("=", 1)[1]))
            except (ValueError, IndexError):
                continue
        return max(dates) if dates else None

    def coverage(self, symbols: Optional[Sequence[str]] = None) -> pl.DataFrame:
        """Per-symbol record count and first/last date — what is actually held, before anyone
        interprets a result computed from it."""
        df = self.read(symbols=symbols)
        if df.is_empty():
            return pl.DataFrame(schema={
                "symbol": pl.Utf8, "records": pl.UInt32,
                "first_date": pl.Date, "last_date": pl.Date,
                "forecasts": pl.UInt32,
            })
        return (
            df.group_by("symbol")
            .agg(
                pl.len().cast(pl.UInt32).alias("records"),
                pl.col("date").min().alias("first_date"),
                pl.col("date").max().alias("last_date"),
                pl.col("predicted_rv_21d").is_not_null().sum().cast(pl.UInt32).alias("forecasts"),
            )
            .sort("symbol")
        )

    @staticmethod
    def _empty() -> pl.DataFrame:
        schema: dict[str, pl.DataType] = {"symbol": pl.Utf8, "date": pl.Date,
                                          "measures_date": pl.Date}
        for c in RV_FORECAST_VALUE_COLUMNS:
            schema[c] = pl.Float64
        schema["methodology_id"] = pl.Utf8
        schema["source"] = pl.Utf8
        return pl.DataFrame(schema=schema)

    # ---- writes (producer only) ------------------------------------------

    def _check_writable(self, methodology_id: str) -> None:
        """Reject a write that would land somewhere it must not.

        Two distinct guards, for two distinct silent-corruption modes:

        * a fingerprint that disagrees with the target directory — this is what makes
          configuration drift *impossible to persist* rather than merely discouraged, and it is
          the reason the id is stamped per record at all;
        * a write to the live history while a swap is staging — those records sit in a directory
          about to be deleted at the flip, so they would vanish without any error.
        """
        lock = self.regeneration_in_progress()
        if lock is not None:
            target = lock.get("target")
            if methodology_id != target:
                raise RegenerationInProgressError(
                    f"regeneration to {target!r} started at {lock.get('started_at')!r} — "
                    f"refusing a write to {methodology_id!r}. The date will be picked up by the "
                    f"next run's missed-date fill."
                )
            return  # the regeneration writing its own staging copy

        active = self.active_methodology()
        if active is not None and methodology_id != active:
            raise MethodologyMismatchError(
                f"store holds methodology {active!r}; refusing a write stamped {methodology_id!r}"
            )

    @staticmethod
    def _validate_frame(df: pl.DataFrame, methodology_id: str) -> pl.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("frame is missing the required 'symbol' column")
        if "methodology_id" in df.columns:
            stamped = set(df["methodology_id"].drop_nulls().unique().to_list())
            if stamped - {methodology_id}:
                raise MethodologyMismatchError(
                    f"frame carries methodology ids {sorted(stamped)}, expected {methodology_id!r}"
                )
        else:
            df = df.with_columns(pl.lit(methodology_id).alias("methodology_id"))
        if "source" not in df.columns:
            df = df.with_columns(pl.lit("unknown").alias("source"))
        keep = [c for c in RV_FORECAST_COLUMNS if c in df.columns]
        return df.select(keep).filter(pl.col("symbol").is_not_null())

    def _write_partition_file(self, part_dir: Path, merged: pl.DataFrame) -> int:
        part_dir.mkdir(parents=True, exist_ok=True)
        target = part_dir / _PART_FILE
        fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=str(part_dir))
        os.close(fd)
        try:
            merged.write_parquet(tmp)
            os.replace(tmp, target)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
        return merged.height

    def write_date_partition(
        self, df: pl.DataFrame, trade_date: date, methodology_id: str
    ) -> int:
        """Write one trading day's records, merging on symbol.

        Merge-on-symbol matters: the daily run writes the screen's own frame first and the
        complement (symbols the screen does not rank) second, and the second must not clobber the
        first. Re-writing a symbol with an identical value is a no-op in effect.
        """
        self._check_writable(methodology_id)
        clean = self._validate_frame(df, methodology_id)
        if clean.is_empty():
            return 0

        # First write into an empty store establishes the active methodology.
        if self.active_methodology() is None and self.regeneration_in_progress() is None:
            self._methodology_dir(methodology_id).mkdir(parents=True, exist_ok=True)
            _atomic_write_text(self.base_dir / _CURRENT, methodology_id)

        part_dir = self._partition_dir(methodology_id, trade_date)
        existing_file = part_dir / _PART_FILE
        if existing_file.exists():
            try:
                existing = pl.read_parquet(existing_file)
                existing = existing.join(clean.select("symbol"), on="symbol", how="anti")
                merged = pl.concat([existing, clean], how="vertical_relaxed")
            except Exception:
                merged = clean
        else:
            merged = clean
        merged = merged.unique(subset=["symbol"], keep="last").sort("symbol")
        return self._write_partition_file(part_dir, merged)

    def rewrite_span(
        self,
        df: pl.DataFrame,
        symbol: str,
        start_date: date,
        end_date: date,
        methodology_id: str,
    ) -> int:
        """Replace one symbol's records across a closed date span, under the active methodology.

        The restatement path. A forecast is otherwise immutable, and a methodology change is
        otherwise the only rewrite — but the *inputs* can also be corrected, and a forecast that
        no longer follows from its own measures is quietly wrong. This is the one operation that
        legitimately replaces values without consuming a methodology identifier.

        Rejects any id other than the active one: a data correction must not be a vehicle for
        slipping in a different model. The span is the caller's to compute — a restated date
        invalidates forecasts forward by the whole fit window, not just itself.
        """
        active = self.active_methodology()
        if active is None:
            raise MethodologyMismatchError("cannot rewrite a span in an empty store")
        if methodology_id != active:
            raise MethodologyMismatchError(
                f"rewrite_span is a same-methodology correction; store holds {active!r}, "
                f"caller passed {methodology_id!r}"
            )
        if self.regeneration_in_progress() is not None:
            raise RegenerationInProgressError("cannot rewrite a span while a regeneration stages")
        if start_date > end_date:
            raise ValueError(f"empty span: {start_date} > {end_date}")

        if "date" not in df.columns:
            raise ValueError("rewrite_span needs a 'date' column to place each replacement record")
        # Validate per date, and only after splitting: the canonical column set deliberately has
        # no `date` (it lives in the partition path), so validating the whole frame first would
        # discard the very column that says where each row belongs — and the span would be
        # emptied rather than rewritten.
        by_date: dict[date, pl.DataFrame] = {}
        for (d,), g in df.group_by(["date"]):
            by_date[d] = self._validate_frame(g.drop("date"), methodology_id)

        written = 0
        for part in sorted(self._methodology_dir(active).glob("date=*")):
            try:
                d = date.fromisoformat(part.name.split("=", 1)[1])
            except (ValueError, IndexError):
                continue
            if not (start_date <= d <= end_date):
                continue
            file = part / _PART_FILE
            if not file.exists():
                continue
            existing = pl.read_parquet(file)
            others = existing.filter(pl.col("symbol") != symbol.upper())
            replacement = by_date.get(d)
            if replacement is not None:
                merged = pl.concat([others, replacement], how="vertical_relaxed")
            else:
                # No replacement offered for a date inside the span: the symbol legitimately has
                # no record there any more (its restated measures no longer support a row), so it
                # drops out rather than keeping a value derived from superseded inputs.
                merged = others
            self._write_partition_file(part, merged.unique(subset=["symbol"], keep="last").sort("symbol"))
            written += 1
        return written

    # ---- regeneration ----------------------------------------------------

    def begin_regeneration(self, methodology_id: str) -> Path:
        """Create the staging directory and take the lock. ``CURRENT`` is untouched, so the
        existing history stays fully readable for the whole rebuild."""
        if methodology_id == self.active_methodology():
            raise ValueError(f"{methodology_id!r} is already the active methodology")
        existing = self.regeneration_in_progress()
        if existing is not None:
            raise RegenerationInProgressError(
                f"a regeneration to {existing.get('target')!r} is already staging "
                f"(started {existing.get('started_at')!r})"
            )
        staging = self._methodology_dir(methodology_id)
        staging.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.base_dir / _LOCK,
            json.dumps({"target": methodology_id,
                        "started_at": datetime.now().isoformat(timespec="seconds")}),
        )
        return staging

    def promote(self, methodology_id: str) -> None:
        """Flip ``CURRENT`` atomically, delete the superseded directory, release the lock.

        **The quality gate is not enforced here, deliberately.** This library holds no domain
        logic and cannot evaluate a forecast. The requirement that a methodology clear its
        accuracy threshold before promotion is met one level up, by the producer's single
        promotion command — which scores, refuses below threshold, and only then calls this.
        Reaching this method by any other route bypasses a guard on an irreversible action.
        """
        staging = self._methodology_dir(methodology_id)
        if not staging.exists():
            raise FileNotFoundError(f"no staged methodology at {staging}")
        superseded = self.active_methodology()
        _atomic_write_text(self.base_dir / _CURRENT, methodology_id)   # the atomic swap
        if superseded and superseded != methodology_id:
            shutil.rmtree(self._methodology_dir(superseded), ignore_errors=True)
        (self.base_dir / _LOCK).unlink(missing_ok=True)

    def discard_regeneration(self, methodology_id: str) -> None:
        """Remove a staging directory and release the lock. Never touches the active history.

        Also the deliberate way to clear a lock a crashed regeneration left behind. There is no
        automatic expiry: auto-clearing would resume appends into a directory a half-finished
        rebuild might still be writing, trading a visible stall for silent corruption.
        """
        if methodology_id == self.active_methodology():
            raise ValueError(f"refusing to discard the active methodology {methodology_id!r}")
        shutil.rmtree(self._methodology_dir(methodology_id), ignore_errors=True)
        (self.base_dir / _LOCK).unlink(missing_ok=True)

    # ---- maintenance -----------------------------------------------------

    def covered_dates(self, methodology_id: Optional[str] = None) -> set[date]:
        """Dates already written — the backfill's resumability check."""
        target = methodology_id or self.active_methodology()
        if target is None:
            return set()
        out: set[date] = set()
        for p in self._methodology_dir(target).glob("date=*"):
            if not (p / _PART_FILE).exists():
                continue
            try:
                out.add(date.fromisoformat(p.name.split("=", 1)[1]))
            except (ValueError, IndexError):
                continue
        return out


__all__ = [
    "RvForecastStore",
    "RegenerationInProgressError",
    "MethodologyMismatchError",
]
