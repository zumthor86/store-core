# store-core

The single owner of the workspace's on-disk **price store** — the schema-of-record,
the read contract, and the append-then-merge write semantics for hive-partitioned
Parquet OHLCV. Sibling to `ingest-core`, but the mirror-image charter: `ingest-core`
owns acquisition and holds **no** storage; `store-core` owns storage and holds no
acquisition, scheduling, HTTP, or domain logic.

## Why it exists

Prices live in Parquet (`exchange=<XX>/<TICKER>.parquet`), not in Postgres, so there
is no SQL view to point a second engine at. Before this library, the producer (Hermes)
read and wrote its own Parquet, and any consumer wanting prices had to either go through
Hermes's HTTP API or reach into Hermes's file tree with its own `pl.read_parquet`. This
library makes the Parquet layout a real contract: **the producer writes through it, and
consumers read through it** — so a re-partition or schema change happens in exactly one
place, and no consumer couples to file paths.

## Contents

| Module | Purpose |
|---|---|
| `store_core.PriceStore` | Read (Polars, lazy scan + predicate pushdown) and write (pandas/pyarrow, merge + atomic rename) for the OHLCV store. `base_dir` is configurable so a consumer points it at the producer's prices directory. |
| `store_core.RvForecastStore` | Versioned, point-in-time realized-volatility forecast history — read/write contract for Hermes's RV forecast store. Raises `MethodologyMismatchError` / `RegenerationInProgressError` rather than returning stale or half-written data. |
| `store_core.types` | Schema-of-record: `PRICE_COLUMNS`, `OHLCV_FLOAT_COLUMNS`, `PRICE_PARQUET_SCHEMA`, `RV_FORECAST_COLUMNS`, `RV_FORECAST_VALUE_COLUMNS`, `RV_FORECAST_HORIZONS`, `RV_FORECAST_PARQUET_SCHEMA`. |

## Usage

```python
from store_core import PriceStore

# Producer (Hermes ingest) — writes:
store = PriceStore()                       # base_dir from PRICES_DIR env
store.upsert(df, exchange="US", ticker="AAPL")

# Consumer (Hephaestus backtester, ares) — reads, no HTTP:
store = PriceStore(base_dir=HERMES_PRICES_DIR)
prices = store.read(["AAPL", "MSFT"], start_date="2020-01-01")
```

## Install (editable, all consumers)

```bash
pip install -e ../store-core
```

## Tests

```bash
pytest tests/unit -q     # hermetic: temp Parquet tree, no network
```

## Changelog

Dated, one-line entries for changes that affect consumers — new modules, contract
or schema changes, breaking behavior. Keep entries short; `git log` has the detail.
Update this **in the same change** that touches this library, and mention it in
whichever consuming project's `CLAUDE.md` you're also updating.

- **2026-07-26** (`b6be45c`) — Added `RvForecastStore`: versioned, point-in-time RV forecast history, consumed by Hephaestus's backtester.
- **2026-07-24** (`52cdaee`) — Initial extraction: `PriceStore` (hive-partitioned Parquet OHLCV), producer Hermes / consumer Hephaestus backtester.
