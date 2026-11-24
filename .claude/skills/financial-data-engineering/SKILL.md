---
name: financial-data-engineering
description: Use when ingesting, validating, normalizing or storing any NEBULA market data - OHLCV, trades, order-book snapshots/diffs, corporate actions, calendars, timezones, point-in-time fundamentals, macro vintages - or touching Parquet/DuckDB/Polars storage, data contracts or data-quality reports.
---

# Financial Data Engineering (NEBULA)

## Purpose
Every dataset in NEBULA is real, provenanced, validated, point-in-time correct and reproducible.
Code: `src/nebula/data/` (adapters, validation, storage, calendars). Docs: `docs/DATA_SOURCES.md`,
`docs/DATA_DICTIONARY.md`.

## Activate when
Adding a provider/adapter, changing a schema, writing to `data/`, handling timestamps or calendars,
adjusting for splits/dividends, or reporting data availability.

## Workflow
1. **Verify the source** before coding: coverage, granularity, rate limits, terms of use, symbol format.
   Record it in `docs/DATA_SOURCES.md`. Never scrape behind logins or bypass limits.
2. **Raw layer** (`data/raw/<provider>/<dataset>/...`): store exactly what the provider returned, plus a
   sidecar metadata JSON (provider, URL/endpoint, retrieval UTC time, request params, row count, sha256).
   Raw is immutable.
3. **Validate** with `nebula.data.validation` -> a `QualityReport` (persisted to `data/reports/`).
   Checks: schema, nulls, duplicates, monotonic time, OHLC consistency (L <= O,C <= H), non-negative
   price/volume, missing sessions vs. exchange calendar, gaps for 24/7 crypto, outlier returns.
4. **Normalize** to the canonical bar contract (`nebula.data.contracts.BAR_SCHEMA`): `ts_utc`
   (bar open, UTC), `ts_local`, `exchange_tz`, `symbol`, `market`, `currency`, OHLCV, `adj_close` if
   provided, `provider`, `ingested_at`. Keep the original timezone.
5. **Store** validated data as zstd Parquet partitioned by `market/symbol/frequency`; query via DuckDB.
6. **Never silently repair**: a fix writes a new column or a new version and logs the transformation.

## Standards
- UTC internally; daily equity bars stamped at session **close** for availability (`available_at`),
  because the close price is not known at the open.
- Corporate actions: keep unadjusted OHLC + separate adjustment factor; adjusted series are derived.
- Fundamentals: key on filing/acceptance timestamp (SEC `acceptedDateTime`), not period end.
- Macro: use ALFRED vintages or publication lags; never the latest revised value for history.
- Polars/DuckDB lazily; never load full history of tick data into RAM (16 GB machine).

## Failure modes
yfinance silently returns adjusted or partial data; `.NS` symbols missing days around Indian holidays;
tz-naive timestamps; Binance kline open-time in ms vs us (spot files switched to microseconds in 2025 -
detect by magnitude); duplicate last bar on re-download; survivorship from today's index list.

## Verification / acceptance
Quality report exists for every batch; tests cover each check with a synthetic broken frame;
re-running ingestion is idempotent (same sha256 or a new version, never an overwrite).

## Limitations
Free sources: no free historical full-depth L2 for any market; Indian intraday history is not freely
available from official sources; yfinance is unofficial and may break (see DATA_SOURCES.md).

## Interacts with
All other skills consume its contracts. `market-microstructure` owns L2 capture semantics.
Reference: `references/provider_notes.md`.
