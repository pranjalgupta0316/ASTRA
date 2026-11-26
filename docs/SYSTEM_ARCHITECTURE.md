# System architecture

ASTRA is a **local-first modular monolith** (one Python package, `src/astra`), storing data as
zstd Parquet queried by DuckDB/pandas/Polars. No services, brokers or cloud dependencies.

```
                 ┌──────────────── config/markets.yaml (markets, instruments, cost assumptions)
                 ▼
 providers ─► data.adapters ─► data/raw (immutable + .meta.json, sha256)
 (Yahoo,         │ normalize
  Binance)       ▼
            data.validation ─► data/reports/quality/*.json
                 │ pass/warn
                 ▼
            data/validated/market=*/symbol=*/freq=*/bars.parquet + manifest.json
                 │                       ▲ DuckDB view `bars`
                 │  data.lock: optional immutable snapshot (data/snapshots, config/data_locks; ADR-0009)
                 ▼
            features.technical (causal features, separate target_* columns)
                 │
      ┌──────────┼───────────────┬───────────────────┐
      ▼          ▼               ▼                   ▼
 stats.tests  research.*     ml.validation +     dl (PyTorch), microstructure,
 stats.inference              ml.classical        portfolio, stats.volatility
      │          │               │
      └──────────┴──► backtest.engine / portfolio.backtest (event-driven, next-open fills, costs,
                              │        volume cap); backtest.audit (leakage canary, cost sensitivity);
                              │        backtest.reference (independent frictionless check, tests only)
                              ▼
                       risk.metrics ─► tracking.tracker ─► experiments/<id>/record.json + artifacts
                                                          │
                                         dashboard.app (Streamlit, reads stored outputs only)
```

## Module boundaries
| Package | Responsibility | Depends on |
|---|---|---|
| common | paths, typed config, logging, seeds, env report | — |
| data | adapters, contracts, calendars, validation, storage, ingest, data locks, fundamentals, macro | common |
| features | causal features + targets | — (pure pandas) |
| stats | tests and inference | — |
| ml / dl | validation splits, model training | stats |
| backtest | event engine, audits | common.config (costs), risk |
| risk / portfolio | metrics, optimizers | stats |
| research | hypothesis runners wiring the above | all of the above |
| tracking | experiment records (incl. data lock), golden regression | common, data.lock |
| dashboard | read-only views | data.storage, tracking |
| agents | deterministic research workflows, optional local LLM narration | research, tracking |

## Timestamp model
- `session_date` = local trading date; `close_time_utc` = availability time of the bar.
- Signal at close t ⇒ order fills at open t+1 (engine invariant, tested).
- Crypto daily bar = [00:00, 24:00) UTC.

## Resource design (16 GB RAM, RTX 4060 Laptop 8 GB)
Daily data is tiny (< 1 MB per instrument); 1-minute crypto klines are ~4.9 M rows per symbol and are
read per symbol, never all at once. Depth capture is streamed to hourly Parquet files with a disk cap. GPU only for PyTorch models; CPU fallback always.
