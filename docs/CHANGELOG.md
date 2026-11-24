# Changelog

## 0.1.0 - 2026-11-24 (unreleased, uncommitted)
- Vertical slice: Yahoo (AAPL, RELIANCE.NS) and Binance Public Data (BTCUSDT) daily ingestion with raw
  provenance, quality reports, validated Parquet + DuckDB view.
- Causal technical features; stationarity / Ljung-Box / NW / bootstrap / PSR / DSR inference.
- Purged walk-forward CV; logistic baseline with ex-ante baseline and DM test.
- Event-driven daily backtester (next-open fills, per-side costs, volume cap, zero-volume deferral,
  cost-aware sizing, rebalance tolerance, exact gross runs); leakage canary; cost sensitivity.
- File-based experiment tracker; Streamlit dashboard (data, market, experiments, backtests, logs).
- 7 project skills, 7 subagents, skill validator, CI workflow.
