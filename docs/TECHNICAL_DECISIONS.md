# Technical decisions (index)

| ADR | Decision | Status |
|---|---|---|
| [0001](adr/0001-modular-monolith-local-first.md) | Local-first modular monolith on Parquet + DuckDB | accepted |
| [0002](adr/0002-python-312-uv.md) | Python 3.12 via uv; PyTorch cu128 wheels | accepted |
| [0003](adr/0003-data-sources-and-timestamps.md) | Yahoo + Binance Public Data; availability-time convention | accepted |
| [0004](adr/0004-experiment-tracking.md) | File-based experiment tracker (MLflow extra later removed as unused) | accepted |
| [0005](adr/0005-deep-learning-scope-and-device.md) | Small PyTorch models, GPU default, device recorded | accepted |
| [0006](adr/0006-portfolios-per-market-local-currency.md) | Portfolios per market in local currency | accepted |

Smaller decisions
- Subagents: 6 instead of 8 recommended roles (see .claude/AGENT_ARCHITECTURE.md).
- XBOM (BSE) calendar used for NSE sessions: exchange_calendars ships no NSE calendar; NSE and BSE
  share the trading-holiday list. Checked by the `sessions` validation check on real data.
- Pandas for daily pipelines (tiny data); Polars reserved for tick/depth workloads where it pays off.
