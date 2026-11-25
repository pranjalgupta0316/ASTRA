# Technical decisions (index)

| ADR | Decision | Status |
|---|---|---|
| [0001](adr/0001-modular-monolith-local-first.md) | Local-first modular monolith on Parquet + DuckDB | accepted |
| [0002](adr/0002-python-312-uv.md) | Python 3.12 via uv; PyTorch cu128 wheels | accepted |
| [0003](adr/0003-data-sources-and-timestamps.md) | Yahoo + Binance Public Data; availability-time convention | accepted |
| [0004](adr/0004-experiment-tracking.md) | File-based experiment tracker (MLflow extra later removed as unused) | accepted |
| [0005](adr/0005-deep-learning-scope-and-device.md) | Small PyTorch models, GPU default, device recorded | accepted |
| [0006](adr/0006-portfolios-per-market-local-currency.md) | Portfolios per market in local currency (cross-market study: ADR-0007) | accepted |
| [0007](adr/0007-cross-market-weekly-alignment.md) | Cross-market study on weekly USD returns | accepted |
| [0008](adr/0008-macro-sources-and-fred-terms.md) | Macro from Yahoo + BLS; FRED manual-only (terms) | accepted |

Further decisions (2026-11-25)
- Core/extended universe tiers; registered core studies read a frozen sample window
  [2018-01-01, 2026-11-23] so they stay reproducible as data grows.
- Portfolio engine default `rebalance_tolerance` 0 (an absolute 2 pp tolerance caused benchmark cash
  drag with many small weights); single-asset `buy_and_hold` re-issues its target until filled.
- Local LLM: Ollama + qwen2.5:7b (q4), context 8192, localhost only, narration only.

Smaller decisions
- Subagents: 6 instead of 8 recommended roles (see .claude/AGENT_ARCHITECTURE.md).
- XBOM (BSE) calendar used for NSE sessions: exchange_calendars ships no NSE calendar; NSE and BSE
  share the trading-holiday list. Checked by the `sessions` validation check on real data.
- Pandas for daily pipelines (tiny data); Polars reserved for tick/depth workloads where it pays off.
