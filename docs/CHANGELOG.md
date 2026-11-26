# Changelog

## Unreleased (0.4 improvement programme, 2026-11-26)
- P1 data locks: immutable content-addressed snapshots, `--lock`, lock stored in every record (ADR-0009);
  `v0.3.0-baseline` reproduces the published records exactly.
- P2 engine verification: property-based tests (hypothesis) against an independent return-algebra reference,
  cash reconciliation, truncation invariance; golden regression of headline numbers under the baseline lock.
- P3 realized volatility (H-RV-001/002/003) from 1-minute and hourly data: all families inconclusive; audited.
- P6 order-level execution simulator (L1 quotes + trades; through/queue/touch fill bounds, latency, mark-outs).
- P7 global multiplicity: BH/BY over 242 tests, deflated Sharpe over 192 strategy curves, calibration.
- P8 point-in-time S&P 500 membership from 2011; survivorship of the US universe quantified.
- P9 computed `superseded` flag; dashboard page for data versions and multiplicity.
- Fixes: `Tracker.save_frame` outside the repo root; architecture/methodology docs drift.

## 0.3.0 (renamed) - 2026-11-25
- Project renamed **NEBULA → ASTRA** everywhere: Python package `astra` (was `nebula`), CLI command
  `astra` (was `nebula`), environment variables `ASTRA_*` (were `NEBULA_*`), folder `C:\Projects\ASTRA`.
  No functional change. Historical git commit messages and stored experiment records/logs keep the old
  name (they are provenance and were not rewritten).

## 0.3.0 - 2026-11-25
- Universe 13 → 111 instruments (core/extended tiers), equities history from 2000; Binance hourly
  (16 pairs) and 1-minute (BTC, ETH) klines; frozen sample window for registered core studies.
- SEC EDGAR point-in-time fundamentals; Yahoo market macro + BLS CPI/unemployment with release-date-
  aware availability; FRED reduced to manual CSVs because of its terms (ADR-0008).
- New pre-registered studies: cross-sectional momentum/reversal, volatility-managed exposure,
  cross-market diversification, 600k-row ML panel, hourly crypto ML, multimodal ablation, 6-hour
  order-book imbalance study.
- Engine fixes from audits: zero default rebalance tolerance (cash drag), buy-and-hold re-issues its
  target (volume-capped first fill), off-grid intraday bars, negative adjusted prices.
- Local LLM (Ollama, qwen2.5:7b) benchmarked and wired with an 8k context; dashboard page for extended
  research; CLI `ingest --intraday`, `ingest-fundamentals`, `ingest-macro`, `ext-study`, `ml-ext`.

## 0.2.0 - 2026-11-24
- Universe expanded to 13 instruments (10 held out); Yahoo in-progress bars dropped with a 2 h settlement buffer.
- Volatility forecasting (EWMA/GARCH/GJR, QLIKE, DM) and walk-forward GMM regimes.
- Portfolio construction (Ledoit-Wolf, min-var, MV, max-Sharpe, risk parity, vol targeting), multi-asset
  event-driven backtester, risk attribution, VaR/ES.
- ML/DL grid (logit, RF, XGBoost, PyTorch MLP/GRU on GPU) with purged walk-forward, BH, pooled panel.
- Market microstructure: Binance L2 sync with gap detection, OFI/microprice, aggTrades analytics,
  disk-capped live capture.
- Held-out confirmatory momentum test (paired stationary-bootstrap Sharpe difference).
- Deterministic runtime research agents + Markdown report; optional localhost LLM narration.
- Dashboard: 12 pages; CLI: study commands and `run-all`.
- Hardening from audits: checksummed streamed downloads with backoff, .gitignore fix (data package now
  tracked), dependency cleanup, amendment A2 for the ML panel.

## 0.1.0 - 2026-11-24
- Vertical slice (AAPL, RELIANCE.NS, BTCUSDT): ingestion, validation, features, stats, logistic
  baseline, event-driven backtester, tracker, dashboard, skills and agents.
