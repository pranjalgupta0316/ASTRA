# Changelog

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
