# Implementation roadmap

| M | Scope | Exit criterion |
|---|---|---|
| M0 | environment, feasibility, risk register | env report + DATA_SOURCES matrix |
| M1 | 7 skills, 6 subagents, registry | skills validated by script |
| M2 | architecture, contracts, ADRs | docs + contracts in code |
| M3 | uv env, ruff, pyright, pytest, CI | all three tools run clean |
| M4 | AAPL + RELIANCE + BTCUSDT ingestion/validation/storage | quality reports pass/warn on real data |
| M5 | features + statistical baselines | causal-feature tests; H-XM-000/001 recorded |
| M6 | classical ML → PyTorch | H-XM-002 recorded; DL vs baseline on identical folds |
| M7 | microstructure: aggTrades analytics, depth capture + reconstruction | gap-detection tests; MB/hour measured |
| M8 | portfolio + risk (CVXPY, shrinkage) | analytic optimizer tests |
| M9 | backtester extensions: multi-asset, calendars across markets, FX | audited multi-asset backtest |
| M10 | runtime research workflows (deterministic agents, optional local LLM) | traceable reports |
| M11 | dashboard expansion, profiling | all pages on real outputs |
| M12 | final audits, report | VALIDATION_REPORT + FINAL_PROJECT_REPORT |

Revision 2026-11-24: M4-M6 vertical slice built together (single pipeline per instrument) because
the slice acceptance requires data → features → stats → ML → backtest → dashboard end to end.
