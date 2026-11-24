# NEBULA

Local-first quantitative research and AI/ML platform for **US equities, Indian equities and crypto**,
built for rigor over demo value: free data with provenance and quality reports, causal features,
purged walk-forward validation, event-driven cost-aware backtesting, portfolio construction, volatility
and regime models, PyTorch deep learning on GPU, Level-2 microstructure analytics, and an experiment
tracker that keeps every failed and inconclusive hypothesis.

> Research and paper trading only. No real-money execution. No paid services (₹0 recurring cost).
> Historical results are not evidence of future profitability. **No tradeable edge was found** -
> see [docs/FINAL_PROJECT_REPORT.md](docs/FINAL_PROJECT_REPORT.md).

## Quick start (Windows, PowerShell)
```powershell
uv sync --all-extras            # Python 3.12 + deps (PyTorch CUDA 12.8 wheels; CPU fallback works)
uv run pytest -m "not network"  # offline test suite (134 tests)
uv run nebula ingest            # 13 instruments, daily bars -> data/ (free providers)
uv run nebula run-all --skip-ingest   # every study, ~20-25 min on an i7-14700HX + RTX 4060
uv run nebula dashboard         # http://127.0.0.1:8501
```

## Commands (`uv run nebula --help`)
| command | what it does |
|---|---|
| `env` | hardware / GPU / library report |
| `ingest [SYMBOLS]` | fetch → raw (sha256) → validate → validated Parquet + quality report |
| `research [SYMBOLS]` | vertical slice: stylized facts, 60d momentum, logistic baseline (H-XM-000/001/002) |
| `holdout` | confirmatory momentum Sharpe test on 10 held-out instruments (H-XM-003) |
| `vol-study` | EWMA/GARCH/GJR volatility forecasts (H-VOL-001) + walk-forward GMM regimes (H-REG-001) |
| `portfolio-study [MARKETS]` | EW / min-var / risk parity / max-Sharpe / mean-variance per market (H-PF-001/002) |
| `ml-study` | 5 models × 13 instruments (H-ML-001) + pooled MLP/GRU panel on GPU (H-ML-002) |
| `micro-study [--capture-seconds N]` | Binance aggTrades analytics (H-MS-000); optional live L2 capture (H-MS-001) |
| `report` | deterministic research agents → `data/reports/research/latest.md` |
| `run-all` | reproduce everything in order |
| `dashboard` | Streamlit dashboard on localhost |

Live order-book capture with a hard disk cap: `uv run python scripts/capture_depth.py --duration 300`.

## Universe
US: AAPL, MSFT, NVDA, SPY, QQQ · India (NSE via Yahoo `.NS`): RELIANCE, TCS, INFY, HDFCBANK, NIFTY 50 ·
Crypto (Binance spot): BTCUSDT, ETHUSDT, SOLUSDT. AAPL/RELIANCE/BTCUSDT were used for design; the other
10 are held out for confirmatory tests.

## Layout
```
config/markets.yaml   markets, instruments, cost assumptions, holdout list
src/nebula/           common, data, features, stats, ml, dl, backtest, portfolio, risk, microstructure,
                      research (study runners), tracking, agents (runtime research roles), dashboard, cli
tests/                unit, data-validation, quantitative-correctness, leakage canaries, network (-m network)
docs/                 charter, architecture, ADRs, data sources, hypotheses (docs/research/*), validation
.claude/              development-time Claude Code skills and subagents (not a runtime dependency)
data/ experiments/ logs/   generated, git-ignored
```

## Documentation
[Final report](docs/FINAL_PROJECT_REPORT.md) · [Status](docs/PROJECT_STATUS.md) ·
[Validation](docs/VALIDATION_REPORT.md) · [Architecture](docs/SYSTEM_ARCHITECTURE.md) ·
[Data sources](docs/DATA_SOURCES.md) · [Data dictionary](docs/DATA_DICTIONARY.md) ·
[Hypotheses](docs/RESEARCH_HYPOTHESES.md) + [docs/research/](docs/research) ·
[Methodology](docs/RESEARCH_METHODOLOGY.md) · [Decisions](docs/TECHNICAL_DECISIONS.md) ·
[Known issues](docs/KNOWN_ISSUES.md) · [Next steps](docs/NEXT_STEPS.md)
