# ASTRA

**Advanced Statistical Trading & Research Architecture** - a multi-market quantitative research platform for
market data analysis, strategy development, backtesting, statistical validation, and market microstructure
research across equities and cryptocurrencies.

Local-first quantitative research and AI/ML platform for **US equities, Indian equities and crypto**,
built for rigor over demo value: free data with provenance and quality reports, causal features,
pre-registered hypotheses, purged walk-forward validation, event-driven cost-aware backtesting,
portfolio construction, volatility/regime models, PyTorch deep learning on GPU, SEC fundamentals and
macro data, live Level-2 order-book capture, independent audits, and an experiment tracker that keeps
every failed and inconclusive result.

> Research and paper trading only. No real-money execution. ₹0 recurring cost.
> **Headline finding:** short-horizon predictability exists (hourly crypto AUC 0.54, order-book
> imbalance AUC 0.91 at 1 s) but no strategy survives realistic trading costs. See
> [docs/FINAL_PROJECT_REPORT.md](docs/FINAL_PROJECT_REPORT.md).

## Quick start (Windows, PowerShell)
```powershell
uv sync --all-extras                  # Python 3.12 + deps (PyTorch CUDA 12.8; CPU fallback works)
uv run pytest -m "not network"        # offline test suite
uv run astra ingest                  # daily bars, 111 instruments (Yahoo + Binance)
uv run astra ingest --intraday       # hourly crypto (16 pairs) + 1-minute BTC/ETH
uv run astra ingest-fundamentals     # SEC EDGAR point-in-time facts (30 US stocks)
uv run astra ingest-macro            # Yahoo yields/VIX/DXY/USDINR + BLS CPI/unemployment
uv run astra run-all --skip-ingest   # every study (~45 min; --heavy-dl adds ~3 h of MLP/GRU)
uv run astra dashboard               # http://127.0.0.1:8501
```

## Commands (`uv run astra --help`)
| command | what it does |
|---|---|
| `env` | hardware / GPU / library report |
| `ingest [SYMBOLS] [--tier] [--intraday]` | fetch → raw (sha256) → validate → validated Parquet + quality report |
| `ingest-fundamentals`, `ingest-macro` | SEC EDGAR facts; Yahoo market macro + BLS (FRED is manual-CSV only - its terms forbid automation) |
| `research`, `holdout` | core vertical slice (H-XM-000/001/002) and held-out momentum test (H-XM-003) |
| `vol-study`, `portfolio-study` | GARCH/EWMA + regimes (H-VOL/H-REG); per-market portfolios (H-PF) |
| `ml-study` | 5 models × 13 instruments + pooled DL panel (H-ML-001/002) |
| `ext-study` | cross-sectional momentum/reversal, vol-managed exposure, cross-market, multimodal (H-CS/H-VT/H-XMKT/H-MM) |
| `ml-ext` | 600k-row daily panel (H-ML-003) + hourly crypto (H-ML-004) |
| `micro-study [--capture-seconds N]` | Binance aggTrades analytics; optional live L2 capture |
| `report [--llm]` | deterministic research agents → `data/reports/research/latest.md` (+ optional local-LLM summary) |
| `rv-study` | realized volatility from 1-minute data: HAR-RV vs GARCH/EWMA (H-RV) |
| `multiplicity` | project-wide FDR (BH/BY), deflated Sharpe, probability calibration over all current records |
| `universe [--fetch]` | point-in-time S&P 500 membership (Wikipedia) + survivorship check of the US universe |
| `records [--all]` | experiment records with computed `superseded` flag, commit and data lock |
| `data-lock create/list/verify`, global `--lock NAME` | immutable data snapshots; run any study on a pinned data version (ADR-0009) |
| `run-all`, `dashboard` | reproduce everything; Streamlit dashboard on localhost |

Long order-book captures with a hard disk cap: `uv run python scripts/capture_depth.py --duration 3600`.
Optional local LLM: install Ollama, `ollama pull qwen2.5:7b`, set `ASTRA_LLM_URL=http://127.0.0.1:11434`
and `ASTRA_LLM_MODEL=qwen2.5:7b`, then `uv run astra report --llm`.

## Universe
**Core (13, frozen sample 2018-01-01…2026-11-23 for registered studies):** AAPL, MSFT, NVDA, SPY, QQQ ·
RELIANCE, TCS, INFY, HDFCBANK, NIFTY 50 · BTCUSDT, ETHUSDT, SOLUSDT.
**Extended (98 more, history from 2000):** 30 US large caps + 15 ETFs, ~49 NSE large caps, 16 crypto
majors - today's names, so **survivorship-biased** (stated on every result).

## Layout
```
config/markets.yaml   markets, 111 instruments (core/extended tiers), costs, frozen core window
src/astra/           common, data (adapters, fundamentals, macro, validation, storage), features, stats,
                      ml, dl, backtest, portfolio, risk, microstructure, research, tracking, agents,
                      dashboard, cli
tests/                unit, data-validation, quantitative-correctness, leakage canaries, dashboard, network
docs/                 final report, status, validation, architecture, ADR 0001-0008, data sources,
                      hypotheses (docs/research/), results (docs/research/results/)
.claude/              development-time Claude Code skills and subagents (not a runtime dependency)
data/ experiments/ logs/   generated, git-ignored
```

## Documentation
[Final report](docs/FINAL_PROJECT_REPORT.md) · [Status](docs/PROJECT_STATUS.md) ·
[Validation](docs/VALIDATION_REPORT.md) · [Results](docs/research/results) ·
[Hypotheses](docs/research/H-EXT.md) · [Architecture](docs/SYSTEM_ARCHITECTURE.md) ·
[Data sources](docs/DATA_SOURCES.md) · [Decisions](docs/TECHNICAL_DECISIONS.md) ·
[Known issues](docs/KNOWN_ISSUES.md) · [Next steps](docs/NEXT_STEPS.md)
