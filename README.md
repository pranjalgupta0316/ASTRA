# NEBULA

Local-first quantitative research and AI/ML platform for **US equities, Indian equities and crypto**,
built for rigor over demo value: real free data with provenance, causal features, purged walk-forward
validation, an event-driven cost-aware backtester, and experiment records that keep failed hypotheses.

> Research and paper-trading only. No real-money execution. No paid services. Results are historical
> and are not evidence of future profitability.

## Status
See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for what is implemented, tested and validated.

## Quick start (Windows, PowerShell)
```powershell
uv sync --all-extras            # Python 3.12 + deps (PyTorch CUDA 12.8 wheels; CPU fallback works)
uv run pytest -m "not network"  # offline test suite
uv run nebula ingest            # AAPL, RELIANCE (NSE), BTCUSDT daily bars -> data/
uv run nebula research          # H-XM-000/001/002 per instrument -> experiments/
uv run nebula dashboard         # http://127.0.0.1:8501
```

## Layout
```
config/markets.yaml        markets, instruments, cost assumptions
src/nebula/                common, data, features, stats, ml, backtest, risk, research, tracking, dashboard
tests/                     unit, data-validation, quantitative-correctness, network integration (-m network)
docs/                      architecture, ADRs, data sources, hypotheses, methodology, status
.claude/                   development-time skills and subagents (not a runtime dependency)
data/ experiments/ logs/   generated, git-ignored
```

## Documentation
[Charter](docs/PROJECT_CHARTER.md) · [Architecture](docs/SYSTEM_ARCHITECTURE.md) ·
[Data sources](docs/DATA_SOURCES.md) · [Data dictionary](docs/DATA_DICTIONARY.md) ·
[Hypotheses](docs/RESEARCH_HYPOTHESES.md) · [Methodology](docs/RESEARCH_METHODOLOGY.md) ·
[Validation](docs/VALIDATION_REPORT.md) · [Decisions](docs/TECHNICAL_DECISIONS.md) ·
[Roadmap](docs/IMPLEMENTATION_ROADMAP.md) · [Known issues](docs/KNOWN_ISSUES.md)
