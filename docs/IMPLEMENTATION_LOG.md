# Implementation log

## 2026-11-24 - Session 1: preflight + vertical slice
**Created**: pyproject.toml, uv.lock, .gitignore, .env.example, README.md, config/markets.yaml,
src/nebula/{common,data,data/adapters,features,stats,ml,backtest,risk,research,tracking,dashboard}/*,
src/nebula/cli.py, tests/* (7 files), scripts/validate_skills.py, .github/workflows/ci.yml,
.claude/skills/* (7 skills + README, REGISTRY, DEPENDENCIES, VALIDATION_REPORT), .claude/agents/* (7),
.claude/AGENT_ARCHITECTURE.md, .claude/AGENT_WORKFLOWS.md, .claude/launch.json, docs/* (all listed in README).

**Commands executed (key)**
- `uv python install 3.12`; `uv sync --python 3.12` (core deps OK); `uv sync --all-extras` (PyTorch cu128 - still downloading at session end)
- `uv run nebula ingest` → 3 instruments ingested (AAPL 2236 rows pass; RELIANCE 2196 warn; BTCUSDT 3249 warn)
- `uv run nebula research` ×2 (second run after audit fixes; both runs retained in experiments/)
- `uv run pytest -m "not network"` → 41 passed; `uv run pytest -m network` → 3/4 then 1/1 + 3 repeats passed
- `uv run ruff check` clean; `uv run pyright` 0 errors
- Streamlit dashboard started on 127.0.0.1:8501; Data and Backtests pages verified rendering stored outputs.

**Decisions**: ADR-0001..0004; pyright pandas-noise rules → warnings; XBOM as NSE calendar proxy;
6 subagents + orchestrator; decision-rule amendment A1 (post-audit, disclosed).

**Independent audit**: general-purpose agent with backtest-auditor brief; 6 defects fixed (see VALIDATION_REPORT).

**Environment changes**: added CPython 3.12.16 under uv; project .venv; no global installs.
