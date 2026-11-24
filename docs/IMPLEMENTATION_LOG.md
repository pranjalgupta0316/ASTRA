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

## 2026-11-24 - Session 1 (continued): M5-M12, autonomous mode
**Work**: GPU verified; universe → 13; parallel specialist agents (quant-researcher: volatility/regimes;
portfolio-risk-analyst: portfolio/risk; microstructure-specialist: L2/aggTrades) on disjoint files;
orchestrator: ML/DL grid + panel, holdout study, runtime agents, dashboard, CLI, run-all.
**Audits**: backtest-auditor (final methodology), test-security-reviewer → fixes in commit 2ec4dbe.
**Key commands**: `nebula ingest` (13 instruments), `nebula ml-study` (13.5 min), `nebula holdout`,
`nebula run-all --skip-ingest` ×3 (17.6 / 16.0 min; last two from clean commits, results identical),
`pytest -m "not network"` → 135 passed, `pytest -m network` → 4 passed, ruff clean, pyright 0 errors.
**Incidents**: PowerShell 5.1 `Get-Content -Raw` re-encoded a UTF-8 file (emoji mojibake) → fixed,
subsequent scripted edits use explicit UTF-8; `.gitignore` excluded `src/nebula/data` → fixed (06b9574);
a run made while docs were being edited produced dirty-tree records → rerun from a clean commit.
**Environment changes**: removed unused deps (pydantic-settings, hypothesis, pytest-cov, mlflow-skinny);
added websockets as a direct dependency.
