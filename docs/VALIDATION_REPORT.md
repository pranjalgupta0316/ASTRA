# Validation report

Final state 2026-11-24. Research numbers: experiment records from commit `642025c` (clean tree),
reproduced bit-for-bit by a second `nebula run-all` (86/86 identical conclusions, 0 metric deviations).

## Engineering checks (executed at the final commit)
| Check | Command | Result |
|---|---|---|
| Offline tests (unit, data validation, quant correctness, leakage canaries, agents, downloads, CLI, dashboard) | `uv run pytest -m "not network"` | **135 passed** |
| Network integration (Yahoo, Binance, DuckDB view) | `uv run pytest -m network` | **4 passed** |
| Fresh clone | clone + run suite against clone sources | 134 passed (before the dashboard test was added); all package files tracked |
| Lint | `uv run ruff check src tests scripts` | clean |
| Types | `uv run pyright` | 0 errors, ~281 warnings (pandas-stub noise, KI-3) |
| Skills / agents | `uv run python scripts/validate_skills.py` | 7/7 skills, 7/7 agents pass |
| Dashboard | headless Streamlit AppTest over all pages | 12/12 pages render, 0 exceptions |
| Reproducibility | two full `run-all` runs from clean commits | identical |

## Independent audits
1. **Vertical-slice methodology audit** - 6 defects fixed (hindsight ML baseline → A1, fragile canary,
   cost borrowing, approximate gross, placeholder rows, missing sensitivity). Engine recompute ≤ 2.6e-4.
2. **Final methodology audit** (all post-slice work) - PASS on H-XM-003 bootstrap & BH, causal portfolio
   covariance and fills (EW recompute 4.4e-16), volatility/regime causality, Binance sync rules,
   aggressor mapping; FAIL on H-ML-002 per-date DM weighting and PARTIAL on panel fold sizes → fixed as
   amendment A2 (plus purged early stopping, Yahoo 2 h settlement buffer). All verdicts stood.
3. **Test & security review** - CRITICAL: unanchored `data/` in .gitignore excluded `src/nebula/data`
   from git → fixed and verified by fresh clone; HIGH: capture disk cap was only a projection → hard cap;
   MEDIUM: undeclared `websockets`, unused deps, stale docs, untested download paths → fixed; LOW:
   validator rewriting a tracked file, cached files not re-verified, no 429 backoff, LLM redirects →
   fixed.

## Research results
See FINAL_PROJECT_REPORT.md §2 (table) and the per-family documents in docs/research/ and
docs/RESEARCH_HYPOTHESES.md. Every record (including rejected/inconclusive) is in experiments/ and on
the dashboard.

## Status vocabulary
Backtest engines, portfolio engine, features, volatility/regime methodology: **validated** (independent
recompute/audit). ML/DL pipeline: **validated** after A2. Microstructure: **tested** (single live run).
Agents/LLM narration: **tested** (LLM against a fake server only). Nothing is production-ready trading
infrastructure.
