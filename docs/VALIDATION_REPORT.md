# Validation report

Last updated 2026-11-24 (vertical slice). All numbers from `experiments/*/record.json` (latest run).

## Engineering checks (executed)
| Check | Command | Result |
|---|---|---|
| Unit / quantitative-correctness / data-validation tests | `uv run pytest -m "not network"` | **41 passed** |
| Network integration (Yahoo, Binance, DuckDB view) | `uv run pytest -m network` | 4 passed on re-run; AAPL failed once transiently (KI-1) - retries added after |
| Lint | `uv run ruff check src tests scripts` | clean |
| Types | `uv run pyright` | 0 errors, ~100 warnings (pandas-stub noise downgraded, KI-3) |
| Skills / agents | `uv run python scripts/validate_skills.py` | 7/7 skills, 7/7 agents pass |

## Independent backtest audit (separate agent, read-only, re-ran code)
- Independent vectorized recompute of H-XM-001 equity matched the engine within 2.6e-4 relative.
- PASS: signal timing (no same-bar fills), causal features (truncation test), scaler per fold, purge +
  embargo gap, per-side costs on traded notional, annualization, Binance ms/µs, adjustment factors.
- Defects found → fixed in this session:
  1. ML gate used hindsight base rate, no significance test → amendment A1 (train-fold baseline + DM t).
  2. Leakage canary fragile (volume cap bound; BTC 3.07 vs threshold 3.0) → canary now cost-free and
     uncapped: Sharpe 8.8–10.6 vs lagged 0.27–1.10.
  3. Buys borrowed to pay costs; daily drift micro-fills → buys sized net of cost, 2% rebalance
     tolerance (AAPL TSMOM fills 133 → 95 = number of signal flips).
  4. "Gross" curve was net + cumulative costs → exact zero-cost re-run (`run_gross`).
  5. Yahoo placeholder rows (V=0, H=L) → excluded from research (recorded per experiment); engine never
     fills on zero-volume bars.
  6. H-XM-002 lacked cost sensitivity/turnover → added.
- Open from audit: H-XM-001's mean-difference test does not test the stated drawdown mechanism
  (→ H-XM-003 proposed); no multiple-testing adjustment across the 9 tests (no result is marginal-supported).

## Research results (net of modelled costs, OOS where fitted)
| Instrument | H-XM-000 stylized facts | H-XM-001 TSMOM-60 vs B&H | H-XM-002 logistic (A1 rule) |
|---|---|---|---|
| AAPL | supported | inconclusive: net SR 0.93 [0.38, 1.62] vs B&H 0.99, NW t −1.49 | inconclusive: AUC 0.508, DM t −0.93; SR 0.86 vs 1.07, t −1.67 (original rule: rejected) |
| RELIANCE | supported | inconclusive: SR 0.29 [−0.45, 1.06] vs 0.66, t −1.90 | **rejected**: AUC 0.483, DM t −2.43; SR −0.35 vs 0.63, t −3.37 |
| BTCUSDT | supported | inconclusive: SR 1.01 [0.29, 1.66] vs 0.63, t 0.28 | inconclusive: AUC 0.528, DM t 0.06; SR 0.60 vs 0.79, t −1.43 (original rule: rejected) |

SR = annualized Sharpe of daily net returns, [95% stationary-bootstrap CI]. Equities 252, crypto 365.

**Interpretation**: no evidence of a tradeable edge. Stylized facts hold everywhere (pipeline sanity).
Simple momentum does not beat buy-and-hold on mean return in any market; ML on technical features shows
no out-of-sample skill and, for RELIANCE, is significantly harmful after 33 bps round-trip costs.
These are the expected null results for liquid large-caps and BTC on daily data.

## Status vocabulary for this slice
Data ingestion, validation, features, engine, stats, ML baseline: **tested**. Research conclusions:
**validated** by independent audit (methodology), with the open items above. Nothing is production-ready.
