# Validation report

Final state 2026-11-25 (v0.3). Research numbers: experiment records reproduced from clean commit
`d3653bb` (see PROJECT_STATUS for provenance and the two documented verdict changes).

## Engineering checks at the final commit
| Check | Result |
|---|---|
| Offline tests (`uv run pytest -m "not network"`) | 203 passed |
| Network integration (`-m network`) | 4 passed (last run) |
| Lint (`ruff check src tests scripts`) | clean |
| Types (`pyright`) | 0 errors (pandas-stub noise as warnings, KI-3) |
| Skills / agents (`scripts/validate_skills.py`) | 7/7, 7/7 |
| Dashboard (AppTest, all pages) | 0 exceptions |
| Full reproduction (`run-all`) | 9/9 steps, 0 dirty records |

## Independent audits (separate read-only agents that re-ran code)
| # | Scope | Key findings → fixes |
|---|---|---|
| 1 | Vertical slice | hindsight ML baseline (A1), fragile canary, cost borrowing, approximate gross, placeholder rows |
| 2 | Post-slice methodology | per-date DM weighting, fold sizes, unpurged early stopping (A2), Yahoo settlement buffer |
| 3 | Tests & security | **`.gitignore` excluded `src/nebula/data`**, disk cap, deps, download hardening |
| 4 | Cross-sectional momentum | **benchmark cash drag from absolute rebalance tolerance** → momentum "supported" became inconclusive; survivorship concentration quantified |
| 5 | Vol-managed / cross-market | **under-invested buy-and-hold benchmark after volume-capped first fill** → single-instrument counts corrected (also fixed SOLUSDT core results) |
| 6 | ML panel / hourly / multimodal | no leakage (canary AUC 0.998); wording: statistical vs economic significance; breakeven ≈ 5 bps |

Every audit's defects were fixed and studies re-run; superseded records remain in `experiments/` and
each results document carries a correction section.

## Results
See FINAL_PROJECT_REPORT.md §3 and `docs/research/results/`. Summary: no strategy survives realistic
costs; statistically significant short-horizon predictability (hourly crypto, order book) is an order
of magnitude smaller than trading costs; well-known statistical properties (stylized facts, volatility
clustering, lower-vol portfolios) are confirmed.
