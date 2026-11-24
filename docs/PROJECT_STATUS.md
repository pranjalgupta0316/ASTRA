# Project status

Last session: 2026-11-24. **All milestones M0-M12 have a delivered, tested scope** (scope limits per
milestone below). Final report: [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md).

| Milestone | State | Scope delivered / not delivered |
|---|---|---|
| M0 environment & feasibility | done | env report, source matrix, risk register |
| M1 skills & agents | done | 7 skills + 7 subagents, validator; agents used for real work and audits |
| M2 architecture | done | SYSTEM_ARCHITECTURE, contracts, ADR 0001-0006 |
| M3 engineering foundation | done | uv lock, ruff, pyright, pytest, CI workflow (**not yet run on GitHub - repo not pushed**) |
| M4 data | tested | 13 instruments daily; fundamentals/macro/news **not built** |
| M5 features + stats, volatility, regimes | validated | H-XM-000/001, H-VOL-001, H-REG-001 |
| M6 classical ML + DL | validated (after A2) | logit/RF/XGB/MLP/GRU; Transformers/TCN deliberately not built (no baseline beat the base rate) |
| M7 microstructure | tested | L2 sync + capture + aggTrades; one 5-min capture only |
| M8 portfolio & risk | validated | per-market, local currency; no FX / cross-market |
| M9 backtesting | validated | single-asset and multi-asset event engines, audits |
| M10 tracking & agents | tested | file tracker; deterministic research roles; LLM optional and unbenchmarked |
| M11 dashboard & integration | tested | 12 pages (AppTest), `run-all` reproduction |
| M12 final validation | done | 3 independent audits, fixes applied, 2 identical reproduction runs |

## Environment (measured)
i7-14700HX 20C/28T · 15.8 GB RAM · RTX 4060 Laptop 8.59 GB (driver 616.92, CUDA 13.4 UMD) · project
Python 3.12.16 via uv 0.13.9 · torch 2.11.0+cu128 (`cuda.is_available() = True`).
Measured footprint: data/ 26.8 MB, experiments/ 33.6 MB; full `run-all` 16-18 min; peak DL VRAM 96 MB;
peak study RAM ≈ 0.6 GB process working set (ML grid), portfolio/vol studies < 0.3 GB.

## Repository
`main`, 14+ commits, clean tree at the final commit; nothing pushed (no remote configured).
