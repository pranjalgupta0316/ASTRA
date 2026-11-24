# Project status

Last session: 2026-11-24. Current milestone: **vertical slice (M0–M6 subset) complete and audited**.

| Milestone | State | Evidence |
|---|---|---|
| M0 environment & feasibility | done | env inspected (below); docs/DATA_SOURCES.md matrix; docs/RISK_REGISTER.md |
| M1 skills & agents | implemented, statically validated | 7 skills + 7 agents pass scripts/validate_skills.py. Discoverability: Claude Code registered all 7 agents and 7 skills when the session moved into C:\Projects\NEBULA (observed in session) |
| M2 architecture | designed | SYSTEM_ARCHITECTURE.md, contracts.py, ADR 0001–0004 |
| M3 engineering foundation | implemented | uv + lock, ruff clean, pyright 0 errors, pytest, CI workflow written (**not yet run on GitHub** - repo not pushed) |
| M4 data (AAPL, RELIANCE.NS, BTCUSDT daily) | tested on real data | quality: AAPL pass; RELIANCE warn (Muhurat sessions, 2 Yahoo gaps, 2 placeholder rows); BTC warn (2020-03-12 crash, real) |
| M5 features + stats baselines | tested | causal-feature truncation test; H-XM-000/001 recorded |
| M6 classical ML | tested; DL **not started** | H-XM-002 recorded; PyTorch install still downloading (KI-2) |
| M7–M12 | planned | see IMPLEMENTATION_ROADMAP.md |

## Environment (measured 2026-11-24)
i7-14700HX 20C/28T · 15.8 GB RAM (5.6 GB free at inspection) · RTX 4060 Laptop 8 GB, driver 616.92
(CUDA 13.4 UMD; no system CUDA toolkit - not needed for PyTorch wheels) · C: 168 GB free · system
Python 3.14.4, project Python 3.12.16 (uv 0.13.9) · git 2.55 · Node 24.
GPU availability to PyTorch: **verified** - torch 2.11.0+cu128, `cuda.is_available() = True`,
RTX 4060 Laptop 8.59 GB; 20 × 4096² fp32 matmuls in 0.45 s, peak 1.4 GB allocated.
Data footprint of the slice: 1.13 MB.

## Repository
Git initialized on `main`, **no commits yet**.
