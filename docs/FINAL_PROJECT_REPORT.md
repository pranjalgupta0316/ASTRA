# NEBULA - final project report

*2026-11-24. All numbers come from experiment records produced from commit `642025c` (clean tree) and
reproduced bit-for-bit by a second full run (86/86 records: identical conclusions, 0 metric deviations).*

## 1. What NEBULA is
A local-first, ₹0-recurring-cost quantitative research platform for US equities, Indian equities and
crypto: free data with provenance → validation → causal features → statistical, ML/DL, volatility,
regime, portfolio and microstructure studies → event-driven cost-aware backtests → experiment records
(including every negative result) → deterministic research agents → Streamlit dashboard. It runs on one
Windows laptop (i7-14700HX, 16 GB RAM, RTX 4060 Laptop 8 GB); `nebula run-all` reproduces every study
in ~16-18 minutes.

## 2. Headline research findings (honest summary)
**No tradeable edge was found.** Every claim of outperformance tested out of sample, net of modelled
costs, was rejected or inconclusive. What *did* hold up are well-known statistical properties.

| Family | Question | Result |
|---|---|---|
| H-XM-000 | Stylized facts (unit-root prices, stationary returns, volatility clustering) | **supported 13/13** |
| H-XM-001 | 60d time-series momentum beats buy-and-hold (mean daily net return) | inconclusive 11, rejected 1 (MSFT, t −2.44), "supported" 1 (SOLUSDT, t 2.12 - does not survive BH across 13; confirmatory H-XM-003 on SOL inconclusive) |
| H-XM-002 | Logistic regression on 13 technical features | inconclusive 9, **rejected 4** (RELIANCE, SPY, INFY, NIFTY50: model or strategy significantly worse) |
| H-XM-003 | Confirmatory: momentum Sharpe > buy-and-hold on 10 held-out instruments | **inconclusive** (0 BH-significant; median ΔSR −0.06). Descriptively lower max drawdown on 9/10 (not a registered test) |
| H-ML-001 | Any of logit / RF / XGBoost / MLP / GRU × 13 instruments beats the ex-ante base rate | **rejected**: 0/65 cells, 0 nominally significant, 21 significantly *worse* (XGBoost on 13/13 - overconfident) |
| H-ML-002 | Pooled-panel MLP / GRU (29.5k rows, GPU) beat baseline and pooled logistic | **inconclusive** (MLP DM t −0.43 vs baseline; GRU −1.73) |
| H-VOL-001 | GJR-GARCH beats EWMA variance forecasts (QLIKE, DM) | **inconclusive 13/13** after BH (GJR lower QLIKE on 12/13 point estimates; NVDA nominal only) |
| H-REG-001 | OOS high-vol GMM state → higher next-day variance | **supported 10/13** (close to restating volatility clustering) |
| H-PF-001 | Min-variance / risk parity have lower realized vol than equal weight | **supported US & crypto**, inconclusive India |
| H-PF-002 | ... and higher net return than equal weight | **rejected US**, inconclusive India & crypto |
| H-MS-000 | BTCUSDT aggTrades price impact (Kyle λ, 1 day) | supported: 0.286 bps/BTC, HAC t 3.49 (one day, same-minute association) |
| H-MS-001 | Top-5 book imbalance predicts next mid move (5-min live L2 capture) | supported by registered rule: AUC 0.81 (1 s), 0.60 (10 s); weak evidence, not tradeable after taker fees |

Portfolio results (net, rf = 0): US EW Sharpe 1.16 vs MinVar 0.89 / RP 1.09; India EW 0.58 vs 0.57 / 0.57;
crypto EW 0.87 vs 0.80 / 0.84. Lower volatility was bought with lower return.

## 3. What was built (and tested)
| Area | Status | Evidence |
|---|---|---|
| Data: Yahoo (US, NSE), Binance Public Data (crypto), 13 instruments, raw sha256 provenance, 9 quality checks, calendars, incomplete-bar and placeholder handling | tested | 4 network + adapter/validation unit tests; quality reports per batch |
| Features (13 causal) + targets | tested | truncation-invariance test; independent audit |
| Statistics & inference (ADF/KPSS, Ljung-Box, NW, bootstrap Sharpe CI & difference, PSR/DSR, BH) | tested | known-value tests |
| Event-driven backtester (next-open fills, per-side costs, volume caps, zero-volume deferral, cost-aware sizing, exact gross) | **validated** | two independent audits; recompute within 2.6e-4; leakage canary |
| Multi-asset portfolio engine + CVXPY optimizers + risk attribution | **validated** | audit recompute of EW equity to 4.4e-16; analytic tests |
| Volatility (EWMA/GARCH/GJR) and regimes (walk-forward GMM) | validated (methodology) | GARCH recursion = arch to 1e-14; causality checks |
| ML (sklearn, XGBoost) + DL (PyTorch MLP/GRU, GPU) with purged walk-forward | validated after amendment A2 | audit; reproducible across runs |
| Microstructure: Binance L2 sync & gap detection, OFI/microprice, aggTrades, disk-capped capture | tested; live run 0 gaps, replay matched 2 snapshots | 21 tests |
| Experiment tracking (incl. negative results, git commit, dirty flag) | tested | 86 records from a clean commit |
| Runtime research agents (deterministic) + report | tested | 7 tests; optional local LLM tested against fake server only |
| Dashboard (12 pages, stored outputs only, localhost) | tested | headless AppTest renders all 12 pages, 0 exceptions |
| Claude Code skills (7) + subagents (7) | statically validated; used in practice | validator; 6 agents performed real work and audits |

Quality gates at the final commit: **135 offline tests + 4 network tests pass**, ruff clean, pyright
0 errors (≈280 pandas-stub warnings), fresh clone passes the suite.

## 4. Process highlights
- Hypotheses registered before results; 2 post-hoc rule amendments disclosed with original verdicts kept.
- Three independent audits (two methodology, one test/security). They found and we fixed: hindsight
  baseline, fragile leakage canary, cost borrowing, approximate gross curve, equal-weighted panel DM,
  fold-size mismatch, unpurged early stopping, **an unanchored `.gitignore` that excluded the data
  package from git**, an incomplete in-progress Yahoo bar, missing download hardening and disk caps.
- Parallel specialist agents built volatility/regimes, portfolio and microstructure on disjoint files;
  an independent auditor reviewed all producers' work.

## 5. Limitations (see KNOWN_ISSUES.md for the full list)
Survivorship-biased 13-name universe; assumed spreads; daily bars only for equities; one venue and one
short capture for microstructure; no fundamentals, macro or news yet (researched, not built); no FX or
cross-market portfolio; DL results device-dependent; optional LLM not benchmarked. NEBULA is a research
platform - **not** production trading infrastructure.

## 6. How to reproduce
```powershell
uv sync --all-extras
uv run nebula ingest
uv run nebula run-all --skip-ingest
uv run pytest -m "not network"
uv run nebula dashboard
```
