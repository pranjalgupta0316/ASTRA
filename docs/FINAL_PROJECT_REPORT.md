# NEBULA - final project report (v0.3)

*2026-11-25. Numbers come from experiment records in `experiments/`; the final `nebula run-all` was run
from a clean commit (see PROJECT_STATUS). Every hypothesis was registered before it was run
(docs/RESEARCH_HYPOTHESES.md, docs/research/H-ML.md, docs/research/H-EXT.md), and every
trading-relevant result went through an independent audit by a separate agent that re-ran the code.*

## 1. What NEBULA is
A local-first, ₹0-recurring-cost quantitative research platform for US equities, Indian equities and
crypto, running on one Windows laptop (i7-14700HX, 16 GB RAM, RTX 4060 Laptop 8 GB):
free data with provenance → validation → causal features → statistical, ML/DL, volatility, regime,
portfolio, cross-sectional, multimodal and order-book studies → event-driven cost-aware backtests →
experiment records that keep every negative result → deterministic research agents (+ optional local
LLM narration) → a 13-page Streamlit dashboard.

## 2. Data actually used
| Data | Source | Size |
|---|---|---|
| Daily bars, 111 instruments (45 US, 50 India, 16 crypto) | Yahoo, Binance Public Data | 637k bars, equities from 2000 |
| Hourly klines, 16 crypto pairs | Binance Public Data | ~1.0M bars |
| 1-minute klines BTC, ETH | Binance Public Data | ~4.87M bars each |
| US fundamentals (30 stocks) | SEC EDGAR (point-in-time) | 59,726 facts |
| Macro | Yahoo yields/VIX/DXY/USDINR + BLS CPI & unemployment | 2000→ |
| Live order book BTCUSDT | Binance WebSocket | 6 h + 5 min, ~211k L2 updates, 0 gaps |

## 3. Findings - the honest summary
**No strategy beat its benchmark after realistic trading costs.** Predictability exists at very short
horizons (hourly crypto, sub-minute order book) but is far smaller than trading costs. Two apparent
wins (cross-sectional momentum; single-asset volatility management) were traced by independent audits
to engine defects and survivorship, then corrected.

| Family | Question | Verdict |
|---|---|---|
| H-XM-000 | Stylized facts (13 core instruments) | supported 13/13 |
| H-XM-001 / 003 | 60d time-series momentum vs buy-and-hold (13 core / 10 held-out) | inconclusive 12 + rejected 1 (MSFT) / inconclusive 10. An earlier SOLUSDT "win" was an artifact of the under-invested benchmark bug |
| H-XM-002 | Logistic regression per instrument | inconclusive / 4 rejected |
| H-ML-001 | 5 models × 13 instruments beat base rate | rejected (0/65) |
| H-ML-002 | Pooled MLP/GRU (13 instruments) | MLP inconclusive; GRU rejected (borderline: t moved from −1.73 to < −1.96 after Yahoo restated adjusted prices) |
| **H-ML-003** | Pooled panel, 111 instruments, 600k rows | **supported statistically**, economically negligible (0.0004 nats, AUC 0.517; logit borderline) |
| **H-ML-004** | Hourly crypto, 1M rows | **supported statistically** (AUC 0.54, DM t 10-12) - **breakeven ≈ 5 bps vs 26 bps cost; net Sharpe −4 to −6** |
| H-MM-001 | Price + SEC fundamentals + macro | rejected - neither modality helps |
| H-VOL-001 | GJR-GARCH vs EWMA volatility forecasts | inconclusive 13/13 |
| H-REG-001 | Regime state predicts next-day variance | supported 10/13 (≈ volatility clustering) |
| H-PF-001 / 002 | Min-var / risk parity vs equal weight | lower vol supported (US, crypto); higher return rejected (US) / inconclusive |
| H-CS-001 | Cross-sectional momentum (US 30, IN 49, crypto 16) | inconclusive (t 1.86 / 1.55 / 0.75) after audit fix; residual excess concentrated in later mega caps (survivorship) |
| H-CS-002 | Short-term reversal | rejected (gross edge destroyed by costs) |
| H-VT-001 | Volatility-managed exposure (114 series) | inconclusive (0 BH-significant; 7 raw-CI positives incl. EW US/IN +0.14/+0.17 ΔSharpe, shallower drawdowns) |
| H-XMKT-001 | US + India + crypto diversification | inconclusive |
| H-MS-000 | BTC price impact (Kyle λ, 1 day) | supported, descriptive |
| **H-MS-001 / 002** | Order-book imbalance → next mid move (6 h) | **supported**: AUC 0.91 (1 s), 0.66 (10 s); **taker strategy −19.6 bps/trade net, 0/2,095 trades profitable** |

The pattern is the textbook one: the closer to the order book, the stronger the statistical signal -
and the larger the cost relative to the move it predicts.

## 4. What was built and how it was validated
- Data layer: adapters (Yahoo, Binance daily/intraday, SEC, BLS, Yahoo macro), checksummed downloads,
  raw sha256 provenance, 9+ quality checks, calendars, point-in-time fundamentals, release-date-aware macro.
- Research: causal features; ADF/KPSS/Ljung-Box/NW/bootstrap/PSR/DSR/BH; purged walk-forward; logistic,
  RF, XGBoost, PyTorch MLP/GRU on GPU; GARCH family; walk-forward GMM regimes; CVXPY portfolios;
  cross-sectional and volatility-managed strategies; L2 book reconstruction.
- Engines: single-asset and multi-asset event-driven backtesters (next-open fills, per-side costs,
  volume caps, zero-volume handling, exact gross runs).
- Validation: 200+ offline tests, network tests, leakage canaries, **six independent audits** that found
  and forced fixes for: a hindsight ML baseline, a fragile canary, cost borrowing, an excluded data
  package (.gitignore), Yahoo in-progress bars, benchmark cash drag (rebalance tolerance), an
  under-invested buy-and-hold benchmark, panel DM weighting, and more. Superseded records are kept.
- Reproducibility: frozen sample windows for registered core studies; `nebula run-all` reproduces all
  studies; two earlier full reruns matched bit-for-bit on 86/86 records.

## 5. Limitations
Survivorship-biased universes (today's names); assumed spreads for equities; one venue and one night of
order-book data; revised (not first-release) macro; no Indian fundamentals; no news data; DL results
device-dependent; GRU on the full panel near the laptop's memory limits. NEBULA is a research
platform, **not** production trading infrastructure.

## 6. Reproduce
```powershell
uv sync --all-extras
uv run nebula ingest; uv run nebula ingest --intraday; uv run nebula ingest-fundamentals; uv run nebula ingest-macro
uv run nebula run-all --skip-ingest            # add --heavy-dl for the ~3 h MLP/GRU runs
uv run pytest -m "not network"
uv run nebula dashboard
```
