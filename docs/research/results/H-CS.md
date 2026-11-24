# H-CS results: cross-sectional momentum (H-CS-001) and short-term reversal (H-CS-002)

## CORRECTION (2026-11-25, after independent audit) - current results
The first run (tables further below, kept for the record) reported momentum as *supported* in US and IN.
The independent backtest audit showed that came from a defect: the portfolio engine's absolute 2 pp
rebalance tolerance never topped up 2-4 % underweights, so the 24-49-name equal-weight **benchmark
drifted to 11-17 % cash** (exposure 0.83-0.89) while the concentrated strategy stayed invested. Re-run
with a full rebalance on every decision date (`rebalance_tolerance=0`; both legs now ≈ 1.00 exposure):

| Hyp | Mkt | Strategy vs benchmark CAGR (net) | Sharpe | NW t (net) | gross t | p_BH | Verdict |
|---|---|---|---|---|---|---|---|
| H-CS-001 momentum | US | 21.5 % vs 17.3 % | 0.99 vs 0.90 | 1.86 | 1.93 | 0.18 | **inconclusive** |
| H-CS-001 momentum | IN | 28.3 % vs 25.1 % | 1.26 vs 1.26 | 1.55 | 1.91 | 0.18 | **inconclusive** |
| H-CS-001 momentum | CRYPTO | 55.6 % vs 46.5 % | 0.92 vs 0.89 | 0.75 | 0.90 | 0.45 | inconclusive |
| H-CS-002 reversal | US | 19.1 % vs 17.5 % | 0.83 vs 0.91 | 1.18 | 2.20 | 0.24 | inconclusive |
| H-CS-002 reversal | IN | 20.8 % vs 26.4 % | 0.97 vs 1.31 | −2.24 | 3.68 | 0.04 | **rejected** (gross edge, destroyed by costs) |
| H-CS-002 reversal | CRYPTO | 15.8 % vs 46.5 % | 0.59 vs 0.89 | −2.21 | −1.49 | 0.04 | **rejected** |

Families: H-CS-001 inconclusive; H-CS-002 rejected. Audit survivorship analysis: ~80 % of the remaining
US momentum excess comes from NFLX, NVDA and AMD (IN: EICHERMOT, ADANIENT, BAJFINANCE) - names in the
universe *because* they later became large caps. Treat any residual momentum as a likely survivorship
artifact until re-tested on a point-in-time universe. The rebalance-tolerance defect was fixed at the
engine level for all multi-asset studies.

---
*Original (superseded) first-run report follows.*

Registered in `docs/research/H-EXT.md` (2026-11-25, before any run). Run 2026-11-25 with
`run_cross_sectional()` (`src/nebula/research/cross_sectional.py`, signals in
`src/nebula/research/signals.py`). Status: **research results, not validated**. They have to be audited
(`backtest-auditor`) before anyone calls them tradeable.

**Survivorship bias.** The universes are today's large caps (30 US stocks), today's NIFTY names
(49 IN stocks) and today's crypto majors (16 USDT pairs), run back over their full history. Names that
failed or were delisted are missing. Both absolute and relative numbers depend on this hindsight
universe. For reversal in particular, "losers that recovered" are over-represented.

## Design (as registered)
- Signals use data up to the close at t, and orders fill at the next open (`nebula.portfolio.backtest`).
  - US/IN momentum: 12-1 log return, log(C[t-21]/C[t-252]), rebalanced on the first session of each month.
  - Crypto momentum: 4-week log return, log(C[t-1]/C[t-28]), rebalanced weekly (first session of the ISO
    week, which is Monday for crypto).
  - Reversal (all markets): -(5-session return), rebalanced weekly.
- Strategy: long-only, equal weight in the top tercile. k = max(1, n//3); names tied at the boundary are
  all included.
- Benchmark: equal weight in every eligible name, with the same dates, engine and costs.
- Eligibility: a valid bar at t, at least 252 valid bars up to t, and a finite signal. No backfilling.
- Costs per side in bps, from config:
  - US: 3, plus a 0.3 sell tax.
  - IN: 6.5, plus 10 tax on each side (33 bps round trip).
  - Crypto: 13 (26 bps round trip).
  - Participation cap: 5% of bar volume. Rebalance tolerance: 2%.
- Study start: the first rebalance with at least 6 rankable names. The run uses full history.
- Decision rule: Newey-West t on the daily net return difference (strategy minus benchmark).
  - Above 1.96: supported. Below -1.96: rejected. Otherwise: inconclusive.
  - Benjamini-Hochberg (q = 0.05) is applied across the 3 markets within each hypothesis. Each record's
    conclusion is the BH-adjusted verdict.
  - `variants_tried` = 3 per hypothesis.
- **Data-hygiene deviation.** The IN panel calendar (the union of per-symbol dates) contained 16 rows
  where only 1 of about 49 listed names had a bar. These are mostly 2003-2005 exchange holidays, plus
  2010-02-06 and 2025-03-18. A rebalance landing on one of them put both legs 100% into that one name.
  - Fix: rows where fewer than 50% of already-listed names have a bar are dropped. This is causal.
    US and crypto have no such rows, and their numbers are byte-identical with or without the filter.
  - The unfiltered run is kept on record as superseded (IDs at the bottom). No verdict changed.

## Results (net, 1x costs; baseline = EW eligible benchmark)

| Hyp | Market | Period | Strat CAGR | Bench CAGR | Strat Sharpe | Bench Sharpe | NW t | p | p_BH | Rule | BH verdict | Strat turnover/yr | Experiment ID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H-CS-001 | US | 2001-01-02..2026-11-23 | 21.41% | 15.99% | 0.994 | 0.920 | 2.71 | 0.0067 | 0.0101 | supported | **supported** | 4.68x | 20261124T200325-H-CS-001-c9a621 |
| H-CS-001 | IN | 2001-02-01..2026-11-23 | 28.61% | 20.32% | 1.275 | 1.208 | 4.15 | 0.00003 | 0.0001 | supported | **supported** | 4.48x | 20261124T200325-H-CS-001-8470f6 |
| H-CS-001 | CRYPTO | 2019-01-14..2026-11-23 | 55.72% | 48.18% | 0.927 | 0.915 | 0.98 | 0.328 | 0.328 | inconclusive | **inconclusive** | 26.6x | 20261124T200325-H-CS-001-0890be |
| H-CS-002 | US | 2001-01-02..2026-11-23 | 18.90% | 16.28% | 0.832 | 0.919 | 1.65 | 0.099 | 0.148 | inconclusive | **inconclusive** | 68.5x | 20261124T200325-H-CS-002-7efec0 |
| H-CS-002 | IN | 2001-01-08..2026-11-23 | 20.52% | 21.04% | 0.975 | 1.227 | 0.25 | 0.807 | 0.807 | inconclusive | **inconclusive** | 68.5x | 20261124T200325-H-CS-002-695dba |
| H-CS-002 | CRYPTO | 2019-01-14..2026-11-23 | 16.30% | 48.18% | 0.591 | 0.915 | -1.95 | 0.051 | 0.148 | inconclusive | **inconclusive** | 65.0x | 20261124T200325-H-CS-002-232ebf |

Family records:
- H-CS-001: **supported** (US and IN supported, crypto inconclusive), `20261124T200325-H-CS-001-310fe9`.
- H-CS-002: **inconclusive** (0 of 3), `20261124T200325-H-CS-002-d5cc73`.

Turnover is two-sided traded notional divided by equity, per year. Benchmark turnover is 0.14-0.17x
for US and IN and 1.19x for crypto.

Secondary statistic (does not decide verdicts): annualised Sharpe difference with a paired stationary
bootstrap 95% CI (block 20, 2000 draws).
- Momentum: US +0.074 [-0.108, 0.248]; IN +0.067 [-0.115, 0.261]; crypto +0.012 [-0.251, 0.245].
- Reversal: US -0.087 [-0.264, 0.095]; IN -0.252 [-0.435, -0.069]; crypto -0.324 [-0.586, -0.075].
- The momentum excess return comes with higher volatility. Strategy vs benchmark annualised vol is
  0.220 vs 0.179 (US) and 0.216 vs 0.165 (IN). **No market shows a significant risk-adjusted
  improvement.** The registered test is on mean return differences, not Sharpe, so it does not control
  for beta or volatility.

## Gross vs net, and cost sensitivity (NW t of the daily difference; ann. mean diff in brackets)

| Hyp | Market | 0x (gross) | 1x | 2x | 4x | Strat CAGR gross / net |
|---|---|---|---|---|---|---|
| H-CS-001 | US | 2.77 (+5.51%) | 2.71 (+5.38%) | 2.64 (+5.24%) | 2.49 (+4.94%) | 21.57% / 21.41% |
| H-CS-001 | IN | 4.54 (+8.35%) | 4.15 (+7.64%) | 3.81 (+7.02%) | 2.95 (+5.41%) | 29.54% / 28.61% |
| H-CS-001 | CRYPTO | 1.06 (+14.06%) | 0.98 (+13.04%) | 0.74 (+10.13%) | 0.21 (+2.91%) | 60.97% / 55.72% |
| H-CS-002 | US | 2.65 (+5.65%) | 1.65 (+3.51%) | 0.64 (+1.35%) | -1.41 (-2.98%) | 21.48% / 18.90% |
| H-CS-002 | IN | 5.72 (+11.36%) | 0.25 (+0.49%) | -5.50 (-11.02%) | -16.77 (-34.09%) | 34.45% / 20.52% |
| H-CS-002 | CRYPTO | -1.39 (-14.03%) | -1.95 (-19.35%) | -3.08 (-30.51%) | -4.91 (-48.99%) | 26.40% / 16.30% |

- Momentum in US and IN is supported at every cost multiple up to 4x.
- Reversal shows a gross edge in US (t 2.65) and IN (t 5.72) that costs erase:
  - IN breaks even at 1x and is strongly negative at 2x.
  - US falls to t 0.64 at 2x and turns negative at 4x.
  - At about 68x turnover a year, IN reversal pays roughly 13.9 pp of CAGR in costs. This matches the
    registered expectation that reversal fails net of costs.
- Crypto reversal underperforms even gross.

## Subperiods (net; first and second half of the daily observations)

| Hyp | Market | Half | Dates | Strat CAGR / Sharpe | Bench CAGR / Sharpe | NW t |
|---|---|---|---|---|---|---|
| H-CS-001 | US | 1 | 2001-01-03..2013-12-11 | 18.27% / 0.873 | 13.23% / 0.731 | 1.62 |
| H-CS-001 | US | 2 | 2013-12-12..2026-11-23 | 24.63% / 1.114 | 18.83% / 1.163 | 2.24 |
| H-CS-001 | IN | 1 | 2001-02-02..2013-12-13 | 33.33% / 1.301 | 25.12% / 1.231 | 2.50 |
| H-CS-001 | IN | 2 | 2013-12-16..2026-11-23 | 24.05% / 1.270 | 15.71% / 1.257 | 3.56 |
| H-CS-001 | CRYPTO | 1 | 2019-01-15..2022-12-19 | 113.89% / 1.215 | 90.43% / 1.191 | 1.13 |
| H-CS-001 | CRYPTO | 2 | 2022-12-20..2026-11-23 | 13.38% / 0.502 | 15.30% / 0.534 | -0.15 |
| H-CS-002 | US | 1 | 2001-01-03..2013-12-11 | 18.49% / 0.752 | 13.83% / 0.737 | 1.67 |
| H-CS-002 | US | 2 | 2013-12-12..2026-11-23 | 19.31% / 0.959 | 18.79% / 1.173 | 0.50 |
| H-CS-002 | IN | 1 | 2001-01-09..2013-12-03 | 28.39% / 1.137 | 26.43% / 1.281 | 0.77 |
| H-CS-002 | IN | 2 | 2013-12-04..2026-11-23 | 13.13% / 0.781 | 15.89% / 1.215 | -0.76 |
| H-CS-002 | CRYPTO | 1 | 2019-01-15..2022-12-19 | 31.17% / 0.770 | 90.43% / 1.191 | -1.79 |
| H-CS-002 | CRYPTO | 2 | 2022-12-20..2026-11-23 | 3.12% / 0.346 | 15.30% / 0.534 | -0.79 |

US momentum is significant only in the second half (first-half t 1.62). IN momentum is significant in
both halves.

## Descriptive long-short (top minus bottom tercile, gross; not executable, no borrow cost)
Each row is the daily top-tercile gross return minus the bottom-tercile gross return. Each leg is
simulated long-only by the engine, so this equals a $1 long / $1 short book re-levered daily.

| Hyp | Market | Ann. mean | Ann. vol | Sharpe | Max DD | NW t |
|---|---|---|---|---|---|---|
| H-CS-001 | US | +5.48% | 20.9% | 0.263 | -43.6% | 1.33 |
| H-CS-001 | IN | +6.27% | 17.1% | 0.367 | -58.8% | 1.74 |
| H-CS-001 | CRYPTO | +43.91% | 62.3% | 0.705 | -61.2% | 2.11 |
| H-CS-002 | US | +5.93% | 20.6% | 0.289 | -60.4% | 1.61 |
| H-CS-002 | IN | +15.05% | 17.3% | 0.868 | -37.8% | 4.57 |
| H-CS-002 | CRYPTO | -19.38% | 50.1% | -0.387 | -94.0% | -1.03 |

These figures do not enter any verdict. The IN reversal spread is large gross, but its turnover
(about 68x per leg) makes it untradeable at the configured IN costs, as the long-only leg above shows.

## Limitations
- Survivorship bias, as stated above.
- Small cross-sections: 24-30 US names, 18-49 IN names, 6-16 crypto pairs. Terciles hold 8-16 names
  in US/IN and 2-5 in crypto.
- The test is on mean return differences, not risk-adjusted ones. Momentum carries more volatility
  than the benchmark.
- Results use daily bars and open fills. Spreads are assumed per market and there is no market impact
  beyond the 5% volume cap. The engine scales buys down when cash falls short, so some legs are
  slightly under-invested; capped and scaled fill counts are in the records.
- The IN calendar filter is a hygiene step decided after the first run exposed the problem. Both runs
  are recorded.
- The US/IN universes include core holdout symbols (MSFT, NVDA, TCS, INFY, HDFCBANK) and ETHUSDT and
  SOLUSDT, as registered in H-EXT. This is a touch of those symbols' history.
- Crypto history starts in 2019 (7.9 years), so power is low there.

## Superseded run (unfiltered IN calendar; recorded, not deleted)
- H-CS-001: family `20261124T200122-H-CS-001-7de8b8`; US `-5fd1cd`, IN `-3b5478`, CRYPTO `-488aeb`.
- H-CS-002: family `20261124T200122-H-CS-002-a90064`; US `-b4f592`, IN `-c1f7fb`, CRYPTO `-ba3c56`.
- IN differences in that run:
  - H-CS-001: NW t 3.96 (strategy CAGR 28.01% vs benchmark 20.15%).
  - H-CS-002: NW t 0.12.
- The verdicts are identical in both runs.

Reproduce: `uv run --no-sync python -c "from nebula.research.cross_sectional import run_cross_sectional; run_cross_sectional()"`
(about 28 s wall time, peak working set about 1.26 GB on this machine).
