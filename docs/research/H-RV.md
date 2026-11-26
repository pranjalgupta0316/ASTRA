# Realized-volatility hypotheses (registered 2026-11-26, committed before implementation or any run)

Motivation: H-VOL-001 compared daily volatility models against the squared daily return, a very noisy
proxy of true variance, and every comparison was inconclusive. ASTRA holds ~4.87 M one-minute bars each
for BTCUSDT and ETHUSDT (and ~1 M hourly bars for 16 pairs) that no study has used. Realized variance
(RV) from intraday returns is a far more precise proxy and enables the standard HAR-RV benchmark
(Corsi 2009, Journal of Financial Econometrics; concept only - not a reproduction).

Common design (fixed a priori; no tuning)
- Data: run under data lock `v0.3.0-baseline` (`astra --lock v0.3.0-baseline ...`); validated Binance
  spot klines; off-grid bars excluded by the existing loader. Days are UTC days.
- Realized variance: RV5_t = sum of squared 5-minute log returns within UTC day t (288 returns), built
  from the 1-minute closes (5-minute sampling is the standard noise/precision compromise; Liu, Patton &
  Sheppard 2015, J. Econometrics - concept). Robustness: RV1 from 1-minute returns. A day is valid only if
  >= 95 % of its 1-minute bars exist; invalid days are excluded from evaluation (never filled).
- Forecasts: one-step-ahead variance for day t+1 using information up to the end of day t.
  - HAR-RV (levels): RV_{t+1} = b0 + b_d RV_t + b_w mean(RV_{t-4..t}) + b_m mean(RV_{t-21..t}) + e, OLS.
  - Benchmarks: EWMA (lambda 0.94) of squared daily close-to-close returns; GARCH(1,1) and GJR-GARCH(1,1,1)
    with Student-t on daily returns (existing `astra.stats.volatility`); RV random walk (RV_{t+1} = RV_t);
    22-day mean of RV.
  - Expanding window, minimum 730 valid days of training, re-fit every 21 days, parameters frozen between
    re-fits. Negative HAR forecasts are floored at the smallest positive training RV (counted, reported).
- Evaluation target: RV5_{t+1}. Losses: QLIKE (primary) and MSE. Diebold-Mariano with Newey-West
  standard errors on the daily loss differential; Benjamini-Hochberg within each hypothesis family.

## H-RV-001 HAR-RV beats GARCH-family and EWMA on BTCUSDT and ETHUSDT
- Comparisons: HAR vs EWMA, HAR vs GARCH(1,1), HAR vs GJR, for BTCUSDT and ETHUSDT = 6 cells.
- Cell supported if HAR has lower QLIKE with DM t > 1.96 after BH; rejected if significantly worse.
- Family supported if HAR is significantly better than EWMA and GARCH(1,1) for both assets.

## H-RV-002 Hourly realized variance, 16 crypto pairs
- RV_h_t = sum of squared hourly log returns in UTC day t (24 returns; valid if >= 22 hourly bars).
- HAR on RV_h vs EWMA on daily returns, evaluated against RV_h_{t+1}; 16 cells, BH. Family supported if
  >= 12 of 16 cells are BH-significant improvements. (Noisier proxy than RV5 - a scalability check.)

## H-RV-003 Re-test of H-VOL-001 with a precise proxy (BTCUSDT, ETHUSDT)
- GJR-GARCH vs EWMA, evaluated against RV5_{t+1} instead of the squared daily return. 2 cells, BH.
- Purpose: was H-VOL-001 inconclusive because of the noisy proxy? The original H-VOL-001 records are
  untouched; this is a new, separately recorded test.

## Descriptive (no hypothesis): forecast quality by volatility regime and Mincer-Zarnowitz regressions
(RV_{t+1} on forecast: intercept, slope, R^2) for every model, reported per asset.
