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

## Amendment 1 (2026-11-26; written and committed after inspecting data coverage only, before any RV, forecast or loss was computed)
The registered text above is unchanged. These points specify what the text leaves open or cannot satisfy
literally. None of them was chosen by looking at results.

1. **Missing minutes inside a valid day.** "288 returns" holds only on complete days. On a valid day with
   missing minutes, RV5 is the sum of squared log returns between *consecutive observed* 5-minute grid
   closes (closes at 00:05, 00:10, ..., 24:00 of day t; the 00:00 point is the close of the previous
   day's 23:59 bar). A gap therefore becomes one longer return spanning the gap. Nothing is filled or
   interpolated. The segment before the first observed grid point of the day is dropped. RV1 (1-minute
   grid) and RV_h (hourly grid, 00:00 point = previous day's 23:00-bar close) follow the same rule.
   Off-grid bars are dropped before anything else. Zero-volume, flat 1-minute bars are real no-trade
   minutes (return 0) and count as existing bars.
2. **Invalid days inside a HAR window.** The daily regressor needs RV_t valid. The weekly and monthly
   regressors are means over the *valid* days in t-4..t and t-21..t. Invalid days are skipped, never
   filled. Origins with RV_t invalid get no HAR, RV-random-walk or 22-day-mean forecast. The 22-day mean
   benchmark uses the same valid-day mean. HAR training rows need RV_s and RV_{s+1} both valid.
3. **Daily returns for EWMA/GARCH/GJR** are close-to-close log returns from the validated 1d klines (no
   gaps under the lock), so r_t covers the same UTC day as RV_t. Full available history per pair (no
   core window), for H-RV-001/003 and for all 16 pairs in H-RV-002 (BTCUSDT and ETHUSDT included).
4. **Refit schedules.** HAR: first fit at the first origin o with >= 730 training rows whose target day
   is <= o, then re-fit every 21 calendar days. GARCH/GJR: the existing `garch_forecast` (zero mean,
   Student-t, percent scaling) with min_train = 730 daily returns and refit_every = 21, anchored at the
   start of the return series. EWMA: the existing `ewma_forecast` (lambda 0.94). The HAR floor is the
   smallest positive valid RV on days <= the refit origin.
5. **Evaluation sample.** For each asset/target, this is the set of origins t where every model
   compared on that asset has a positive forecast and day t+1 is valid. One common sample is used for
   all models of that asset. Losses: normalised QLIKE (RV/h - ln(RV/h) - 1) and MSE in percent^2
   (variance x 1e4). If a target RV were 0, DM on QLIKE would use the ranking form RV/h + ln h (identical
   differences). Such cases are counted.
6. **DM and BH.** DM t is the Newey-West t of mean(loss_benchmark - loss_HAR), with the default lag
   floor(4 (n/100)^(2/9)) as in H-VOL-001. Positive t means HAR is better (in H-RV-003: GJR is better).
   Two-sided normal p. BH at q = 0.05 within each family (6, 16 and 2 cells). Cell "supported": BH
   rejects and t > 0. "Rejected": BH rejects and t < 0. Otherwise "inconclusive". MSE DMs are secondary
   and reported only.
7. **Family verdicts not defined above.** A family is "supported" if its registered criterion is met. It
   is "rejected" if the mirrored criterion is met (the benchmark is significantly better in the same
   cells: H-RV-001, EWMA and GARCH(1,1) beat HAR on both assets; H-RV-002, >= 12 of 16 cells
   significantly worse). Otherwise it is "inconclusive". H-RV-003: "supported" if both cells are
   supported, "rejected" if both are rejected, otherwise "inconclusive".
8. **RV1 robustness** (descriptive, no verdict, not in any BH family): HAR re-fit on RV1 and evaluated
   against RV1_{t+1}, with the benchmarks unchanged.
9. **Regimes** (descriptive): terciles of the origin's 22-day valid-mean RV5 over the evaluation sample.
   Mincer-Zarnowitz is plain OLS of RV_{t+1} on the forecast (intercept, slope, R^2) on the evaluation
   sample.
