# Volatility forecasting and regime hypotheses (milestone M5)

Registered 2026-11-24 **before** any forecast, loss or regime statistic was computed on real data.
All parameters below are fixed a priori; nothing is tuned. Results are filled in from experiment
records (`experiments/`), never edited to look better. Negative and inconclusive results are kept.

Universe (13): US AAPL MSFT NVDA SPY QQQ; IN RELIANCE TCS INFY HDFCBANK NIFTY50;
CRYPTO BTCUSDT ETHUSDT SOLUSDT. Validated daily bars, total-return adjusted OHLC
(`astra.features.technical.adjusted_ohlc`). Excluded rows: provider placeholders
(volume == 0 and high == low); for NIFTY50 (Yahoo index volume unreliable) high == low only.
Excluded rows are dropped, not repaired (the following return spans two sessions), and listed in each
experiment record.

Common definitions
- r_t = ln(C_t / C_{t-1}) on adjusted closes (decimal). Models are fit on 100 * r_t (arch scaling).
- Forecast origin t (close of session t, information <= t) -> target session t+1.
- Realized-variance proxy (primary): RV_{t+1} = r_{t+1}^2 (close-to-close; conditionally unbiased for
  the quantity every model forecasts). Secondary, descriptive only: Garman-Klass and Parkinson
  per-period variances (they omit the overnight gap, so they are biased proxies for close-to-close
  variance and are NOT used for any decision).
- QLIKE(RV, h) = RV/h - ln(RV/h) - 1 (Patton 2011). Because RV = 0 on unchanged-close days makes the
  normalised form infinite, DM tests use the equivalent form RV/h + ln h, whose per-day differences
  between two models are identical. MSE = (RV - h)^2.
- Diebold-Mariano: t = mean(d) / HAC se, d_t = L(model)_t - L(EWMA)_t, Newey-West with
  floor(4 (n/100)^(2/9)) lags (`astra.stats.inference.newey_west_tstat`); p two-sided, normal.
- Out-of-sample window: every origin from index 503 (504 returns available, ~2 equity years) to the
  second-to-last session. Walk-forward only; the OOS window is evaluated exactly once
  (test-set touches = 1 per instrument per hypothesis).

---

```
ID:              H-VOL-001
Title:           GJR-GARCH(1,1,1) beats EWMA(0.94) for one-step daily variance under QLIKE
Status:          inconclusive (13/13 instruments)
Registered:      2026-11-24 (before results were seen)
1. Intuition:          Volatility mean-reverts and responds asymmetrically to negative returns
                       (leverage / volatility-feedback effect). EWMA has no mean reversion
                       (integrated, alpha + beta = 1, omega = 0) and no asymmetry, so a fitted
                       GJR-GARCH should forecast one-step variance better, mainly for equities;
                       crypto asymmetry is weaker / sign-ambiguous in the literature.
2. Formulation:        h^GJR_{t+1|t} = omega + (alpha + gamma 1[e_t<0]) e_t^2 + beta h_t, zero mean,
                       Student-t innovations, MLE on 100*r over the expanding window [0, T_k],
                       refit at origins T_k = 503 + 21k; between refits the parameters are frozen
                       and the recursion is run forward on new data (arch-identical backcast:
                       0.94-weighted mean of the first 75 squared returns).
                       Baseline h^EWMA_{t+1|t} = 0.94 h^EWMA_{t|t-1} + 0.06 r_t^2 (seeded with r_0^2,
                       first 19 origins masked).
                       Other forecasters reported (secondary, descriptive): GARCH(1,1)-t (same
                       schedule), rolling mean of r^2 over 20 and 60 sessions.
3. Data:               validated daily bars 2018-01-01 .. 2026-11-23 (SOLUSDT from 2020-08-11),
                       close known at the session close; no availability lag beyond that.
4. Features:           past log returns only.
5. Target:             RV_{t+1} = r_{t+1}^2, horizon 1 session, log return, raw (not excess).
6. Mechanism:          Statistical forecasting claim, not a trade. Any risk-targeting use is a
                       separate hypothesis that must go to backtest-auditor.
7. H0:                 E[QLIKE(GJR) - QLIKE(EWMA)] = 0 (DM, two-sided).
8. Validation design:  expanding walk-forward, min train 504 returns, refit every 21 sessions,
                       one-step-ahead, no overlap of target with training data (target t+1 >
                       last training return T_k <= t). Decision per instrument on BH-adjusted
                       (Benjamini-Hochberg, q = 0.05, family = 13 instruments) two-sided DM p:
                         p_BH < 0.05 and t < 0 -> supported (GJR lower QLIKE)
                         p_BH < 0.05 and t > 0 -> rejected  (EWMA lower QLIKE)
                         otherwise             -> inconclusive.
                       Family summary = count of supported / rejected / inconclusive; raw (unadjusted)
                       |t| > 1.96 counts also reported.
9. Costs:              n/a (forecast accuracy, no trading).
10. Failure modes:     noisy r^2 proxy lowers DM power; t-MLE non-convergence on short windows;
                       structural breaks (2020 COVID, crypto 2021-22) dominating the loss mean;
                       a few extreme days dominating QLIKE when h is tiny; dropped placeholder rows.
Variants tried:        1 decision variant (GJR vs EWMA, QLIKE, r^2 proxy) per instrument, 13
                       instruments. Secondary descriptive comparisons: 3 other models x 2 losses
                       + GK/Parkinson proxies - reported, never used to change the verdict.
Result:                see "Results" below (filled from experiment records).
Conclusion:            inconclusive - GJR has the lower mean QLIKE on 12/13 instruments but no DM test
                       survives BH (1/13 raw |t| > 1.96: NVDA).
```

Distribution choice (fixed): Student-t. Daily returns of all 13 instruments are heavy-tailed
(H-XM-000 kurtosis); t-MLE is more efficient than Gaussian QMLE under fat tails while the variance
recursion, and therefore the forecast formula, is identical. Gaussian QMLE was NOT also tried.

---

```
ID:              H-REG-001
Title:           Out-of-sample high-vol GMM state predicts higher next-day realized variance
Status:          supported on 10/13 instruments, inconclusive on 3/13
Registered:      2026-11-24 (before results were seen)
1. Intuition:          Volatility clusters; a 2-state mixture on (20d realized vol, 20d return)
                       should separate calm and turbulent periods, and because volatility is
                       persistent the state at t carries information about variance at t+1.
2. Formulation:        x_t = (vol20_t, ret20_t), vol20_t = sample std of r_{t-19..t},
                       ret20_t = sum r_{t-19..t}. sklearn GaussianMixture(n_components=2,
                       covariance_type="full", n_init=3, random_state=7) on features standardised
                       with training-window mean/std. Expanding window, first fit at origin 503,
                       refit every 63 sessions; state for origins in a block = argmax posterior of
                       the model fit at the block start. Labels: components sorted by their
                       training-sample mean vol20 (state 1 = high vol), so labels are stable across
                       refits.
3. Data:               as H-VOL-001.
4. Features:           vol20, ret20 (causal; computed from returns <= t).
5. Target:             RV_{t+1} = r_{t+1}^2 (and |r_{t+1}|, same rank test).
6. Mechanism:          Statistical claim (volatility clustering), not a trade.
7. H0:                 E[RV_{t+1} | state_t = high] <= E[RV_{t+1} | state_t = low].
8. Validation design:  OOS origins only (>= 503). Tests on OOS days:
                         - primary: one-sided Mann-Whitney U (high > low) on RV_{t+1};
                         - HAC check: OLS RV_{t+1} = a + b 1[high_t], Newey-West t of b
                           (autocorrelation-robust; iid tests overstate significance with
                           persistent states and clustered RV);
                         - reported: Welch t (one-sided).
                       Decision per instrument: BH (q = 0.05, 13 instruments) on MW p:
                         p_BH < 0.05 and HAC t > 1.96 -> supported
                         HAC t < -1.96                -> rejected
                         otherwise (incl. a state never visited OOS) -> inconclusive.
                       Also reported: OOS persistence P(s_{t+1} = s_t) vs. the independence
                       benchmark p_low^2 + p_high^2; QLIKE of a state-conditional mean-RV forecast
                       vs. the unconditional expanding-mean forecast (baseline), descriptive.
9. Costs:              n/a.
10. Failure modes:     labels fit on the whole sample (prevented: expanding windows only);
                       label switching across refits (prevented: vol sort); one state absorbing
                       outliers only (tiny high-state share); heavy-tailed RV making Welch fragile.
Variants tried:        1 (fixed features, K = 2, refit 63, min train 504). K and feature set
                       were not searched.
Result:                see "Results" below.
Conclusion:            supported on 10/13 (BH-significant MW and HAC t > 1.96); MSFT, TCS, INFY
                       inconclusive (MW BH-significant, HAC t 1.57-1.90). None rejected.
```

---

## Results

Run 2026-11-24 (`run_volatility_study()`, `run_regime_study()`, working tree on top of d379710, dirty).
Runtime 37 s (volatility) + 24 s (regime); peak process working set 280 MB. Experiment IDs are the
`experiments/<id>/record.json` folders; forecast/state frames under `artifacts/`.

### H-VOL-001 (OOS origins from 2020-01-03 US, 2020-01-22 IN stocks, 2020-01-27 NIFTY50, 2019-05-20
BTC/ETH, 2021-12-28 SOL, all to 2026-11-20/22; mean normalised QLIKE on r^2 > 0
days, percent units; DM t < 0 = model better than EWMA; BH over 13 two-sided p)

| symbol | n | QLIKE EWMA (baseline) | roll20 | roll60 | GARCH | GJR | DM t GJR | p | p_BH | DM t GJR (MSE) | verdict | experiment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | 1731 | 1.5959 | 1.6454 | 1.6738 | 1.5614 | 1.5474 | -1.92 | 0.055 | 0.359 | -1.14 | inconclusive | 20261124T145135-H-VOL-001-bda3d4 |
| RELIANCE | 1689 | 1.6336 | 1.6731 | 1.6834 | 1.6201 | 1.6139 | -0.79 | 0.428 | 0.475 | -1.49 | inconclusive | 20261124T145135-H-VOL-001-c1bd6a |
| BTCUSDT | 2744 | 1.9830 | 2.1115 | 2.0364 | 1.9597 | 1.9592 | -1.18 | 0.238 | 0.475 | 2.06 | inconclusive | 20261124T145135-H-VOL-001-4df8be |
| MSFT | 1731 | 1.6371 | 1.7031 | 1.7113 | 1.6114 | 1.6177 | -0.78 | 0.438 | 0.475 | -0.67 | inconclusive | 20261124T145136-H-VOL-001-cb15b9 |
| NVDA | 1731 | 1.5068 | 1.5324 | 1.5914 | 1.4632 | 1.4540 | -2.50 | 0.012 | 0.160 | -1.37 | inconclusive | 20261124T145136-H-VOL-001-561c97 |
| SPY | 1731 | 1.5431 | 1.5877 | 1.6831 | 1.4873 | 1.4677 | -1.72 | 0.086 | 0.374 | -0.56 | inconclusive | 20261124T145136-H-VOL-001-c5aa68 |
| QQQ | 1731 | 1.5389 | 1.5571 | 1.6628 | 1.5035 | 1.4970 | -1.11 | 0.266 | 0.475 | -0.93 | inconclusive | 20261124T145136-H-VOL-001-20ee70 |
| TCS | 1689 | 1.7097 | 1.7751 | 1.7406 | 1.6840 | 1.6825 | -1.57 | 0.116 | 0.377 | -0.02 | inconclusive | 20261124T145136-H-VOL-001-250a53 |
| INFY | 1689 | 1.6301 | 1.6802 | 1.6772 | 1.6135 | 1.6092 | -0.93 | 0.354 | 0.475 | 0.42 | inconclusive | 20261124T145136-H-VOL-001-bfd05d |
| HDFCBANK | 1689 | 1.6113 | 1.6731 | 1.6806 | 1.6484 | 1.6449 | 0.86 | 0.389 | 0.475 | 0.54 | inconclusive | 20261124T145136-H-VOL-001-ef671f |
| NIFTY50 | 1686 | 1.5124 | 1.5297 | 1.6323 | 1.4944 | 1.4822 | -1.12 | 0.262 | 0.475 | -0.52 | inconclusive | 20261124T145136-H-VOL-001-0d2db7 |
| ETHUSDT | 2744 | 1.8720 | 2.0038 | 1.8975 | 1.8461 | 1.8479 | -0.78 | 0.436 | 0.475 | 0.92 | inconclusive | 20261124T145136-H-VOL-001-f01592 |
| SOLUSDT | 1791 | 1.5778 | 1.6539 | 1.6586 | 1.5691 | 1.5688 | -0.24 | 0.810 | 0.810 | -0.53 | inconclusive | 20261124T145136-H-VOL-001-e53f99 |

Family: supported 0, rejected 0, inconclusive 13 after BH; raw |t| > 1.96: 1 (NVDA, GJR better),
0 with EWMA better. Point estimates favour GJR on 12/13 (direction consistent, power low with the
noisy r^2 proxy). Secondary (descriptive): rolling-mean forecasters are worse than EWMA (DM t > 0
on all 13 for roll20). 0 GARCH/GJR refits failed or did not converge.

### H-REG-001 (mean next-day r^2 in percent^2; MW one-sided p, BH over 13; HAC = Newey-West t of the
high-state dummy slope; persistence vs independence benchmark p_low^2 + p_high^2)

| symbol | n | share high | persistence (indep.) | mean RV low (baseline) | mean RV high | ratio | Welch t | MW p_BH | HAC t | DM t cond vs uncond QLIKE | verdict | experiment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | 1731 | 0.206 | 0.957 (0.673) | 2.68 | 8.27 | 3.08 | 5.12 | 1.0e-13 | 3.31 | -2.28 | supported | 20261124T145159-H-REG-001-c62843 |
| RELIANCE | 1689 | 0.061 | 0.986 (0.886) | 2.43 | 13.21 | 5.44 | 3.25 | 3.0e-05 | 2.05 | -2.44 | supported | 20261124T145159-H-REG-001-3cc30f |
| BTCUSDT | 2744 | 0.070 | 0.985 (0.870) | 9.89 | 18.89 | 1.91 | 3.03 | 2.0e-05 | 2.52 | -0.48 | supported | 20261124T145159-H-REG-001-5c4b84 |
| MSFT | 1731 | 0.057 | 0.985 (0.892) | 3.00 | 14.35 | 4.79 | 3.28 | 5.4e-07 | 1.90 | -1.23 | inconclusive | 20261124T145159-H-REG-001-a157fc |
| NVDA | 1731 | 0.366 | 0.960 (0.536) | 7.17 | 17.85 | 2.49 | 5.66 | 3.9e-22 | 4.54 | -2.93 | supported | 20261124T145159-H-REG-001-91aa3f |
| SPY | 1731 | 0.057 | 0.988 (0.892) | 1.05 | 11.06 | 10.51 | 4.33 | 4.9e-12 | 2.55 | -2.51 | supported | 20261124T145159-H-REG-001-c3fdfc |
| QQQ | 1731 | 0.277 | 0.957 (0.599) | 1.52 | 5.14 | 3.39 | 6.02 | 1.8e-18 | 3.64 | -2.52 | supported | 20261124T145159-H-REG-001-6e1c08 |
| TCS | 1689 | 0.065 | 0.986 (0.878) | 2.30 | 5.05 | 2.19 | 2.12 | 1.2e-02 | 1.76 | -0.35 | inconclusive | 20261124T145159-H-REG-001-7d9d17 |
| INFY | 1689 | 0.108 | 0.973 (0.807) | 2.93 | 5.11 | 1.74 | 1.95 | 3.8e-03 | 1.57 | 0.09 | inconclusive | 20261124T145159-H-REG-001-492e58 |
| HDFCBANK | 1689 | 0.178 | 0.977 (0.707) | 2.31 | 4.07 | 1.76 | 3.85 | 6.8e-17 | 2.90 | -0.02 | supported | 20261124T145159-H-REG-001-4be97a |
| NIFTY50 | 1686 | 0.084 | 0.987 (0.847) | 0.81 | 6.44 | 8.01 | 3.38 | 7.1e-12 | 2.23 | -2.26 | supported | 20261124T145159-H-REG-001-315bfd |
| ETHUSDT | 2744 | 0.073 | 0.987 (0.865) | 16.64 | 35.80 | 2.15 | 2.71 | 3.7e-06 | 2.06 | -3.97 | supported | 20261124T145159-H-REG-001-e9940b |
| SOLUSDT | 1791 | 0.069 | 0.988 (0.871) | 21.82 | 47.98 | 2.20 | 2.75 | 6.4e-06 | 2.89 | -8.20 | supported | 20261124T145200-H-REG-001-491f4d |

Family: supported 10, rejected 0, inconclusive 3 (all 13 MW-significant after BH; the HAC gate
fails for MSFT, TCS, INFY). Caveat: the effect is volatility clustering, which is already known
(H-XM-000); the regime classifier was not compared against EWMA here, so "the state adds
information beyond EWMA" is NOT established.
