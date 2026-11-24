# Methods reference (quantitative-research)

## Stationarity
- ADF: H0 unit root. KPSS: H0 stationary. Agreement table:
  ADF reject & KPSS not reject -> stationary; ADF not reject & KPSS reject -> unit root;
  both reject -> likely fractional / structural break; neither -> inconclusive (low power).
- statsmodels: `adfuller(x, autolag="AIC")`, `kpss(x, regression="c", nlags="auto")`.

## Cointegration
- Engle-Granger two-step: `statsmodels.tsa.stattools.coint(y, x)`; critical values differ from ADF.
- Johansen: `statsmodels.tsa.vector_ar.vecm.coint_johansen(df, det_order=0, k_ar_diff=1)`; use trace
  and max-eigen statistics; report rank.
- Half-life of mean reversion from AR(1) on spread: hl = -ln(2)/ln(phi).

## Information coefficient
- Spearman rank correlation of signal and forward return per date (cross-section) or rolling (time
  series). t-stat of mean IC with Newey-West lags >= horizon - 1.

## Multiple testing
- Benjamini-Hochberg FDR: `statsmodels.stats.multitest.multipletests(p, method="fdr_bh")`.
- Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014, J. Portfolio Management): adjusts the observed
  Sharpe for number of trials, skew and kurtosis. Implemented in `nebula.stats.inference`.

## Volatility estimators (per-period variance)
- Parkinson: (1/(4 ln 2)) * ln(H/L)^2
- Garman-Klass: 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2
- EWMA (RiskMetrics): s2_t = lambda s2_{t-1} + (1-lambda) r_{t-1}^2, lambda = 0.94 daily.
- Forecast loss: QLIKE = s2_real/s2_fc - ln(s2_real/s2_fc) - 1 (robust to noisy proxies, Patton 2011).

## Regimes
- GaussianMixture / GaussianHMM on (return, realized vol). Fit on train only; map states by sorting
  on train-sample mean vol so labels are stable across refits.
