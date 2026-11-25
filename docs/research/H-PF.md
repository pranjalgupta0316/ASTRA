# H-PF: Portfolio construction hypotheses (milestone M8)

Registered on 2026-11-24 **before** any portfolio result was computed (no portfolio backtest had been
run on ASTRA data when this file was written). All parameters below are fixed a priori; no tuning.
Results are filled in from experiment records (`experiments/`, hypothesis ids `H-PF-001`, `H-PF-002`),
never edited by hand. Code: `src/astra/portfolio/`, `src/astra/risk/attribution.py`,
`src/astra/research/portfolio_study.py`.

## Common design (fixed a priori)
- **Per-market portfolios in local currency; no FX mixing.** US (USD): AAPL, MSFT, NVDA, SPY, QQQ.
  IN (INR): RELIANCE, TCS, INFY, HDFCBANK (NIFTY50 is a benchmark, not tradable). CRYPTO (USDT):
  BTCUSDT, ETHUSDT, SOLUSDT. Each market is aligned on its own calendar (union of the market's validated
  session dates); equities and crypto are never mixed and no prices are forward-filled across markets.
- Prices: total-return adjusted O/C (`adjusted_ohlc`). Provider placeholder rows (volume 0 and H == L)
  are excluded (same policy as the vertical slice). An asset missing a bar cannot trade that day; its
  position is marked at its last close.
- **Universe entry:** an asset becomes eligible at decision date t only once it has >= 252 valid daily
  returns up to and including t. No backfilling (SOLUSDT enters roughly one year after its 2020-08
  listing). Portfolios start at the first decision date with >= 2 eligible assets.
- **Covariance:** Ledoit-Wolf shrinkage (sklearn) of simple daily returns over the trailing 252
  observations ending at t (inclusive of t's close), rows where every eligible asset has a return;
  nearest-PSD repair; condition number recorded. 252 observations are used for crypto too (about 8
  calendar months); annualization is 252 for US/IN and 365 for CRYPTO.
- **Rebalance:** monthly. Decision at the close of the first session of each month, fill at the next
  session's open (engine convention). Costs from `config/markets.yaml` (per side, bps of traded
  notional), max participation 5% of bar volume, zero-volume / missing bars never fill, buys sized net
  of cost, cash never negative, rebalance tolerance 2% per asset (engine default).
- **Constraints:** long-only, fully invested, per-asset cap 0.40 (equities) / 0.60 (crypto).
- **Strategies:** equal weight (EW, benchmark), minimum variance (MinVar), risk parity (RP, Spinu 2013
  log-barrier, equal risk budgets). Descriptive only (no hypothesis, because mean estimates are mostly
  noise): mean-variance (risk aversion 5) and maximum Sharpe (convex reformulation, rf = 0), both with
  a shrunk mean: 252-day sample mean shrunk 50% toward its cross-sectional average; max Sharpe falls
  back to MinVar if no asset has a positive shrunk mean.

## H-PF-001 MinVar and RP have lower realized volatility than EW (per market)
1. Intuition: the covariance matrix is far better estimated than means; optimizing on it alone should
   reduce out-of-sample risk vs 1/N (e.g. Clarke, de Silva & Thorley 2006; Maillard, Roncalli &
   Teiletche 2010).
2. Statistic per strategy s: ratio of realized annualized volatility of daily net returns, vol(s)/vol(EW),
   over the common evaluation period.
7. H0: vol(s)/vol(EW) >= 1.
8. Inference: paired stationary bootstrap (Politis-Romano, mean block 20 days, 2000 resamples, seed 7)
   95% CI of the ratio. Per strategy: CI upper < 1 -> supported; CI lower > 1 -> rejected; else
   inconclusive. Secondary (reported, not decisive): Newey-West t on r_EW^2 - r_s^2.
   Record verdict per market: supported iff both MinVar and RP are supported; rejected if either is
   rejected; otherwise inconclusive. Sub-verdicts are stored in the record.
10. Failure modes: tiny universes (3-5 highly correlated assets) leave little to diversify; caps of
    0.40/0.60 bind and push weights toward EW; regime shifts make trailing covariance stale.

## H-PF-002 MinVar and RP have higher net Sharpe than EW (per market)
7. H0: mean(r_s,net - r_EW,net) <= 0.
8. Test (same rule as the vertical slice): Newey-West t on daily net return differences;
   t > 1.96 supported, t < -1.96 rejected, else inconclusive. Record verdict per market combines the
   two strategies as in H-PF-001.
   **Known mismatch, disclosed at registration:** the registered test is on mean return differences,
   not on Sharpe ratios; a lower-volatility portfolio can have a higher Sharpe with a lower mean. As a
   pre-registered *secondary, non-decisive* statistic we report a paired stationary bootstrap 95% CI
   of the annualized Sharpe difference (same bootstrap settings as H-PF-001).
10. Failure modes: low-vol assets in this universe (SPY, HDFCBANK, BTC) also had lower returns over
    2019-2026, which the return-difference test penalizes; single historical path; survivorship.

## Survivorship and other limitations (stated up front)
The universe is today's well-known survivors (large caps, the two largest crypto assets plus SOL).
That biases absolute returns upward for all strategies; relative comparisons are less affected but not
immune. Sample is 2018-2026 only (one regime path per market). Daily bars cannot model intraday
execution; costs are assumptions. Sensitivity runs (sample vs Ledoit-Wolf covariance, 126/252/504
windows, cost multipliers 0/1/2/4) are descriptive and do not change verdicts.

## Results (run 2026-11-24, appended after the run; registration text above unchanged)
Source: experiment records listed below; daily net returns, annualization 252 (US/IN) / 365 (CRYPTO).

| Market | Period | MinVar vol/EW [95% CI] | RP vol/EW [95% CI] | H-PF-001 | NW t (MinVar, RP) | H-PF-002 |
|---|---|---|---|---|---|---|
| US | 2019-02-01..2026-11-23 | 0.824 [0.785, 0.855] | 0.922 [0.897, 0.942] | supported | -4.07, -3.46 | rejected |
| IN | 2019-02-01..2026-11-23 | 0.988 [0.967, 1.004] | 0.994 [0.981, 1.004] | inconclusive | -0.28, -0.59 | inconclusive |
| CRYPTO | 2018-10-01..2026-11-23 | 0.903 [0.877, 0.928] | 0.956 [0.945, 0.967] | supported | -1.52, -1.73 | inconclusive |

Secondary (non-decisive) Sharpe differences vs EW: US MinVar -0.27 [-0.45, -0.08], RP -0.07
[-0.15, 0.01]; IN MinVar -0.02, RP -0.02 (CIs span 0); CRYPTO MinVar -0.07, RP -0.02 (CIs span 0).
Records: US 20261124T145424-H-PF-001-36f4b0 / 20261124T145425-H-PF-002-3a50a2; IN
20261124T145456-H-PF-001-8d617c / 20261124T145456-H-PF-002-e63479; CRYPTO
20261124T145531-H-PF-001-bed552 / 20261124T145531-H-PF-002-4f38c9.
Known execution property: the registered 2% per-asset tolerance band can leave sale proceeds in cash
until the next rebalance (EW average exposure 0.983 US, 0.994 IN, 0.997 CRYPTO); same rule for all
strategies.
