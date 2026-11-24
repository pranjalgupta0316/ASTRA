# Research hypotheses register

Registered on 2026-11-24 **before** any result was computed. Parameters fixed a priori; no tuning.
Results are filled in from experiment records (`experiments/`), never edited by hand to look better.

## H-XM-000 Stylized facts (all three instruments)
1. Intuition: efficient-ish markets → prices ≈ random walk; volatility clusters.
2. Formulation: ADF+KPSS on log price and log returns; Ljung-Box(10) on r and r².
3. Data: validated daily bars. 7. H0 (per component): opposite of each stated fact.
8. Validation: full sample, descriptive. Supported iff price=unit_root, returns=stationary, LB(r²) p<0.05.

## H-XM-001 Time-series momentum, 60 days, long/flat
1. Intuition: trend persistence from under-reaction / herding (Moskowitz, Ooi & Pedersen 2012, JFE -
   cited for the concept; NEBULA's daily, single-asset, long/flat version is not a reproduction).
2. Signal s_t = 1[log P_t − log P_{t−60} > 0]. Target: next open-to-open returns while held.
3. Data: adjusted daily bars. 5. Execution: decision at close t, fill at open t+1.
6. Mechanism: avoid prolonged drawdowns; cost: turnover on signal flips.
7. H0: mean(r_tsmom,net − r_bh,net) ≤ 0. 8. Newey-West t on daily differences; |t|>1.96 decides.
9. Costs: config/markets.yaml per market. 10. Failure modes: whipsaw in ranging markets, one-period
   dependence (2020/2022), small sample of trend episodes.

## H-XM-002 Logistic regression on 13 causal features
1. Intuition: weak, nonlinear-ish short-term predictability in technicals; likely near zero for liquid
   large caps (null expected).
2. P(r_{t+1→t+2}^{open} > 0 | features_t) via L2-logistic (C=0.1), standardized within training fold.
4. Features: ret_1, mom_5/20/60, sma_ratio_20/50, vol_20, parkinson_20, rsi_14, macd_hist_norm,
   bb_z_20, volume_z_20, autocorr_20.
7. H0: OOS log loss ≥ base-rate log loss, or strategy (long if p>0.5) not better than B&H net.
8. Purged expanding walk-forward: min_train = 2 years, test = 1 quarter, purge 2, embargo 5.
10. Failure modes: base-rate drift; regime dependence; costs dominate any small edge.

### Amendment A1 to H-XM-002 (2026-11-24, after results - disclosed as post hoc)
Independent audit found the registered gate compared model log loss with the **OOS sample's own** base
rate (hindsight) and decided on a point estimate. Amended rule: baseline = each row's training-fold base
rate; Diebold-Mariano-style Newey-West t on per-row log-loss differences. Rejected if model (DM t) or
strategy (NW t) is significantly worse (t < −1.96); supported only if both are significantly better;
else inconclusive. Each record stores `verdict_original_rule` next to the amended verdict.

## H-XM-003 (proposed, NOT yet run) Momentum as a risk-reducer
Motivated by observed H-XM-001 results (BTC Sharpe 1.01 vs 0.63 while mean-return test is
inconclusive) - therefore **not a clean pre-registration**; must be tested on held-out instruments
(ETHUSDT, SOLUSDT, MSFT, TCS.NS) before any claim. H0: Sharpe(TSMOM) ≤ Sharpe(volatility-matched
buy-and-hold); test: Ledoit-Wolf (2008) Sharpe-difference test with HAC/bootstrap inference.

Results: see `docs/VALIDATION_REPORT.md` and the dashboard "Research experiments" page.
