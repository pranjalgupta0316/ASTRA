---
name: quantitative-research
description: Use when designing, running or judging an ASTRA alpha/signal hypothesis - momentum, mean reversion, stat-arb, pairs/cointegration, factor models, PCA, regime detection, GARCH-family volatility, stationarity and hypothesis tests, multiple-testing correction, or out-of-sample validation of any quantitative claim.
---

# Quantitative Research (ASTRA)

## Purpose
Turn a market intuition into a falsifiable, cost-aware, out-of-sample-tested research result that is
recorded whether it succeeds or fails. Code lives in `src/astra/research/`, `src/astra/stats/`,
hypotheses in `docs/RESEARCH_HYPOTHESES.md`.

## Activate when
- Proposing or evaluating a signal, factor, pair, regime model or volatility forecast.
- Interpreting a t-stat, p-value, IC, Sharpe or any "this works" claim.
- Choosing between statistical tests (ADF/KPSS, Engle-Granger/Johansen, bootstrap, HAC).

## Workflow
1. **Register the hypothesis first** (template in `references/hypothesis_template.md`): intuition,
   formula, data, features, target, mechanism, H0, validation design, cost assumptions, failure modes.
   Assign an ID `H-<market>-<nnn>` before looking at results.
2. **Freeze the split** before any fitting: train / validation / test by date. The test window is
   touched once; record each touch in the experiment log.
3. **Descriptives and stationarity**: returns not prices; ADF *and* KPSS (they have opposite nulls -
   report both); check autocorrelation (Ljung-Box) and fat tails.
4. **Baseline**: compare against buy-and-hold, zero-signal, and a trivially simple rule.
5. **Inference**: use HAC (Newey-West) standard errors for overlapping returns; block bootstrap for
   Sharpe/IC confidence intervals; never iid bootstrap autocorrelated series.
6. **Multiple testing**: count every variant tried (parameters, universes, horizons). Apply
   Benjamini-Hochberg across the family, and report the deflated Sharpe ratio when > 5 variants.
7. **Economic significance**: net of ASTRA cost model (`backtesting-audit` skill) - a significant IC
   that dies after 10 bps is a negative result, record it as such.
8. **Stability**: sub-period and per-regime results, parameter-neighbourhood sensitivity.
9. **Record** the outcome (supported / rejected / inconclusive) in the experiment tracker.

## Standards
- Signals at time t use only information available at t (availability timestamp, not event timestamp).
- Cointegration: Engle-Granger for pairs, Johansen for >2 legs; re-estimate hedge ratios on rolling
  windows - in-sample hedge ratios applied out of sample are a leakage source.
- Regimes: fit on train only; report out-of-sample regime persistence and whether conditioning on the
  regime improves OOS metrics vs. unconditioned baseline. Pretty regime plots are not evidence.
- GARCH: fit on returns in percent (arch package scaling), evaluate with QLIKE and MSE vs. EWMA and
  realized-vol baselines.

## Failure modes to check
Data snooping across many variants; using full-sample z-score normalization; survivorship (universe
picked from today's index members); confusing statistical and economic significance; regime labels
fit on the whole sample; reporting the best of N parameter sets.

## Verification / acceptance
- Hypothesis registered before results; test-set touches counted.
- Results include baseline, CI, number of variants tried, net-of-cost metric.
- Negative/inconclusive results are recorded, not deleted.

## Limitations
Small universes (5-15 instruments) give low cross-sectional power; say so instead of over-reading ICs.

## Interacts with
`financial-data-engineering` (data availability timestamps), `backtesting-audit` (independent check of
any tradeable claim - this skill may not approve its own backtest), `financial-ml-research` (ML signals),
`portfolio-risk-engineering` (turning signals into weights). Methods detail: `references/methods.md`.
