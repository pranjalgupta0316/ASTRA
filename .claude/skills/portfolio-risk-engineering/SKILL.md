---
name: portfolio-risk-engineering
description: Use when constructing ASTRA portfolios or computing risk - equal weight, minimum variance, mean-variance, max Sharpe, risk parity, volatility targeting, position sizing and limits, covariance estimation/shrinkage, factor exposure, VaR, Expected Shortfall, drawdown, concentration, turnover, or annualization of risk metrics.
---

# Portfolio & Risk Engineering (ASTRA)

## Purpose
Numerically stable, constraint-aware portfolio construction and correctly annualized risk metrics.
Code: `src/astra/portfolio/`, `src/astra/risk/`.

## Activate when
Turning signals into weights, computing any performance/risk statistic, comparing optimizers, setting
leverage/position/concentration limits.

## Workflow
1. **Covariance**: estimate on a trailing window ending at t-1; default Ledoit-Wolf shrinkage
   (`sklearn.covariance.LedoitWolf`). Check condition number; refuse to optimize if not PSD after
   nearest-PSD projection.
2. **Optimizer** (CVXPY, solver CLARABEL/OSQP/ECOS): long-only fully-invested unless specified;
   box constraints per asset; group (market/sector) caps; turnover penalty. Max-Sharpe via the convex
   reformulation (scale weights), not a non-convex ratio.
3. **Risk parity**: solve the convex log-barrier formulation (Spinu 2013) and normalize.
4. **Mixed currencies** (USD, INR, USDT): convert returns to a base currency with FX on the same
   timestamps, or keep portfolios per market; state which.
5. **Sizing**: volatility targeting `w_scaled = w * target_vol / forecast_vol`, capped leverage.
6. **Metrics** via `astra.risk.metrics` with explicit `periods_per_year`.

## Standards
- CAGR from equity curve and calendar time; Sharpe/Sortino with stated rf; max drawdown on net equity.
- VaR/ES: historical and Gaussian/Cornish-Fisher; report confidence and horizon; ES >= VaR check.
- Distinguish **forecast** risk (ex-ante w' S w) from **realized** risk.
- Sensitivity: sample vs shrunk covariance, window length.

## Failure modes
Covariance using the current day; mean-variance on noisy means (estimation error maximization);
252 for crypto; aligning US and India calendars by forward-filling prices (creates stale correlation);
ignoring FX.

## Verification / acceptance
Tests: 2-asset analytic min-variance weights; risk-parity equal risk contributions within 1e-6;
known-series Sharpe/MDD/VaR values; weights sum to 1 and respect bounds.

## Limitations
Small universes (3-15 assets) make covariance estimates noisy and optimizer gains over equal weight
statistically hard to show; mixed-calendar portfolios (24/7 crypto vs exchange sessions) need explicit
alignment rules and are not yet implemented.

## Interacts with
`backtesting-audit` (weights go through the engine with costs), `quantitative-research` (signals),
`financial-data-engineering` (calendar alignment).
