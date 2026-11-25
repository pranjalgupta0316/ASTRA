---
name: backtesting-audit
description: Use when building or changing the ASTRA event-driven backtester, modelling execution (fees, spread, slippage, impact), running walk-forward / purged CV, or auditing any backtest or strategy result for look-ahead, survivorship, leakage, unrealistic fills or overfitting before it is reported.
---

# Backtesting & Audit (ASTRA)

## Purpose
Make every reported strategy number defensible. Code: `src/astra/backtest/`. This skill is the
independent check: a result produced under `quantitative-research` or `financial-ml-research` is not
"validated" until it passes the audit checklist below.

## Activate when
Writing/altering the engine, cost model or fill logic; producing performance tables; reviewing a
strategy claim; choosing a CV scheme for time series.

## Engine contract
1. Events flow strictly in time: `MarketEvent(t)` -> strategy sees data with `available_at <= t` ->
   `SignalEvent` -> `OrderEvent` -> fill at the **next** tradable price (next bar open by default),
   never the price that generated the signal.
2. Costs = commission (bps or per-share, market-specific) + half-spread + slippage (bps or k * vol) +
   optional square-root impact: `impact = c * sigma * sqrt(Q / ADV)`.
3. Liquidity cap: fill quantity <= participation_rate * bar volume; remainder cancelled (logged).
4. Calendar-aware: no fills on closed sessions; crypto 24/7.
5. Accounting: cash, positions, realized/unrealized PnL, fees tracked separately; gross and net equity.

## Audit checklist (all must be answered in the report)
- [ ] Signal/execution alignment test: shifting signals one bar later changes results as expected,
      and a deliberately leaky signal (future return) is caught by `astra.backtest.audit`.
- [ ] Features computed with rolling/expanding windows only; scalers fit on train only.
- [ ] Universe defined point-in-time (or survivorship limitation stated explicitly).
- [ ] Corporate actions: trading on unadjusted prices or consistently adjusted; dividends credited.
- [ ] Costs non-zero and sensitivity reported (0x, 1x, 2x, 4x cost).
- [ ] Gross vs net, turnover, exposure, max DD, Sharpe with CI, number of variants tried.
- [ ] Walk-forward or purged K-fold with embargo >= label horizon for any fitted component.
- [ ] Out-of-sample period reported separately and touched once.

## Failure modes
Same-bar fill; `shift(-1)` in features; global normalization; dropping delisted names; costs applied to
notional instead of traded notional; annualizing crypto with 252; Sharpe from overlapping returns
without adjustment; optimizing on the test window.

## Verification / acceptance
Unit tests with hand-computed PnL for a 3-bar toy series; leakage canary test; cost sensitivity table.

## Limitations
Daily-bar fills cannot model intraday queue position or real spread; state this. Free data has no
historical quotes, so spread is an assumption per market (documented in `docs/RESEARCH_METHODOLOGY.md`).

## Interacts with
Consumes signals from `quantitative-research` / `financial-ml-research`, weights from
`portfolio-risk-engineering`, calendars from `financial-data-engineering`. Detail:
`references/cost_and_cv.md`.
