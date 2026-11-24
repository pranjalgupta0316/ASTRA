# Results: H-VT-001 (volatility-managed exposure) and H-XMKT-001 (cross-market diversification)

The hypotheses were registered in `docs/research/H-EXT.md` before any run, and that file is unchanged.
Run 2026-11-25 (UTC timestamps in the experiment IDs). Code: `src/nebula/research/vol_managed.py` and
`src/nebula/research/cross_market.py`. Tests: `tests/test_vol_managed_xmkt.py`. Alignment rules:
`docs/adr/0007-cross-market-weekly-alignment.md`.

**Survivorship.** Every series comes from today's universe: 30 US mega-caps, 15 ETFs, the 49 current
NIFTY names plus the NIFTY50 index, and 16 surviving crypto majors. Delisted, demoted and dead names are
absent. Absolute returns are biased upward, and managed-vs-unmanaged and diversification comparisons are
conditional on survival. Survivors may also have had fewer fatal high-volatility collapses, and those
collapses are where vol management helps most.

Common settings:
- rf = 0.
- Annualization: daily 252 (US, IN) and 365 (crypto); weekly 52.
- Paired stationary bootstrap: mean block 20, 2000 draws, seed 7.
- Costs are the per-market config values (bps per side): US 3 buy / 3.3 sell, IN 16.5 / 16.5, crypto
  13 / 13.

## H-VT-001 Volatility-managed exposure

**Rule.**
- σ̂_t = sqrt(EWMA(λ=0.94) variance) of daily log returns, known at the close of t.
- σ_target = median σ̂ over the first 3 calendar years of each series, then frozen. That calibration
  window is excluded from evaluation.
- w_t = min(1, σ_target/σ̂_t).
- Decision at close t, fill at open t+1.
- Unmanaged benchmark = buy-and-hold for single instruments, and monthly-rebalanced EW for portfolios,
  over the same post-calibration window.

**Execution.**
- Single instruments use `nebula.backtest.engine`. It trades when |target − current| ≥ 0.02, its
  default tolerance. NIFTY50 is an index with unreliable volume, so it has no participation cap and
  volume is not used.
- EW portfolios use `nebula.portfolio.backtest` with per-leg tolerance 0. The 0.02 tolerance is applied
  to the exposure scale instead: a managed order is issued when |scale_t − last issued| > 0.02, or on a
  monthly date. On non-monthly trigger days the managed book keeps the unmanaged book's drifted
  composition, so the two differ only in exposure. σ̂ for an EW portfolio comes from its own unmanaged
  gross daily returns.
- Gross results come from exact zero-cost reruns.

**Decision rule.**
- Per series: CI > 0 supported, CI < 0 rejected, otherwise inconclusive.
- BH across the family of 114 series (111 instruments + 3 EW portfolios).
- Family aggregation, fixed before the run: supported if at least one BH-significant positive and no
  BH-significant negative; rejected in the mirror case; otherwise inconclusive.

### Counts (verdict from the raw 95% CI; after BH, every series is inconclusive)

| Group | n | supported | rejected | inconclusive | median ΔSharpe | share ΔSharpe > 0 | median ΔMDD (pp, + = shallower) |
|---|---|---|---|---|---|---|---|
| US (30 stocks + 15 ETFs) | 45 | 5 | 0 | 40 | +0.028 | 91% | +2.4 |
| IN (49 stocks + NIFTY50) | 50 | 6 | 0 | 44 | +0.021 | 76% | +2.3 |
| Crypto | 16 | 0 | 0 | 16 | +0.011 | 63% | +1.0 |
| EW portfolios | 3 | 2 | 0 | 1 | +0.141 | 100% | +12.8 |

- Raw-CI "supported" series: BRK_B, XLF, XLE, XLP, XLU, KOTAKBANK, AXISBANK, ASIANPAINT, BAJAJFINSV,
  BRITANNIA, APOLLOHOSP, EW_US, EW_IN.
- Smallest BH-adjusted p = 0.114 (KOTAKBANK). There are **0 BH-significant** series, positive or
  negative. 22 of 114 point estimates are negative.
- **Family conclusion: inconclusive** (`20261124T200923-H-VT-001-64d246`; its `table` artifact lists
  every series with its experiment ID).

### Equal-weight portfolios (daily net, post-calibration)

| Portfolio | Eval window | Sharpe managed / unmanaged | ΔSharpe [95% CI] | p_BH | ΔMDD | turnover/yr managed / unmanaged | cost drag (CAGR) managed / unmanaged | record |
|---|---|---|---|---|---|---|---|---|
| EW US (30) | 2004-01-05..2026-11-23 | 1.115 / 0.974 | +0.141 [+0.004, +0.286] | 0.351 | +14.2 pp | 1.25 / 0.67 | 0.05% / 0.02% | `20261124T200923-H-VT-001-6cba61` |
| EW IN (49) | 2004-02-02..2026-11-23 | 1.390 / 1.218 | +0.172 [+0.012, +0.339] | 0.351 | +12.8 pp | 1.94 / 0.78 | 0.39% / 0.16% | `20261124T200923-H-VT-001-1d0c68` |
| EW crypto (16) | 2021-05-02..2026-11-23 | 0.248 / 0.237 | +0.011 [−0.109, +0.132] | 0.996 | +1.8 pp | 3.69 / 1.45 | 0.46% / 0.17% | `20261124T200923-H-VT-001-076888` |

- ΔMDD = MDD(managed) − MDD(unmanaged), so a positive value means a shallower drawdown.
- Turnover is two-sided, as a fraction of equity per year.
- Exposure:
  - Unmanaged EW averaged 1.000 (US), 0.9997 (IN) and 1.000 (crypto).
  - Managed EW averaged 0.968, 0.937 and 0.945, with minimums of 0.27 (US) and 0.29 (crypto).
  - On IN the first evaluation day is all cash for both arms, because no IN leg had a fillable bar on
    2004-02-02.
- Cost sensitivity (0x/1x/2x/4x) for the EW portfolios is in the family record's `ew_cost_sensitivity`
  artifact.

**Forecast vs realized risk** (managed, median over series): ex-ante w_t·σ̂_t annualized vs realized
vol of daily net returns.
- US: 20.8% forecast vs 21.8% realized.
- IN: 28.0% vs 30.2%.
- Crypto: 71.9% vs 77.8%.
- EW portfolios: 14.8% vs 15.7% (median of the three).

EWMA mildly under-forecasts next-day risk.

**Interpretation.** Point estimates lean positive: 91% of US and 76% of IN series, and drawdowns are
mostly shallower. No single series survives BH, so H-VT-001 is not supported at the family level.
The registered calibration makes the rule nearly inert for many series. The first 3 years often
contain a crisis (2000-02 for US stocks, 2007-10 for NIFTY50, 2017-20 for crypto), which yields a high
σ_target:
- 19 series have a mean weight above 0.99. For example, AAPL's σ_target is 61% annualized and its mean
  weight is 0.996.
- The median mean weight is 0.96-0.98.

Crypto shows no effect and doubles turnover.

## H-XMKT-001 Cross-market diversification (weekly USD)

**Setup.**
- Sleeves are the unmanaged EW portfolios above (net of within-sleeve costs, local currency, full
  history).
- Sampling and FX follow ADR-0007:
  - Each sleeve is sampled at its last session ≤ Friday.
  - IN is converted at the latest USDINR value available strictly before the IN session.
  - USDT is treated as USD.
- Mix = inverse trailing-52-week-vol weights. Decisions are made at the first week of each month and
  executed one week later.
- Costs are charged on sleeve turnover at each market's rates.
- US-only = 100% US sleeve. Both portfolios start in cash and pay their initial purchase.
- Window: 2019-05-17..2026-11-20, 393 weekly returns (it starts once crypto has 52 weekly returns).

| | Mix | US-only |
|---|---|---|
| CAGR | 26.8% | 19.1% |
| Ann. vol (realized, weekly ×√52) | 18.3% | 17.3% |
| Sharpe | 1.387 (gross 1.391) | 1.099 |
| Max drawdown | −30.5% | −27.6% |
| Turnover / yr (two-sided) | 0.69 | 0 |
| Cost drag (CAGR) | 0.09% | — |

- Mean mix weights: US 0.45, IN 0.45, crypto 0.10.
- Weekly USD correlations: US-IN 0.44, US-crypto 0.25, IN-crypto 0.23.
- Sleeve vols: US 17.3%, IN 18.1%, crypto 78.6%.
- Forecast vs realized for the mix: ex-ante sqrt(w'Σw·52) from the trailing 52-week sample covariance
  averaged 16.3%. Realized vol per holding month averaged 14.4% (median ratio 0.75). The full-period
  realized vol was 18.3%.

| Test | Estimate [95% CI] | Verdict | Record |
|---|---|---|---|
| H1a vol(mix)/vol(US-only) < 1 | 1.062 [0.913, 1.244] | **inconclusive** | `20261124T200926-H-XMKT-001-9141f7` |
| H1b ΔSharpe(mix − US-only) > 0 | +0.288 [−0.317, +0.854] (p = 0.37) | **inconclusive** | `20261124T200926-H-XMKT-001-ec1fcf` |

Inverse-vol weighting ignores correlation. The 10% crypto weight at about 79% vol adds roughly as much
risk as it diversifies away, so the mix is not less volatile than US-only. The higher Sharpe point
estimate rests on one 7.5-year path, and its CI is wide.

Alignment audit (in the `alignment_audit` artifact):
- 14 US weeks and 25 IN weeks ended on a session before Friday (holidays). Every crypto week ended on
  its Friday bar. No week lacked a session.
- The FX observation was 3 days older than the IN close at the median, 5 days at most.

## Engine change made during this task (reported)

- The portfolio engine's default `rebalance_tolerance` changed from 0.02 to 0.0. An absolute band never
  tops up underweights smaller than the band.
- These studies pass tolerance 0 explicitly for every EW run, so the results above are unaffected.
- `portfolio_study.py` now passes its registered 0.02 explicitly, which keeps H-PF reproducible.
  Rerunning at 0.02 reproduces the recorded values, e.g. US MinVar/EW vol 0.824 [0.785, 0.855].
- Descriptive rerun at tolerance 0 (no records written):
  - Average exposure at 0.02 was 0.983-0.999 (3-5 assets). At 0 it is 1.000.
  - US and crypto verdicts are unchanged.
  - IN H-PF-001 would move from inconclusive to supported: MinVar/EW 0.982 [0.965, 0.997], RP/EW
    0.988 [0.980, 0.994]. This is descriptive only; the registered verdict stays.
  - H-PF-002 verdicts are unchanged everywhere.

## Limitations

- Survivorship bias, stated above.
- Daily-bar fills cannot model intraday spread or queue position.
- The NIFTY50 index is not tradable; it serves as a proxy.
- IN zero-volume bars delay fills.
- Data quality: ADANIENT has 46 negative adjusted closes in 2002 (calibration window only), which made
  its σ̂ undefined on those days. This is flagged for the data owner and not repaired here.
- H-XMKT: the three closes are non-synchronous; USDT is taken as USD; INR conversion costs are not
  modelled; FX is lagged 2-5 days.
- One historical path per market. Weekly sampling gives about 393 observations, so CIs are wide.
- The FRED DEXINUS → Yahoo INR=X deviation is disclosed in ADR-0007.
