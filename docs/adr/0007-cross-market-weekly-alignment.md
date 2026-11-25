# ADR-0007: Cross-market portfolios are aligned on weekly (Friday) closes in USD

- **Status**: accepted 2026-11-25 (first used by H-XMKT-001). Amends ADR-0006 for cross-market studies
  only. Per-market studies stay in local currency on their own daily calendar.
- **Decision**:
  1. **Sampling.** Weeks run Saturday to Friday (pandas `W-FRI`, closed and labelled right). Each market
     sleeve is sampled at its own last session dated on or before the Friday:
     - US: the XNYS Friday close (20:00 or 21:00 UTC).
     - IN: the Friday close (15:30 IST, 10:00 UTC).
     - Crypto: the Binance daily bar dated Friday (UTC day; it closes Saturday 00:00 UTC). Saturday and
       Sunday crypto bars belong to the next week.

     If the Friday is a holiday, the last earlier session of that week is used. If a market has no
     session in a week, its sleeve keeps its last value for that market only. A closed exchange has no
     price change, so this is not cross-market forward-filling. Daily prices are never forward-filled
     across calendars.
  2. **FX.** INR values are converted with USDINR (Yahoo `INR=X`, INR per USD, from
     `data/validated/macro/market.parquet`). The rate used is the latest observation whose
     `available_from` is **strictly before** the IN session date. `available_from` is the second XNYS
     session after the bar date, and the IN close comes before the New York session of the same date.
     `weekly_usd_sleeves` asserts this rule. The realized FX observation lag was median 3 days and max
     5 days over 2019-2026. The registration text named FRED DEXINUS, but FRED is manual-CSV-only in
     ASTRA and no FRED data exist, so the Yahoo series is used (a disclosed deviation).
  3. **Crypto USDT is treated as USD.** USDT/USD basis risk is ignored.
  4. **Execution lag.** A portfolio decision that uses week-t closes is executed at the closes of week
     t+1. This is strictly after every sleeve close used, including the crypto close at Saturday
     00:00 UTC.
  5. **Annualization.** 52 for weekly returns. Rebalancing costs are charged on each sleeve's traded USD
     notional at that market's configured buy and sell bps. INR conversion costs are not modelled.
- **Problem**: US, Indian and 24/7 crypto calendars never align daily. Forward-filling daily prices
  across closed sessions creates stale returns and biases correlation down (see the
  portfolio-risk-engineering skill). FX has to be causal.
- **Alternatives**:
  - Daily alignment on the union calendar with forward-fill: rejected because of stale correlations.
  - Daily alignment on the intersection calendar: drops many sessions, and the closes would still be
    up to 14 h apart.
  - Synchronous intraday snapshots: no free intraday data for IN or US over the history.
  - Monthly sampling: too few observations, about 90 over 2019-2026.
- **Consequences**:
  - The three closes in a week are non-synchronous by up to about 14 h. Cross-market weekly
    correlations are mildly biased and lead/lag effects are not captured.
  - The FX rate is 2-5 days older than the IN equity close it converts. The FX move shows up in the
    USD return about one week late in part. This is unbiased in mean but mis-timed.
  - Weekly sampling leaves about 52 observations per year, so bootstrap CIs are wide.
- **Reversibility**: high. Alignment is isolated in `astra.research.cross_market` (`weekly_last`,
  `fx_asof`, `weekly_usd_sleeves`), and tests in `tests/test_vol_managed_xmkt.py` cover it.
