# ADR-0006: Portfolios are built per market in local currency

- **Decision**: portfolio construction and backtests run separately for US (USD), India (INR) and crypto
  (USDT) on each market's own calendar; no cross-market portfolio and no FX conversion yet.
- **Problem**: mixing a 24/7 market with exchange calendars requires an explicit alignment rule and FX
  series at matching timestamps; forward-filling prices across closed sessions fabricates stale
  correlations.
- **Alternatives**: a USD-based global portfolio with FRED FX (deferred - needs an FX adapter and an
  alignment ADR); weekly sampling to align calendars (loses information).
- **Consequences**: no diversification across markets is measured; results are local-currency.
- **Reversibility**: high - add an FX adapter and an alignment rule, then a cross-market study.
