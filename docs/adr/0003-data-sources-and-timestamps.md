# ADR-0003: Initial data sources and the availability-time convention

- **Decision**: Yahoo Finance (via yfinance) for US and Indian daily bars; Binance Public Data static
  files for crypto. Every bar carries `close_time_utc` = the earliest time the bar's close is known,
  derived from the exchange calendar (`exchange_calendars`; XNYS for US, XBOM as NSE proxy).
- **Problem**: need free, legal, multi-decade daily data for three markets with correct availability.
- **Alternatives**: Stooq (CSV, coverage uncertain for India), NSE bhavcopy (ToS restrict automation),
  Binance REST klines (rate-limited; no checksums), Alpha Vantage/Tiingo free tiers (keys, quotas,
  terms change - violates "no dependency on disappearing free tier" unless a local alternative exists).
- **Reasoning**: Binance static files are checksummed and immutable → reproducible. Yahoo is the only
  free source covering both AAPL and RELIANCE.NS daily history in one API; it is a *replaceable*
  adapter, and raw responses are archived locally so research does not depend on it staying online.
- **Trade-offs**: Yahoo is unofficial and restates adjusted history; mitigated by storing raw snapshots
  with retrieval time and sha256.
- **Risks**: Yahoo breakage/blocking → adapter interface allows a drop-in replacement.
- **Reversibility**: high (adapter pattern).
