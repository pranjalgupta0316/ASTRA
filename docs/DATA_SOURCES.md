# Data sources and feasibility matrix

Status legend: **verified** = NEBULA fetched real data from it in code; **probed** = endpoint answered
an HTTP request on 2026-11-24; **researched** = documentation reviewed only; **rejected** = not used.

## Source-availability matrix (vertical slice + planned)

| Instrument | Provider | History | Frequency | Access | Terms / licensing | Rate limits | Missing fields | Limitations | Validation plan | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | Yahoo Finance (yfinance) | 1980-12-12 → | 1d (1m ≈7-30 days, 1h ≈730 days) | unofficial JSON endpoints | Yahoo ToS: personal, non-commercial use; no redistribution | undocumented; throttles bursts | quotes/spread, point-in-time adj. | unofficial, can break; OHLC split-adjusted only; adj. history restated retroactively | calendar vs XNYS, OHLC checks, dividend factor monotone | see PROJECT_STATUS |
| RELIANCE.NS | Yahoo Finance | 1996-01-01 → (first trade per meta) | 1d | same | same | same | same | early history quality unknown; holiday coverage checked vs XBOM | calendar vs XBOM (BSE proxy for NSE) | see PROJECT_STATUS |
| ^NSEI (NIFTY 50) | Yahoo Finance | 2007-09-17 → | 1d | same | index data licensed by NSE Indices; personal research only | same | volume unreliable for index | not ingested yet | — | probed |
| BTCUSDT | Binance Public Data (data.binance.vision) | 2017-08 → | 1d…1s klines, trades, aggTrades | static zips + SHA-256 CHECKSUM | Binance terms; public market data | static files, polite 0.25 s pause | historical order book (spot) **not provided** | single venue; ms→µs timestamp switch on 2025-01-01 | checksum, continuous UTC days, OHLC | see PROJECT_STATUS |
| BTCUSDT depth | Binance REST `data-api.binance.vision/api/v3/depth` + WS `@depth@100ms` | **prospective only** | snapshot / 100 ms diffs | public, no key | public market data | request weight per minute | historical L2 | L2 only, no L3; capture costs disk | sequence-gap detection (U/u) | probed |
| Fundamentals (US) | SEC EDGAR `data.sec.gov` companyfacts | ~2009 → (XBRL) | per filing | JSON, requires descriptive User-Agent | US public domain | ≤ 10 req/s | — | XBRL tagging inconsistent across firms | key on `filed` date | researched |
| Macro (US) | FRED / ALFRED | decades | daily…quarterly | free API key or CSV | FRED terms; some series third-party-licensed | 120 req/min | — | must use vintages for history | `realtime_start` alignment | researched |
| India official | NSE / BSE websites, bhavcopy | — | 1d | website | ToS restrict automated access | — | — | **not scraped**; manual download only if ever needed | — | rejected (automation) |
| News | public RSS feeds | varies | event | RSS | per-publisher; headlines only | — | full text usually not licensed | optional; deferred until M5+ | — | researched |

## Key verified facts (2026-11-24 probes)
- Yahoo chart meta: AAPL tz America/New_York USD; RELIANCE.NS and ^NSEI tz Asia/Kolkata INR.
- Binance public data files for 2017-08 (first BTCUSDT month), 2026-10 monthly, and 2026-11-22 daily
  klines return HTTP 200; monthly aggTrades available.
- `data-api.binance.vision/api/v3/depth` answers with `lastUpdateId` and bid/ask levels (L2).

## What is NOT available for free (do not design around it)
- Historical full-depth order books for any of the three markets.
- Historical bid/ask quotes for US/India equities (spread must be assumed - see cost table).
- Indian intraday history from official sources.
- Point-in-time historical index constituents (survivorship risk for any cross-sectional study).
