# Data sources and feasibility matrix

Status legend: **verified** = ASTRA fetched real data from it in code; **probed** = endpoint answered
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

## Extended data (2026-11-25) - verified by ingestion
| Dataset | Source | Coverage | Size | Notes |
|---|---|---|---|---|
| Daily bars, 111 instruments | Yahoo (US 45, IN 50) + Binance (16) | equities 2000→, crypto 2017→ | 637,180 bars | survivorship-biased (today's names) |
| Hourly klines, 16 crypto pairs | Binance Public Data | 2017→ | ~1.0M bars | off-grid bars after 2018-02-09 outage flagged, excluded in research |
| 1-minute klines BTCUSDT, ETHUSDT | Binance Public Data | 2017-08→ | ~4.87M bars each | |
| US fundamentals (30 stocks) | SEC EDGAR companyfacts (public domain) | filings 2009→ | 59,726 facts | first-filed values; usable from session after `filed`; generic User-Agent accepted |
| Market macro | Yahoo ^TNX ^FVX ^IRX ^VIX DX-Y.NYB INR=X | 2000→ (USDINR 2003→) | 39,792 rows | usable next session |
| US CPI, unemployment | BLS API v1 (public domain) | 2000-01→2026-10 | 321 months each | availability = max(lag, actual release date) |
| FRED | — | — | — | **not automated**: terms prohibit scripts, storage, ML training (ADR-0008) |
| BTCUSDT live L2 + trades | Binance WebSocket | 6 × 1 h + 5 min (2026-11-24/25) | ≈ 211k depth events, 167k trades | 0 sequence gaps |
Details: [docs/data/FUNDAMENTALS_MACRO.md](data/FUNDAMENTALS_MACRO.md).

## Endpoints actually used by code
| Purpose | Endpoint | Module |
|---|---|---|
| US/IN daily bars | Yahoo chart API via `yfinance` | data/adapters/yahoo.py |
| Crypto daily klines | `https://data.binance.vision/data/spot/{monthly,daily}/klines/...` (+ `.CHECKSUM`) | data/adapters/binance.py, data/download.py |
| Crypto aggTrades (daily) | `https://data.binance.vision/data/spot/daily/aggTrades/...` (+ `.CHECKSUM`) | microstructure/trades.py |
| L2 snapshot | `https://data-api.binance.vision/api/v3/depth?limit=1000` | microstructure/capture.py |
| L2 diffs + trades (live) | `wss://stream.binance.com:9443` (fallback `wss://data-stream.binance.vision`) | microstructure/capture.py |
| Optional LLM | localhost only (`ASTRA_LLM_URL`), disabled by default | agents/llm.py |
| US fundamentals | `https://data.sec.gov/api/xbrl/companyfacts/CIK*.json`, `https://www.sec.gov/files/company_tickers.json` | data/fundamentals/sec.py |
| CPI / unemployment | `https://api.bls.gov/publicAPI/v1/timeseries/data/` | data/macro/bls.py |
| Market macro | Yahoo via yfinance (^TNX ^FVX ^IRX ^VIX DX-Y.NYB INR=X) | data/macro/market.py |

## Verified ingestion (2026-11-24)
All 13 instruments ingested daily bars from 2018-01-01 (SOLUSDT from its 2020-08 listing): AAPL, MSFT,
NVDA, QQQ pass; SPY, the Indian names and crypto pass with explained warnings (Muhurat sessions, Yahoo
gaps/placeholder rows, real crash days). Yahoo's in-progress session bar is dropped (2 h settlement buffer).

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
