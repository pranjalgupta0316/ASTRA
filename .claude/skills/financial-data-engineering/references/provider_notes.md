# Provider notes (verify before relying - re-check when an adapter breaks)

| Provider | Access | Notes |
|---|---|---|
| Yahoo Finance via `yfinance` | unofficial scraping of public JSON endpoints | Personal/research use per Yahoo ToS; no SLA; daily history decades; 1m data ~7-30 days, 1h ~730 days; `auto_adjust` default changed to True in 0.2.5x - always pass explicitly. India: `.NS` (NSE) / `.BO` (BSE), index `^NSEI`. |
| Binance Public Data | `https://data.binance.vision/` static zip + CHECKSUM | Spot klines, aggTrades, trades daily/monthly files; no historical order book for spot. Timestamps ms until 2024-12-31, microseconds from 2025-01-01 for spot. |
| Binance REST | `https://api.binance.com/api/v3/klines` etc. | Weight-based rate limits (check `X-MBX-USED-WEIGHT-1M`). May be geo-restricted in some regions; `data-api.binance.vision` is a market-data-only mirror. |
| Binance WebSocket | `wss://stream.binance.com:9443/ws/<sym>@depth@100ms` | Diff depth with `U`/`u` update IDs; must sync with REST `/api/v3/depth` snapshot (`lastUpdateId`). |
| SEC EDGAR | `https://data.sec.gov/` JSON APIs | Requires descriptive User-Agent header; <= 10 requests/s. companyfacts gives XBRL facts with `filed` dates. |
| FRED / ALFRED | API key (free) or `fredgraph.csv` download | ALFRED provides vintages (`realtime_start`). |
| NSE / BSE official | website, bhavcopy archives | Terms restrict automated access; do not scrape. Bhavcopy archive links change; treat as manual-download source only. |
