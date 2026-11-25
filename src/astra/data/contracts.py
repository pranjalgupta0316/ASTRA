"""Canonical data contracts.

Daily bar contract (one row per instrument-session):

| column            | type                 | meaning                                                         |
|-------------------|----------------------|-----------------------------------------------------------------|
| session_date      | date                 | exchange-local trading date (crypto: UTC date)                  |
| open_time_utc     | timestamp[us, UTC]   | session open                                                    |
| close_time_utc    | timestamp[us, UTC]   | session close == availability time of the bar's close/volume    |
| symbol            | str                  | ASTRA symbol (e.g. RELIANCE)                                   |
| provider_symbol   | str                  | provider symbol (e.g. RELIANCE.NS)                              |
| market            | str                  | US / IN / CRYPTO                                                |
| exchange          | str                  | NASDAQ / NSE / BINANCE                                          |
| exchange_tz       | str                  | IANA tz of the exchange                                         |
| currency          | str                  | quote currency                                                  |
| open,high,low,close | float64            | provider prices (Yahoo: split-adjusted, NOT dividend-adjusted)  |
| adj_close         | float64 (nullable)   | split+dividend adjusted close (equities); = close (crypto)     |
| volume            | float64              | base-asset volume (shares / coins)                              |
| quote_volume      | float64 (nullable)   | quote-currency volume when provided (crypto)                    |
| trade_count       | float64 (nullable)   | number of trades when provided (crypto)                         |
| dividend          | float64              | cash dividend with ex-date on this session (0 if none)          |
| split_ratio       | float64              | split ratio effective this session (1 if none)                  |
| provider          | str                  | yahoo / binance_vision                                          |
| ingested_at_utc   | timestamp[us, UTC]   | when ASTRA retrieved the raw data                              |
"""

from __future__ import annotations

import pyarrow as pa

BAR_COLUMNS: list[str] = [
    "session_date",
    "open_time_utc",
    "close_time_utc",
    "symbol",
    "provider_symbol",
    "market",
    "exchange",
    "exchange_tz",
    "currency",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "quote_volume",
    "trade_count",
    "dividend",
    "split_ratio",
    "provider",
    "ingested_at_utc",
]

BAR_SCHEMA = pa.schema(
    [
        ("session_date", pa.date32()),
        ("open_time_utc", pa.timestamp("us", tz="UTC")),
        ("close_time_utc", pa.timestamp("us", tz="UTC")),
        ("symbol", pa.string()),
        ("provider_symbol", pa.string()),
        ("market", pa.string()),
        ("exchange", pa.string()),
        ("exchange_tz", pa.string()),
        ("currency", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("adj_close", pa.float64()),
        ("volume", pa.float64()),
        ("quote_volume", pa.float64()),
        ("trade_count", pa.float64()),
        ("dividend", pa.float64()),
        ("split_ratio", pa.float64()),
        ("provider", pa.string()),
        ("ingested_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)

PRICE_COLUMNS = ["open", "high", "low", "close"]
