"""Offline tests of calendars and adapter normalization (no network)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from nebula.common.config import get_config
from nebula.data.adapters.base import RawBatch
from nebula.data.adapters.binance import BinanceVisionAdapter, _to_utc
from nebula.data.adapters.yahoo import YahooAdapter
from nebula.data.calendars import session_schedule
from nebula.data.contracts import BAR_COLUMNS
from nebula.data.validation import validate_bars

CFG = get_config()


def test_us_calendar_excludes_holidays_and_handles_dst():
    sched = session_schedule(CFG.markets["US"], "2024-07-01", "2024-07-08")
    assert pd.Timestamp("2024-07-04") not in sched.index  # Independence Day
    # 16:00 New York = 20:00 UTC in summer (EDT), 21:00 UTC in winter (EST)
    assert sched.loc["2024-07-05", "close"].hour == 20
    winter = session_schedule(CFG.markets["US"], "2024-01-08", "2024-01-08")
    assert winter["close"].iloc[0].hour == 21


def test_india_calendar_close_is_1530_ist():
    sched = session_schedule(CFG.markets["IN"], "2024-03-04", "2024-03-08")
    close = sched["close"].iloc[0].tz_convert("Asia/Kolkata")
    assert (close.hour, close.minute) == (15, 30)


def test_binance_timestamp_units_ms_and_us():
    ms = pd.Series([1704067200000])  # 2024-01-01 in ms
    us = pd.Series([1735689600000000])  # 2025-01-01 in us
    assert _to_utc(ms).iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert _to_utc(us).iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")


def test_binance_normalize_contract():
    inst = CFG.instrument("BTCUSDT")
    raw = pd.DataFrame(
        {
            "open_time": [1735603200000, 1735689600000000],  # 2024-12-31 ms, 2025-01-01 us
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 11.0],
            "close_time": [0, 0],
            "quote_volume": [1000.0, 1100.0],
            "trade_count": [5, 6],
            "taker_buy_base": [1.0, 1.0],
            "taker_buy_quote": [1.0, 1.0],
            "ignore": [0, 0],
        }
    )
    batch = RawBatch("binance_vision", inst, datetime.now(UTC), raw)
    out = BinanceVisionAdapter().normalize_daily(batch, CFG.markets["CRYPTO"])
    assert list(out.columns) == BAR_COLUMNS
    assert str(out["session_date"].iloc[1]) == "2025-01-01"
    assert (out["close_time_utc"] - out["open_time_utc"]).eq(pd.Timedelta(days=1)).all()
    assert validate_bars(out, CFG.markets["CRYPTO"]).status in {"pass", "warn"}


def test_yahoo_normalize_contract_and_close_time():
    inst = CFG.instrument("RELIANCE")
    idx = pd.DatetimeIndex(["2024-03-04", "2024-03-05"]).tz_localize("Asia/Kolkata")
    raw = pd.DataFrame(
        {
            "Date": idx,
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [100.0, 101.0],
            "Volume": [1e6, 1.1e6],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        }
    )
    batch = RawBatch("yahoo", inst, datetime.now(UTC), raw)
    out = YahooAdapter().normalize_daily(batch, CFG.markets["IN"])
    assert list(out.columns) == BAR_COLUMNS
    assert (out["split_ratio"] == 1.0).all()
    # available at 15:30 IST = 10:00 UTC
    assert out["close_time_utc"].iloc[0] == pd.Timestamp("2024-03-04 10:00", tz="UTC")
