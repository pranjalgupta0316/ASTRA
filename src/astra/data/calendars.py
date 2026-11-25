"""Market calendars: exchange sessions for equities (exchange_calendars), continuous UTC days for crypto."""

from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

from astra.common.config import MarketConfig


@lru_cache(maxsize=8)
def _calendar(code: str) -> xcals.ExchangeCalendar:
    return xcals.get_calendar(code)


def session_schedule(
    market: MarketConfig, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> pd.DataFrame:
    """Return a frame indexed by session date (tz-naive) with UTC `open` and `close` columns."""
    start_ts = pd.Timestamp(start).normalize().tz_localize(None)
    end_ts = pd.Timestamp(end).normalize().tz_localize(None)
    if market.is_24_7:
        days = pd.date_range(start_ts, end_ts, freq="D")
        opens = days.tz_localize("UTC")
        return pd.DataFrame({"open": opens, "close": opens + pd.Timedelta(days=1)}, index=days)
    cal = _calendar(market.calendar)
    start_ts = max(start_ts, cal.first_session)
    end_ts = min(end_ts, cal.last_session)
    sched = cal.schedule.loc[start_ts:end_ts, ["open", "close"]].copy()
    sched.index = pd.DatetimeIndex(sched.index).tz_localize(None)
    return sched


def is_session(market: MarketConfig, day: pd.Timestamp) -> bool:
    if market.is_24_7:
        return True
    return bool(_calendar(market.calendar).is_session(pd.Timestamp(day).normalize().tz_localize(None)))
