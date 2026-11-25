from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astra.common.config import get_config
from astra.data.calendars import session_schedule
from astra.data.contracts import BAR_COLUMNS


def make_bars(market: str = "US", n: int = 300, seed: int = 0, start: str = "2020-01-02") -> pd.DataFrame:
    """Synthetic canonical bars on the real exchange calendar (GBM prices)."""
    cfg = get_config()
    m = cfg.markets[market]  # type: ignore[index]
    sched = session_schedule(m, start, pd.Timestamp(start) + pd.Timedelta(days=int(n * 1.6) + 10)).iloc[:n]
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.015, len(sched))
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.003, len(sched)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, len(sched))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, len(sched))))
    df = pd.DataFrame(
        {
            "session_date": sched.index.date,
            "open_time_utc": sched["open"].to_numpy(),
            "close_time_utc": sched["close"].to_numpy(),
            "symbol": "TEST",
            "provider_symbol": "TEST",
            "market": market,
            "exchange": "TEST",
            "exchange_tz": m.exchange_tz,
            "currency": m.currency,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(sched)).astype(float),
            "quote_volume": np.nan,
            "trade_count": np.nan,
            "dividend": 0.0,
            "split_ratio": 1.0,
            "provider": "synthetic",
            "ingested_at_utc": pd.Timestamp("2026-01-01", tz="UTC"),
        }
    )
    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"], utc=True)
    df["close_time_utc"] = pd.to_datetime(df["close_time_utc"], utc=True)
    return df[BAR_COLUMNS]


@pytest.fixture
def us_bars() -> pd.DataFrame:
    return make_bars("US")


@pytest.fixture
def crypto_bars() -> pd.DataFrame:
    return make_bars("CRYPTO", start="2021-01-01")
