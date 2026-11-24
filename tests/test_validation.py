from __future__ import annotations

import pandas as pd

from nebula.common.config import get_config
from nebula.data.validation import validate_bars

US = get_config().markets["US"]
CRYPTO = get_config().markets["CRYPTO"]


def _status(report, name):
    return next(c for c in report.checks if c.name == name).status


def test_clean_synthetic_passes(us_bars):
    rep = validate_bars(us_bars, US)
    assert rep.status == "pass", rep.summary()


def test_crypto_calendar_is_continuous(crypto_bars):
    rep = validate_bars(crypto_bars, CRYPTO)
    assert _status(rep, "sessions") == "pass", rep.summary()


def test_duplicate_detected(us_bars):
    bad = pd.concat([us_bars, us_bars.iloc[[10]]]).sort_values("close_time_utc").reset_index(drop=True)
    assert _status(validate_bars(bad, US), "duplicates") == "fail"


def test_missing_session_detected(us_bars):
    bad = us_bars.drop(index=list(range(50, 60))).reset_index(drop=True)
    rep = validate_bars(bad, US)
    assert _status(rep, "sessions") in {"warn", "fail"}
    sess = next(c for c in rep.checks if c.name == "sessions")
    assert sess.count == 10


def test_weekend_row_is_off_calendar(us_bars):
    bad = us_bars.copy()
    # move one session onto a Saturday
    sat = pd.Timestamp("2020-01-04").date()
    bad.loc[1, "session_date"] = sat
    rep = validate_bars(bad.sort_values("session_date").reset_index(drop=True), US)
    assert "off-calendar" in next(c for c in rep.checks if c.name == "sessions").details


def test_ohlc_inconsistency_and_negative_price(us_bars):
    bad = us_bars.copy()
    bad.loc[5, "high"] = bad.loc[5, "low"] * 0.5
    bad.loc[7, "close"] = -1.0
    check = next(c for c in validate_bars(bad, US).checks if c.name == "price_consistency")
    assert check.status == "fail" and check.count == 2


def test_out_of_order_detected(us_bars):
    bad = us_bars.iloc[::-1].reset_index(drop=True)
    assert _status(validate_bars(bad, US), "monotonic_time") == "fail"


def test_naive_timestamps_detected(us_bars):
    bad = us_bars.copy()
    bad["close_time_utc"] = bad["close_time_utc"].dt.tz_localize(None)
    assert _status(validate_bars(bad, US), "monotonic_time") == "fail"


def test_negative_volume(us_bars):
    bad = us_bars.copy()
    bad.loc[3, "volume"] = -5
    assert _status(validate_bars(bad, US), "volume") == "fail"


def test_schema_missing_column(us_bars):
    rep = validate_bars(us_bars.drop(columns=["volume"]), US)
    assert rep.status == "fail" and len(rep.checks) == 1


def _hourly(n=500):
    from nebula.data.contracts import BAR_COLUMNS

    t = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({c: 1.0 for c in BAR_COLUMNS}, index=range(n))
    df["open_time_utc"], df["close_time_utc"] = t, t + pd.Timedelta(hours=1)
    df["session_date"] = t.date
    df[["symbol", "provider_symbol", "market", "exchange", "exchange_tz", "currency", "provider"]] = "X"
    df["ingested_at_utc"] = pd.Timestamp("2026-01-01", tz="UTC")
    df["open"] = df["high"] = df["low"] = df["close"] = df["adj_close"] = 100.0
    return df


def test_intraday_continuity_and_duplicate_key():
    ok = validate_bars(_hourly(), CRYPTO, freq="1h")
    assert _status(ok, "sessions") == "pass" and _status(ok, "duplicates") == "pass"
    # an outage of 5 hours = one gap run, 1% missing -> warn
    gap = _hourly().drop(index=range(100, 105)).reset_index(drop=True)
    s = next(c for c in validate_bars(gap, CRYPTO, freq="1h").checks if c.name == "sessions")
    assert s.status == "warn" and s.count == 5 and "in 1 gap" in s.details
    # many bars per session_date must not count as duplicates intraday
    assert _status(validate_bars(_hourly(), CRYPTO, freq="1h"), "duplicates") == "pass"
    big = _hourly().drop(index=range(50, 120)).reset_index(drop=True)
    assert _status(validate_bars(big, CRYPTO, freq="1h"), "sessions") == "fail"
    # a few off-grid bars (provider artifact) -> warn; many -> fail
    few = _hourly(1000)
    few.loc[500:502, "open_time_utc"] += pd.Timedelta(minutes=28)
    assert _status(validate_bars(few, CRYPTO, freq="1h"), "sessions") == "warn"
    many = _hourly(1000)
    many.loc[100:200, "open_time_utc"] += pd.Timedelta(minutes=28)
    assert _status(validate_bars(many, CRYPTO, freq="1h"), "sessions") == "fail"


def test_validation_does_not_mutate(us_bars):
    before = us_bars.copy()
    validate_bars(us_bars, US)
    pd.testing.assert_frame_equal(before, us_bars)
