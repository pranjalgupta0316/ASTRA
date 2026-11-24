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


def test_validation_does_not_mutate(us_bars):
    before = us_bars.copy()
    validate_bars(us_bars, US)
    pd.testing.assert_frame_equal(before, us_bars)
