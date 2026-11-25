from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astra.features.technical import FEATURE_COLUMNS, build_features, garman_klass_var, parkinson_var, rsi


def test_features_are_causal_truncation_invariant(us_bars):
    """Features at row t must not change when future rows are appended."""
    full = build_features(us_bars)
    cut = 200
    partial = build_features(us_bars.iloc[:cut])
    pd.testing.assert_frame_equal(
        full.loc[: cut - 1, FEATURE_COLUMNS].reset_index(drop=True),
        partial[FEATURE_COLUMNS].reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
    )


def test_targets_are_forward_looking(us_bars):
    f = build_features(us_bars)
    o = f["adj_open"]
    assert f["target_ret_oo"].iloc[10] == pytest.approx(np.log(o.iloc[12] / o.iloc[11]))
    assert f["target_ret_oo"].iloc[-1:].isna().all() and f["target_ret_oo"].iloc[-2:].isna().all()


def test_nonpositive_adjustment_becomes_missing(us_bars):
    """Provider error (ADANIENT 2002): adj_close <= 0 must give missing prices, never negative ones."""
    from astra.common.config import get_config
    from astra.data.validation import validate_bars
    from astra.features.technical import adjusted_ohlc

    bad = us_bars.copy()
    bad.loc[10:12, "adj_close"] = -0.01
    adj = adjusted_ohlc(bad)
    assert adj.loc[10:12, "close"].isna().all() and (adj["close"].dropna() > 0).all()
    chk = next(
        c for c in validate_bars(bad, get_config().markets["US"]).checks if c.name == "corporate_actions"
    )
    assert chk.status == "warn" and chk.count == 3


def test_rsi_known_values():
    up = pd.Series(np.arange(1.0, 40.0))
    assert rsi(up).dropna().eq(100.0).all()
    down = pd.Series(np.arange(40.0, 1.0, -1))
    assert rsi(down).dropna().max() == pytest.approx(0.0)


def test_parkinson_and_gk_nonnegative_on_flat_open_close():
    h, lo = pd.Series([105.0]), pd.Series([95.0])
    assert parkinson_var(h, lo).iloc[0] == pytest.approx(np.log(105 / 95) ** 2 / (4 * np.log(2)))
    gk = garman_klass_var(pd.Series([100.0]), h, lo, pd.Series([100.0]))
    assert gk.iloc[0] == pytest.approx(0.5 * np.log(105 / 95) ** 2)


def test_dividend_adjustment_does_not_create_return_gap(us_bars):
    """A dividend lowers adj factor for history; adjusted returns must be continuous across ex-date."""
    bars = us_bars.copy()
    ex = 150
    # Simulate provider: unadjusted close drops by dividend on ex-date, adj_close back-adjusted
    div = 2.0
    bars.loc[ex:, ["open", "high", "low", "close"]] -= div
    factor = 1 - div / us_bars.loc[ex - 1, "close"]
    bars["adj_close"] = bars["close"]
    bars.loc[: ex - 1, "adj_close"] = bars.loc[: ex - 1, "close"] * factor
    f = build_features(bars)
    f0 = build_features(us_bars)
    # Adjusted return on ex-date ~ original return (dividend reinvested), not a -2 point drop
    assert f.loc[ex, "ret_1"] == pytest.approx(
        np.log((us_bars.loc[ex, "close"] - div) / (us_bars.loc[ex - 1, "close"] * factor)), rel=1e-9
    )
    assert abs(f.loc[ex, "ret_1"] - f0.loc[ex, "ret_1"]) < 1e-3
