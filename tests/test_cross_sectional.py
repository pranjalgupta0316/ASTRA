"""Tests for astra.research.signals and the cross-sectional study helpers (H-CS-001 / H-CS-002)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from astra.research.cross_sectional import (
    bh_verdicts,
    compute_signal,
    rebalance_dates,
    signal_spec,
    verdict_t,
)
from astra.research.signals import (
    eligibility,
    equal_weight_targets,
    momentum_signal,
    monthly_first_sessions,
    n_rankable,
    reversal_signal,
    tercile_masks,
    weekly_first_sessions,
)


def _panel(n: int = 400, k: int = 7, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, k)), axis=0))
    df = pd.DataFrame(px, index=idx, columns=[f"S{i}" for i in range(k)])
    df.iloc[:120, 1] = np.nan  # late listing
    df.iloc[200:205, 2] = np.nan  # gap
    return df


@pytest.mark.parametrize(
    "fn",
    [
        lambda c: momentum_signal(c, 252, 21),
        lambda c: momentum_signal(c, 28, 1),
        lambda c: reversal_signal(c, 5),
        lambda c: eligibility(c, 252).astype(float),
    ],
)
def test_signal_truncation_invariance(fn) -> None:
    close = _panel()
    full = fn(close)
    for cut in (260, 300, 371, 399):
        trunc = fn(close.iloc[: cut + 1])
        pd.testing.assert_frame_equal(full.iloc[: cut + 1], trunc)


def test_future_perturbation_does_not_change_past_signal() -> None:
    close = _panel()
    t = 300
    shocked = close.copy()
    shocked.iloc[t + 1 :] *= 3.0
    for fn in (lambda c: momentum_signal(c, 252, 21), lambda c: reversal_signal(c, 5)):
        pd.testing.assert_frame_equal(fn(close).iloc[: t + 1], fn(shocked).iloc[: t + 1])


def test_signal_values_hand_computed() -> None:
    idx = pd.bdate_range("2021-01-01", periods=30)
    c = pd.DataFrame({"A": np.arange(1.0, 31.0)}, index=idx)
    mom = momentum_signal(c, 28, 1)
    assert np.isnan(mom["A"].iloc[27])
    assert mom["A"].iloc[28] == pytest.approx(np.log(28.0 / 1.0))  # C[t-1]=28, C[t-28]=1
    rev = reversal_signal(c, 5)
    assert rev["A"].iloc[10] == pytest.approx(-(11.0 / 6.0 - 1.0))
    m121 = momentum_signal(pd.DataFrame({"A": np.arange(1.0, 301.0)}), 252, 21)
    assert m121["A"].iloc[260] == pytest.approx(np.log(240.0 / 9.0))
    with pytest.raises(ValueError):
        momentum_signal(c, 5, 5)


def test_eligibility_252_rule_no_backfill() -> None:
    close = _panel()
    el = eligibility(close, 252)
    # S0 valid from row 0 -> eligible from row 251 (252nd valid bar)
    assert not el["S0"].iloc[250] and el["S0"].iloc[251]
    # S1 lists at row 120 -> eligible from row 371
    assert not el["S1"].iloc[370] and el["S1"].iloc[371]
    # S2 missing rows 200-204: 5 fewer valid bars, and never eligible on a missing bar
    assert not el["S2"].iloc[255] and el["S2"].iloc[256]
    assert not el["S2"].iloc[200:205].any()


def test_tercile_assignment_and_ties() -> None:
    names = list("ABCDEFGHI")
    sig = pd.Series([9, 8, 7, 6, 5, 4, 3, 2, 1], index=names, dtype=float)
    elig = pd.Series(True, index=names)
    top, bot = tercile_masks(sig, elig)
    assert list(sig.index[top]) == ["A", "B", "C"]
    assert list(sig.index[bot]) == ["G", "H", "I"]
    # tie at the top boundary: C and D share the 3rd-largest value -> both included
    sig2 = sig.copy()
    sig2["D"] = 7.0
    top2, _ = tercile_masks(sig2, elig)
    assert set(sig2.index[top2]) == {"A", "B", "C", "D"}
    # order independence
    perm = sig2.sample(frac=1.0, random_state=3)
    top3, _ = tercile_masks(perm, elig.reindex(perm.index))
    assert set(perm.index[top3]) == {"A", "B", "C", "D"}
    # ineligible and NaN names are not ranked; n=7 -> k=2
    sig4 = sig.copy()
    sig4["A"] = np.nan
    elig4 = elig.copy()
    elig4["I"] = False
    top4, bot4 = tercile_masks(sig4, elig4)
    assert set(sig4.index[top4]) == {"B", "C"}
    assert set(sig4.index[bot4]) == {"G", "H"}
    # all equal -> everyone in both terciles; none eligible -> empty
    t5, b5 = tercile_masks(pd.Series(1.0, index=names), elig)
    assert t5.all() and b5.all()
    t6, b6 = tercile_masks(sig, pd.Series(False, index=names))
    assert not t6.any() and not b6.any()


def test_rebalance_schedules() -> None:
    idx = pd.DatetimeIndex(
        ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05", "2024-03-01", "2024-03-04"]
    )
    assert list(monthly_first_sessions(idx)) == list(pd.DatetimeIndex(["2024-02-01", "2024-03-01"]))
    # ISO weeks: Tue 30 Jan (first row, excluded), Mon 5 Feb, Fri 1 Mar, Mon 4 Mar
    assert list(weekly_first_sessions(idx)) == list(
        pd.DatetimeIndex(["2024-02-05", "2024-03-01", "2024-03-04"])
    )
    # 24/7: Monday every week, ISO year boundary handled (2025-12-29 is ISO 2026-W01)
    days = pd.date_range("2025-12-24", "2026-01-13", freq="D")
    wk = weekly_first_sessions(days)
    assert list(wk) == list(pd.DatetimeIndex(["2025-12-29", "2026-01-05", "2026-01-12"]))
    assert all(d.day_of_week == 0 for d in wk)
    # holiday Monday -> first session is Tuesday
    hol = pd.DatetimeIndex(["2024-05-24", "2024-05-28", "2024-05-29"])
    assert list(weekly_first_sessions(hol)) == [pd.Timestamp("2024-05-28")]


def test_spec_routing() -> None:
    assert signal_spec("H-CS-001", "US").rebalance == "monthly"
    assert signal_spec("H-CS-001", "IN").rebalance == "monthly"
    assert signal_spec("H-CS-001", "CRYPTO").rebalance == "weekly"
    for m in ("US", "IN", "CRYPTO"):
        assert signal_spec("H-CS-002", m).rebalance == "weekly"
    close = _panel()
    pd.testing.assert_frame_equal(compute_signal("H-CS-001", "CRYPTO", close), momentum_signal(close, 28, 1))
    pd.testing.assert_frame_equal(compute_signal("H-CS-001", "US", close), momentum_signal(close, 252, 21))
    pd.testing.assert_frame_equal(compute_signal("H-CS-002", "IN", close), reversal_signal(close, 5))
    d = rebalance_dates(signal_spec("H-CS-001", "US"), close.index)
    assert all(a.month != b.month for a, b in itertools.pairwise(d))


def test_benchmark_and_strategy_weights() -> None:
    close = _panel(k=7)
    sig = momentum_signal(close, 28, 1)
    el = eligibility(close, 252)
    dates = weekly_first_sessions(close.index)
    dates = dates[dates >= close.index[371]]
    bench = equal_weight_targets(sig, el, dates, "all")
    top = equal_weight_targets(sig, el, dates, "top")
    bot = equal_weight_targets(sig, el, dates, "bottom")
    n = n_rankable(sig, el).loc[dates]
    assert (n == 7).all()
    np.testing.assert_allclose(bench.sum(axis=1), 1.0)
    np.testing.assert_allclose(bench.to_numpy(), 1 / 7)
    np.testing.assert_allclose(top.sum(axis=1), 1.0)
    assert ((top > 0).sum(axis=1) == 2).all()  # k = 7 // 3, continuous signal -> no ties
    np.testing.assert_allclose(top.to_numpy()[top.to_numpy() > 0], 0.5)
    assert not ((top > 0) & (bot > 0)).to_numpy().any()
    # top names have the highest signal at each date
    for d in dates:
        s = sig.loc[d]
        assert s[top.loc[d] > 0].min() > s[top.loc[d] == 0].max()
    # before S1 is eligible, benchmark excludes it
    early = weekly_first_sessions(close.index)
    early = early[(early >= close.index[260]) & (early < close.index[371])]
    b_early = equal_weight_targets(sig, el, early, "all")
    assert (b_early["S1"] == 0).all()
    np.testing.assert_allclose(b_early.drop(columns="S1").to_numpy()[b_early.drop(columns="S1") > 0], 1 / 6)


def test_verdicts_and_bh() -> None:
    assert verdict_t(2.0) == "supported"
    assert verdict_t(-2.0) == "rejected"
    assert verdict_t(1.0) == "inconclusive"
    out = bh_verdicts({"US": 3.5, "IN": -2.0, "CRYPTO": 0.3})
    assert out["US"]["verdict_bh"] == "supported"
    assert out["CRYPTO"]["verdict_bh"] == "inconclusive"
    # raw p(2.0)=0.0455, BH-adjusted = 0.0455*3/2 = 0.068 > 0.05 -> not rejected after BH
    assert out["IN"]["p_bh"] == pytest.approx(0.0455 * 1.5, abs=1e-3)
    assert out["IN"]["verdict_bh"] == "inconclusive"
    assert bh_verdicts({"A": -4.0})["A"]["verdict_bh"] == "rejected"


def test_regular_sessions_drops_single_name_rows_causally() -> None:
    from astra.research.signals import regular_sessions

    close = _panel(n=300, k=6)
    close.iloc[150, 1:] = np.nan  # spurious holiday row: only S0 has a bar
    keep = regular_sessions(close, 0.5)
    assert not keep.iloc[150]
    assert keep.drop(close.index[150]).all()  # S1 unlisted before row 120 does not count against coverage
    pd.testing.assert_series_equal(keep.iloc[:201], regular_sessions(close.iloc[:201], 0.5))
