"""Pure, causal cross-sectional signal helpers (H-CS-001 momentum, H-CS-002 short-term reversal).

All functions take wide frames indexed by one market's session calendar (rows = sessions, columns =
symbols, NaN = no valid bar). The value at row t only reads rows <= t, so truncating the input after t
never changes the output at t (tested in tests/test_cross_sectional.py). Lags are in panel rows, i.e.
market sessions (calendar days for 24/7 crypto).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Leg = Literal["top", "bottom", "all"]


def _carried_close(close: pd.DataFrame) -> pd.DataFrame:
    """Last valid close known at each row (forward fill only - causal)."""
    return close.ffill()


def momentum_signal(close: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """log(C[t-skip] / C[t-lookback]) with C the last valid close known at each lagged row.

    Equities 12-1: lookback=252, skip=21. Crypto 4-week: lookback=28, skip=1 (H-EXT.md).
    """
    if not 0 <= skip < lookback:
        raise ValueError("need 0 <= skip < lookback")
    c = _carried_close(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(c.shift(skip) / c.shift(lookback))
    return out.where(np.isfinite(out))


def reversal_signal(close: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """-(C[t] / C[t-horizon] - 1): past losers score highest (H-CS-002)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    c = _carried_close(close)
    out = -(c / c.shift(horizon) - 1.0)
    return out.where(np.isfinite(out))


def eligibility(close: pd.DataFrame, min_history: int = 252) -> pd.DataFrame:
    """True at (t, s) iff s has a valid bar at t and >= `min_history` valid bars up to and including t.

    No backfilling: a name enters only once its own history is long enough.
    """
    valid = close.notna()
    return valid & (valid.cumsum() >= min_history)


def regular_sessions(close: pd.DataFrame, min_coverage: float = 0.5) -> pd.Series:
    """True for rows where >= `min_coverage` of the names already listed (first valid bar <= t) have a
    bar. Causal (row t only). Removes spurious calendar rows created by the union of per-symbol dates,
    e.g. an exchange holiday on which a single provider series has a bar."""
    valid = close.notna()
    listed = valid.cummax().sum(axis=1)
    cov = valid.sum(axis=1) / listed.where(listed > 0)
    return cov.fillna(0.0) >= min_coverage


def monthly_first_sessions(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First session of each calendar month. The panel's first row is excluded (we cannot know it is
    the first session of its month without data before the sample)."""
    idx = pd.DatetimeIndex(index)
    per = idx.to_period("M")
    return idx[np.r_[False, per[1:] != per[:-1]]]


def weekly_first_sessions(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First session of each ISO week (Monday-Sunday); Monday for 24/7 crypto. First row excluded."""
    idx = pd.DatetimeIndex(index)
    iso = idx.isocalendar()
    key = (iso["year"].to_numpy() * 100 + iso["week"].to_numpy()).astype(np.int64)
    return idx[np.r_[False, key[1:] != key[:-1]]]


def tercile_masks(signal: pd.Series, eligible: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Top / bottom tercile membership among eligible names with a finite signal.

    k = max(1, n // 3) with n the number of rankable names. Ties at a boundary are all included (a name
    is in the top tercile iff its signal >= the k-th largest value), so a tercile can exceed k names when
    the boundary value is tied; membership never depends on column order.
    """
    ok = eligible.reindex(signal.index).fillna(False).astype(bool) & np.isfinite(signal)
    s = signal[ok]
    top = pd.Series(False, index=signal.index)
    bottom = pd.Series(False, index=signal.index)
    n = len(s)
    if n == 0:
        return top, bottom
    k = max(1, n // 3)
    vals = np.sort(s.to_numpy(float))
    hi_cut, lo_cut = vals[n - k], vals[k - 1]
    top.loc[s.index[s.to_numpy() >= hi_cut]] = True
    bottom.loc[s.index[s.to_numpy() <= lo_cut]] = True
    return top, bottom


def equal_weight_targets(
    signal: pd.DataFrame, eligible: pd.DataFrame, dates: pd.DatetimeIndex, leg: Leg
) -> pd.DataFrame:
    """Equal-weight target rows at `dates`: the top tercile, bottom tercile or all eligible names
    (benchmark; uses eligibility and a finite signal so it covers exactly the ranked set)."""
    rows = {}
    for d in dates:
        sig, elig = signal.loc[d], eligible.loc[d]
        if leg == "all":
            mask = elig.fillna(False).astype(bool) & np.isfinite(sig)
        else:
            top, bottom = tercile_masks(sig, elig)
            mask = top if leg == "top" else bottom
        n = int(mask.sum())
        rows[d] = mask.astype(float) / n if n else mask.astype(float)
    return pd.DataFrame(rows).T.reindex(columns=signal.columns).fillna(0.0)


def n_rankable(signal: pd.DataFrame, eligible: pd.DataFrame) -> pd.Series:
    """Number of eligible names with a finite signal per row."""
    return (eligible.astype(bool) & np.isfinite(signal)).sum(axis=1)
