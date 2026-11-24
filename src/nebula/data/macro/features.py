"""Combined as-of macro features: Yahoo market macro + BLS (+ FRED only if built from manual CSVs).

Derived series are computed on the OBSERVATION sequence of each source series and inherit the
`available_from` of their latest component, then everything is joined as-of: on date t a value is
used only if `available_from <= t` (forward-filled from availability, never before).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nebula.common.paths import Paths, get_paths
from nebula.data.macro.bls import bls_path
from nebula.data.macro.fred import fred_path
from nebula.data.macro.fred import macro_features as asof_join
from nebula.data.macro.market import market_path

DERIVED = ("slope_10y_3m", "tnx_chg_20d", "vix_z20", "cpi_3m_ann", "unrate_chg_3m")


def load_macro_tables(paths: Paths | None = None) -> pd.DataFrame:
    """Long table of every available macro source; FRED series are prefixed `FRED_`."""
    paths = paths or get_paths()
    frames = []
    for p, prefix in ((market_path(paths), ""), (bls_path(paths), ""), (fred_path(paths), "FRED_")):
        if p.exists():
            t = pq.read_table(p).to_pandas()
            t["series"] = prefix + t["series"].astype(str)
            frames.append(t)
    if not frames:
        raise FileNotFoundError("No macro tables; run ingest_market_macro() / ingest_bls() first")
    return pd.concat(frames, ignore_index=True)


def _obs(table: pd.DataFrame, s: str) -> pd.DataFrame:
    g = table[table["series"] == s].copy()
    g["date"] = pd.to_datetime(g["date"])
    g["available_from"] = pd.to_datetime(g["available_from"])
    return g.sort_values("date").reset_index(drop=True)


def _rows(name: str, g: pd.DataFrame, value: pd.Series, avail: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"series": name, "date": g["date"], "value": value, "available_from": avail}).dropna(
        subset=["value", "available_from"]
    )


def derived_series(table: pd.DataFrame) -> pd.DataFrame:
    """Long rows for the derived features (each with a conservative available_from)."""
    out = []
    tnx, irx, vix = _obs(table, "TNX"), _obs(table, "IRX"), _obs(table, "VIX")
    if len(tnx) and len(irx):
        m = tnx.merge(irx, on="date", suffixes=("_t", "_i"))
        avail = m[["available_from_t", "available_from_i"]].max(axis=1)
        out.append(_rows("slope_10y_3m", m, m["value_t"] - m["value_i"], avail))
    if len(tnx):
        out.append(_rows("tnx_chg_20d", tnx, tnx["value"] - tnx["value"].shift(20), tnx["available_from"]))
    if len(vix):
        r = vix["value"].rolling(20, min_periods=20)
        sd = r.std(ddof=1).replace(0.0, np.nan)
        out.append(_rows("vix_z20", vix, (vix["value"] - r.mean()) / sd, vix["available_from"]))
    for src, name, fn in (
        ("CPI", "cpi_3m_ann", lambda cur, prev: (cur / prev) ** 4 - 1.0),
        ("UNRATE", "unrate_chg_3m", lambda cur, prev: cur - prev),
    ):
        g = _obs(table, src)
        if g.empty:
            continue
        prev = g[["date", "value", "available_from"]].assign(date=g["date"] + pd.DateOffset(months=3))
        m = g.merge(prev, on="date", how="left", suffixes=("", "_prev"))  # exact month t-3 (gaps -> NaN)
        avail = m[["available_from", "available_from_prev"]].max(axis=1)
        out.append(_rows(name, m, fn(m["value"], m["value_prev"]), avail))
    return (
        pd.concat(out, ignore_index=True)
        if out
        else pd.DataFrame(columns=["series", "date", "value", "available_from"])
    )


def macro_features(
    dates: pd.Series | pd.DatetimeIndex | list,
    table: pd.DataFrame | None = None,
    paths: Paths | None = None,
) -> pd.DataFrame:
    """As-of macro features indexed by `dates`: raw levels (TNX, FVX, IRX, VIX, DXY, USDINR, CPI,
    UNRATE, FRED_* if present) plus DERIVED columns."""
    table = load_macro_tables(paths) if table is None else table
    long = pd.concat(
        [table[["series", "date", "value", "available_from"]], derived_series(table)], ignore_index=True
    )
    base = [s for s in table["series"].drop_duplicates().tolist()]
    der = [s for s in DERIVED if s in set(long["series"])]
    return asof_join(dates, series=base + der, table=long)
