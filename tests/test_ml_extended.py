"""H-ML-003/004 helpers and H-MM-001 multimodal construction (synthetic, offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astra.data.fundamentals import sec
from astra.features.technical import FEATURE_COLUMNS, build_features
from astra.research import ml_extended as me
from astra.research import multimodal as mm
from conftest import make_bars

# ------------------------------------------------------------------------------ H-MM-001 rows / target


def _bars_dict(n: int = 400, k: int = 3) -> dict[str, pd.DataFrame]:
    return {f"S{i}": make_bars("US", n=n, seed=i, start="2019-01-02") for i in range(k)}


def test_month_end_rows_are_last_session_of_complete_months():
    sessions = pd.Series(
        pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28", "2020-03-02"])
    )
    me_dates = mm.month_end_dates(sessions)
    # March 2020 is the month of the final session -> incomplete -> excluded
    assert list(me_dates) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]

    bars = _bars_dict()
    df, _ = mm.build_monthly_frame(bars, with_fundamentals=False, with_macro=False, min_history=0)
    all_sessions = pd.to_datetime(bars["S0"]["session_date"])
    expected = all_sessions.groupby(all_sessions.dt.to_period("M")).max()
    assert set(df["session_date"]) <= set(expected)
    # one row per symbol per decision date, every decision date is a month end
    assert not df.duplicated(["session_date", "symbol"]).any()
    assert df.groupby("session_date")["symbol"].nunique().eq(3).all()


def test_entry_rule_drops_first_252_sessions():
    bars = _bars_dict(n=500)
    df, _ = mm.build_monthly_frame(bars, with_fundamentals=False, with_macro=False)
    first_allowed = pd.to_datetime(bars["S0"]["session_date"]).iloc[mm.MIN_HISTORY]
    assert df["session_date"].min() >= first_allowed


def test_excess_return_target_is_causal_and_matches_definition():
    bars = _bars_dict()
    rows = pd.concat([mm.symbol_daily(s, b) for s, b in bars.items()], ignore_index=True)
    t = 100
    o = {s: b["open"].to_numpy() for s, b in bars.items()}  # adj_close == close -> adjusted open == open
    r = {s: o[s][t + 1 + mm.HORIZON] / o[s][t + 1] - 1 for s in bars}
    date = pd.Timestamp(bars["S0"]["session_date"].iloc[t])
    rows["session_date"] = pd.to_datetime(rows["session_date"])
    tgt = mm.add_excess_target(rows[rows["session_date"] == date]).set_index("symbol")
    ew = np.mean(list(r.values()))
    for s in bars:
        assert tgt.loc[s, "fwd_ret"] == pytest.approx(r[s])
        assert tgt.loc[s, "target"] == float(r[s] - ew > 0)

    # prices AFTER the label window (and the features' future) do not change the row at t
    shocked = {s: b.copy() for s, b in bars.items()}
    for b in shocked.values():
        for c in ("open", "high", "low", "close", "adj_close"):
            b.loc[t + 2 + mm.HORIZON :, c] *= 3.0
    rows2 = pd.concat([mm.symbol_daily(s, b) for s, b in shocked.items()], ignore_index=True)
    rows2["session_date"] = pd.to_datetime(rows2["session_date"])
    tgt2 = mm.add_excess_target(rows2[rows2["session_date"] == date]).set_index("symbol")
    pd.testing.assert_series_equal(tgt["target"], tgt2["target"])
    pd.testing.assert_frame_equal(tgt[FEATURE_COLUMNS], tgt2[FEATURE_COLUMNS])

    # ... but the exit open of the window does (the target really is forward-looking)
    moved = {s: b.copy() for s, b in bars.items()}
    moved["S0"].loc[t + 1 + mm.HORIZON, "open"] *= 1.5
    r0 = mm.symbol_daily("S0", moved["S0"])
    assert r0["fwd_ret"].iloc[t] == pytest.approx((1 + r["S0"]) * 1.5 - 1)


def test_month_folds_purge_and_embargo():
    months = pd.Series(np.repeat(pd.period_range("2000-01", "2009-12", freq="M"), 3))
    folds = list(mm.month_folds(months, initial=60, test=12, purge=1, embargo=1))
    assert len(folds) == 5
    for tr, te, info in folds:
        last_train, first_test = months.iloc[tr].max(), months.iloc[te].min()
        assert (first_test - last_train).n == 3  # two decision months dropped (purge 1 + embargo 1)
        assert not set(tr) & set(te)
        assert months.iloc[te].nunique() == 12
        assert info["test_first_month"] == str(first_test)
    assert str(months.iloc[folds[0][1]].min()) == "2005-01"


def test_imputer_and_scaler_fit_on_training_fold_only():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    X[:200, 2] = np.where(rng.random(200) < 0.3, np.nan, X[:200, 2] + 10.0)  # train regime
    X[200:, 2] = np.where(rng.random(200) < 0.3, np.nan, X[200:, 2] - 50.0)  # test regime
    y = (rng.random(n) > 0.5).astype(float)
    df = pd.DataFrame(X, columns=["a", "b", "c"]).assign(target=y)
    seen: list[np.ndarray] = []

    def factory(name: str):
        model = mm.make_model(name)
        orig = model.fit

        def fit(Xf, yf):
            seen.append(Xf.copy())
            return orig(Xf, yf)

        model.fit = fit  # type: ignore[method-assign]
        factory.last = model  # type: ignore[attr-defined]
        return model

    tr, te = np.arange(0, 200), np.arange(200, 400)
    mm.walk_forward_set(df, ["a", "b", "c"], "logit", [(tr, te, {"fold": 0})], factory=factory)
    np.testing.assert_allclose(seen[0], X[:200])  # model only ever sees training rows
    imp = factory.last.named_steps["impute"]  # type: ignore[attr-defined]
    assert imp.statistics_[2] == pytest.approx(np.nanmedian(X[:200, 2]))
    assert imp.statistics_[2] > 5  # not the test-period median (~ -50)
    assert imp.indicator_ is not None and list(imp.indicator_.features_) == [2]


def test_feature_set_column_selection():
    macro = ["m_VIX", "m_TNX"]
    assert mm.feature_columns("P", macro) == FEATURE_COLUMNS
    assert mm.feature_columns("P+F", macro) == FEATURE_COLUMNS + mm.FUND
    assert mm.feature_columns("P+M", macro) == FEATURE_COLUMNS + macro
    assert mm.feature_columns("P+F+M", macro) == FEATURE_COLUMNS + mm.FUND + macro
    with pytest.raises(ValueError):
        mm.feature_columns("F+M", macro)


def _facts_doc() -> dict:
    def f(start, end, val, filed, form="10-Q", fp="Q1", fy=2019):
        d = {
            "end": end,
            "val": val,
            "accn": f"a-{form}-{filed}",
            "fy": fy,
            "fp": fp,
            "form": form,
            "filed": filed,
        }
        if start:
            d["start"] = start
        return d

    ni = [
        f("2019-01-01", "2019-03-31", 100, "2019-05-01"),
        f("2019-04-01", "2019-06-30", 110, "2019-08-01", fp="Q2"),
        f("2019-07-01", "2019-09-30", 120, "2019-11-01", fp="Q3"),
        f("2019-01-01", "2019-12-31", 460, "2020-02-05", form="10-K", fp="FY"),
        f("2020-01-01", "2020-03-31", 140, "2020-05-01", fy=2020),
    ]
    return {
        "cik": 1,
        "entityName": "Synthetic",
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [f(None, "2020-01-31", 100, "2020-02-05", form="10-K", fp="FY")]}
                }
            },
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": ni}},
                "Revenues": {"units": {"USD": [{**x, "val": x["val"] * 10} for x in ni]}},
                "StockholdersEquity": {
                    "units": {"USD": [f(None, "2019-12-31", 2000, "2020-02-05", form="10-K", fp="FY")]}
                },
            },
        },
    }


def test_no_fundamentals_before_filed():
    facts = sec.normalize_companyfacts(_facts_doc(), "SYN")
    bars = {"SYN": make_bars("US", n=420, seed=1, start="2019-01-02")}
    df, _ = mm.build_monthly_frame(bars, facts=facts, with_macro=False, min_history=0)
    has = df.dropna(subset=["net_margin"])
    assert len(has) > 0
    assert (df["session_date"] > pd.to_datetime(df["fund_filed"])).where(df["fund_filed"].notna(), True).all()
    # FY2019 10-K filed 2020-02-05: the 2020-01-31 month-end row must not see the TTM it introduced
    jan = df.loc[df["session_date"] == pd.Timestamp("2020-01-31")]
    assert jan[mm.FUND].isna().all(axis=None)
    assert np.isfinite(df[mm.FUND].to_numpy(dtype=float)[~np.isnan(df[mm.FUND].to_numpy(dtype=float))]).all()


def test_leakage_canary_future_feature_is_flagged():
    rng = np.random.default_rng(3)
    n = 600
    y = (rng.random(n) > 0.5).astype(float)
    df = pd.DataFrame({"noise": rng.normal(size=n), "leak": y + rng.normal(0, 0.05, n), "target": y})
    folds = [(np.arange(0, 300), np.arange(300, 600), {"fold": 0})]
    p, _, _, _ = mm.walk_forward_set(df, ["noise", "leak"], "logit", folds)
    assert me._safe_auc(y[300:], p[300:]) > 0.99  # the pipeline would flag a leaked feature


# ------------------------------------------------------------------------------ H-ML-003/004 helpers


def test_calendar_folds_gap_and_blocks():
    dates = pd.Series(np.repeat(pd.date_range("2000-01-03", "2004-12-31", freq="B"), 2))
    folds = list(
        me.calendar_folds(dates, pd.Timedelta(days=730), pd.Timedelta(days=91), pd.Timedelta(days=7))
    )
    first = dates.min()
    assert folds[0].test_start >= first + pd.Timedelta(days=730)
    prev_end = None
    for f in folds:
        tr, te = dates.iloc[f.train], dates.iloc[f.test]
        assert (te.min() - tr.max()).days > 7
        assert (te.max() - te.min()).days < 91
        assert dates.iloc[f.test].value_counts().eq(2).all()  # rows of a date never split
        if prev_end is not None:
            assert te.min() > prev_end
        prev_end = te.max()


def test_eligible_frame_entry_rule_and_dtypes(us_bars):
    f = build_features(make_bars("US", n=400))
    f["symbol"] = "TEST"
    df = me.eligible_frame(f, seq_len=10)
    assert pd.to_datetime(df["session_date"]).min() >= pd.to_datetime(f["session_date"]).iloc[me.MIN_HISTORY]
    assert not df.isna().any().any()
    assert df[f"{FEATURE_COLUMNS[0]}_lag0"].dtype == np.float32
    np.testing.assert_allclose(
        df[f"{FEATURE_COLUMNS[0]}_lag0"].to_numpy(), df[FEATURE_COLUMNS[0]].to_numpy(), rtol=1e-6
    )


def test_resource_probe_reports_memory():
    with me.ResourceProbe() as p:
        a = np.ones(5_000_000)
        a.sum()
    r = p.result
    assert r["peak_tracemalloc_mb"] >= 39
    assert r["peak_working_set_mb_sampled"] > 0 and "gpu" in r


def test_hourly_backtest_costs_and_turnover():
    t = pd.date_range("2024-01-01", periods=4, freq="h")
    preds = pd.DataFrame(
        {"close_time_utc": t, "symbol": "X", "p_logit": [0.6, 0.6, 0.4, 0.6], "target_ret_oo": np.log(1.01)}
    )
    bt = me.hourly_backtest(preds, "logit", round_trip_bps=26.0)
    # entries at bars 0 and 3, exit at bar 2: 3 position changes x 13 bps
    exp = (3 * 0.01 - 3 * 13e-4) / 4
    assert bt["mean_bar_return"] == pytest.approx(exp)


def test_walk_forward_checkpoint_resume_reuses_folds(tmp_path):
    rng = np.random.default_rng(5)
    dates = pd.Series(np.repeat(pd.date_range("2000-01-03", "2003-12-31", freq="B"), 2))
    X = rng.normal(size=(len(dates), 3))
    y = (rng.random(len(dates)) > 0.5).astype(float)
    folds = list(
        me.calendar_folds(dates, pd.Timedelta(days=730), pd.Timedelta(days=91), pd.Timedelta(days=7))
    )
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return me.make_factory("logit", None, 0)()

    p1, b1, _, _ = me.walk_forward("logit", X, y, folds, factory, checkpoint=tmp_path)
    assert calls["n"] == len(folds)
    p2, b2, pf2, _ = me.walk_forward("logit", X, y, folds, factory, checkpoint=tmp_path)
    assert calls["n"] == len(folds)  # nothing refit on resume
    np.testing.assert_allclose(p1, p2, equal_nan=True)
    np.testing.assert_allclose(b1, b2, equal_nan=True)
    assert pf2["resumed"].all()


def test_calendar_folds_tz_aware_hourly_gap():
    t = pd.Series(pd.date_range("2020-01-01", periods=24 * 900, freq="h", tz="UTC"))
    folds = list(me.calendar_folds(t, pd.Timedelta(days=730), pd.Timedelta(days=91), pd.Timedelta(hours=48)))
    assert folds
    for f in folds:
        assert t.iloc[f.test].min() - t.iloc[f.train].max() > pd.Timedelta(hours=48)
