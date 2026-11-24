"""NEBULA research dashboard (Streamlit).

Reads ONLY stored outputs (validated data, quality reports, experiment records/artifacts, logs, reports).
Nothing is computed for display purposes beyond plotting, and nothing is hard-coded. Missing outputs are
shown as missing.

Run: uv run nebula dashboard   (binds 127.0.0.1)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nebula.common.config import get_config
from nebula.common.paths import get_paths
from nebula.data.storage import read_bars, read_manifest
from nebula.tracking.tracker import ExperimentRecord, Tracker, record_subject

st.set_page_config(page_title="NEBULA", layout="wide")
PATHS = get_paths()
CFG = get_config()
TR = Tracker(PATHS)

PAGES = [
    "Overview",
    "Data availability & quality",
    "Market overview",
    "Research experiments",
    "Backtests",
    "Machine learning",
    "Volatility forecasting",
    "Market regimes",
    "Portfolio & risk",
    "Market microstructure",
    "Research report & agents",
    "System logs",
]
page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption(
    "All numbers are read from data/, experiments/ and logs/. Missing items are shown as missing."
)


# ---------------------------------------------------------------------------------------------- helpers
@st.cache_data(ttl=60)
def index() -> pd.DataFrame:
    df = TR.list()
    if not df.empty:
        df["subject"] = [_subject(e) for e in df["experiment_id"]]
    return df


def _subject(eid: str) -> str:
    try:
        rec = TR.load(eid)
    except FileNotFoundError:
        return ""
    return record_subject(rec)


def latest(hypothesis: str) -> pd.DataFrame:
    df = index()
    if df.empty:
        return df
    df = df[df["hypothesis_id"] == hypothesis]
    return df.drop_duplicates(subset=["name", "subject"], keep="last")


def artifact(eid: str, name: str) -> pd.DataFrame | None:
    p = PATHS.experiments / eid / "artifacts" / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def need(df: pd.DataFrame | None, what: str) -> bool:
    if df is None or df.empty:
        st.info(f"No stored results for {what} yet.")
        return False
    return True


def record(eid: str) -> ExperimentRecord:
    return TR.load(eid)


def conclusion_badge(c: str) -> str:
    return {"supported": "SUPPORTED", "rejected": "REJECTED", "inconclusive": "INCONCLUSIVE"}.get(c, c)


# ---------------------------------------------------------------------------------------------- pages
if page == "Overview":
    st.header("NEBULA - research status")
    df = index()
    if not need(df, "experiments"):
        st.stop()
    last = df.drop_duplicates(subset=["hypothesis_id", "name", "subject"], keep="last")
    piv = last.pivot_table(
        index="hypothesis_id", columns="conclusion", values="experiment_id", aggfunc="count", fill_value=0
    )
    st.subheader("Latest conclusions per hypothesis (count of experiments)")
    st.dataframe(piv, width="stretch")
    st.caption(
        "Negative and inconclusive results are retained by design. Registered hypotheses: "
        "docs/RESEARCH_HYPOTHESES.md and docs/research/*.md."
    )

elif page == "Data availability & quality":
    st.header("Data availability & quality")
    rows = []
    for inst in CFG.instruments:
        try:
            m = read_manifest(inst.symbol, inst.market, PATHS)
            rows.append({"symbol": inst.symbol, "market": inst.market, "provider": inst.provider, **m})
        except FileNotFoundError:
            rows.append(
                {
                    "symbol": inst.symbol,
                    "market": inst.market,
                    "provider": inst.provider,
                    "quality_status": "NOT INGESTED",
                }
            )
    man = pd.DataFrame(rows)
    cols = [
        "symbol",
        "market",
        "provider",
        "rows",
        "first_session",
        "last_session",
        "quality_status",
        "written_at_utc",
    ]
    st.dataframe(man[[c for c in cols if c in man]], width="stretch")
    for _, r in man.iterrows():
        if isinstance(r.get("quality_report"), str):
            rep = json.loads((PATHS.root / r["quality_report"]).read_text(encoding="utf-8"))
            with st.expander(f"{r['symbol']} - {rep['created_at_utc'][:19]} - {r['quality_status']}"):
                st.dataframe(pd.DataFrame(rep["checks"]), width="stretch")
                if r.get("transform_notes"):
                    st.write("Transform notes:", r["transform_notes"])

elif page == "Market overview":
    st.header("Market overview")
    sym = st.selectbox("Instrument", [i.symbol for i in CFG.instruments])
    inst = CFG.instrument(sym)
    try:
        bars = read_bars(sym, inst.market, PATHS)
    except FileNotFoundError as e:
        st.warning(str(e))
        st.stop()
    bars["session_date"] = pd.to_datetime(bars["session_date"])
    fig = go.Figure(
        go.Candlestick(
            x=bars["session_date"],
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name=sym,
        )
    )
    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        yaxis_title=f"{sym} ({CFG.markets[inst.market].currency}, provider prices)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Yahoo OHLC are split-adjusted, not dividend-adjusted; research uses total-return adjustment.")

elif page == "Research experiments":
    st.header("Research experiments")
    df = index()
    if not need(df, "experiments"):
        st.stop()
    hyps = sorted(df["hypothesis_id"].unique())
    pick = st.multiselect("Hypotheses", hyps, default=hyps)
    only_latest = st.checkbox("Latest run per experiment/subject only", value=True)
    view = df[df["hypothesis_id"].isin(pick)]
    if only_latest:
        view = view.drop_duplicates(subset=["hypothesis_id", "name", "subject"], keep="last")
    st.dataframe(
        view[["experiment_id", "hypothesis_id", "name", "subject", "conclusion", "created_at_utc"]],
        width="stretch",
    )
    eid = st.selectbox("Inspect experiment", view["experiment_id"].tolist()[::-1])
    if eid:
        rec = record(str(eid))
        st.subheader(f"{rec.hypothesis_id}: {conclusion_badge(rec.conclusion)}")
        st.write(rec.hypothesis)
        st.write("Reproduce:", f"`{rec.reproduce}`" if rec.reproduce else "-")
        st.json(rec.model_dump(), expanded=False)

elif page == "Backtests":
    st.header("Single-instrument backtests (net of modelled costs)")
    df = index()
    if not need(df, "experiments"):
        st.stop()
    bt = df[
        df["experiment_id"].map(lambda e: (PATHS.experiments / e / "artifacts" / "equity.parquet").exists())
    ]
    bt = bt[bt["hypothesis_id"].str.startswith("H-XM")]
    bt = bt.drop_duplicates(subset=["hypothesis_id", "name", "subject"], keep="last")
    labels = {
        f"{r.hypothesis_id} {r.name} {r.subject} [{r.conclusion}]": r.experiment_id for r in bt.itertuples()
    }
    if not need(bt, "backtests"):
        st.stop()
    eid = labels[st.selectbox("Backtest", list(labels))]
    rec = record(eid)
    eq = artifact(eid, "equity")
    assert eq is not None
    fig = go.Figure()
    fig.add_scatter(x=eq.index, y=eq["equity_net"], name="strategy net")
    if "equity_gross" in eq:
        fig.add_scatter(
            x=eq.index, y=eq["equity_gross"], name="strategy gross (zero-cost run)", line={"dash": "dot"}
        )
    if "bh_equity_net" in eq:
        fig.add_scatter(x=eq.index, y=eq["bh_equity_net"], name="buy & hold net")
    fig.update_layout(height=450, yaxis_title="equity")
    st.plotly_chart(fig, width="stretch")
    m = rec.metrics
    c = st.columns(4)
    c[0].metric("Net Sharpe", f"{m.get('net_sharpe', m.get('tsmom_sharpe', float('nan'))):.2f}")
    c[1].metric("Buy&hold Sharpe", f"{m.get('bh_sharpe', float('nan')):.2f}")
    c[2].metric("Test statistic", f"{m.get('nw_t_diff', m.get('delta_sr', float('nan'))):.2f}")
    c[3].metric("Conclusion", rec.conclusion)
    sens = artifact(eid, "cost_sensitivity")
    if sens is not None:
        st.subheader("Transaction-cost sensitivity")
        st.dataframe(sens, width="stretch")

elif page == "Machine learning":
    st.header("Machine learning & deep learning")
    g = latest("H-ML-001")
    if need(g, "H-ML-001 (model x instrument grid)"):
        eid = g["experiment_id"].iloc[-1]
        rec = record(eid)
        st.subheader(f"H-ML-001: {conclusion_badge(rec.conclusion)}")
        st.write({k: v for k, v in rec.metrics.items() if k != "gpu"})
        grid = artifact(eid, "grid")
        if grid is not None:
            st.dataframe(grid, width="stretch")
            heat = grid.pivot_table(index="symbol", columns="model", values="dm_t")
            st.plotly_chart(
                go.Figure(
                    go.Heatmap(
                        z=heat.values,
                        x=heat.columns,
                        y=heat.index,
                        colorscale="RdBu",
                        zmid=0,
                        colorbar={"title": "DM t"},
                    )
                ),
                width="stretch",
            )
            st.caption(
                "DM t > 0: model beats its ex-ante (train-fold) base rate on OOS log loss. "
                "Significance is judged after Benjamini-Hochberg across all cells."
            )
    pan = latest("H-ML-002")
    if need(pan, "H-ML-002 (pooled panel DL)"):
        rows = []
        for e in pan["experiment_id"]:
            r = record(e)
            rows.append(
                {
                    "experiment": r.name,
                    "conclusion": r.conclusion,
                    **{k: v for k, v in r.metrics.items() if isinstance(v, int | float)},
                }
            )
        st.subheader("H-ML-002 pooled panel")
        st.dataframe(pd.DataFrame(rows), width="stretch")

elif page == "Volatility forecasting":
    st.header("Volatility forecasting (H-VOL-001)")
    v = latest("H-VOL-001")
    if not need(v, "H-VOL-001"):
        st.stop()
    st.dataframe(v[["subject", "conclusion", "experiment_id"]], width="stretch")
    sym = st.selectbox("Instrument", v["subject"].tolist())
    eid = v.loc[v["subject"] == sym, "experiment_id"].iloc[-1]
    fc = artifact(eid, "forecasts")
    rec = record(eid)
    st.write(f"Conclusion: {conclusion_badge(rec.conclusion)}")
    if fc is not None:
        fc = fc[fc["oos"]] if "oos" in fc else fc
        fig = go.Figure()
        x = pd.to_datetime(fc["target_date"])
        fig.add_scatter(x=x, y=fc["rv_cc_next"] ** 0.5, name="realized |r| proxy", opacity=0.3)
        for col in ["ewma", "gjr"]:
            fig.add_scatter(x=x, y=fc[col] ** 0.5, name=f"{col} forecast")
        fig.update_layout(height=450, yaxis_title="daily volatility")
        st.plotly_chart(fig, width="stretch")
    st.json({k: v for k, v in rec.metrics.items()}, expanded=False)

elif page == "Market regimes":
    st.header("Market regimes (H-REG-001, walk-forward Gaussian mixture)")
    rg = latest("H-REG-001")
    if not need(rg, "H-REG-001"):
        st.stop()
    st.dataframe(rg[["subject", "conclusion", "experiment_id"]], width="stretch")
    sym = st.selectbox("Instrument", rg["subject"].tolist())
    eid = rg.loc[rg["subject"] == sym, "experiment_id"].iloc[-1]
    s = artifact(eid, "states")
    if s is not None:
        s = s.dropna(subset=["state"])
        fig = go.Figure()
        x = pd.to_datetime(s["origin_date"])
        fig.add_scatter(x=x, y=s["vol20"], name="20d vol", line={"color": "gray"})
        hi = s["state"] == s["state"].max()
        fig.add_scatter(
            x=x[hi],
            y=s.loc[hi, "vol20"],
            mode="markers",
            name="OOS high-vol state",
            marker={"size": 3, "color": "crimson"},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "States are out-of-sample (model refit on expanding windows). Visual regimes are not "
            "evidence of economic meaning; see the record's statistical tests."
        )
    st.json(record(eid).metrics, expanded=False)

elif page == "Portfolio & risk":
    st.header("Portfolio construction & risk (per market, local currency)")
    pf = latest("H-PF-001")
    if not need(pf, "H-PF-001"):
        st.stop()
    labels = {r.subject: r.experiment_id for r in pf.itertuples()}
    mkt = st.selectbox("Market", list(labels))
    eid = labels[mkt]
    perf = artifact(eid, "performance")
    if perf is not None:
        st.subheader("Performance (net)")
        st.dataframe(perf, width="stretch")
    eq = artifact(eid, "equity")
    if eq is not None:
        fig = go.Figure()
        for c in [c for c in eq.columns if c.endswith("_equity_net")]:
            fig.add_scatter(x=eq.index, y=eq[c], name=c.replace("_equity_net", ""))
        fig.update_layout(height=420, yaxis_title="equity (net)")
        st.plotly_chart(fig, width="stretch")
    for name, title in [
        ("risk_contributions_last", "Risk contributions (last rebalance)"),
        ("var_es", "VaR / Expected Shortfall"),
        ("cost_sensitivity", "Cost sensitivity"),
    ]:
        a = artifact(eid, name)
        if a is not None:
            st.subheader(title)
            st.dataframe(a, width="stretch")
    h2 = latest("H-PF-002")
    st.subheader("Verdicts")
    st.dataframe(
        pd.concat([pf, h2])[["hypothesis_id", "subject", "conclusion", "experiment_id"]],
        width="stretch",
    )

elif page == "Market microstructure":
    st.header("Market microstructure (Binance L2 + aggTrades)")
    st.caption("Level 2 only (no L3); depth data is prospective capture, not historical.")
    a = latest("H-MS-000")
    if need(a, "aggTrades study (H-MS-000)"):
        eid = a["experiment_id"].iloc[-1]
        st.subheader(f"aggTrades study: {conclusion_badge(record(eid).conclusion)}")
        b = artifact(eid, "bars_1m")
        if b is not None:
            fig = go.Figure()
            fig.add_scatter(x=pd.to_datetime(b["ts"]), y=b["vwap"], name="1m VWAP")
            fig.add_bar(
                x=pd.to_datetime(b["ts"]), y=b["signed_volume"], name="signed volume", yaxis="y2", opacity=0.4
            )
            fig.update_layout(height=420, yaxis2={"overlaying": "y", "side": "right"})
            st.plotly_chart(fig, width="stretch")
        k = artifact(eid, "kyle_lambda")
        if k is not None:
            st.dataframe(k, width="stretch")
    c = latest("H-MS-001")
    if need(c, "depth capture study (H-MS-001)"):
        eid = c["experiment_id"].iloc[-1]
        st.subheader(f"H-MS-001: {conclusion_badge(record(eid).conclusion)}")
        r = artifact(eid, "h_ms_001_results")
        if r is not None:
            st.dataframe(r, width="stretch")
        bk = artifact(eid, "book_1s_summary")
        if bk is not None:
            fig = go.Figure()
            fig.add_scatter(x=bk["sec"], y=bk["mid"], name="mid")
            fig.add_scatter(x=bk["sec"], y=bk["imbalance_5"], name="top-5 imbalance", yaxis="y2")
            fig.update_layout(height=380, yaxis2={"overlaying": "y", "side": "right"})
            st.plotly_chart(fig, width="stretch")
        st.json(record(eid).metrics, expanded=False)

elif page == "Research report & agents":
    st.header("Research report (deterministic agents)")
    rep = PATHS.reports / "research" / "latest.md"
    if rep.exists():
        st.markdown(rep.read_text(encoding="utf-8"))
    else:
        st.info("No report yet - run `uv run nebula report`.")
    act = PATHS.logs / "agent_activity.jsonl"
    st.subheader("Agent activity")
    if act.exists():
        rows = [json.loads(x) for x in act.read_text(encoding="utf-8").splitlines() if x.strip()]
        st.dataframe(pd.DataFrame(rows).tail(50), width="stretch")
    else:
        st.info("No agent activity logged yet.")

elif page == "System logs":
    st.header("System logs (last 200 lines)")
    p: Path = PATHS.logs / "nebula.jsonl"
    if p.exists():
        st.code("\n".join(p.read_text(encoding="utf-8").splitlines()[-200:]))
    else:
        st.info("No log file yet.")
