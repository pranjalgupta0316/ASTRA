"""NEBULA research dashboard (Streamlit). Reads ONLY stored outputs: validated data, quality reports,
experiment records. Nothing here is computed for display-only purposes or hard-coded.

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
from nebula.tracking.tracker import Tracker

st.set_page_config(page_title="NEBULA", layout="wide")
PATHS = get_paths()
CFG = get_config()

PAGES = ["Data availability & quality", "Market overview", "Research experiments", "Backtests", "System logs"]
page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption("All numbers are read from data/ and experiments/. Missing items are shown as missing.")


def _manifests() -> pd.DataFrame:
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
                    "rows": None,
                    "quality_status": "NOT INGESTED",
                }
            )
    return pd.DataFrame(rows)


if page == PAGES[0]:
    st.header("Data availability & quality")
    man = _manifests()
    st.dataframe(
        man[
            [
                c
                for c in [
                    "symbol",
                    "market",
                    "provider",
                    "rows",
                    "first_session",
                    "last_session",
                    "quality_status",
                    "sha256",
                ]
                if c in man
            ]
        ],
        use_container_width=True,
    )
    for _, r in man.iterrows():
        if isinstance(r.get("quality_report"), str):
            rep = json.loads((PATHS.root / r["quality_report"]).read_text(encoding="utf-8"))
            with st.expander(f"{r['symbol']} quality report ({rep['created_at_utc'][:19]})"):
                st.dataframe(pd.DataFrame(rep["checks"]), use_container_width=True)

elif page == PAGES[1]:
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
        yaxis_title=f"{inst.symbol} ({CFG.markets[inst.market].currency}, provider prices)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Yahoo OHLC are split-adjusted, not dividend-adjusted; research uses total-return adjustment.")
    st.dataframe(bars.tail(10), use_container_width=True)

elif page == PAGES[2]:
    st.header("Research experiments")
    tr = Tracker(PATHS)
    idx = tr.list()
    if idx.empty:
        st.info("No experiments recorded yet. Run `uv run nebula research`.")
        st.stop()
    st.dataframe(
        idx[["experiment_id", "hypothesis_id", "name", "conclusion", "created_at_utc"]],
        use_container_width=True,
    )
    eid = st.selectbox("Experiment", idx["experiment_id"].tolist()[::-1])
    if eid is None:
        st.stop()
    rec = tr.load(str(eid))
    st.subheader(f"{rec.hypothesis_id}: {rec.conclusion.upper()}")
    st.write(rec.hypothesis)
    st.json(rec.model_dump(exclude={"datasets"}), expanded=False)

elif page == PAGES[3]:
    st.header("Backtests (net of modelled costs)")
    tr = Tracker(PATHS)
    idx = tr.list()
    if idx.empty:
        st.info("No experiments recorded yet.")
        st.stop()
    bt_ids = [
        e for e in idx["experiment_id"] if (PATHS.experiments / e / "artifacts" / "equity.parquet").exists()
    ]
    eid = st.selectbox("Backtest experiment", bt_ids[::-1])
    if eid is None:
        st.info("No experiment has a stored equity curve yet.")
        st.stop()
    eid = str(eid)
    rec = tr.load(eid)
    eq = pd.read_parquet(PATHS.experiments / eid / "artifacts" / "equity.parquet")
    fig = go.Figure()
    fig.add_scatter(x=eq.index, y=eq["equity_net"], name="strategy net")
    fig.add_scatter(x=eq.index, y=eq["equity_gross"], name="strategy gross", line={"dash": "dot"})
    fig.add_scatter(x=eq.index, y=eq["bh_equity_net"], name="buy & hold net")
    fig.update_layout(height=450, yaxis_title="equity")
    st.plotly_chart(fig, use_container_width=True)
    m = rec.metrics
    cols = st.columns(4)
    cols[0].metric("Net Sharpe", f"{m.get('net_sharpe', float('nan')):.2f}")
    cols[1].metric("Buy&hold Sharpe", f"{m.get('bh_sharpe', float('nan')):.2f}")
    cols[2].metric("NW t (diff)", f"{m.get('nw_t_diff', float('nan')):.2f}")
    cols[3].metric("Conclusion", rec.conclusion)
    sens = PATHS.experiments / eid / "artifacts" / "cost_sensitivity.parquet"
    if sens.exists():
        st.subheader("Transaction-cost sensitivity")
        st.dataframe(pd.read_parquet(sens), use_container_width=True)

elif page == PAGES[4]:
    st.header("System logs (last 200 lines)")
    p: Path = PATHS.logs / "nebula.jsonl"
    if p.exists():
        lines = p.read_text(encoding="utf-8").splitlines()[-200:]
        st.code("\n".join(lines))
    else:
        st.info("No log file yet.")
