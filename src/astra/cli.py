"""ASTRA command line: `uv run astra --help`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

from astra.common.config import get_config
from astra.common.logging import configure_logging

app = typer.Typer(add_completion=False, help="ASTRA quantitative research platform")
lock_app = typer.Typer(add_completion=False, help="Pin validated data to immutable snapshots (data locks)")
app.add_typer(lock_app, name="data-lock")


@app.callback()
def _main(
    log_level: str = "WARNING",
    lock: str | None = typer.Option(
        None, help="run against a data lock (pinned snapshot); default: ASTRA_DATA_LOCK or live data"
    ),
) -> None:
    configure_logging(log_level)
    if lock:
        import os

        from astra.data.lock import load_lock

        load_lock(lock)  # fail fast on an unknown lock
        os.environ["ASTRA_DATA_LOCK"] = lock  # inherited by run-all's subprocesses


@lock_app.command("create")
def lock_create(name: str, note: str = "") -> None:
    """Snapshot every validated file and write config/data_locks/NAME.json (immutable)."""
    from astra.data.lock import create_lock

    typer.echo(json.dumps(create_lock(name, note), indent=2))


@lock_app.command("list")
def lock_list() -> None:
    """List data locks."""
    from astra.data.lock import list_locks

    typer.echo(json.dumps(list_locks(), indent=2))


@lock_app.command("verify")
def lock_verify(name: str) -> None:
    """Re-hash every snapshot of a lock; exit 1 if any is missing or corrupted."""
    from astra.data.lock import verify_lock

    res = verify_lock(name)
    typer.echo(json.dumps(res, indent=2))
    raise typer.Exit(code=0 if res["ok"] else 1)


@app.command()
def env() -> None:
    """Print the hardware / library environment report."""
    from astra.common.env_report import environment_report

    typer.echo(json.dumps(environment_report(), indent=2, default=str))


@app.command()
def ingest(
    symbols: list[str] = typer.Argument(None),
    start: str | None = None,
    end: str | None = None,
    tier: str | None = typer.Option(None, help="core | extended (default: all)"),
    intraday: bool = typer.Option(False, help="ingest each instrument's configured intraday frequencies"),
) -> None:
    """Fetch, validate and store bars (default: daily bars for all configured instruments)."""
    from astra.data.ingest import ingest_symbol

    cfg = get_config()
    syms = symbols or cfg.symbols(tier=tier)
    failed: list[str] = []
    for s in syms:
        freqs = cfg.instrument(s).intraday if intraday else ["1d"]
        for f in freqs:
            try:
                res = ingest_symbol(s, start=start, end=end, freq=f)
            except Exception as exc:  # keep going; report every failure at the end
                typer.echo(f"{s} [{f}] FAILED: {exc!r}")
                failed.append(f"{s}:{f}")
                continue
            typer.echo(f"[{f}] " + res.report.summary())
            if res.validated_file is None:
                failed.append(f"{s}:{f}")
    if failed:
        typer.echo(f"failed: {failed}")
    raise typer.Exit(code=1 if failed else 0)


@app.command("ingest-fundamentals")
def ingest_fundamentals() -> None:
    """SEC EDGAR companyfacts for US stocks (point-in-time; generic User-Agent unless SEC_USER_AGENT set)."""
    from astra.data.fundamentals import ingest_sec

    typer.echo(json.dumps(ingest_sec(), indent=2, default=str)[:20000])


@app.command("ingest-macro")
def ingest_macro() -> None:
    """Market macro (Yahoo: yields, VIX, DXY, USD/INR) + BLS CPI/unemployment. FRED is manual-CSV only."""
    from astra.data.macro import ingest_bls, ingest_market_macro

    for res in (ingest_market_macro(), ingest_bls()):
        typer.echo(json.dumps(res, indent=2, default=str)[:20000])


@app.command()
def research(symbols: list[str] = typer.Argument(None)) -> None:
    """Run the vertical-slice research (H-XM-000/001/002) for each symbol."""
    from astra.research.vertical_slice import run_symbol

    syms = symbols or get_config().symbols(tier="core")
    for s in syms:
        typer.echo(json.dumps(run_symbol(s), indent=2, default=str))


@app.command()
def report(
    llm: bool = typer.Option(False, help="append local-LLM narration if ASTRA_LLM_URL is set"),
) -> None:
    """Run the deterministic research agents and write data/reports/research/latest.md."""
    from astra.agents.orchestrator import write_report

    typer.echo(str(write_report(use_llm=llm)))


@app.command()
def holdout() -> None:
    """Run the confirmatory H-XM-003 Sharpe-difference test on held-out instruments."""
    from astra.research.holdout_study import run_holdout_tsmom

    res = run_holdout_tsmom()
    t = res.pop("table")
    typer.echo(t.to_string(index=False))
    typer.echo(json.dumps(res, indent=2, default=str))


@app.command("vol-study")
def vol_study() -> None:
    """H-VOL-001 volatility forecasts and H-REG-001 regimes on all instruments."""
    from astra.research.regime_study import run_regime_study
    from astra.research.volatility_study import run_volatility_study

    for res in (run_volatility_study(), run_regime_study()):
        t = res.pop("table", None)
        if t is not None:
            typer.echo(t.to_string(index=False))
        typer.echo(json.dumps(res.get("counts", {}), indent=2, default=str))


@app.command("rv-study")
def rv_study() -> None:
    """H-RV-001/002/003 realized-volatility study (1m BTC/ETH, 1h crypto pairs). Run under a data lock."""
    import pandas as pd

    from astra.research.realized_vol import run_rv_study

    res = run_rv_study()
    cols = [
        "symbol",
        "cell",
        "n",
        "qlike_model",
        "qlike_bench",
        "dm_t_qlike",
        "p_bh",
        "dm_t_mse",
        "conclusion",
    ]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        for hid in ("H-RV-001", "H-RV-002", "H-RV-003"):
            r = res[hid]
            typer.echo(f"== {hid}: family {r['verdict']} ({r['family_id']})")
            typer.echo(r["table"][cols].to_string(index=False))


@app.command("universe")
def universe(
    fetch: bool = typer.Option(False, help="re-download the two Wikipedia pages (2 API requests)"),
    dates: str = typer.Option("2011-01-03,2015-01-02,2018-01-02,2022-01-03", help="comma-separated dates"),
) -> None:
    """P8: point-in-time S&P 500 membership (Wikipedia, CC BY-SA) and a survivorship check of the US
    universe. Writes data/validated/universe/sp500_membership.parquet + manifest."""
    from astra.data.universe import build, fetch_pages, survivorship_report

    man = build(files=fetch_pages() if fetch else None)
    typer.echo(
        json.dumps({k: v for k, v in man.items() if k not in ("sources", "changes_per_year")}, indent=1)
    )
    typer.echo(survivorship_report([d.strip() for d in dates.split(",")]).to_string(index=False))


@app.command("multiplicity")
def multiplicity() -> None:
    """P7: global FDR over all current records, deflated Sharpe of every strategy curve, calibration of
    stored OOS probabilities. Re-reads records only; writes data/reports/research/multiplicity.*"""
    from astra.research.multiplicity import write_report

    typer.echo(json.dumps(write_report(), indent=2))


@app.command("records")
def records(all_records: bool = typer.Option(False, "--all", help="include superseded records")) -> None:
    """List experiment records with the computed superseded flag (records are never modified)."""
    from astra.tracking.tracker import Tracker

    st = Tracker().status()
    if not all_records and len(st):
        st = st.loc[~st["superseded"]]
    typer.echo(st.drop(columns=["superseded_by"], errors="ignore").to_string(index=False))


@app.command("portfolio-study")
def portfolio_study(markets: list[str] = typer.Argument(None)) -> None:
    """H-PF-001/002 portfolio construction study per market."""
    from astra.research.portfolio_study import run_portfolio_study

    typer.echo(json.dumps(run_portfolio_study(markets or None), indent=2, default=str)[:20000])


@app.command("micro-study")
def micro_study(
    capture_seconds: int = typer.Option(0, help="also run a live depth capture (seconds)"),
) -> None:
    """Microstructure: aggTrades study (H-MS-000); optional live capture + H-MS-001."""
    from astra.microstructure.capture import run_capture
    from astra.research.microstructure_study import run_aggtrades_study, run_capture_study

    typer.echo(run_aggtrades_study().experiment_id)
    if capture_seconds > 0:
        rep = run_capture("BTCUSDT", float(capture_seconds))
        typer.echo(str(rep))
        typer.echo(run_capture_study().experiment_id)


@app.command("ext-study")
def ext_study() -> None:
    """Extended-universe studies: cross-sectional (H-CS), vol-managed (H-VT), cross-market (H-XMKT),
    multimodal (H-MM-001). ~10 min."""
    from astra.research.cross_market import run_cross_market
    from astra.research.cross_sectional import run_cross_sectional
    from astra.research.multimodal import run_multimodal
    from astra.research.vol_managed import run_vol_managed

    for name, fn in (
        ("H-CS", run_cross_sectional),
        ("H-VT", run_vol_managed),
        ("H-XMKT", run_cross_market),
        ("H-MM", run_multimodal),
    ):
        res = fn()
        typer.echo(
            f"{name}: " + json.dumps({k: v for k, v in res.items() if isinstance(v, str | int | float)})
        )


@app.command("ml-ext")
def ml_ext(
    models: str = typer.Option(
        "logit,xgb,mlp,gru", help="comma-separated; MLP+GRU take ~2.5 h on an RTX 4060"
    ),
    hourly: bool = typer.Option(True, help="also run H-ML-004 hourly crypto"),
    device: str | None = typer.Option(None, help="cuda | cpu (default: auto)"),
) -> None:
    """Large daily panel (H-ML-003) and hourly crypto (H-ML-004)."""
    from astra.research.ml_extended import run_hourly_crypto, run_ml_panel_extended

    ms = [m.strip() for m in models.split(",") if m.strip()]
    r = run_ml_panel_extended(models=ms, device=device)
    typer.echo(json.dumps({k: v for k, v in r.items() if isinstance(v, str | int | float)}, default=str))
    if hourly:
        hm = [m for m in ms if m != "gru"]  # GRU was not registered for H-ML-004
        r = run_hourly_crypto(models=hm, device=device)
        typer.echo(json.dumps({k: v for k, v in r.items() if isinstance(v, str | int | float)}, default=str))


@app.command("run-all")
def run_all(
    skip_ingest: bool = False,
    heavy_dl: bool = typer.Option(False, help="include the ~3 h MLP/GRU runs of H-ML-003/004"),
) -> None:
    """Reproduce every study in order (ingest -> core -> extended -> ML -> report)."""
    ingest = (
        []
        if skip_ingest
        else [["ingest"], ["ingest", "--intraday"], ["ingest-fundamentals"], ["ingest-macro"]]
    )
    steps = [
        *ingest,
        ["research"],
        ["holdout"],
        ["vol-study"],
        ["rv-study"],
        ["portfolio-study"],
        ["ml-study"],
        ["micro-study"],
        ["ext-study"],
        ["ml-ext", "--models", "logit,xgb,mlp,gru" if heavy_dl else "logit,xgb"],
        ["multiplicity"],
        ["report"],
    ]
    for s in steps:
        typer.echo(f"== astra {' '.join(s)}")
        r = subprocess.run([sys.executable, "-m", "astra.cli", *s], check=False)
        if r.returncode != 0:
            typer.echo(f"step {s} failed with exit code {r.returncode}")
            raise typer.Exit(code=r.returncode)


@app.command("ml-study")
def ml_study(
    panel: bool = typer.Option(True, help="also run the pooled-panel H-ML-002 study"),
    grid: bool = typer.Option(True, help="run the per-instrument H-ML-001 grid"),
    device: str | None = typer.Option(None, help="cuda | cpu (default: auto)"),
) -> None:
    """Run H-ML-001 (5 models x 13 instruments) and H-ML-002 (pooled panel DL)."""
    from astra.research.ml_study import run_instrument_grid, run_panel

    if grid:
        res = run_instrument_grid(device=device)
        g = res.pop("grid")
        typer.echo(g.to_string(index=False))
        typer.echo(json.dumps(res, indent=2, default=str))
    if panel:
        typer.echo(json.dumps(run_panel(device=device), indent=2, default=str))


@app.command()
def dashboard(port: int = 8501) -> None:
    """Launch the Streamlit dashboard (localhost only)."""
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
        ],
        check=False,
    )


if __name__ == "__main__":
    app()
