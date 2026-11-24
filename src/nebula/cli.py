"""NEBULA command line: `uv run nebula --help`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

from nebula.common.config import get_config
from nebula.common.logging import configure_logging

app = typer.Typer(add_completion=False, help="NEBULA quantitative research platform")


@app.callback()
def _main(log_level: str = "WARNING") -> None:
    configure_logging(log_level)


@app.command()
def env() -> None:
    """Print the hardware / library environment report."""
    from nebula.common.env_report import environment_report

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
    from nebula.data.ingest import ingest_symbol

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
    from nebula.data.fundamentals import ingest_sec

    typer.echo(json.dumps(ingest_sec(), indent=2, default=str)[:20000])


@app.command("ingest-macro")
def ingest_macro() -> None:
    """Market macro (Yahoo: yields, VIX, DXY, USD/INR) + BLS CPI/unemployment. FRED is manual-CSV only."""
    from nebula.data.macro import ingest_bls, ingest_market_macro

    for res in (ingest_market_macro(), ingest_bls()):
        typer.echo(json.dumps(res, indent=2, default=str)[:20000])


@app.command()
def research(symbols: list[str] = typer.Argument(None)) -> None:
    """Run the vertical-slice research (H-XM-000/001/002) for each symbol."""
    from nebula.research.vertical_slice import run_symbol

    syms = symbols or get_config().symbols(tier="core")
    for s in syms:
        typer.echo(json.dumps(run_symbol(s), indent=2, default=str))


@app.command()
def report(
    llm: bool = typer.Option(False, help="append local-LLM narration if NEBULA_LLM_URL is set"),
) -> None:
    """Run the deterministic research agents and write data/reports/research/latest.md."""
    from nebula.agents.orchestrator import write_report

    typer.echo(str(write_report(use_llm=llm)))


@app.command()
def holdout() -> None:
    """Run the confirmatory H-XM-003 Sharpe-difference test on held-out instruments."""
    from nebula.research.holdout_study import run_holdout_tsmom

    res = run_holdout_tsmom()
    t = res.pop("table")
    typer.echo(t.to_string(index=False))
    typer.echo(json.dumps(res, indent=2, default=str))


@app.command("vol-study")
def vol_study() -> None:
    """H-VOL-001 volatility forecasts and H-REG-001 regimes on all instruments."""
    from nebula.research.regime_study import run_regime_study
    from nebula.research.volatility_study import run_volatility_study

    for res in (run_volatility_study(), run_regime_study()):
        t = res.pop("table", None)
        if t is not None:
            typer.echo(t.to_string(index=False))
        typer.echo(json.dumps(res.get("counts", {}), indent=2, default=str))


@app.command("portfolio-study")
def portfolio_study(markets: list[str] = typer.Argument(None)) -> None:
    """H-PF-001/002 portfolio construction study per market."""
    from nebula.research.portfolio_study import run_portfolio_study

    typer.echo(json.dumps(run_portfolio_study(markets or None), indent=2, default=str)[:20000])


@app.command("micro-study")
def micro_study(
    capture_seconds: int = typer.Option(0, help="also run a live depth capture (seconds)"),
) -> None:
    """Microstructure: aggTrades study (H-MS-000); optional live capture + H-MS-001."""
    from nebula.microstructure.capture import run_capture
    from nebula.research.microstructure_study import run_aggtrades_study, run_capture_study

    typer.echo(run_aggtrades_study().experiment_id)
    if capture_seconds > 0:
        rep = run_capture("BTCUSDT", float(capture_seconds))
        typer.echo(str(rep))
        typer.echo(run_capture_study().experiment_id)


@app.command("run-all")
def run_all(skip_ingest: bool = False) -> None:
    """Reproduce every study in order (ingest -> research -> holdout -> vol -> portfolio -> ML -> report)."""
    steps = ([] if skip_ingest else [["ingest"]]) + [
        ["research"],
        ["holdout"],
        ["vol-study"],
        ["portfolio-study"],
        ["ml-study"],
        ["micro-study"],
        ["report"],
    ]
    for s in steps:
        typer.echo(f"== nebula {' '.join(s)}")
        r = subprocess.run([sys.executable, "-m", "nebula.cli", *s], check=False)
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
    from nebula.research.ml_study import run_instrument_grid, run_panel

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
