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
    symbols: list[str] = typer.Argument(None), start: str | None = None, end: str | None = None
) -> None:
    """Fetch, validate and store daily bars (default: all configured instruments)."""
    from nebula.data.ingest import ingest_symbol

    syms = symbols or [i.symbol for i in get_config().instruments]
    failed = False
    for s in syms:
        res = ingest_symbol(s, start=start, end=end)
        typer.echo(res.report.summary())
        typer.echo(f"  raw: {res.raw_file}\n  validated: {res.validated_file}")
        failed |= res.validated_file is None
    raise typer.Exit(code=1 if failed else 0)


@app.command()
def research(symbols: list[str] = typer.Argument(None)) -> None:
    """Run the vertical-slice research (H-XM-000/001/002) for each symbol."""
    from nebula.research.vertical_slice import run_symbol

    syms = symbols or [i.symbol for i in get_config().instruments]
    for s in syms:
        typer.echo(json.dumps(run_symbol(s), indent=2, default=str))


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
