"""Integration tests that hit free public providers. Run with: uv run pytest -m network"""

from __future__ import annotations

import pytest

from nebula.common.paths import Paths
from nebula.data.ingest import ingest_symbol
from nebula.data.storage import query


@pytest.mark.network
@pytest.mark.parametrize("symbol", ["AAPL", "RELIANCE", "BTCUSDT"])
def test_ingest_recent_window(symbol, tmp_path):
    tmp = Paths(root=tmp_path)  # data written under tmp; instrument config still read from repo
    res = ingest_symbol(symbol, start="2024-01-01", end="2024-03-31", paths=tmp)
    assert res.report.rows > 40
    assert res.report.status in {"pass", "warn"}, res.report.summary()
    assert res.validated_file is not None and res.validated_file.exists()
    assert res.raw_file.with_suffix(".meta.json").exists()


@pytest.mark.network
def test_duckdb_view_over_validated(tmp_path):
    tmp = Paths(root=tmp_path)
    ingest_symbol("BTCUSDT", start="2024-01-01", end="2024-02-01", paths=tmp)  # Jan monthly file only
    df = query("SELECT symbol, count(*) AS n FROM bars GROUP BY symbol", tmp)
    assert int(df.loc[df.symbol == "BTCUSDT", "n"].iloc[0]) == 31
