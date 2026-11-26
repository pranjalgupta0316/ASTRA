"""Data locks: immutable snapshots, strict resolution, records pinned to a data version."""

from __future__ import annotations

import pandas as pd
import pytest

from astra.common.config import get_config
from astra.common.paths import Paths
from astra.data import lock as lk
from astra.data.storage import query, read_bars, read_manifest, write_validated
from astra.data.validation import validate_bars
from astra.tracking.tracker import Tracker
from conftest import make_bars  # tests/ is on sys.path under pytest (no package __init__)


def _store(paths: Paths, bars: pd.DataFrame, symbol: str = "TEST") -> None:
    bars = bars.assign(symbol=symbol)
    rep = validate_bars(bars, get_config().markets["US"])
    raw = paths.raw / "x.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"raw")
    write_validated(bars, rep, raw, paths)


@pytest.fixture
def tmp_paths(tmp_path) -> Paths:
    p = Paths(root=tmp_path)
    p.ensure()
    return p


def test_lock_pins_data_against_later_changes(tmp_paths):
    bars = make_bars("US", n=50)
    _store(tmp_paths, bars)
    info = lk.create_lock("v1", note="t", paths=tmp_paths)
    assert info["n_files"] >= 2  # bars.parquet + manifest.json
    # a re-download changes the live file
    _store(
        tmp_paths,
        bars.assign(
            close=bars["close"] * 2,
            adj_close=bars["adj_close"] * 2,
            high=bars["high"] * 2,
            open=bars["open"] * 2,
            low=bars["low"] * 2,
        ),
    )
    live = read_bars("TEST", "US", tmp_paths)
    with lk.use_lock("v1"):
        locked = read_bars("TEST", "US", tmp_paths)
        man = read_manifest("TEST", "US", tmp_paths)
    assert locked["close"].iloc[0] == pytest.approx(bars["close"].iloc[0])
    assert live["close"].iloc[0] == pytest.approx(2 * bars["close"].iloc[0])
    assert man["rows"] == 50


def test_lock_is_strict_and_immutable(tmp_paths):
    _store(tmp_paths, make_bars("US", n=40))
    lk.create_lock("v1", paths=tmp_paths)
    _store(tmp_paths, make_bars("US", n=40), symbol="NEW")  # added after the lock
    with lk.use_lock("v1"), pytest.raises(lk.LockError, match="not in data lock"):
        read_bars("NEW", "US", tmp_paths)
    with pytest.raises(lk.LockError, match="immutable"):
        lk.create_lock("v1", paths=tmp_paths)


def test_verify_detects_corruption_and_missing(tmp_paths):
    _store(tmp_paths, make_bars("US", n=40))
    lk.create_lock("v1", paths=tmp_paths)
    assert lk.verify_lock("v1", tmp_paths)["ok"]
    lock = lk.load_lock("v1", tmp_paths)
    rel, meta = next(iter(lock["files"].items()))
    obj = lk.object_path(meta["sha256"], "." + rel.rsplit(".", 1)[-1], tmp_paths)
    obj.write_bytes(b"tampered")
    res = lk.verify_lock("v1", tmp_paths)
    assert not res["ok"] and rel in res["corrupt"]


def test_snapshots_are_deduplicated(tmp_paths):
    _store(tmp_paths, make_bars("US", n=40))
    a = lk.create_lock("v1", paths=tmp_paths)
    b = lk.create_lock("v2", paths=tmp_paths)  # nothing changed
    assert a["new_snapshot_bytes"] > 0 and b["new_snapshot_bytes"] == 0
    assert lk.lock_fingerprint("v1", tmp_paths) == lk.lock_fingerprint("v2", tmp_paths)


def test_query_uses_locked_snapshots(tmp_paths):
    _store(tmp_paths, make_bars("US", n=30))
    lk.create_lock("v1", paths=tmp_paths)
    _store(tmp_paths, make_bars("US", n=60))
    with lk.use_lock("v1"):
        n_locked = int(query("SELECT count(*) AS n FROM bars", tmp_paths)["n"].iloc[0])
    n_live = int(query("SELECT count(*) AS n FROM bars", tmp_paths)["n"].iloc[0])
    assert (n_locked, n_live) == (30, 60)


def test_records_store_the_active_lock(tmp_paths):
    _store(tmp_paths, make_bars("US", n=30))
    lk.create_lock("v1", paths=tmp_paths)
    tr = Tracker(tmp_paths)
    with lk.use_lock("v1"):
        rec = tr.new("x", "H-T-1", "t")
    assert rec.data_lock["name"] == "v1" and len(rec.data_lock["fingerprint"]) == 64
    with lk.use_lock(None):
        assert tr.new("x", "H-T-1", "t").data_lock["name"] is None


def test_paths_outside_validated_are_untouched(tmp_paths):
    _store(tmp_paths, make_bars("US", n=30))
    lk.create_lock("v1", paths=tmp_paths)
    other = tmp_paths.raw / "x.parquet"
    with lk.use_lock("v1"):
        assert lk.resolve(other, tmp_paths) == other


def test_unknown_lock_and_bad_names(tmp_paths):
    with pytest.raises(lk.LockError):
        lk.load_lock("nope", tmp_paths)
    with pytest.raises(lk.LockError, match="invalid"):
        lk.create_lock("bad name", paths=tmp_paths)
