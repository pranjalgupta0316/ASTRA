"""Offline, synthetic tests for nebula.microstructure (no network)."""

from __future__ import annotations

import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nebula.microstructure.orderbook import (
    DepthEvent,
    DepthSnapshot,
    OrderBook,
    SyncStatus,
    book_imbalance,
    compare_with_snapshot,
    microprice,
    ofi_increment,
    ofi_series,
    replay,
    spread_bps,
)
from nebula.microstructure.trades import (
    aggressor_sign,
    detect_epoch_unit,
    extract_csv,
    kyle_lambda,
    scan_aggtrades_csv,
    trade_bars,
)

SNAP = DepthSnapshot(
    last_update_id=100,
    bids=((99.0, 1.0), (98.0, 2.0), (97.0, 3.0)),
    asks=((101.0, 1.5), (102.0, 2.5), (103.0, 4.0)),
)


def ev(U: int, u: int, bids=(), asks=(), E: int = 0) -> DepthEvent:
    return DepthEvent(E=E or 1_700_000_000_000 + u, U=U, u=u, bids=tuple(bids), asks=tuple(asks), recv_ns=u)


# ------------------------------------------------------------------ sync rules
def test_snapshot_and_diffs_apply_correctly() -> None:
    book = OrderBook()
    book.load_snapshot(SNAP)
    assert book.best_bid() == (99.0, 1.0) and book.best_ask() == (101.0, 1.5)
    assert book.apply(ev(95, 99, bids=[(99.0, 9.0)])) is SyncStatus.STALE  # u <= lastUpdateId
    assert book.apply(ev(100, 103, bids=[(99.5, 0.7)], asks=[(102.0, 5.0)])) is SyncStatus.APPLIED
    assert book.apply(ev(104, 104, bids=[(98.0, 0.0)], asks=[(100.5, 0.2)])) is SyncStatus.APPLIED
    assert book.best_bid() == (99.5, 0.7)
    assert book.best_ask() == (100.5, 0.2)
    assert book.top_bids(3) == [(99.5, 0.7), (99.0, 1.0), (97.0, 3.0)]
    assert book.top_asks(3) == [(100.5, 0.2), (101.0, 1.5), (102.0, 5.0)]
    assert book.last_update_id == 104 and book.valid and book.synced
    assert book.last_event_ms == 1_700_000_000_104 and book.last_recv_ns == 104


def test_first_event_must_straddle_last_update_id() -> None:
    # first non-stale event starts after L+1 -> snapshot too old, book invalid
    book = OrderBook()
    book.load_snapshot(SNAP)
    assert book.apply(ev(103, 105)) is SyncStatus.SNAPSHOT_TOO_OLD
    assert not book.valid and not book.synced
    # U <= L+1 <= u is accepted even if U < L+1
    book2 = OrderBook()
    book2.load_snapshot(SNAP)
    assert book2.apply(ev(98, 101, asks=[(101.0, 0.0)])) is SyncStatus.APPLIED
    assert book2.synced and book2.best_ask() == (102.0, 2.5)
    # exactly U == L+1 is accepted too
    book3 = OrderBook()
    book3.load_snapshot(SNAP)
    assert book3.apply(ev(101, 101)) is SyncStatus.APPLIED


def test_events_before_snapshot_are_buffered() -> None:
    assert OrderBook().apply(ev(1, 2)) is SyncStatus.BUFFERED


def test_sequence_gap_detected_and_book_invalid() -> None:
    book = OrderBook()
    book.load_snapshot(SNAP)
    assert book.apply(ev(100, 102)) is SyncStatus.APPLIED
    assert book.apply(ev(105, 106, bids=[(99.0, 7.0)])) is SyncStatus.GAP  # expected U == 103
    assert not book.valid and book.gaps == 1
    assert book.best_bid() == (99.0, 1.0)  # gapped event not applied
    # nothing is applied until a new snapshot is loaded
    assert book.apply(ev(107, 108)) is SyncStatus.GAP
    book.load_snapshot(DepthSnapshot(110, ((99.0, 4.0),), ((101.0, 1.0),)))
    assert book.valid and book.apply(ev(111, 111)) is SyncStatus.APPLIED


def test_replay_detects_gap_and_resyncs_with_next_snapshot() -> None:
    snap2 = DepthSnapshot(last_update_id=110, bids=((99.0, 5.0),), asks=((101.0, 1.0),))
    events = [ev(100, 102), ev(103, 104), ev(107, 108), ev(109, 111, bids=[(99.0, 6.0)]), ev(112, 112)]
    rows, st = replay([SNAP, snap2], events)
    assert st.gaps == 1
    assert st.resyncs == 1
    assert st.snapshots_used == 2
    assert st.stale == 1  # ev(107,108) after resync: u <= 110
    assert st.applied == 4
    assert rows[-1]["best_bid"] == 99.0 and rows[-1]["bid_qty"] == 6.0
    assert all(r["valid"] for r in rows)


def test_replay_without_resync_snapshot_reports_unsynced() -> None:
    rows, st = replay([SNAP], [ev(100, 102), ev(110, 111), ev(112, 113)])
    assert st.gaps == 1 and st.resyncs == 0 and st.unsynced_events == 2 and len(rows) == 1


def test_qty_zero_removes_level() -> None:
    book = OrderBook()
    book.load_snapshot(SNAP)
    book.apply(ev(101, 101, bids=[(99.0, 0.0)], asks=[(101.0, 0.0), (150.0, 0.0)]))  # 150 absent: no-op
    assert book.best_bid() == (98.0, 2.0)
    assert book.best_ask() == (102.0, 2.5)
    assert 99.0 not in book.bids.qty and len(book.bids) == 2 and len(book.asks) == 2


def test_crossed_book_detected() -> None:
    book = OrderBook()
    book.load_snapshot(SNAP)
    assert book.apply(ev(101, 101, bids=[(101.5, 1.0)])) is SyncStatus.INTEGRITY
    assert not book.valid and book.integrity_failures == 1
    assert "crossed" in book.invalid_reason
    locked = OrderBook()
    locked.load_snapshot(SNAP)
    assert locked.apply(ev(101, 101, bids=[(101.0, 1.0)])) is SyncStatus.INTEGRITY  # bid == ask


def test_negative_qty_detected() -> None:
    book = OrderBook()
    book.load_snapshot(SNAP)
    assert book.apply(ev(101, 101, asks=[(105.0, -1.0)])) is SyncStatus.INTEGRITY
    assert "negative" in book.invalid_reason


def test_verification_snapshot_compare() -> None:
    book = OrderBook()
    book.load_snapshot(SNAP)
    checked, mism = compare_with_snapshot(book, SNAP, k=3)
    assert checked == 6 and mism == 0
    other = DepthSnapshot(100, ((99.0, 1.1), (98.0, 2.0), (97.0, 3.0)), SNAP.asks)
    assert compare_with_snapshot(book, other, k=3) == (6, 1)
    assert compare_with_snapshot(book, other, k=3, exclude_bids={99.0}) == (5, 0)


# ------------------------------------------------------------------ analytics
def test_microprice_and_imbalance_hand_examples() -> None:
    # bid 100 x 3, ask 101 x 1: microprice = (101*3 + 100*1)/4 = 100.75 (leans to the thin ask)
    assert microprice(100.0, 101.0, 3.0, 1.0) == pytest.approx(100.75)
    assert microprice(100.0, 101.0, 1.0, 1.0) == pytest.approx(100.5)
    assert book_imbalance([3.0, 1.0], [1.0, 1.0]) == pytest.approx((4 - 2) / 6)
    assert book_imbalance([0.0], [2.0]) == pytest.approx(-1.0)
    assert np.isnan(book_imbalance([0.0], [0.0]))
    assert spread_bps(99.0, 101.0) == pytest.approx(200.0)
    book = OrderBook()
    book.load_snapshot(SNAP)
    assert book.mid() == pytest.approx(100.0)
    assert book.microprice() == pytest.approx((101.0 * 1.0 + 99.0 * 1.5) / 2.5)
    assert book.imbalance(2) == pytest.approx((3.0 - 4.0) / 7.0)
    assert book.depth(3) == (pytest.approx(6.0), pytest.approx(8.0))


def test_ofi_sign_hand_example() -> None:
    # bid queue grows at same price: +2 buying pressure
    assert ofi_increment(100, 1, 101, 1, 100, 3, 101, 1) == pytest.approx(2.0)
    # bid price improves: + new bid size
    assert ofi_increment(100, 1, 101, 1, 100.5, 0.4, 101, 1) == pytest.approx(0.4)
    # ask price drops (sellers step in): - new ask size
    assert ofi_increment(100, 1, 101, 1, 100, 1, 100.5, 2) == pytest.approx(-2.0)
    # ask queue depleted, ask price moves up: + old ask size
    assert ofi_increment(100, 1, 101, 1, 100, 1, 101.5, 5) == pytest.approx(1.0)
    # bid price falls: - old bid size
    assert ofi_increment(100, 1, 101, 1, 99.5, 7, 101, 1) == pytest.approx(-1.0)
    s = ofi_series([100, 100, 100.5], [1, 3, 0.4], [101, 101, 101], [1, 1, 1])
    assert np.isnan(s[0]) and s[1:] == [pytest.approx(2.0), pytest.approx(0.4)]


# ------------------------------------------------------------------ trades
def test_aggressor_sign_mapping() -> None:
    assert aggressor_sign(True) == -1  # buyer is maker -> seller aggressed
    assert aggressor_sign(False) == 1


def test_epoch_unit_detection() -> None:
    assert detect_epoch_unit([1_700_000_000_000, 1_700_000_000_001]) == "ms"
    assert detect_epoch_unit([1_735_689_600_000_000]) == "us"
    with pytest.raises(ValueError):
        detect_epoch_unit([1_700_000_000_000, 1_735_689_600_000_000])


def _write_zip(tmp: Path, header: bool, us: bool) -> Path:
    base = 1_735_689_600_000  # 2025-01-01 00:00:00 UTC in ms
    scale = 1000 if us else 1
    rows = [
        (1, 100.0, 1.0, 1, 1, base + 100, "False"),
        (2, 101.0, 2.0, 2, 3, base + 500, "True"),
        (3, 102.0, 1.0, 4, 4, base + 1_200, "false"),
        (4, 101.0, 3.0, 5, 5, base + 61_000, "TRUE"),
    ]
    lines = [
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match"
    ]
    lines = lines if header else []
    lines += [f"{a},{p},{q},{f},{l_},{t * scale},{m},True" for a, p, q, f, l_, t, m in rows]
    z = tmp / f"X-aggTrades-{header}-{us}.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("X-aggTrades.csv", "\n".join(lines) + "\n")
    return z


@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("us", [True, False])
def test_aggtrades_parse_and_bars(tmp_path: Path, header: bool, us: bool) -> None:
    z = _write_zip(tmp_path, header, us)
    out = tmp_path / f"x_{header}_{us}"
    out.mkdir()
    lf = scan_aggtrades_csv(extract_csv(z, out))
    df = lf.collect()
    assert df.height == 4
    assert df["ts"][0] == datetime.fromisoformat("2025-01-01T00:00:00.100+00:00")
    assert df["side"].to_list() == [1, -1, 1, -1]
    b1s = trade_bars(lf, "1s")
    assert b1s["n_trades"].to_list() == [2, 1, 1]
    first = b1s.row(0, named=True)
    assert first["vwap"] == pytest.approx((100 + 202) / 3)
    assert first["signed_volume"] == pytest.approx(1.0 - 2.0)
    assert first["trade_imbalance"] == pytest.approx(-1 / 3)
    assert first["realized_vol"] == pytest.approx(abs(np.log(101 / 100)))
    b1m = trade_bars(lf, "1m")
    assert b1m["n_trades"].to_list() == [3, 1]
    assert b1m["signed_volume"].to_list() == [pytest.approx(0.0), pytest.approx(-3.0)]


def test_kyle_lambda_recovers_slope() -> None:
    rng = np.random.default_rng(0)
    n = 500
    sv = rng.normal(0, 10, n)
    dp = 0.3 * sv + rng.normal(0, 1, n)
    vwap = 100 * np.exp(np.cumsum(np.concatenate([[0.0], dp[1:]])) / 1e4)
    ts = [datetime(2025, 1, 1) + timedelta(minutes=i) for i in range(n)]
    bars = pl.DataFrame({"ts": ts, "vwap": vwap, "signed_volume": sv})
    k = kyle_lambda(bars)
    assert k["n"] == n - 1
    assert k["lambda_bps_per_unit"] == pytest.approx(0.3, abs=0.03)
    assert k["t_hac"] > 10


# ------------------------------------------------------------------ capture (offline parts)
def test_live_sync_requests_resync_on_gap() -> None:
    from nebula.microstructure.capture import CaptureReport, _LiveSync

    rep = CaptureReport(symbol="X", run_dir="")
    s = _LiveSync(rep)
    s.on_event(ev(99, 100))
    s.on_event(ev(101, 102))
    assert s.need_snapshot and len(s.pending) == 2
    s.on_snapshot(SNAP)  # L=100: first event stale, second straddles 101
    assert not s.need_snapshot and rep.events_applied_online == 1 and rep.resyncs == 0
    s.on_event(ev(110, 111))  # gap
    assert s.need_snapshot and rep.gaps_detected == 1 and s.pending[0].U == 110
    s.on_event(ev(112, 112))
    s.on_snapshot(DepthSnapshot(110, ((99.0, 1.0),), ((101.0, 1.0),)))
    assert rep.resyncs == 1 and not s.need_snapshot and rep.events_applied_online == 3


def test_long_capture_requires_disk_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from nebula.microstructure.capture import CaptureReport

    spec = importlib.util.spec_from_file_location(
        "capture_depth", Path(__file__).resolve().parents[1] / "scripts" / "capture_depth.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls: list[float] = []

    def fake_run(symbol, duration_s, out_dir=None, **kw):
        calls.append(duration_s)
        return CaptureReport(symbol=symbol, run_dir="", parquet_mb_per_hour=20.0)

    monkeypatch.setattr(mod, "run_capture", fake_run)
    # default cap (500 MB) applies to every capture: 20 MB projected fits, so it runs, with the cap passed on
    seen_caps: list[float] = []

    def fake_run_cap(symbol, duration_s, out_dir=None, **kw):
        calls.append(duration_s)
        seen_caps.append(kw.get("max_raw_mb"))
        return CaptureReport(symbol=symbol, run_dir="", parquet_mb_per_hour=20.0)

    monkeypatch.setattr(mod, "run_capture", fake_run_cap)
    assert mod.main(["--duration", "3600"]) == 0
    assert calls == [mod.PROBE_S, 3600.0] and seen_caps[-1] == 500.0
    calls.clear()
    with pytest.raises(SystemExit):  # beyond the 24 h maximum
        mod.main(["--duration", str(mod.MAX_DURATION_S + 1)])
    monkeypatch.setattr(mod, "run_capture", fake_run)
    assert mod.main(["--duration", "3600", "--max-disk-mb", "10"]) == 2  # 20 MB projected > 10
    calls.clear()
    assert mod.main(["--duration", "3600", "--max-disk-mb", "50"]) == 0
    assert calls == [mod.PROBE_S, 3600.0]
    calls.clear()
    assert mod.main(["--duration", "300"]) == 0 and calls == [300.0]  # <= 10 min: no probe
