"""Bounded live capture of Binance spot L2 diff depth + aggTrades, with online sync bookkeeping.

Streams: `<symbol>@depth@100ms` and `<symbol>@aggTrade` over one combined WebSocket connection
(wss://stream.binance.com:9443, fallback wss://data-stream.binance.vision). REST snapshots from
https://data-api.binance.vision/api/v3/depth?limit=1000 (weight 50 each; at most one per second and a
hard cap per run).

Everything is written raw to Parquet (zstd) with the exchange time (`E`, ms) and the local receipt time
(`recv_ns`, ns since epoch):
  data/microstructure/<SYMBOL>/<run_id>/depth_events.parquet   one row per diff event (L2 only)
  data/microstructure/<SYMBOL>/<run_id>/agg_trades.parquet     one row per aggTrade
  data/microstructure/<SYMBOL>/<run_id>/snapshots.parquet      one row per snapshot level (sync/verify)
  data/microstructure/<SYMBOL>/<run_id>/capture_report.json    sizes, MB/hour, gaps, resyncs

The book is kept synchronized online (to know when a resync snapshot is needed) and can be rebuilt
offline with `load_capture` + `orderbook.replay`. The capture has a hard stop at `duration_s`.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from nebula.common.logging import get_logger
from nebula.common.paths import get_paths
from nebula.microstructure.orderbook import DepthEvent, DepthSnapshot, OrderBook, SyncStatus

log = get_logger(__name__)

WS_URLS = ("wss://stream.binance.com:9443", "wss://data-stream.binance.vision")
REST_DEPTH = "https://data-api.binance.vision/api/v3/depth"
USER_AGENT = "nebula-research/0.1 (+local research; respects rate limits)"
HTTP_TIMEOUT_S = 10
WS_OPEN_TIMEOUT_S = 10
RECV_TIMEOUT_S = 10
MIN_SNAPSHOT_INTERVAL_S = 1.0
MAX_SNAPSHOTS = 30

DEPTH_SCHEMA = pa.schema(
    [
        ("E", pa.int64()),
        ("U", pa.int64()),
        ("u", pa.int64()),
        ("recv_ns", pa.int64()),
        ("bid_px", pa.list_(pa.float64())),
        ("bid_qty", pa.list_(pa.float64())),
        ("ask_px", pa.list_(pa.float64())),
        ("ask_qty", pa.list_(pa.float64())),
    ]
)
TRADE_SCHEMA = pa.schema(
    [
        ("E", pa.int64()),
        ("T", pa.int64()),
        ("agg_id", pa.int64()),
        ("price", pa.float64()),
        ("qty", pa.float64()),
        ("first_id", pa.int64()),
        ("last_id", pa.int64()),
        ("is_buyer_maker", pa.bool_()),
        ("recv_ns", pa.int64()),
    ]
)
SNAP_SCHEMA = pa.schema(
    [
        ("snapshot_id", pa.int32()),
        ("kind", pa.string()),
        ("last_update_id", pa.int64()),
        ("request_ns", pa.int64()),
        ("recv_ns", pa.int64()),
        ("side", pa.string()),
        ("price", pa.float64()),
        ("qty", pa.float64()),
    ]
)


@dataclass
class CaptureReport:
    symbol: str
    run_dir: str
    ws_url: str = ""
    started_utc: str = ""
    ended_utc: str = ""
    requested_duration_s: float = 0.0
    actual_duration_s: float = 0.0  # data window: start until the hard stop / last message
    wall_duration_s: float = 0.0  # including connection close and file finalization
    depth_events: int = 0
    agg_trades: int = 0
    ws_messages: int = 0
    ws_bytes: int = 0
    reconnects: int = 0
    snapshots_sync: int = 0
    snapshots_verify: int = 0
    snapshot_errors: int = 0
    gaps_detected: int = 0
    snapshot_too_old: int = 0
    integrity_failures: int = 0
    resyncs: int = 0
    events_applied_online: int = 0
    trade_id_gaps: int = 0
    parquet_bytes: int = 0
    parquet_mb_per_hour: float = 0.0
    ws_json_mb_per_hour: float = 0.0
    projected_gb_per_day: float = 0.0
    errors: list[str] = field(default_factory=list)


class _ParquetSink:
    """Append-only parquet writer; buffers rows and flushes row groups."""

    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path = path
        self.schema = schema
        self.rows: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
        self.rows = []

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()


def _fetch_snapshot(http: requests.Session, symbol: str, limit: int = 1000) -> tuple[dict, int, int]:
    t0 = time.time_ns()
    r = http.get(REST_DEPTH, params={"symbol": symbol, "limit": limit}, timeout=HTTP_TIMEOUT_S)
    r.raise_for_status()
    return r.json(), t0, time.time_ns()


class _LiveSync:
    """Online application of the Binance sync rules: decides when a (re)sync snapshot is needed."""

    def __init__(self, rep: CaptureReport) -> None:
        self.book = OrderBook()
        self.pending: list[DepthEvent] = []
        self.need_snapshot = True
        self.rep = rep
        self.ever_synced = False

    def on_event(self, ev: DepthEvent) -> None:
        if self.need_snapshot:
            self.pending.append(ev)
            return
        self._apply([ev])

    def on_snapshot(self, snap: DepthSnapshot) -> None:
        self.book.load_snapshot(snap)
        if self.ever_synced:
            self.rep.resyncs += 1
        evs, self.pending = self.pending, []
        self.need_snapshot = False
        self._apply(evs)

    def _apply(self, evs: list[DepthEvent]) -> None:
        for i, ev in enumerate(evs):
            res = self.book.apply(ev)
            if res is SyncStatus.APPLIED:
                self.ever_synced = True
                self.rep.events_applied_online += 1
                continue
            if res is SyncStatus.STALE:
                continue
            if res is SyncStatus.GAP:
                self.rep.gaps_detected += 1
                log.warning("depth.gap", reason=self.book.invalid_reason)
            elif res is SyncStatus.SNAPSHOT_TOO_OLD:
                self.rep.snapshot_too_old += 1
            elif res is SyncStatus.INTEGRITY:
                self.rep.integrity_failures += 1
                log.warning("depth.integrity", reason=self.book.invalid_reason)
                self.need_snapshot = True
                self.pending = list(evs[i + 1 :])
                return
            self.need_snapshot = True
            self.pending = list(evs[i:])
            return

    def connection_lost(self) -> None:
        """A reconnect may have lost events: force a resync."""
        if not self.need_snapshot:
            self.book.invalidate("websocket reconnect")
            self.need_snapshot = True
            self.pending = []


async def capture(
    symbol: str,
    duration_s: float,
    out_dir: Path | None = None,
    ws_urls: tuple[str, ...] = WS_URLS,
    verify_every_s: float = 120.0,
    flush_every_s: float = 30.0,
    max_raw_mb: float | None = None,
) -> CaptureReport:
    """Capture depth + aggTrades for `duration_s` seconds (hard stop). Returns the capture report.

    `max_raw_mb` stops the capture once the raw WebSocket bytes received exceed the cap. Parquet on
    disk is always smaller than the raw JSON (~7x in the measured run), so disk use stays below it.
    """
    sym = symbol.upper()
    stream = sym.lower()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = out_dir or (get_paths().data / "microstructure")
    run_dir = base / sym / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rep = CaptureReport(symbol=sym, run_dir=str(run_dir), requested_duration_s=duration_s)
    depth_sink = _ParquetSink(run_dir / "depth_events.parquet", DEPTH_SCHEMA)
    trade_sink = _ParquetSink(run_dir / "agg_trades.parquet", TRADE_SCHEMA)
    snap_sink = _ParquetSink(run_dir / "snapshots.parquet", SNAP_SCHEMA)
    http = requests.Session()
    http.headers["User-Agent"] = USER_AGENT
    sync = _LiveSync(rep)
    snap_id = 0
    last_snap_t = 0.0
    last_verify_t = time.monotonic()
    last_flush_t = time.monotonic()
    last_trade_id: int | None = None
    snap_task: asyncio.Task | None = None
    snap_kind = "sync"

    from websockets.asyncio.client import connect

    t_start = time.monotonic()
    t_last_msg = t_start
    deadline = t_start + duration_s
    rep.started_utc = datetime.now(UTC).isoformat()

    def store_snapshot(payload: dict, req_ns: int, recv_ns: int, kind: str) -> DepthSnapshot:
        nonlocal snap_id
        snap = DepthSnapshot.from_rest(payload, recv_ns=recv_ns)
        for side, levels in (("bid", snap.bids), ("ask", snap.asks)):
            for p, q in levels:
                snap_sink.add(
                    {
                        "snapshot_id": snap_id,
                        "kind": kind,
                        "last_update_id": snap.last_update_id,
                        "request_ns": req_ns,
                        "recv_ns": recv_ns,
                        "side": side,
                        "price": p,
                        "qty": q,
                    }
                )
        snap_id += 1
        return snap

    def total_snapshots() -> int:
        return rep.snapshots_sync + rep.snapshots_verify + rep.snapshot_errors

    async def handle_snapshot_task() -> None:
        nonlocal snap_task
        if snap_task is None or not snap_task.done():
            return
        task, snap_task = snap_task, None
        try:
            payload, req_ns, recv_ns = task.result()
        except Exception as exc:
            rep.snapshot_errors += 1
            rep.errors.append(f"snapshot: {exc!r}")
            return
        snap = store_snapshot(payload, req_ns, recv_ns, snap_kind)
        if snap_kind == "sync":
            rep.snapshots_sync += 1
            sync.on_snapshot(snap)
        else:
            rep.snapshots_verify += 1

    def maybe_request_snapshot() -> None:
        nonlocal snap_task, last_snap_t, snap_kind, last_verify_t
        now = time.monotonic()
        if snap_task is not None or now - last_snap_t < MIN_SNAPSHOT_INTERVAL_S:
            return
        if total_snapshots() >= MAX_SNAPSHOTS or now > deadline - 1.0:
            return
        if sync.need_snapshot and sync.pending:
            snap_kind = "sync"
        elif not sync.need_snapshot and verify_every_s > 0 and now - last_verify_t >= verify_every_s:
            snap_kind = "verify"
            last_verify_t = now
        else:
            return
        last_snap_t = now
        snap_task = asyncio.create_task(asyncio.to_thread(_fetch_snapshot, http, sym))

    url_idx = 0
    try:
        while time.monotonic() < deadline:
            url = f"{ws_urls[url_idx % len(ws_urls)]}/stream?streams={stream}@depth@100ms/{stream}@aggTrade"
            try:
                async with connect(
                    url, open_timeout=WS_OPEN_TIMEOUT_S, close_timeout=2, max_size=2**22
                ) as ws:
                    rep.ws_url = url
                    log.info("capture.connected", url=url)
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, RECV_TIMEOUT_S))
                        except TimeoutError:
                            if time.monotonic() >= deadline:
                                break
                            raise
                        recv_ns = time.time_ns()
                        t_last_msg = time.monotonic()
                        rep.ws_messages += 1
                        rep.ws_bytes += len(raw)
                        if max_raw_mb is not None and rep.ws_bytes > max_raw_mb * 1e6:
                            rep.errors.append(f"disk cap reached: {rep.ws_bytes / 1e6:.1f} MB raw received")
                            deadline = time.monotonic()  # hard stop; finalization below still runs
                            break
                        msg = json.loads(raw)
                        data = msg.get("data", msg)
                        etype = data.get("e")
                        if etype == "depthUpdate":
                            ev = DepthEvent.from_ws(data, recv_ns=recv_ns)
                            rep.depth_events += 1
                            depth_sink.add(
                                {
                                    "E": ev.E,
                                    "U": ev.U,
                                    "u": ev.u,
                                    "recv_ns": recv_ns,
                                    "bid_px": [p for p, _ in ev.bids],
                                    "bid_qty": [q for _, q in ev.bids],
                                    "ask_px": [p for p, _ in ev.asks],
                                    "ask_qty": [q for _, q in ev.asks],
                                }
                            )
                            sync.on_event(ev)
                        elif etype == "aggTrade":
                            aid = int(data["a"])
                            if last_trade_id is not None and aid != last_trade_id + 1:
                                rep.trade_id_gaps += 1
                            last_trade_id = aid
                            rep.agg_trades += 1
                            trade_sink.add(
                                {
                                    "E": int(data["E"]),
                                    "T": int(data["T"]),
                                    "agg_id": aid,
                                    "price": float(data["p"]),
                                    "qty": float(data["q"]),
                                    "first_id": int(data["f"]),
                                    "last_id": int(data["l"]),
                                    "is_buyer_maker": bool(data["m"]),
                                    "recv_ns": recv_ns,
                                }
                            )
                        await handle_snapshot_task()
                        maybe_request_snapshot()
                        if time.monotonic() - last_flush_t >= flush_every_s:
                            depth_sink.flush()
                            trade_sink.flush()
                            snap_sink.flush()
                            last_flush_t = time.monotonic()
                    break  # deadline reached
            except Exception as exc:  # any WS/network failure -> count, reconnect until the deadline
                if time.monotonic() >= deadline:
                    break
                rep.reconnects += 1
                rep.errors.append(f"ws {url}: {exc!r}")
                log.warning("capture.ws_error", url=url, error=repr(exc))
                sync.connection_lost()
                last_trade_id = None
                url_idx += 1
                await asyncio.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    finally:
        if snap_task is not None:
            snap_task.cancel()
        depth_sink.close()
        trade_sink.close()
        snap_sink.close()
        http.close()

    rep.wall_duration_s = time.monotonic() - t_start
    elapsed = min(max(t_last_msg, min(time.monotonic(), deadline)) - t_start, duration_s)
    rep.actual_duration_s = elapsed
    rep.ended_utc = datetime.now(UTC).isoformat()
    rep.parquet_bytes = sum(p.stat().st_size for p in run_dir.glob("*.parquet"))
    hours = elapsed / 3600 if elapsed > 0 else float("nan")
    rep.parquet_mb_per_hour = rep.parquet_bytes / 1e6 / hours
    rep.ws_json_mb_per_hour = rep.ws_bytes / 1e6 / hours
    rep.projected_gb_per_day = rep.parquet_mb_per_hour * 24 / 1e3
    (run_dir / "capture_report.json").write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")
    log.info("capture.done", **{k: v for k, v in asdict(rep).items() if k != "errors"})
    return rep


def run_capture(symbol: str, duration_s: float, out_dir: Path | None = None, **kw: Any) -> CaptureReport:
    """Synchronous wrapper with an outer hard timeout (duration + 30 s for flush/close)."""

    async def _main() -> CaptureReport:
        return await asyncio.wait_for(capture(symbol, duration_s, out_dir, **kw), timeout=duration_s + 30)

    return asyncio.run(_main())


# ------------------------------------------------------------------ offline loading
def load_capture(run_dir: Path) -> tuple[list[DepthSnapshot], list[DepthSnapshot], list[DepthEvent]]:
    """Read a capture dir -> (sync snapshots in fetch order, verify snapshots, events in arrival order)."""
    snaps_sync: list[DepthSnapshot] = []
    snaps_verify: list[DepthSnapshot] = []
    sp = run_dir / "snapshots.parquet"
    if sp.exists():
        st = pq.read_table(sp).to_pandas()
        for _sid, g in st.groupby("snapshot_id", sort=True):
            bids = g[g["side"] == "bid"]
            asks = g[g["side"] == "ask"]
            snap = DepthSnapshot(
                last_update_id=int(g["last_update_id"].iloc[0]),
                bids=tuple(zip(bids["price"].tolist(), bids["qty"].tolist(), strict=True)),
                asks=tuple(zip(asks["price"].tolist(), asks["qty"].tolist(), strict=True)),
                recv_ns=int(g["recv_ns"].iloc[0]),
            )
            (snaps_sync if g["kind"].iloc[0] == "sync" else snaps_verify).append(snap)
    events: list[DepthEvent] = []
    dp = run_dir / "depth_events.parquet"
    if dp.exists():
        t = pq.read_table(dp).to_pylist()
        for r in t:
            events.append(
                DepthEvent(
                    E=r["E"],
                    U=r["U"],
                    u=r["u"],
                    bids=tuple(zip(r["bid_px"], r["bid_qty"], strict=True)),
                    asks=tuple(zip(r["ask_px"], r["ask_qty"], strict=True)),
                    recv_ns=r["recv_ns"],
                )
            )
    return snaps_sync, snaps_verify, events
