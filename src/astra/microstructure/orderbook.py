"""L2 (market-by-price) order book with Binance spot diff-depth synchronization.

Data level: Binance public depth streams are **L2 only** (aggregated quantity per price level). There
are no order IDs, so nothing here is L3 / market-by-order and queue position is not observable.

Synchronization (Binance spot "How to manage a local order book correctly"):
1. Buffer `<symbol>@depth` events from the WebSocket.
2. Fetch a REST snapshot `/api/v3/depth` -> `lastUpdateId` (L).
3. Drop buffered events with `u <= L`. If the first remaining event has `U > L + 1` the snapshot is too
   old -> fetch a new snapshot (`SyncStatus.SNAPSHOT_TOO_OLD`).
4. The first applied event must satisfy `U <= L + 1 <= u`.
5. Every later event must satisfy `U == prev_u + 1`; otherwise it is a **gap**: the book is marked
   invalid and a resync (new snapshot) is required. Events with `u <= prev_u` (already covered) are
   ignored as stale.
6. Quantities are absolute; quantity 0 removes the level.

Each applied event keeps the exchange event time `E` (ms) and the local receipt time (ns since epoch).
Integrity checks after every applied event: crossed/locked book (best bid >= best ask) and negative
quantities both mark the book invalid.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

Level = tuple[float, float]  # (price, quantity)


@dataclass(frozen=True)
class DepthEvent:
    """One Binance diff-depth event (`depthUpdate`)."""

    E: int  # exchange event time, ms since epoch
    U: int  # first update id in event
    u: int  # final update id in event
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    recv_ns: int = 0  # local receipt time, ns since epoch

    @classmethod
    def from_ws(cls, msg: dict, recv_ns: int = 0) -> DepthEvent:
        return cls(
            E=int(msg["E"]),
            U=int(msg["U"]),
            u=int(msg["u"]),
            bids=tuple((float(p), float(q)) for p, q in msg.get("b", [])),
            asks=tuple((float(p), float(q)) for p, q in msg.get("a", [])),
            recv_ns=recv_ns,
        )


@dataclass(frozen=True)
class DepthSnapshot:
    """REST `/api/v3/depth` snapshot."""

    last_update_id: int
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    recv_ns: int = 0

    @classmethod
    def from_rest(cls, payload: dict, recv_ns: int = 0) -> DepthSnapshot:
        return cls(
            last_update_id=int(payload["lastUpdateId"]),
            bids=tuple((float(p), float(q)) for p, q in payload["bids"]),
            asks=tuple((float(p), float(q)) for p, q in payload["asks"]),
            recv_ns=recv_ns,
        )


class SyncStatus(StrEnum):
    BUFFERED = "buffered"  # no snapshot yet: event kept for later
    STALE = "stale"  # u <= last applied id: ignored
    APPLIED = "applied"
    GAP = "gap"  # continuity broken: book invalid, resync required
    SNAPSHOT_TOO_OLD = "snapshot_too_old"  # first buffered event starts after L+1: refetch snapshot
    INTEGRITY = "integrity"  # applied, but book crossed or negative qty: invalid, resync required


class _Side:
    """Price -> quantity map with a sorted price list (ascending)."""

    __slots__ = ("prices", "qty")

    def __init__(self) -> None:
        self.qty: dict[float, float] = {}
        self.prices: list[float] = []

    def clear(self) -> None:
        self.qty.clear()
        self.prices.clear()

    def set(self, price: float, q: float) -> None:
        if q == 0.0:
            if price in self.qty:
                del self.qty[price]
                i = bisect.bisect_left(self.prices, price)
                del self.prices[i]
            return
        if price not in self.qty:
            bisect.insort(self.prices, price)
        self.qty[price] = q

    def __len__(self) -> int:
        return len(self.prices)


@dataclass
class OrderBook:
    """L2 book state plus Binance diff-depth sync state machine."""

    symbol: str = ""
    bids: _Side = field(default_factory=_Side)
    asks: _Side = field(default_factory=_Side)
    last_update_id: int | None = None  # id of the last state reflected in the book (L or prev u)
    synced: bool = False  # True once the first post-snapshot event satisfied U <= L+1 <= u
    valid: bool = False  # False before snapshot, after a gap or after an integrity failure
    last_event_ms: int | None = None
    last_recv_ns: int | None = None
    gaps: int = 0
    integrity_failures: int = 0
    snapshots_loaded: int = 0
    events_applied: int = 0
    events_stale: int = 0
    invalid_reason: str = "no snapshot"

    # ------------------------------------------------------------------ sync
    def load_snapshot(self, snap: DepthSnapshot) -> None:
        self.bids.clear()
        self.asks.clear()
        for p, q in snap.bids:
            self.bids.set(p, q)
        for p, q in snap.asks:
            self.asks.set(p, q)
        self.last_update_id = snap.last_update_id
        self.synced = False
        self.valid = True
        self.invalid_reason = ""
        self.snapshots_loaded += 1
        if self.integrity_error() is not None:
            self.valid = False
            self.invalid_reason = f"snapshot integrity: {self.integrity_error()}"

    def invalidate(self, reason: str) -> None:
        self.valid = False
        self.synced = False
        self.invalid_reason = reason

    def apply(self, ev: DepthEvent) -> SyncStatus:
        """Apply one diff event according to the Binance sync rules."""
        if self.last_update_id is None:
            return SyncStatus.BUFFERED
        if not self.valid:
            # after a gap/integrity failure nothing is applied until a new snapshot is loaded
            return SyncStatus.GAP
        L = self.last_update_id
        if ev.u <= L:
            self.events_stale += 1
            return SyncStatus.STALE
        if not self.synced:
            if ev.U > L + 1:
                self.invalidate(f"snapshot too old: U={ev.U} > L+1={L + 1}")
                return SyncStatus.SNAPSHOT_TOO_OLD
            # here U <= L+1 and u >= L+1 -> first event rule satisfied
            self.synced = True
        elif ev.U != L + 1:
            self.gaps += 1
            self.invalidate(f"sequence gap: U={ev.U} != prev_u+1={L + 1}")
            return SyncStatus.GAP
        for p, q in ev.bids:
            self.bids.set(p, q)
        for p, q in ev.asks:
            self.asks.set(p, q)
        self.last_update_id = ev.u
        self.last_event_ms = ev.E
        self.last_recv_ns = ev.recv_ns
        self.events_applied += 1
        err = self.integrity_error()
        if err is not None:
            self.integrity_failures += 1
            self.invalidate(f"integrity: {err}")
            return SyncStatus.INTEGRITY
        return SyncStatus.APPLIED

    # ------------------------------------------------------------ integrity
    def integrity_error(self) -> str | None:
        if any(q < 0 for q in self.bids.qty.values()) or any(q < 0 for q in self.asks.qty.values()):
            return "negative quantity"
        if any(not math.isfinite(q) for q in self.bids.qty.values()) or any(
            not math.isfinite(q) for q in self.asks.qty.values()
        ):
            return "non-finite quantity"
        bb, ba = self.best_bid(), self.best_ask()
        if bb is not None and ba is not None and bb[0] >= ba[0]:
            return f"crossed book: bid {bb[0]} >= ask {ba[0]}"
        return None

    # ------------------------------------------------------------ accessors
    def best_bid(self) -> Level | None:
        if not self.bids.prices:
            return None
        p = self.bids.prices[-1]
        return p, self.bids.qty[p]

    def best_ask(self) -> Level | None:
        if not self.asks.prices:
            return None
        p = self.asks.prices[0]
        return p, self.asks.qty[p]

    def top_bids(self, k: int) -> list[Level]:
        return [(p, self.bids.qty[p]) for p in reversed(self.bids.prices[-k:])]

    def top_asks(self, k: int) -> list[Level]:
        return [(p, self.asks.qty[p]) for p in self.asks.prices[:k]]

    def mid(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        return float("nan") if bb is None or ba is None else (bb[0] + ba[0]) / 2.0

    def spread_bps(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        return float("nan") if bb is None or ba is None else spread_bps(bb[0], ba[0])

    def depth(self, k: int) -> tuple[float, float]:
        """Summed quantity of the top-k bid and ask levels."""
        return sum(q for _, q in self.top_bids(k)), sum(q for _, q in self.top_asks(k))

    def microprice(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return float("nan")
        return microprice(bb[0], ba[0], bb[1], ba[1])

    def imbalance(self, k: int = 5) -> float:
        return book_imbalance([q for _, q in self.top_bids(k)], [q for _, q in self.top_asks(k)])

    def state_row(self, k: int = 5) -> dict:
        """Flat analytics row for the current state (used for reconstruction output)."""
        bb, ba = self.best_bid(), self.best_ask()
        db, da = self.depth(k)
        return {
            "E_ms": self.last_event_ms,
            "recv_ns": self.last_recv_ns,
            "update_id": self.last_update_id,
            "valid": self.valid,
            "segment": self.snapshots_loaded,  # changes on every (re)sync -> contiguous valid segment id
            "best_bid": bb[0] if bb else float("nan"),
            "bid_qty": bb[1] if bb else float("nan"),
            "best_ask": ba[0] if ba else float("nan"),
            "ask_qty": ba[1] if ba else float("nan"),
            "mid": self.mid(),
            "spread_bps": self.spread_bps(),
            "microprice": self.microprice(),
            "imbalance_1": self.imbalance(1),
            f"imbalance_{k}": self.imbalance(k),
            f"depth_bid_{k}": db,
            f"depth_ask_{k}": da,
            "n_bid_levels": len(self.bids),
            "n_ask_levels": len(self.asks),
        }


# ---------------------------------------------------------------- analytics
def spread_bps(bid: float, ask: float) -> float:
    m = (bid + ask) / 2.0
    return (ask - bid) / m * 1e4


def microprice(bid: float, ask: float, bid_qty: float, ask_qty: float) -> float:
    """Size-weighted mid: (a*Qb + b*Qa)/(Qb + Qa). Leans towards the side with less resting size."""
    tot = bid_qty + ask_qty
    if tot <= 0:
        return float("nan")
    return (ask * bid_qty + bid * ask_qty) / tot


def book_imbalance(bid_qtys: Sequence[float], ask_qtys: Sequence[float]) -> float:
    """(sum Qb - sum Qa)/(sum Qb + sum Qa) in [-1, 1]."""
    b, a = float(sum(bid_qtys)), float(sum(ask_qtys))
    return float("nan") if b + a <= 0 else (b - a) / (b + a)


def ofi_increment(
    prev_bid: float,
    prev_bid_qty: float,
    prev_ask: float,
    prev_ask_qty: float,
    bid: float,
    bid_qty: float,
    ask: float,
    ask_qty: float,
) -> float:
    """Order-flow-imbalance increment e_n (Cont, Kukanov & Stoikov 2014, eq. 2).

    e_n = 1{b_n >= b_{n-1}} Qb_n - 1{b_n <= b_{n-1}} Qb_{n-1}
        - 1{a_n <= a_{n-1}} Qa_n + 1{a_n >= a_{n-1}} Qa_{n-1}
    Positive = net buying pressure at the touch.
    """
    e = 0.0
    if bid >= prev_bid:
        e += bid_qty
    if bid <= prev_bid:
        e -= prev_bid_qty
    if ask <= prev_ask:
        e -= ask_qty
    if ask >= prev_ask:
        e += prev_ask_qty
    return e


def ofi_series(
    bid: Sequence[float], bid_qty: Sequence[float], ask: Sequence[float], ask_qty: Sequence[float]
) -> list[float]:
    """Per-update OFI increments for a sequence of best-level states (first element = nan)."""
    out = [float("nan")]
    for i in range(1, len(bid)):
        out.append(
            ofi_increment(
                bid[i - 1], bid_qty[i - 1], ask[i - 1], ask_qty[i - 1], bid[i], bid_qty[i], ask[i], ask_qty[i]
            )
        )
    return out


# -------------------------------------------------------------- replay
@dataclass
class ReplayStats:
    events: int = 0
    applied: int = 0
    stale: int = 0
    gaps: int = 0
    integrity_failures: int = 0
    snapshot_too_old: int = 0
    snapshots_used: int = 0
    resyncs: int = 0  # successful re-synchronizations after the initial one
    unsynced_events: int = 0  # events that could not be applied (waiting for a usable snapshot)
    verify_checks: int = 0  # verification snapshots compared against the reconstructed book
    verify_levels_checked: int = 0
    verify_mismatches: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def replay(
    snapshots: Sequence[DepthSnapshot],
    events: Iterable[DepthEvent],
    k: int = 5,
    verify: Sequence[DepthSnapshot] = (),
    verify_levels: int = 20,
) -> tuple[list[dict], ReplayStats]:
    """Offline reconstruction: replay events (in arrival order) against sync snapshots (in fetch order).

    Mirrors the live logic: start from snapshots[0]; whenever the book becomes invalid (gap, integrity
    failure, snapshot too old) the next snapshot is loaded and the triggering event and all later events
    are re-synchronized against it. Returns one analytics row per applied event plus stats.

    `verify` snapshots (independent REST fetches) are compared with the reconstructed book at their
    `lastUpdateId` L: exactly before the event with U == L+1, or - if L falls inside an event's
    [U, u] range - after applying it, excluding the price levels that event touched (those may legitimately
    differ). Untouched top-`verify_levels` levels must match exactly.
    """
    pending_verify = sorted(verify, key=lambda s: s.last_update_id)
    vi = 0
    book = OrderBook()
    st = ReplayStats()
    rows: list[dict] = []
    snaps = list(snapshots)
    si = 0
    if snaps:
        book.load_snapshot(snaps[0])
        si = 1
        st.snapshots_used = 1
    had_sync = False  # True once any event has been applied
    for ev in events:
        st.events += 1
        while True:
            if book.last_update_id is None:
                st.unsynced_events += 1
                break
            # verification: snapshot state equals the state just before this event
            while (
                vi < len(pending_verify)
                and book.valid
                and book.synced
                and pending_verify[vi].last_update_id < ev.U
            ):
                vs = pending_verify[vi]
                if vs.last_update_id == ev.U - 1 and book.last_update_id == vs.last_update_id:
                    _verify(book, vs, verify_levels, set(), set(), st)
                vi += 1
            res = book.apply(ev)
            if res is SyncStatus.APPLIED:
                while vi < len(pending_verify) and pending_verify[vi].last_update_id <= ev.u:
                    vs = pending_verify[vi]
                    if vs.last_update_id >= ev.U:
                        _verify(book, vs, verify_levels, {p for p, _ in ev.bids}, {p for p, _ in ev.asks}, st)
                    vi += 1
                st.applied += 1
                had_sync = True
                rows.append(book.state_row(k))
                break
            if res is SyncStatus.STALE:
                st.stale += 1
                break
            if res is SyncStatus.INTEGRITY:
                st.integrity_failures += 1
                break
            # GAP or SNAPSHOT_TOO_OLD (or invalid book): need a newer snapshot covering this event
            if res is SyncStatus.SNAPSHOT_TOO_OLD:
                st.snapshot_too_old += 1
            nxt = _next_usable(snaps, si, ev)
            if nxt is None:
                st.unsynced_events += 1
                break
            si = nxt + 1
            book.load_snapshot(snaps[nxt])
            st.snapshots_used += 1
            if had_sync:
                st.resyncs += 1
            # loop: retry this event against the new snapshot
    st.gaps = book.gaps
    return rows, st


def compare_with_snapshot(
    book: OrderBook,
    snap: DepthSnapshot,
    k: int = 20,
    exclude_bids: set[float] | None = None,
    exclude_asks: set[float] | None = None,
) -> tuple[int, int]:
    """Compare top-k levels of a snapshot (and of the book) with the book -> (checked, mismatches)."""
    exclude_bids = exclude_bids or set()
    exclude_asks = exclude_asks or set()
    snap_b = dict(snap.bids)
    snap_a = dict(snap.asks)
    sb = sorted(snap_b, reverse=True)[:k]
    sa = sorted(snap_a)[:k]
    # only compare book levels inside the snapshot's price coverage (a limit=1000 snapshot is truncated)
    bb = [p for p, _ in book.top_bids(k) if sb and p >= sb[-1]]
    ba = [p for p, _ in book.top_asks(k) if sa and p <= sa[-1]]
    checked = mism = 0
    for prices, mine, theirs, excl in (
        (set(sb) | set(bb), book.bids.qty, snap_b, exclude_bids),
        (set(sa) | set(ba), book.asks.qty, snap_a, exclude_asks),
    ):
        for p in prices - excl:
            checked += 1
            if mine.get(p, 0.0) != theirs.get(p, 0.0):
                mism += 1
    return checked, mism


def _verify(
    book: OrderBook, snap: DepthSnapshot, k: int, ex_b: set[float], ex_a: set[float], st: ReplayStats
) -> None:
    checked, mism = compare_with_snapshot(book, snap, k, ex_b, ex_a)
    st.verify_checks += 1
    st.verify_levels_checked += checked
    st.verify_mismatches += mism


def _next_usable(snaps: Sequence[DepthSnapshot], start: int, ev: DepthEvent) -> int | None:
    """Index of the next snapshot (fetch order) that this event or a later one can sync against."""
    for j in range(start, len(snaps)):
        if snaps[j].last_update_id + 1 >= ev.U:
            return j
    return None
