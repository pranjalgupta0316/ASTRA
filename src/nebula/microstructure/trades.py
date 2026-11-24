"""Binance Public Data spot aggTrades: download (sha256-verified), parse, interval bars, price impact.

aggTrades CSV columns: agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time,
is_buyer_maker, is_best_match. Files may or may not carry a header row. `transact_time` is ms up to
2024-12-31 and microseconds from 2025-01-01 for spot - detected by magnitude.

Aggressor side: `is_buyer_maker = True` means the buyer's order was resting (maker), so the **seller was
the aggressor** -> aggressive SELL (sign -1). False -> aggressive BUY (+1).

These are trades only (no quotes): VWAP changes are used as a price proxy, not mid changes.
"""

from __future__ import annotations

import json
import math
import tempfile
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import requests

from nebula.common.logging import get_logger
from nebula.common.paths import get_paths
from nebula.data.download import checked_download, sha256_of

log = get_logger(__name__)

BASE = "https://data.binance.vision/data/spot/daily/aggTrades"
USER_AGENT = "nebula-research/0.1 (+local research; respects rate limits)"
REQUEST_PAUSE_S = 0.25
TIMEOUT_S = 60
AGG_COLS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
]
US_THRESHOLD = 10**14  # epoch values above this are microseconds (1e14 ms would be year 5138)


# ------------------------------------------------------------------ helpers
def detect_epoch_unit(values: Any) -> str:
    """'us' if epoch integers are microseconds, 'ms' if milliseconds (decided by magnitude)."""
    v = np.asarray(values, dtype=np.int64)
    if v.size == 0:
        raise ValueError("empty timestamp array")
    us = v > US_THRESHOLD
    if us.all():
        return "us"
    if not us.any():
        return "ms"
    raise ValueError("mixed ms/us timestamps in one array")


def epoch_to_us_expr(col: str) -> pl.Expr:
    """Polars expression normalizing ms-or-us epoch ints (per row, by magnitude) to microseconds."""
    c = pl.col(col).cast(pl.Int64)
    return pl.when(c > US_THRESHOLD).then(c).otherwise(c * 1000)


def aggressor_sign(is_buyer_maker: bool) -> int:
    """+1 aggressive buy, -1 aggressive sell. isBuyerMaker True -> seller aggressed -> -1."""
    return -1 if is_buyer_maker else 1


def aggtrades_url(symbol: str, day: date) -> str:
    return f"{BASE}/{symbol}/{symbol}-aggTrades-{day.isoformat()}.zip"


def raw_dir(symbol: str) -> Path:
    return get_paths().raw / "binance_vision" / symbol / "aggTrades"


# ------------------------------------------------------------------ download
def download_aggtrades(
    symbol: str, day: date, dest_dir: Path | None = None, session: requests.Session | None = None
) -> Path | None:
    """Download one daily aggTrades zip, verify its published sha256, write a sidecar metadata JSON.

    Returns the zip path, or None if the file is not published (404). Existing files are reused
    (raw layer is immutable).
    """
    dest_dir = dest_dir or raw_dir(symbol)
    url = aggtrades_url(symbol, day)
    dest = dest_dir / url.rsplit("/", 1)[-1]
    http = session or requests.Session()
    http.headers["User-Agent"] = USER_AGENT
    existed = dest.exists()
    if checked_download(http, url, dest) is None:  # streamed, sha256-verified; cache re-verified
        return None
    if existed and dest.with_suffix(".meta.json").exists():
        return dest
    actual = sha256_of(dest)
    meta = {
        "provider": "binance_vision",
        "dataset": "spot/daily/aggTrades",
        "url": url,
        "symbol": symbol,
        "day": day.isoformat(),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "sha256": actual,
        "checksum_verified": True,
        "bytes": dest.stat().st_size,
    }
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("aggtrades.download", url=url, bytes=meta["bytes"])
    return dest


def download_latest_complete(
    symbol: str, today: date | None = None, max_back: int = 2, session: requests.Session | None = None
) -> tuple[date, Path]:
    """Most recent complete UTC day published: try yesterday, then the day before (up to max_back)."""
    today = today or datetime.now(UTC).date()
    for back in range(1, max_back + 1):
        d = today - timedelta(days=back)
        p = download_aggtrades(symbol, d, session=session)
        if p is not None:
            return d, p
    raise RuntimeError(f"No aggTrades file published for {symbol} in the last {max_back} days")


# ------------------------------------------------------------------ parsing
def _has_header(first_line: str) -> bool:
    return first_line.strip().lower().startswith("agg_trade_id")


def scan_aggtrades_csv(csv_path: Path) -> pl.LazyFrame:
    """Lazy scan of an extracted aggTrades CSV -> normalized columns.

    Output: ts (Datetime us, UTC), agg_trade_id, price, qty, is_buyer_maker (bool), side (+1/-1).
    """
    with csv_path.open("r", encoding="utf-8") as fh:
        first = fh.readline()
    header = _has_header(first)
    lf = pl.scan_csv(
        csv_path,
        has_header=header,
        new_columns=AGG_COLS,
        schema_overrides={
            "agg_trade_id": pl.Int64,
            "price": pl.Float64,
            "quantity": pl.Float64,
            "first_trade_id": pl.Int64,
            "last_trade_id": pl.Int64,
            "transact_time": pl.Int64,
            "is_buyer_maker": pl.String,
            "is_best_match": pl.String,
        },
    )
    ibm = pl.col("is_buyer_maker").str.to_lowercase() == "true"
    return lf.select(
        epoch_to_us_expr("transact_time").cast(pl.Datetime("us", "UTC")).alias("ts"),
        pl.col("agg_trade_id"),
        pl.col("price"),
        pl.col("quantity").alias("qty"),
        ibm.alias("is_buyer_maker"),
        pl.when(ibm).then(-1).otherwise(1).cast(pl.Int8).alias("side"),
    )


def extract_csv(zip_path: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"expected exactly one CSV in {zip_path}, found {names}")
        name = Path(names[0]).name  # never trust archive paths
        target = out_dir / name
        with zf.open(names[0]) as src, target.open("wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    return target


# ------------------------------------------------------------------ bars
def trade_bars(trades: pl.LazyFrame, every: str) -> pl.DataFrame:
    """Per-interval trade bars (empty intervals are absent).

    Columns: ts (interval start), n_trades, volume, quote_volume, vwap, buy_volume, sell_volume,
    signed_volume, trade_imbalance, first_price, last_price, realized_vol (sqrt of summed squared
    trade-to-trade log returns inside the interval; includes bid-ask bounce).
    """
    t = trades.sort("ts", "agg_trade_id").with_columns(
        pl.col("price").log().diff().alias("_r"),
        pl.col("ts").dt.truncate(every).alias("_bucket"),
    )
    # trade-to-trade return only within the same bucket
    t = t.with_columns(
        pl.when(pl.col("_bucket") == pl.col("_bucket").shift(1))
        .then(pl.col("_r"))
        .otherwise(None)
        .alias("_r")
    )
    buy = pl.when(pl.col("side") > 0).then(pl.col("qty")).otherwise(0.0)
    sell = pl.when(pl.col("side") < 0).then(pl.col("qty")).otherwise(0.0)
    out = (
        t.group_by("_bucket")
        .agg(
            pl.len().alias("n_trades"),
            pl.col("qty").sum().alias("volume"),
            (pl.col("price") * pl.col("qty")).sum().alias("quote_volume"),
            buy.sum().alias("buy_volume"),
            sell.sum().alias("sell_volume"),
            pl.col("price").first().alias("first_price"),
            pl.col("price").last().alias("last_price"),
            (pl.col("_r") ** 2).sum().sqrt().alias("realized_vol"),
        )
        .rename({"_bucket": "ts"})
        .with_columns(
            (pl.col("quote_volume") / pl.col("volume")).alias("vwap"),
            (pl.col("buy_volume") - pl.col("sell_volume")).alias("signed_volume"),
            ((pl.col("buy_volume") - pl.col("sell_volume")) / pl.col("volume")).alias("trade_imbalance"),
        )
        .sort("ts")
        .collect()
    )
    return out


def kyle_lambda(bars: pl.DataFrame, interval: timedelta = timedelta(minutes=1)) -> dict[str, float]:
    """OLS of dP_t (bps, 1e4 * d log VWAP between consecutive intervals) on signed volume_t, HAC SE.

    Only pairs of adjacent intervals are used. Newey-West maxlags = floor(4 (n/100)^(2/9)).
    Reduced-form, contemporaneous: impact association, not a causal or tradable estimate.
    """
    import statsmodels.api as sm

    df = bars.sort("ts").with_columns(
        (1e4 * pl.col("vwap").log().diff()).alias("dp_bps"),
        (pl.col("ts").diff() == pl.lit(interval)).alias("_adj"),
    )
    df = df.filter(pl.col("_adj") & pl.col("dp_bps").is_not_null())
    y = df["dp_bps"].to_numpy()
    x = df["signed_volume"].to_numpy()
    n = len(y)
    if n < 30:
        return {"n": float(n), "lambda_bps_per_unit": float("nan"), "t_hac": float("nan")}
    lags = math.floor(4 * (n / 100) ** (2 / 9))
    res = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    params, bse, tv, pv = (np.asarray(a) for a in (res.params, res.bse, res.tvalues, res.pvalues))
    return {
        "n": float(n),
        "hac_maxlags": float(lags),
        "intercept_bps": float(params[0]),
        "lambda_bps_per_unit": float(params[1]),
        "lambda_se_hac": float(bse[1]),
        "t_hac": float(tv[1]),
        "p_hac": float(pv[1]),
        "r2": float(res.rsquared),
    }


def aggtrades_summary(zip_path: Path) -> dict[str, Any]:
    """Parse one daily zip lazily and return {'bars_1s', 'bars_1m', 'stats', 'kyle'}."""
    with tempfile.TemporaryDirectory(prefix="nebula_aggtrades_") as tmp:
        csv = extract_csv(zip_path, Path(tmp))
        with csv.open("r", encoding="utf-8") as fh:
            header = _has_header(fh.readline())
        lf = scan_aggtrades_csv(csv)
        raw_ts = (
            pl.scan_csv(
                csv, has_header=header, new_columns=AGG_COLS, schema_overrides={"transact_time": pl.Int64}
            )
            .select(pl.col("transact_time"))
            .head(1000)
            .collect()["transact_time"]
            .to_numpy()
        )
        unit = detect_epoch_unit(raw_ts)
        totals = lf.select(
            pl.len().alias("n_trades"),
            pl.col("qty").sum().alias("volume"),
            (pl.col("price") * pl.col("qty")).sum().alias("quote_volume"),
            pl.col("ts").min().alias("first_ts"),
            pl.col("ts").max().alias("last_ts"),
            (pl.col("side") > 0).mean().alias("buy_trade_share"),
            pl.when(pl.col("side") > 0).then(pl.col("qty")).otherwise(0.0).sum().alias("buy_volume"),
            pl.col("agg_trade_id").is_duplicated().sum().alias("dup_ids"),
            (pl.col("agg_trade_id").max() - pl.col("agg_trade_id").min() + 1 - pl.len()).alias("id_holes"),
        ).collect()
        bars_1s = trade_bars(lf, "1s")
        bars_1m = trade_bars(lf, "1m")
    row = totals.row(0, named=True)
    r1m = np.diff(np.log(bars_1m["vwap"].to_numpy()))
    stats = {
        "timestamp_unit": unit,
        "n_trades": int(row["n_trades"]),
        "volume": float(row["volume"]),
        "quote_volume": float(row["quote_volume"]),
        "first_ts": str(row["first_ts"]),
        "last_ts": str(row["last_ts"]),
        "buy_trade_share": float(row["buy_trade_share"]),
        "buy_volume_share": float(row["buy_volume"]) / float(row["volume"]),
        "duplicate_agg_ids": int(row["dup_ids"]),
        "agg_id_holes": int(row["id_holes"]),
        "seconds_with_trades": int(bars_1s.height),
        "minutes_with_trades": int(bars_1m.height),
        "daily_rv_1m_vwap_bps": float(1e4 * np.sqrt(np.sum(r1m**2))),
        "imbalance_1m_mean": float(bars_1m["trade_imbalance"].mean()),  # type: ignore[arg-type]
        "imbalance_1m_std": float(bars_1m["trade_imbalance"].std()),  # type: ignore[arg-type]
        "imbalance_1s_std": float(bars_1s["trade_imbalance"].std()),  # type: ignore[arg-type]
        "median_trades_per_1s_bar": float(bars_1s["n_trades"].median()),  # type: ignore[arg-type]
    }
    return {"bars_1s": bars_1s, "bars_1m": bars_1m, "stats": stats, "kyle": kyle_lambda(bars_1m)}
