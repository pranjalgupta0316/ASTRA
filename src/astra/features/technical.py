"""Causal technical / statistical features for daily bars.

Every feature at row t uses only data with close_time_utc <= close of t (verified by the
truncation-invariance test in tests/test_features.py). Targets are forward-looking by definition and
are kept in separately named `target_*` columns.

Price basis: total-return adjusted. adj_factor = adj_close / close (Yahoo close is split-adjusted
only). The factor multiplies O/H/L. Using today's factor for history does not change returns between
corporate-action dates, so returns carry no look-ahead; absolute adjusted price LEVELS do embed later
dividends and must not be used as features (we only use ratios/returns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1",
    "mom_5",
    "mom_20",
    "mom_60",
    "sma_ratio_20",
    "sma_ratio_50",
    "vol_20",
    "parkinson_20",
    "rsi_14",
    "macd_hist_norm",
    "bb_z_20",
    "volume_z_20",
    "autocorr_20",
]


def adjusted_ohlc(bars: pd.DataFrame) -> pd.DataFrame:
    factor = (bars["adj_close"] / bars["close"]).fillna(1.0)
    # A non-positive adjustment factor is a provider back-adjustment error (seen: ADANIENT 2002, Yahoo
    # adj_close < 0). Those rows become missing prices instead of producing negative/invalid returns.
    factor = factor.where(factor > 0)
    out = pd.DataFrame(index=bars.index)
    out["session_date"] = pd.to_datetime(bars["session_date"])
    out["close_time_utc"] = bars["close_time_utc"]
    for c in ("open", "high", "low", "close"):
        out[c] = bars[c] * factor
    out["volume"] = bars["volume"]
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI (alpha = 1/n smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_f = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def parkinson_var(high: pd.Series, low: pd.Series) -> pd.Series:
    return (np.log(high / low) ** 2) / (4.0 * np.log(2.0))


def garman_klass_var(o: pd.Series, h: pd.Series, lo: pd.Series, c: pd.Series) -> pd.Series:
    return 0.5 * np.log(h / lo) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Return a frame indexed like `bars` with feature and target columns."""
    adj = adjusted_ohlc(bars).reset_index(drop=True)
    px = adj["close"]
    logp = np.log(px)
    f = pd.DataFrame({"session_date": adj["session_date"], "close_time_utc": adj["close_time_utc"]})
    f["ret_1"] = logp.diff()
    for k in (5, 20, 60):
        f[f"mom_{k}"] = logp - logp.shift(k)
    for k in (20, 50):
        f[f"sma_ratio_{k}"] = px / px.rolling(k, min_periods=k).mean() - 1.0
    f["vol_20"] = f["ret_1"].rolling(20, min_periods=20).std()
    f["parkinson_20"] = np.sqrt(parkinson_var(adj["high"], adj["low"]).rolling(20, min_periods=20).mean())
    f["rsi_14"] = rsi(px, 14)
    m = macd(px)
    f["macd_hist_norm"] = m["macd_hist"] / px
    sma20 = px.rolling(20, min_periods=20).mean()
    sd20 = px.rolling(20, min_periods=20).std()
    f["bb_z_20"] = (px - sma20) / sd20
    logv = np.log(adj["volume"].replace(0, np.nan))
    f["volume_z_20"] = (logv - logv.rolling(20, min_periods=20).mean()) / logv.rolling(
        20, min_periods=20
    ).std()
    f["autocorr_20"] = f["ret_1"].rolling(20, min_periods=20).corr(f["ret_1"].shift(1))

    # Targets. Decision at close t -> execute at open t+1 -> hold to open t+2.
    f["target_ret_oo"] = np.log(adj["open"].shift(-2) / adj["open"].shift(-1))
    f["target_ret_cc"] = f["ret_1"].shift(-1)
    f["target_up"] = (f["target_ret_oo"] > 0).astype("float").where(f["target_ret_oo"].notna())
    f["adj_open"] = adj["open"]
    f["adj_close"] = adj["close"]
    f["volume"] = adj["volume"]
    return f
