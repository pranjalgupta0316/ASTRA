"""Model-ready datasets: per-instrument and pooled panels, plus causal lagged (sequence) matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nebula.common.config import get_config
from nebula.common.paths import Paths, get_paths
from nebula.data.storage import read_bars
from nebula.features.technical import FEATURE_COLUMNS, build_features


def load_features(
    symbol: str,
    paths: Paths | None = None,
    window: tuple[str | None, str | None] | None = None,
    freq: str = "1d",
) -> pd.DataFrame:
    """Features for one instrument with provider placeholder rows excluded (V == 0 and H == L;
    for indices, whose Yahoo volume is unreliable, H == L only)."""
    inst = get_config().instrument(symbol)
    bars = read_bars(symbol, inst.market, paths or get_paths(), freq=freq, window=window)
    flat = bars["high"] == bars["low"]
    placeholder = flat if inst.kind == "index" else flat & (bars["volume"] == 0)
    bars = bars.loc[~placeholder].reset_index(drop=True)
    f = build_features(bars)
    f["symbol"] = symbol
    return f


def lagged_matrix(feats: pd.DataFrame, columns: list[str], seq_len: int) -> pd.DataFrame:
    """Row t = [x_{t-L+1}, ..., x_t] flattened (oldest first). Uses only past rows -> causal."""
    blocks = [feats[columns].shift(k).add_suffix(f"_lag{k}") for k in range(seq_len - 1, -1, -1)]
    return pd.concat(blocks, axis=1)


def model_frame(feats: pd.DataFrame, seq_len: int = 10) -> pd.DataFrame:
    """Rows with a complete feature window and a defined target; columns: flat + lagged features."""
    lag = lagged_matrix(feats, FEATURE_COLUMNS, seq_len)
    df = pd.concat(
        [feats[["session_date", "symbol", "target_up", "target_ret_oo"]], feats[FEATURE_COLUMNS], lag], axis=1
    )
    return df.dropna().reset_index(drop=True)


def lag_columns(seq_len: int = 10) -> list[str]:
    return [f"{c}_lag{k}" for k in range(seq_len - 1, -1, -1) for c in FEATURE_COLUMNS]


def panel(
    symbols: list[str],
    seq_len: int = 10,
    paths: Paths | None = None,
    window: tuple[str | None, str | None] | None = None,
    freq: str = "1d",
) -> pd.DataFrame:
    frames = [model_frame(load_features(s, paths, window, freq), seq_len) for s in symbols]
    df = pd.concat(frames, ignore_index=True)
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def date_folds(dates: pd.Series, min_train_dates: int, test_dates: int, gap_dates: int):
    """Date-based purged expanding walk-forward for pooled panels: all rows of a date share a fold.
    Yields (train_row_idx, test_row_idx)."""
    udates = np.array(sorted(pd.unique(dates)))
    d = dates.to_numpy()
    start = min_train_dates
    while start < len(udates):
        stop = min(start + test_dates, len(udates))
        train_cut = udates[start - gap_dates] if start - gap_dates > 0 else None
        if train_cut is not None:
            tr = np.flatnonzero(d < train_cut)
            te = np.flatnonzero((d >= udates[start]) & (d <= udates[stop - 1]))
            if len(tr) and len(te):
                yield tr, te
        start = stop
