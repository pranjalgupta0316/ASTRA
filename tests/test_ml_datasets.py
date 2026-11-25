"""Dataset construction and walk-forward helpers (no torch required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from astra.features.technical import FEATURE_COLUMNS, build_features
from astra.ml.classical import walk_forward_from_splits
from astra.ml.datasets import date_folds, lag_columns, lagged_matrix, model_frame


def test_lagged_matrix_is_causal(us_bars):
    f = build_features(us_bars)
    lag = lagged_matrix(f, FEATURE_COLUMNS, 5)
    # row t, newest block equals features at t; oldest block equals features at t-4
    t = 150
    np.testing.assert_allclose(
        lag.loc[t, [f"{c}_lag0" for c in FEATURE_COLUMNS]].to_numpy(dtype=float),
        f.loc[t, FEATURE_COLUMNS].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        lag.loc[t, [f"{c}_lag4" for c in FEATURE_COLUMNS]].to_numpy(dtype=float),
        f.loc[t - 4, FEATURE_COLUMNS].to_numpy(dtype=float),
    )
    assert lag_columns(5)[: len(FEATURE_COLUMNS)] == [f"{c}_lag4" for c in FEATURE_COLUMNS]  # oldest first


def test_model_frame_has_no_nans_and_targets(us_bars):
    f = build_features(us_bars)
    f["symbol"] = "TEST"
    df = model_frame(f, 10)
    assert not df.isna().any().any() and set(df["target_up"].unique()) <= {0.0, 1.0}


def test_date_folds_purge_gap_and_no_overlap():
    dates = pd.Series(np.repeat(pd.date_range("2020-01-01", periods=100, freq="D"), 3))
    folds = list(date_folds(dates, min_train_dates=40, test_dates=20, gap_dates=7))
    assert len(folds) == 3
    for tr, te in folds:
        last_train, first_test = dates.iloc[tr].max(), dates.iloc[te].min()
        assert (first_test - last_train).days >= 8  # 7 purged dates in between
        assert len(set(tr) & set(te)) == 0
        # every row of a test date is in the same fold
        assert dates.iloc[te].value_counts().eq(3).all()


def test_walk_forward_from_splits_baseline_is_ex_ante():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    X = np.zeros((8, 1))
    splits = [(np.arange(0, 4), np.arange(4, 8))]

    class Const:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

    _, base, _ = walk_forward_from_splits(X, y, Const, splits)
    assert np.all(base[4:] == 1.0)  # train-fold base rate, not the test-period rate (0.0)
