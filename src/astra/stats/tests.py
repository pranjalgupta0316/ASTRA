"""Stationarity and serial-dependence tests with explicit interpretation."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity(series: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    x = pd.Series(series).dropna().astype(float)
    adf_stat, adf_p, *_ = adfuller(x, autolag="AIC", result_object=False)  # type: ignore[call-arg]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)  # p-value clipped at table bounds
        kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto", result_object=False)  # type: ignore[call-arg]
    adf_rej, kpss_rej = adf_p < alpha, kpss_p < alpha
    if adf_rej and not kpss_rej:
        verdict = "stationary"
    elif not adf_rej and kpss_rej:
        verdict = "unit_root"
    elif adf_rej and kpss_rej:
        verdict = "conflicting (possible breaks / long memory)"
    else:
        verdict = "inconclusive (low power)"
    return {
        "n": len(x),
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "kpss_p_note": "statsmodels clips KPSS p-values to [0.01, 0.10]",
        "verdict": verdict,
    }


def ljung_box(returns: pd.Series, lags: int = 10) -> dict[str, float]:
    x = pd.Series(returns).dropna().astype(float)
    res = acorr_ljungbox(x, lags=[lags], return_df=True)
    res_sq = acorr_ljungbox(x**2, lags=[lags], return_df=True)
    return {
        "lags": lags,
        "q_stat": float(res["lb_stat"].iloc[0]),
        "p_value": float(res["lb_pvalue"].iloc[0]),
        "q_stat_squared": float(res_sq["lb_stat"].iloc[0]),
        "p_value_squared": float(res_sq["lb_pvalue"].iloc[0]),
    }


def describe_returns(returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    x = pd.Series(returns).dropna().astype(float)
    return {
        "n": len(x),
        "mean_ann": float(x.mean() * periods_per_year),
        "vol_ann": float(x.std(ddof=1) * np.sqrt(periods_per_year)),
        "skew": float(x.skew()),
        "excess_kurtosis": float(x.kurt()),
        "acf_1": float(x.autocorr(1)),
        "min": float(x.min()),
        "max": float(x.max()),
    }
