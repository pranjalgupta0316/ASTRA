"""Project-wide multiple-testing view (P7): global FDR, deflated Sharpe, probability calibration.

Why: every study controls false discoveries *within* its own family (BH), but ASTRA has run hundreds of
tests. A result that survives its family can still be one of the expected false positives of the whole
programme. This module re-reads the current (non-superseded) experiment records - nothing is re-run and
no record is changed - and reports:

1. Global ledger: one row per primary test, its p-value, and Benjamini-Hochberg (independence / positive
   dependence) and Benjamini-Yekutieli (any dependence) q-values across ALL current tests.
2. Deflated Sharpe ratio (Bailey & Lopez de Prado 2014, concept) for every net strategy equity curve,
   with n_trials = all strategy curves in the current ledger and the cross-trial Sharpe variance.
3. Calibration of stored out-of-sample probabilities: Brier score vs the base-rate forecast, expected
   calibration error (10 equal-count bins) and the logistic calibration slope.

These are descriptive re-assessments: registered verdicts stay as recorded; the report states which
locally significant results do not survive the global view.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from astra.stats.inference import deflated_sharpe_ratio
from astra.tracking.golden import _flatten
from astra.tracking.tracker import ExperimentRecord, Tracker, _cell_subject


@dataclass(frozen=True)
class PrimaryRule:
    """Which flattened metric keys of a hypothesis are its registered primary tests.
    kind 'p' = already a p-value; 't2' = a t/z statistic converted to a two-sided normal p-value."""

    hypothesis: str  # regex on hypothesis_id
    key: str  # regex on the flattened metric key
    kind: str  # "p" | "t2"


# Explicit registry (fixed before computing the global view). Diagnostics (H-XM-000 stationarity / Ljung-
# Box) and purely descriptive studies (H-MS-001/002 rates) are not hypothesis tests and are excluded.
PRIMARY_RULES: tuple[PrimaryRule, ...] = (
    PrimaryRule(r"^H-XM-00[12]$", r"^nw_t_diff$", "t2"),
    PrimaryRule(r"^H-XM-002$", r"^clf_ll_improvement_nw_t$", "t2"),
    PrimaryRule(r"^H-XM-003$", r"^p_value$|^p_two_sided$", "p"),
    PrimaryRule(r"^H-VOL-001$", r"^dm_p$", "p"),
    PrimaryRule(r"^H-REG-001$", r"^mw_p$", "p"),
    PrimaryRule(r"^H-ML-00[12]$", r"^dm_t_vs_baseline$", "t2"),
    PrimaryRule(r"^H-ML-00[34]$", r"^p_one_sided$", "p"),
    PrimaryRule(r"^H-CS-", r"^markets\.[A-Z]+\.p_two_sided$", "p"),
    PrimaryRule(r"^H-VT-", r"^p_two_sided$", "p"),
    # H-PF-001's registered decision is a bootstrap CI of the vol ratio (no p-value); its pre-registered
    # secondary NW t on r_EW^2 - r_s^2 stands in for it here. H-PF-002 is the NW t on return differences.
    PrimaryRule(r"^H-PF-001$", r"^tests\.[a-z_]+\.nw_t_sq_return_diff$", "t2"),
    PrimaryRule(r"^H-PF-002$", r"^tests\.[a-z_]+\.nw_t_return_diff$", "t2"),
    PrimaryRule(r"^H-XMKT-", r"^delta_sr_p_two_sided$", "p"),
    PrimaryRule(r"^H-RV-", r"^dm_p_qlike$", "p"),
    PrimaryRule(r"^H-MS-000$", r"^kyle_p_hac$", "p"),
)
EXCLUDED = (r"^H-XM-000$", r"^H-MS-00[12]")


def _primary_tests(rec: ExperimentRecord) -> list[dict[str, Any]]:
    flat = _flatten(rec.metrics)
    out = []
    for rule in PRIMARY_RULES:
        if not re.search(rule.hypothesis, rec.hypothesis_id):
            continue
        for k, v in flat.items():
            if not re.search(rule.key, k) or not isinstance(v, int | float) or isinstance(v, bool):
                continue
            if v is None or not math.isfinite(float(v)):
                continue
            p = float(v) if rule.kind == "p" else float(2 * stats.norm.sf(abs(float(v))))
            out.append({"metric": k, "p": min(max(p, 0.0), 1.0), "kind": rule.kind})
    return out


def global_ledger(tracker: Tracker | None = None, q: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tests, unmapped): tests = one row per primary test of every current record with global q-values;
    unmapped = current records that contributed no test (family summaries, diagnostics, descriptive)."""
    tr = tracker or Tracker()
    rows, unmapped = [], []
    for rec in tr.current():
        if any(re.search(x, rec.hypothesis_id) for x in EXCLUDED):
            unmapped.append(
                {
                    "experiment_id": rec.experiment_id,
                    "hypothesis_id": rec.hypothesis_id,
                    "name": rec.name,
                    "reason": "diagnostic/descriptive (excluded by rule)",
                }
            )
            continue
        tests = _primary_tests(rec)
        if not tests:
            unmapped.append(
                {
                    "experiment_id": rec.experiment_id,
                    "hypothesis_id": rec.hypothesis_id,
                    "name": rec.name,
                    "reason": "no primary test key in record",
                }
            )
        for t in tests:
            rows.append(
                {
                    "experiment_id": rec.experiment_id,
                    "hypothesis_id": rec.hypothesis_id,
                    "name": rec.name,
                    "subject": _cell_subject(rec),
                    "local_conclusion": rec.conclusion,
                    **t,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(unmapped)
    # the same test can appear in a cell record and its family summary: count it once
    df = df.drop_duplicates(["hypothesis_id", "subject", "metric", "p"]).reset_index(drop=True)
    p = df["p"].to_numpy()
    df["q_bh_global"] = multipletests(p, alpha=q, method="fdr_bh")[1]
    df["q_by_global"] = multipletests(p, alpha=q, method="fdr_by")[1]
    df["sig_raw"] = p < q
    df["sig_bh_global"] = df["q_bh_global"] < q
    df["sig_by_global"] = df["q_by_global"] < q
    return df.sort_values("p").reset_index(drop=True), pd.DataFrame(unmapped)


# ----------------------------------------------------------------------------- deflated Sharpe

_BENCH = re.compile(r"^(bh_|unmanaged_|benchmark_|bottom_)|_gross$|gross_|plus_costs")
_NET = re.compile(r"(^|_)equity_net$|^strategy_net$")


def _equity_curves(tr: Tracker) -> list[dict[str, Any]]:
    out = []
    for rec in tr.current():
        for a in rec.artifacts:
            path = Path(a) if Path(a).is_absolute() else tr.paths.root / a
            if path.name != "equity.parquet" or not path.exists():
                continue
            eq = pd.read_parquet(path)
            for col in eq.columns:
                if _NET.search(col) and not _BENCH.search(col):
                    r = eq[col].astype(float).pct_change().dropna()
                    r = r[np.isfinite(r)]
                    if len(r) > 30 and r.std() > 0:
                        out.append(
                            {
                                "experiment_id": rec.experiment_id,
                                "hypothesis_id": rec.hypothesis_id,
                                "name": rec.name,
                                "subject": _cell_subject(rec),
                                "curve": col,
                                "returns": r,
                            }
                        )
    return out


def deflated_sharpe_table(tracker: Tracker | None = None) -> pd.DataFrame:
    """DSR of every current net strategy curve, deflated by the number of strategy trials in the
    programme and the cross-trial variance of their per-period Sharpe ratios."""
    tr = tracker or Tracker()
    curves = _equity_curves(tr)
    if not curves:
        return pd.DataFrame()
    srs = np.array([c["returns"].mean() / c["returns"].std(ddof=1) for c in curves])
    n_trials, var = len(curves), float(np.var(srs, ddof=1)) if len(curves) > 1 else None
    rows = []
    for c, sr in zip(curves, srs, strict=True):
        r = c.pop("returns")
        rows.append(
            {
                **c,
                "n_obs": len(r),
                "sr_per_period": float(sr),
                "sr_annual_365": float(sr * math.sqrt(365)),
                "sr_annual_252": float(sr * math.sqrt(252)),
                "dsr": deflated_sharpe_ratio(r, n_trials=n_trials, trial_sr_variance=var),
                "n_trials": n_trials,
            }
        )
    return pd.DataFrame(rows).sort_values("dsr", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- calibration


def calibration_metrics(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Brier, base-rate Brier, Brier skill, ECE (equal-count bins) and logistic calibration slope."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean((y.mean() - y) ** 2))
    order = np.argsort(p, kind="stable")
    ece = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            ece += len(idx) / len(p) * abs(p[idx].mean() - y[idx].mean())
    logit = np.log(p / (1 - p))
    import statsmodels.api as sm

    try:
        fit = sm.Logit(y, sm.add_constant(logit)).fit(disp=0)
        slope = float(fit.params[1])
    except Exception:  # perfect separation / degenerate predictions
        slope = float("nan")
    return {
        "n": float(len(p)),
        "base_rate": float(y.mean()),
        "brier": brier,
        "brier_base_rate": base,
        "brier_skill": 1 - brier / base if base > 0 else float("nan"),
        "ece": float(ece),
        "calibration_slope": slope,
    }


def calibration_table(tracker: Tracker | None = None) -> pd.DataFrame:
    tr = tracker or Tracker()
    rows = []
    for rec in tr.current():
        for a in rec.artifacts:
            path = Path(a) if Path(a).is_absolute() else tr.paths.root / a
            if path.name != "oos_predictions.parquet" or not path.exists():
                continue
            pr = pd.read_parquet(path)
            ycol = next((c for c in ("y", "target_up", "target") if c in pr.columns), None)
            if ycol is None:
                continue
            named = ("proba", "baseline_proba", "baseline")
            pcols = [c for c in pr.columns if c.startswith("p_") or c in named]
            for pc in pcols:
                m = calibration_metrics(pr[ycol].to_numpy(), pr[pc].to_numpy())
                rows.append(
                    {
                        "experiment_id": rec.experiment_id,
                        "hypothesis_id": rec.hypothesis_id,
                        "subject": _cell_subject(rec),
                        "prediction": pc,
                        **m,
                    }
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)  # noqa: E731
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_report(out_dir: Path | None = None, tracker: Tracker | None = None) -> dict[str, Any]:
    """Compute all three views, save parquet tables and a markdown summary; returns headline counts."""
    tr = tracker or Tracker()
    out = out_dir or tr.paths.data / "reports" / "research"
    out.mkdir(parents=True, exist_ok=True)
    tests, unmapped = global_ledger(tr)
    dsr = deflated_sharpe_table(tr)
    cal = calibration_table(tr)
    for name, df in (
        ("global_tests", tests),
        ("global_unmapped", unmapped),
        ("deflated_sharpe", dsr),
        ("calibration", cal),
    ):
        df.to_parquet(out / f"multiplicity_{name}.parquet")
    st = tr.status()
    lost = tests
    if len(tests):
        lost = tests.loc[(tests["local_conclusion"] == "supported") & ~tests["sig_bh_global"]]
    head = {
        "records_total": len(st),
        "records_current": int((~st["superseded"]).sum()) if len(st) else 0,
        "tests": len(tests),
        "sig_raw": int(tests["sig_raw"].sum()) if len(tests) else 0,
        "sig_bh_global": int(tests["sig_bh_global"].sum()) if len(tests) else 0,
        "sig_by_global": int(tests["sig_by_global"].sum()) if len(tests) else 0,
        "supported_tests_not_surviving_global_bh": len(lost),
        "strategy_curves": len(dsr),
        "dsr_above_0_95": int((dsr["dsr"] > 0.95).sum()) if len(dsr) else 0,
    }
    lines = [
        "# Global multiplicity report (generated; descriptive - registered verdicts unchanged)",
        "",
        *(f"- {k}: {v}" for k, v in head.items()),
        "",
        "## Tests that are significant within their family but not after global BH",
        "",
        _md(lost[["hypothesis_id", "subject", "metric", "p", "q_bh_global"]]) if len(lost) else "(none)",
        "",
        "## Top deflated Sharpe ratios",
        "",
        _md(dsr.head(15)[["hypothesis_id", "name", "curve", "sr_annual_252", "dsr"]])
        if len(dsr)
        else "(none)",
        "",
        "## Calibration",
        "",
        _md(cal.round(4)),
        "",
    ]
    (out / "multiplicity.md").write_text("\n".join(lines), encoding="utf-8")
    return head
