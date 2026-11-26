"""P7 global multiplicity + P9 superseded flag, on synthetic records in a temporary tracker."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from astra.common.paths import Paths
from astra.research.multiplicity import (
    calibration_metrics,
    deflated_sharpe_table,
    global_ledger,
)
from astra.tracking.tracker import Tracker


@pytest.fixture
def tr(tmp_path) -> Tracker:
    p = Paths(root=tmp_path)
    p.ensure()
    return Tracker(p)


def _log(tr: Tracker, hid: str, sym: str, metrics: dict, name: str = "cell", conclusion="inconclusive"):
    rec = tr.new(
        name,
        hid,
        "h",
        datasets=[{"symbol": sym, "market": "US", "freq": "1d"}],
        metrics=metrics,
        conclusion=conclusion,
    )
    tr.log(rec)
    time.sleep(0.002)  # distinct created_at ordering
    return rec


def test_superseded_is_computed_per_cell_and_records_untouched(tr):
    old = _log(tr, "H-VT-001", "AAPL", {"p_two_sided": 0.5})
    new = _log(tr, "H-VT-001", "AAPL", {"p_two_sided": 0.01})
    other = _log(tr, "H-VT-001", "MSFT", {"p_two_sided": 0.2})
    st = tr.status().set_index("experiment_id")
    assert st.loc[old.experiment_id, "superseded"]
    assert st.loc[old.experiment_id, "superseded_by"] == new.experiment_id
    assert not st.loc[new.experiment_id, "superseded"] and not st.loc[other.experiment_id, "superseded"]
    assert {r.experiment_id for r in tr.current()} == {new.experiment_id, other.experiment_id}
    assert tr.load(old.experiment_id).metrics == {"p_two_sided": 0.5}  # never rewritten


def test_one_symbol_at_two_frequencies_is_one_subject(tr):
    for model in ("a", "b"):
        rec = tr.new(
            f"rv_{model}",
            "H-RV-001",
            "h",
            metrics={"dm_p_qlike": 0.3},
            datasets=[{"symbol": "ETHUSDT", "market": "CRYPTO", "freq": f} for f in ("1m", "1d")],
        )
        tr.log(rec)
    rec = tr.new(
        "rv_a",
        "H-RV-001",
        "h",
        metrics={"dm_p_qlike": 0.3},
        datasets=[{"symbol": "BTCUSDT", "market": "CRYPTO", "freq": f} for f in ("1m", "1d")],
    )
    tr.log(rec)
    assert len(tr.current()) == 3  # BTC and ETH cells are not collapsed into "CRYPTO"


def test_global_ledger_q_values_and_rules(tr):
    ps = [0.001, 0.01, 0.03, 0.2, 0.9]
    for i, p in enumerate(ps):
        _log(
            tr,
            "H-VT-001",
            f"S{i}",
            {"p_two_sided": p},
            conclusion="supported" if p < 0.05 else "inconclusive",
        )
    _log(tr, "H-XM-001", "AAPL", {"nw_t_diff": 2.5})  # t -> two-sided normal p
    _log(tr, "H-XM-000", "AAPL", {"ljung_box": {"p_value": 1e-9}})  # diagnostic: excluded
    _log(tr, "H-VT-001", "FAM", {"something": 1.0}, name="family")  # no primary key
    tests, unmapped = global_ledger(tr)
    assert len(tests) == 6
    t_row = tests.loc[tests["hypothesis_id"] == "H-XM-001"].iloc[0]
    assert t_row["p"] == pytest.approx(2 * stats.norm.sf(2.5))
    expected = multipletests(tests["p"].to_numpy(), method="fdr_bh")[1]
    np.testing.assert_allclose(tests["q_bh_global"], expected)
    assert (tests["q_by_global"] >= tests["q_bh_global"] - 1e-15).all()  # BY is more conservative
    assert set(unmapped["hypothesis_id"]) == {"H-XM-000", "H-VT-001"}


def test_calibration_metrics_on_calibrated_and_constant_forecasts():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    m = calibration_metrics(y, p)
    assert m["ece"] < 0.01 and m["calibration_slope"] == pytest.approx(1.0, abs=0.03)
    assert m["brier_skill"] > 0.1
    base = calibration_metrics(y, np.full_like(p, y.mean()))
    assert base["brier_skill"] == pytest.approx(0.0, abs=1e-9)


def test_deflated_sharpe_ranks_a_real_edge_above_noise(tr):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=2000)
    for i in range(12):
        drift = 0.0015 if i == 0 else 0.0
        rec = tr.new(f"strat{i}", "H-VT-001", "h", datasets=[{"symbol": f"S{i}", "market": "US"}])
        eq = pd.DataFrame(
            {
                "managed_equity_net": np.cumprod(1 + rng.normal(drift, 0.01, len(idx))),
                "unmanaged_equity_net": np.cumprod(1 + rng.normal(0.003, 0.01, len(idx))),
            },
            index=idx,
        )
        tr.save_frame(rec, "equity", eq)
        tr.log(rec)
    d = deflated_sharpe_table(tr)
    assert len(d) == 12 and set(d["curve"]) == {"managed_equity_net"}  # benchmark curves excluded
    assert d.iloc[0]["name"] == "strat0" and d.iloc[0]["dsr"] > 0.95
    assert (d["n_trials"] == 12).all()
