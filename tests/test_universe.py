"""P8 point-in-time S&P 500 membership: parsing and backward reconstruction (offline, synthetic)."""

from __future__ import annotations

import pandas as pd

from astra.data.universe import members_on, parse_changes, reconstruct, reliable_from

CHANGES_HTML = """
<table><thead>
<tr><th rowspan=2>Effective Date</th><th colspan=2>Added</th><th colspan=2>Removed</th>
<th rowspan=2>Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr></thead>
<tbody>
<tr><td>January 1, 2020</td><td>C</td><td>Cco</td><td>D |</td><td>Dco</td><td>x</td></tr>
<tr><td>March 5, 2019</td><td>FB</td><td>Facebook</td><td>E</td><td>Eco</td><td>x</td></tr>
<tr><td>June 1, 2018</td><td></td><td></td><td>F</td><td>Fco</td><td>spin-off</td></tr>
<tr><td>May 1, 2018</td><td>ZZ</td><td>Gone</td><td></td><td></td><td>later renamed, unrecorded</td></tr>
</tbody></table>
"""


def test_parse_changes_cleans_tickers_and_applies_renames():
    ch = parse_changes(CHANGES_HTML)
    assert list(ch["date"]) == list(pd.to_datetime(["2020-01-01", "2019-03-05", "2018-06-01", "2018-05-01"]))
    assert ch.loc[0, "removed"] == "D"  # stray "|" stripped
    assert ch.loc[1, "added"] == "META"  # FB -> META rename
    assert pd.isna(ch.loc[2, "added"])


def test_backward_reconstruction_intervals_and_orphans():
    ch = parse_changes(CHANGES_HTML)
    iv, orphans = reconstruct({"A", "META", "C"}, ch, pd.Timestamp("2026-01-01"))
    assert members_on(iv, "2026-01-01") == {"A", "META", "C"}
    assert members_on(iv, "2020-01-01") == {"A", "META", "C"}  # effective date: added member from d
    assert members_on(iv, "2019-12-31") == {"A", "META", "D"}  # removed member until d (exclusive)
    assert members_on(iv, "2019-01-01") == {"A", "D", "E"}
    assert members_on(iv, "2018-05-15") == {"A", "D", "E", "F"}
    assert [o["ticker"] for o in orphans] == ["ZZ"]  # added but never a member afterwards


def test_reliable_from_requires_dense_years():
    dates = [
        *pd.date_range("2005-01-01", periods=3, freq="90D"),
        *pd.date_range("2011-01-01", "2014-12-31", freq="14D"),
    ]
    ch = pd.DataFrame({"date": pd.to_datetime(dates), "added": "X", "removed": "Y"})
    assert reliable_from(ch, min_per_year=15) == pd.Timestamp("2011-01-01")
