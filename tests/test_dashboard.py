"""Render every dashboard page headlessly (Streamlit AppTest) and assert no exception is raised.

Uses the real stored outputs under data/ and experiments/; skipped on a fresh clone without them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nebula.common.paths import get_paths

APP = Path(__file__).resolve().parents[1] / "src" / "nebula" / "dashboard" / "app.py"

pytestmark = pytest.mark.skipif(
    not (get_paths().experiments / "index.jsonl").exists(),
    reason="no stored experiments (run `nebula run-all`)",
)


def _pages() -> list[str]:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    return list(at.sidebar.radio[0].options)


@pytest.mark.slow
def test_every_dashboard_page_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    failures = {}
    for page in _pages():
        at = AppTest.from_file(str(APP), default_timeout=120).run()
        at.sidebar.radio[0].set_value(page).run()
        if at.exception:
            failures[page] = [e.message for e in at.exception]
    assert not failures, failures
