"""Golden regression: headline vertical-slice numbers under data lock v0.3.0-baseline must not move.

Skipped when the lock's snapshot objects are not on this machine (fresh clone without data).
A deliberate change regenerates the golden file with scripts/update_golden.py in the same commit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astra.data.lock import LockError, load_lock, object_path
from astra.tracking.golden import GOLDEN_LOCK, collect_headlines, compare, golden_path

pytestmark = pytest.mark.regression


def _lock_available() -> bool:
    """Cheap presence check (no re-hashing; `astra data-lock verify` does that)."""
    if not golden_path().exists():
        return False
    try:
        files = load_lock(GOLDEN_LOCK)["files"]
    except LockError:
        return False
    return all(object_path(m["sha256"], Path(r).suffix).exists() for r, m in files.items())


@pytest.mark.skipif(not _lock_available(), reason="data lock snapshots or golden file not present")
def test_vertical_slice_headlines_unchanged():
    expected = json.loads(golden_path().read_text(encoding="utf-8"))
    diffs = compare(expected, collect_headlines())
    assert not diffs, "headline numbers moved:\n" + "\n".join(diffs[:50])
