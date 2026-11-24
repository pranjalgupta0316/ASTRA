"""Offline tests for checksummed downloads (fake HTTP session) and a CLI smoke test."""

from __future__ import annotations

import hashlib

import pytest
from typer.testing import CliRunner

from nebula.cli import app
from nebula.data import download
from nebula.data.download import ChecksumError, checked_download

PAYLOAD = b"zip-bytes" * 1000
GOOD = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResp:
    def __init__(self, status: int, body: bytes = b"", headers: dict | None = None) -> None:
        self.status_code, self._body, self.headers = status, body, headers or {}

    @property
    def text(self) -> str:
        return self._body.decode()

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self) -> None:
        pass


class FakeSession:
    def __init__(self, routes: dict[str, list[FakeResp]]) -> None:
        self.routes, self.calls = routes, []

    def get(self, url: str, timeout: float, stream: bool = False) -> FakeResp:
        assert timeout > 0  # every request must carry a timeout
        self.calls.append(url)
        q = self.routes[url]
        return q.pop(0) if len(q) > 1 else q[0]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda s: None)


def test_download_verifies_and_stores_checksum(tmp_path):
    s = FakeSession(
        {"u.zip.CHECKSUM": [FakeResp(200, f"{GOOD}  u.zip".encode())], "u.zip": [FakeResp(200, PAYLOAD)]}
    )
    dest = tmp_path / "u.zip"
    assert checked_download(s, "u.zip", dest) == dest
    assert dest.read_bytes() == PAYLOAD and (tmp_path / "u.zip.CHECKSUM").exists()
    assert not (tmp_path / "u.zip.part").exists()


def test_checksum_mismatch_raises_and_leaves_nothing(tmp_path):
    s = FakeSession({"u.zip.CHECKSUM": [FakeResp(200, b"0" * 64)], "u.zip": [FakeResp(200, PAYLOAD)]})
    with pytest.raises(ChecksumError):
        checked_download(s, "u.zip", tmp_path / "u.zip")
    assert list(tmp_path.iterdir()) == []


def test_unpublished_file_returns_none(tmp_path):
    s = FakeSession({"u.zip.CHECKSUM": [FakeResp(404)], "u.zip": [FakeResp(404)]})
    assert checked_download(s, "u.zip", tmp_path / "u.zip") is None


def test_cached_file_reverified_and_corruption_detected(tmp_path):
    s = FakeSession(
        {"u.zip.CHECKSUM": [FakeResp(200, f"{GOOD}  u.zip".encode())], "u.zip": [FakeResp(200, PAYLOAD)]}
    )
    dest = tmp_path / "u.zip"
    checked_download(s, "u.zip", dest)
    n = len(s.calls)
    assert checked_download(s, "u.zip", dest) == dest and len(s.calls) == n  # no network on valid cache
    dest.write_bytes(b"corrupted")
    with pytest.raises(ChecksumError):
        checked_download(s, "u.zip", dest)


def test_rate_limit_is_retried(tmp_path):
    s = FakeSession(
        {
            "u.zip.CHECKSUM": [
                FakeResp(429, headers={"Retry-After": "1"}),
                FakeResp(200, f"{GOOD}  u.zip".encode()),
            ],
            "u.zip": [FakeResp(503), FakeResp(200, PAYLOAD)],
        }
    )
    assert checked_download(s, "u.zip", tmp_path / "u.zip") is not None
    assert s.calls.count("u.zip.CHECKSUM") == 2 and s.calls.count("u.zip") == 2


def test_cli_lists_all_commands():
    res = CliRunner().invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in [
        "env",
        "ingest",
        "research",
        "holdout",
        "vol-study",
        "portfolio-study",
        "ml-study",
        "micro-study",
        "report",
        "run-all",
        "dashboard",
    ]:
        assert cmd in res.output
