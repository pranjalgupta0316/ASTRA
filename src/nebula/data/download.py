"""Checksummed, streamed, rate-limit-aware downloads for Binance Public Data style files.

- The provider's `<file>.CHECKSUM` (sha256) is fetched first and stored next to the file.
- The file is streamed to `<dest>.part` while hashing, verified, then atomically renamed.
- Cached files are re-verified against the stored checksum on every reuse (detects corruption).
- HTTP 429 / 418 / 5xx are retried with exponential backoff honoring `Retry-After`.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

TIMEOUT_S = 30
PAUSE_S = 0.25
MAX_ATTEMPTS = 5
BACKOFF_S = 2.0
RETRY_STATUS = {418, 429, 500, 502, 503, 504}


class ChecksumError(RuntimeError):
    pass


def _get(http: requests.Session, url: str, stream: bool = False) -> requests.Response:
    r = http.get(url, timeout=TIMEOUT_S, stream=stream)
    for attempt in range(MAX_ATTEMPTS - 1):
        if r.status_code not in RETRY_STATUS:
            break
        retry_after = r.headers.get("Retry-After")
        wait = float(retry_after) if retry_after and retry_after.isdigit() else BACKOFF_S * 2**attempt
        r.close()
        time.sleep(min(wait, 120.0))
        r = http.get(url, timeout=TIMEOUT_S, stream=stream)
    if r.status_code in RETRY_STATUS:
        r.raise_for_status()
    time.sleep(PAUSE_S)
    return r


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_download(http: requests.Session, url: str, dest: Path) -> Path | None:
    """Return `dest` (verified) or None if the provider has not published the file (404)."""
    sidecar = dest.with_name(dest.name + ".CHECKSUM")
    if dest.exists():
        if not sidecar.exists():  # cache from before checksums were stored: fetch and persist once
            c = _get(http, url + ".CHECKSUM")
            c.raise_for_status()
            sidecar.write_text(c.text, encoding="utf-8")
        expected = sidecar.read_text(encoding="utf-8").split()[0].strip().lower()
        if sha256_of(dest) != expected:
            raise ChecksumError(f"cached file {dest} does not match its checksum - delete it and re-run")
        return dest
    c = _get(http, url + ".CHECKSUM")
    if c.status_code == 404:
        return None
    c.raise_for_status()
    expected = c.text.split()[0].strip().lower()
    r = _get(http, url, stream=True)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    with part.open("wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            h.update(chunk)
            fh.write(chunk)
    if h.hexdigest() != expected:
        part.unlink(missing_ok=True)
        raise ChecksumError(f"checksum mismatch for {url}: {h.hexdigest()} != {expected}")
    part.replace(dest)
    sidecar.write_text(c.text, encoding="utf-8")
    return dest
