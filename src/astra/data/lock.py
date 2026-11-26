"""Dataset versioning: immutable content-addressed snapshots + named locks.

Why: studies used to read the *latest* validated files, so a re-download (e.g. Yahoo restating adjusted
prices) could silently change a result that had already been recorded. A lock pins every validated file
to a snapshot; experiment records store the lock they ran under, so they can be replayed exactly.

Layout
- data/snapshots/objects/<sha[:2]>/<sha256><ext>   immutable copies (git-ignored, deduplicated by content)
- config/data_locks/<name>.json                    the lock manifest: relative path -> sha256 (tracked in git)

Activation: env var ASTRA_DATA_LOCK=<name> (the CLI's global `--lock` sets it, so subprocesses inherit it)
or the `use_lock(name)` context manager. While a lock is active, `resolve()` maps a live validated path to
its snapshot and refuses paths the lock does not contain (strict: never a silent fallback to live data).
Ingestion always reads and writes live data.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from astra.common.paths import Paths, get_paths

ENV = "ASTRA_DATA_LOCK"
_override: str | None = None


class LockError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locks_dir(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).config / "data_locks"


def objects_dir(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).data / "snapshots" / "objects"


def object_path(sha: str, suffix: str, paths: Paths | None = None) -> Path:
    return objects_dir(paths) / sha[:2] / f"{sha}{suffix}"


def _rel(path: Path, paths: Paths) -> str:
    return Path(path).resolve().relative_to(paths.validated.resolve()).as_posix()


def create_lock(name: str, note: str = "", paths: Paths | None = None) -> dict[str, Any]:
    """Snapshot every file under data/validated and write config/data_locks/<name>.json."""
    paths = paths or get_paths()
    if not name or any(c in name for c in r'\/:*?"<>| '):
        raise LockError(f"invalid lock name {name!r}")
    dest = locks_dir(paths) / f"{name}.json"
    if dest.exists():
        raise LockError(f"lock {name!r} already exists - locks are immutable; choose a new name")
    files: dict[str, dict[str, Any]] = {}
    new_bytes = 0
    for p in sorted(paths.validated.rglob("*")):
        if not p.is_file():
            continue
        sha = _sha256(p)
        obj = object_path(sha, p.suffix, paths)
        if not obj.exists():
            obj.parent.mkdir(parents=True, exist_ok=True)
            tmp = obj.with_name(obj.name + ".part")
            shutil.copyfile(p, tmp)
            if _sha256(tmp) != sha:
                tmp.unlink(missing_ok=True)
                raise LockError(f"snapshot copy of {p} does not match its hash")
            tmp.replace(obj)
            new_bytes += p.stat().st_size
        files[_rel(p, paths)] = {"sha256": sha, "bytes": p.stat().st_size}
    if not files:
        raise LockError("no validated files to lock")
    manifest = {
        "name": name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "note": note,
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "files": files,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return {"name": name, "n_files": len(files), "new_snapshot_bytes": new_bytes, "path": str(dest)}


def load_lock(name: str, paths: Paths | None = None) -> dict[str, Any]:
    p = locks_dir(paths) / f"{name}.json"
    if not p.exists():
        raise LockError(f"unknown data lock {name!r} (no {p})")
    return json.loads(p.read_text(encoding="utf-8"))


def list_locks(paths: Paths | None = None) -> list[dict[str, Any]]:
    out = []
    for p in sorted(locks_dir(paths).glob("*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        out.append({k: m.get(k) for k in ("name", "created_at_utc", "n_files", "total_bytes", "note")})
    return out


def verify_lock(name: str, paths: Paths | None = None) -> dict[str, Any]:
    """Re-hash every snapshot object of a lock; reports missing or corrupted objects."""
    lock = load_lock(name, paths)
    missing, corrupt = [], []
    for rel, meta in lock["files"].items():
        obj = object_path(meta["sha256"], Path(rel).suffix, paths)
        if not obj.exists():
            missing.append(rel)
        elif _sha256(obj) != meta["sha256"]:
            corrupt.append(rel)
    return {
        "name": name,
        "n_files": len(lock["files"]),
        "missing": missing,
        "corrupt": corrupt,
        "ok": not missing and not corrupt,
    }


def lock_fingerprint(name: str, paths: Paths | None = None) -> str:
    """sha256 of the lock manifest's file map (stable identifier of the exact data version)."""
    files = load_lock(name, paths)["files"]
    blob = json.dumps({k: v["sha256"] for k, v in files.items()}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def active_lock() -> str | None:
    return _override if _override is not None else (os.environ.get(ENV) or None)


@contextmanager
def use_lock(name: str | None) -> Generator[None]:
    """Activate a lock for the duration of a block (None = live data)."""
    global _override
    prev = _override
    _override = name if name is not None else ""
    try:
        yield
    finally:
        _override = prev


@dataclass(frozen=True)
class _Cache:
    key: str  # lock file path: a name alone is ambiguous across project roots
    files: dict[str, dict[str, Any]]


_cache: _Cache | None = None


def _lock_files(name: str, paths: Paths) -> dict[str, dict[str, Any]]:
    global _cache
    key = str((locks_dir(paths) / f"{name}.json").resolve())
    if _cache is None or _cache.key != key:
        _cache = _Cache(key, load_lock(name, paths)["files"])
    return _cache.files


def resolve(path: Path, paths: Paths | None = None) -> Path:
    """Live validated path -> locked snapshot when a lock is active; unchanged otherwise.
    Paths outside data/validated are returned unchanged (raw files, captures, experiments)."""
    name = active_lock()
    if not name:
        return Path(path)
    paths = paths or get_paths()
    try:
        rel = _rel(Path(path), paths)
    except ValueError:
        return Path(path)
    files = _lock_files(name, paths)
    if rel not in files:
        raise LockError(f"{rel} is not in data lock {name!r}; create a new lock that includes it")
    obj = object_path(files[rel]["sha256"], Path(rel).suffix, paths)
    if not obj.exists():
        raise LockError(f"snapshot for {rel} missing ({obj}); run `astra data-lock verify {name}`")
    return obj


def resolved_exists(path: Path, paths: Paths | None = None) -> bool:
    """Existence check that honours the active lock (False if the lock does not contain the path)."""
    try:
        return resolve(path, paths).exists()
    except LockError:
        return False


def locked_files_matching(pattern_suffix: str, paths: Paths | None = None) -> list[Path]:
    """Snapshot paths of locked files whose relative path ends with `pattern_suffix` (e.g. 'bars.parquet');
    live validated files when no lock is active."""
    paths = paths or get_paths()
    name = active_lock()
    if not name:
        return sorted(p for p in paths.validated.rglob("*") if p.as_posix().endswith(pattern_suffix))
    files = _lock_files(name, paths)
    return [
        object_path(m["sha256"], Path(r).suffix, paths)
        for r, m in sorted(files.items())
        if r.endswith(pattern_suffix)
    ]
