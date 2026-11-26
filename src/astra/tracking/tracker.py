"""Local file-based experiment tracking (see docs/adr/0004-experiment-tracking.md).

experiments/
  index.jsonl                 one summary line per experiment (append-only)
  <experiment_id>/record.json full record
  <experiment_id>/artifacts/  tables, equity curves (parquet / csv)
Failed and negative experiments are recorded exactly like successful ones.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from astra.common.paths import Paths, get_paths

Conclusion = Literal["supported", "rejected", "inconclusive", "failed_run", "pending"]


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = run("rev-parse", "HEAD") or "no-commit"
    dirty = bool(run("status", "--porcelain"))
    return {"commit": commit, "dirty": dirty}


class ExperimentRecord(BaseModel):
    experiment_id: str
    name: str
    hypothesis_id: str
    hypothesis: str
    created_at_utc: str
    code_version: dict[str, Any]
    datasets: list[dict[str, Any]] = Field(default_factory=list)  # manifests incl. sha256
    features: list[str] = Field(default_factory=list)
    model: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    costs: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""
    conclusion: Conclusion = "pending"
    variants_tried: int = 1
    reproduce: str = ""
    # Exact data version: {"name", "fingerprint"} of the active data lock, or {"name": None} = live data
    # (records written before data locks existed have {}).
    data_lock: dict[str, Any] = Field(default_factory=dict)


def current_data_lock(paths: Paths | None = None) -> dict[str, Any]:
    from astra.data.lock import active_lock, lock_fingerprint

    name = active_lock()
    if not name:
        return {"name": None, "note": "live validated data (not pinned)"}
    return {"name": name, "fingerprint": lock_fingerprint(name, paths)}


def record_subject(rec: ExperimentRecord) -> str:
    """Instrument for single-dataset records; market (or 'multi') for multi-instrument records."""
    ds = [d for d in rec.datasets if isinstance(d, dict)]
    if not ds:
        return ""
    if len(ds) == 1 and ds[0].get("symbol"):
        return str(ds[0]["symbol"])
    markets = {str(d["market"]) for d in ds if d.get("market")}
    if len(markets) == 1:
        return markets.pop()
    if len(ds) == 1 and ds[0].get("market"):
        return str(ds[0]["market"])
    return "multi"


def _cell_subject(rec: ExperimentRecord) -> str:
    """Like record_subject, but one symbol at several frequencies (1m + 1d) is still that symbol."""
    syms = {str(d["symbol"]) for d in rec.datasets if isinstance(d, dict) and d.get("symbol")}
    return syms.pop() if len(syms) == 1 else record_subject(rec)


class Tracker:
    def __init__(self, paths: Paths | None = None) -> None:
        self.paths = (paths or get_paths()).ensure()
        self.root = self.paths.experiments

    def new(self, name: str, hypothesis_id: str, hypothesis: str, **kw: Any) -> ExperimentRecord:
        stamp = datetime.now(UTC)
        eid = f"{stamp:%Y%m%dT%H%M%S}-{hypothesis_id}-{uuid.uuid4().hex[:6]}"
        return ExperimentRecord(
            experiment_id=eid,
            name=name,
            hypothesis_id=hypothesis_id,
            hypothesis=hypothesis,
            created_at_utc=stamp.isoformat(),
            code_version=git_state(self.paths.root),
            data_lock=kw.pop("data_lock", None) or current_data_lock(self.paths),
            **kw,
        )

    def artifact_dir(self, rec: ExperimentRecord) -> Path:
        d = self.root / rec.experiment_id / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_frame(self, rec: ExperimentRecord, name: str, df: pd.DataFrame) -> Path:
        p = self.artifact_dir(rec) / f"{name}.parquet"
        df.to_parquet(p)
        # relative to the repo when inside it; absolute when ASTRA_EXPERIMENTS_DIR points elsewhere
        rel = p.relative_to(self.paths.root) if p.is_relative_to(self.paths.root) else p
        rec.artifacts.append(str(rel))
        return p

    def log(self, rec: ExperimentRecord) -> Path:
        d = self.root / rec.experiment_id
        d.mkdir(parents=True, exist_ok=True)
        p = d / "record.json"
        p.write_text(rec.model_dump_json(indent=2), encoding="utf-8")
        summary = {
            "experiment_id": rec.experiment_id,
            "name": rec.name,
            "hypothesis_id": rec.hypothesis_id,
            "created_at_utc": rec.created_at_utc,
            "conclusion": rec.conclusion,
            "headline": {k: v for k, v in rec.metrics.items() if isinstance(v, int | float | str)},
        }
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, default=str) + "\n")
        return p

    def list(self) -> pd.DataFrame:
        idx = self.root / "index.jsonl"
        if not idx.exists():
            return pd.DataFrame()
        rows = [json.loads(line) for line in idx.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)

    def load(self, experiment_id: str) -> ExperimentRecord:
        p = self.root / experiment_id / "record.json"
        return ExperimentRecord.model_validate_json(p.read_text(encoding="utf-8"))

    def status(self) -> pd.DataFrame:
        """Every record with a computed `superseded` flag (records themselves are never modified).

        A record is superseded when a later record exists for the same (hypothesis_id, subject, name):
        re-runs after a fix or a re-ingest replace earlier ones for reporting, but stay on disk as
        provenance. `superseded_by` names the current record of the group."""
        idx = self.list()
        if idx.empty:
            return idx
        rows = []
        for eid in idx["experiment_id"]:
            rec = self.load(eid)
            rows.append(
                {
                    "experiment_id": eid,
                    "hypothesis_id": rec.hypothesis_id,
                    "name": rec.name,
                    "subject": _cell_subject(rec),
                    "created_at_utc": rec.created_at_utc,
                    "conclusion": rec.conclusion,
                    "commit": str(rec.code_version.get("commit", ""))[:7],
                    "dirty": bool(rec.code_version.get("dirty", False)),
                    "data_lock": (rec.data_lock or {}).get("name"),
                }
            )
        df = pd.DataFrame(rows).drop_duplicates("experiment_id").sort_values("created_at_utc")
        key = ["hypothesis_id", "subject", "name"]
        latest = df.groupby(key, sort=False)["experiment_id"].transform("last")
        df["superseded"] = df["experiment_id"] != latest
        df["superseded_by"] = latest.where(df["superseded"], None)
        return df.reset_index(drop=True)

    def current(self) -> list[ExperimentRecord]:
        """The non-superseded records."""
        st = self.status()
        if st.empty:
            return []
        return [self.load(e) for e in st.loc[~st["superseded"], "experiment_id"]]
