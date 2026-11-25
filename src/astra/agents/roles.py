"""Deterministic research roles. Each returns `Finding`s that reference their evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd

from astra.common.config import get_config
from astra.common.paths import Paths, get_paths
from astra.data.storage import read_manifest
from astra.tracking.tracker import ExperimentRecord, Tracker, record_subject

Severity = Literal["info", "warn", "error"]


@dataclass
class Finding:
    agent: str
    severity: Severity
    subject: str
    message: str
    evidence: list[str] = field(default_factory=list)  # file paths / experiment ids


def _latest_records(tracker: Tracker) -> list[ExperimentRecord]:
    """Latest record per (hypothesis_id, name, subject) - older reruns stay on disk, unchanged."""
    idx = tracker.list()
    if idx.empty:
        return []
    recs = []
    for eid in idx["experiment_id"]:
        try:
            recs.append(tracker.load(eid))
        except FileNotFoundError:
            continue
    latest: dict[tuple[str, str, str], ExperimentRecord] = {}
    for r in recs:
        latest[(r.hypothesis_id, r.name, _subject(r))] = r
    return list(latest.values())


def _subject(r: ExperimentRecord) -> str:
    return record_subject(r)


class DataQualityAgent:
    name = "data_quality"

    def __init__(self, paths: Paths | None = None, stale_days: int = 7) -> None:
        self.paths, self.stale_days = paths or get_paths(), stale_days

    def run(self) -> list[Finding]:
        out: list[Finding] = []
        now = datetime.now(UTC)
        for inst in get_config().instruments:
            try:
                m = read_manifest(inst.symbol, inst.market, self.paths)
            except FileNotFoundError:
                out.append(Finding(self.name, "error", inst.symbol, "no validated data - run `astra ingest`"))
                continue
            ev = [m["quality_report"]]
            if m["quality_status"] == "fail":
                out.append(Finding(self.name, "error", inst.symbol, "latest quality report FAILED", ev))
            report = json.loads((self.paths.root / m["quality_report"]).read_text(encoding="utf-8"))
            for c in report["checks"]:
                if c["status"] != "pass":
                    out.append(
                        Finding(
                            self.name,
                            "warn" if c["status"] == "warn" else "error",
                            inst.symbol,
                            f"{c['name']}: {c['details']} e.g. {c['examples'][:3]}",
                            ev,
                        )
                    )
            age = (now - datetime.fromisoformat(m["written_at_utc"])).days
            if age > self.stale_days:
                out.append(Finding(self.name, "warn", inst.symbol, f"data is {age} days old", ev))
            for note in m.get("transform_notes", []):
                if "incomplete" in note:
                    out.append(Finding(self.name, "info", inst.symbol, note, ev))
        return out


class ModelEvaluationAgent:
    """Methodology checks on experiment records (no re-computation of results)."""

    name = "model_evaluation"

    def __init__(self, paths: Paths | None = None) -> None:
        self.tracker = Tracker(paths)

    def run(self) -> list[Finding]:
        out: list[Finding] = []
        for r in _latest_records(self.tracker):
            ev = [r.experiment_id]
            subj = f"{r.hypothesis_id}/{_subject(r)}"
            if r.code_version.get("dirty"):
                out.append(
                    Finding(
                        self.name,
                        "warn",
                        subj,
                        "produced from an uncommitted working tree - rerun after commit",
                        ev,
                    )
                )
            if r.code_version.get("commit") == "no-commit":
                out.append(Finding(self.name, "warn", subj, "no git commit recorded", ev))
            if r.variants_tried > 1 and "bh" not in json.dumps(r.metrics).lower():
                out.append(
                    Finding(
                        self.name,
                        "warn",
                        subj,
                        f"{r.variants_tried} variants without a recorded multiple-testing adjustment",
                        ev,
                    )
                )
            if r.conclusion == "supported" and r.hypothesis_id.startswith(("H-XM", "H-ML")):
                out.append(
                    Finding(
                        self.name,
                        "info",
                        subj,
                        "supported result - requires independent backtest-auditor validation",
                        ev,
                    )
                )
        return out


class BacktestAnalysisAgent:
    name = "backtest_analysis"

    def __init__(self, paths: Paths | None = None) -> None:
        self.paths = paths or get_paths()
        self.tracker = Tracker(self.paths)

    def run(self) -> list[Finding]:
        out: list[Finding] = []
        for r in _latest_records(self.tracker):
            m = r.metrics
            if "net" not in m or not isinstance(m.get("net"), dict):
                continue
            ev, subj = [r.experiment_id], f"{r.hypothesis_id}/{_subject(r)}"
            gross = m.get("gross", {}).get("sharpe")
            net = m["net"].get("sharpe")
            if gross is not None and net is not None and gross > 0 > net:
                out.append(
                    Finding(
                        self.name,
                        "warn",
                        subj,
                        f"costs flip the sign: gross Sharpe {gross:.2f} -> net {net:.2f}",
                        ev,
                    )
                )
            canary = m.get("leakage_canary")
            if isinstance(canary, dict) and not canary.get("passed", True):
                out.append(Finding(self.name, "error", subj, "leakage canary FAILED", ev))
            sens = self.paths.experiments / r.experiment_id / "artifacts" / "cost_sensitivity.parquet"
            if not sens.exists():
                out.append(Finding(self.name, "warn", subj, "no transaction-cost sensitivity table", ev))
            else:
                s = pd.read_parquet(sens)
                one, two = (
                    s.loc[s.cost_multiplier == 1.0, "sharpe"],
                    s.loc[s.cost_multiplier == 2.0, "sharpe"],
                )
                if len(one) and len(two) and float(one.iloc[0]) > 0 >= float(two.iloc[0]):
                    out.append(
                        Finding(
                            self.name,
                            "warn",
                            subj,
                            "edge disappears at 2x assumed costs (fragile to cost assumptions)",
                            ev,
                        )
                    )
        return out


class RiskAnalysisAgent:
    name = "risk_analysis"

    def __init__(self, paths: Paths | None = None, mdd_limit: float = -0.5) -> None:
        self.tracker, self.mdd_limit = Tracker(paths), mdd_limit

    def run(self) -> list[Finding]:
        out: list[Finding] = []
        for r in _latest_records(self.tracker):
            net = r.metrics.get("net")
            if not isinstance(net, dict):
                continue
            mdd = net.get("max_drawdown")
            if mdd is not None and mdd < self.mdd_limit:
                out.append(
                    Finding(
                        self.name,
                        "warn",
                        f"{r.hypothesis_id}/{_subject(r)}",
                        f"max drawdown {mdd:.1%} exceeds limit {self.mdd_limit:.0%}",
                        [r.experiment_id],
                    )
                )
        return out


class ResearchPlanningAgent:
    """Rule-based next-experiment recommendations from the conclusion table."""

    name = "research_planning"

    def __init__(self, paths: Paths | None = None) -> None:
        self.tracker = Tracker(paths)

    def run(self) -> list[Finding]:
        recs = _latest_records(self.tracker)
        if not recs:
            return [Finding(self.name, "info", "-", "no experiments yet: start with `astra research`")]
        df = pd.DataFrame([{"h": r.hypothesis_id, "c": r.conclusion, "id": r.experiment_id} for r in recs])
        out = []
        for h, g in df.groupby("h"):
            counts = g["c"].value_counts().to_dict()
            ids = g["id"].tolist()[:5]
            if counts.get("supported", 0) and not counts.get("rejected", 0):
                msg = "supported somewhere: replicate on held-out instruments/periods before any claim"
            elif counts.get("inconclusive", 0) >= max(counts.get("rejected", 0), 1):
                msg = "mostly inconclusive: low power - extend sample (more instruments/history) or drop"
            elif counts.get("rejected", 0):
                msg = "rejected: deprioritize; record as negative result, do not re-tune on the same data"
            else:
                msg = "no decision yet"
            out.append(Finding(self.name, "info", str(h), f"{counts} -> {msg}", ids))
        return out


def findings_frame(findings: list[Finding]) -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in findings])


def conclusions_table(paths: Paths | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in _latest_records(Tracker(paths)):
        rows.append(
            {
                "hypothesis": r.hypothesis_id,
                "experiment": r.name,
                "subject": _subject(r),
                "conclusion": r.conclusion,
                "experiment_id": r.experiment_id,
                "created_at_utc": r.created_at_utc[:19],
            }
        )
    return pd.DataFrame(rows).sort_values(["hypothesis", "subject"]) if rows else pd.DataFrame()
