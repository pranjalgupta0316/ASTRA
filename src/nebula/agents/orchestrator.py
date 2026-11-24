"""Project orchestration: runs the deterministic roles, logs activity, writes a Markdown research report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nebula.agents.llm import narrate
from nebula.agents.roles import (
    BacktestAnalysisAgent,
    DataQualityAgent,
    Finding,
    ModelEvaluationAgent,
    ResearchPlanningAgent,
    RiskAnalysisAgent,
    conclusions_table,
    findings_frame,
)
from nebula.common.paths import Paths, get_paths


def md_table(df: pd.DataFrame) -> str:
    """Minimal Markdown table (avoids a `tabulate` dependency)."""

    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    head = "| " + " | ".join(cell(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _activity_log(paths: Paths) -> Path:
    paths.logs.mkdir(parents=True, exist_ok=True)
    return paths.logs / "agent_activity.jsonl"


def run_review(paths: Paths | None = None) -> dict[str, Any]:
    """Run all roles; returns findings and writes one activity line per role."""
    paths = paths or get_paths()
    roles = [
        DataQualityAgent(paths),
        ModelEvaluationAgent(paths),
        BacktestAnalysisAgent(paths),
        RiskAnalysisAgent(paths),
        ResearchPlanningAgent(paths),
    ]
    findings: list[Finding] = []
    log = _activity_log(paths)
    for role in roles:
        started = datetime.now(UTC)
        res = role.run()
        findings += res
        with log.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts": started.isoformat(),
                        "agent": role.name,
                        "findings": len(res),
                        "errors": sum(f.severity == "error" for f in res),
                        "warnings": sum(f.severity == "warn" for f in res),
                    }
                )
                + "\n"
            )
    return {"findings": findings}


def write_report(paths: Paths | None = None, use_llm: bool = False) -> Path:
    """Markdown report of all latest conclusions + agent findings, every row traceable to its source."""
    paths = paths or get_paths()
    review = run_review(paths)
    f = findings_frame(review["findings"])
    concl = conclusions_table(paths)
    now = datetime.now(UTC)
    lines = [
        "# NEBULA research report",
        "",
        f"Generated {now:%Y-%m-%d %H:%M} UTC by `nebula report` (deterministic; no LLM-computed numbers).",
        "",
        "## Conclusions (latest run per hypothesis / experiment / subject)",
        "",
    ]
    if concl.empty:
        lines.append("_No experiments recorded._")
    else:
        counts = concl["conclusion"].value_counts().to_dict()
        lines += [f"Totals: {counts}", "", md_table(concl)]
    for sev, title in (("error", "Errors"), ("warn", "Warnings"), ("info", "Notes and recommendations")):
        sub = f[f["severity"] == sev] if not f.empty else f
        lines += ["", f"## {title} ({len(sub)})", ""]
        if len(sub):
            sub = sub.assign(evidence=sub["evidence"].map(lambda e: ", ".join(e[:3])))
            lines.append(md_table(sub[["agent", "subject", "message", "evidence"]]))
        else:
            lines.append("_None._")
    body = "\n".join(lines) + "\n"
    if use_llm:
        summary = narrate(body)
        if summary:
            body += (
                "\n## Narrative summary (local LLM, unverified prose - numbers above are authoritative)\n\n"
            )
            body += summary + "\n"
    out_dir = paths.reports / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"research_report_{now:%Y%m%dT%H%M%SZ}.md"
    out.write_text(body, encoding="utf-8")
    (out_dir / "latest.md").write_text(body, encoding="utf-8")
    return out
