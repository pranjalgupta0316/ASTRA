from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from nebula.agents import llm
from nebula.agents.orchestrator import md_table, write_report
from nebula.agents.roles import BacktestAnalysisAgent, ModelEvaluationAgent, ResearchPlanningAgent
from nebula.common.paths import Paths
from nebula.tracking.tracker import Tracker


def _seed(tmp_path):
    paths = Paths(root=tmp_path)
    tr = Tracker(paths)
    a = tr.new(
        "x",
        "H-XM-900",
        "test",
        datasets=[{"symbol": "AAA"}],
        conclusion="supported",
        metrics={
            "net": {"sharpe": -0.2, "max_drawdown": -0.7},
            "gross": {"sharpe": 0.4},
            "leakage_canary": {"passed": False},
        },
    )
    tr.log(a)
    b = tr.new(
        "y", "H-XM-901", "test", datasets=[{"symbol": "BBB"}], conclusion="inconclusive", variants_tried=12
    )
    tr.log(b)
    return paths, a, b


def test_agents_flag_cost_sign_flip_canary_and_methodology(tmp_path):
    paths, a, b = _seed(tmp_path)
    bt = BacktestAnalysisAgent(paths).run()
    msgs = " ".join(f.message for f in bt)
    assert "costs flip the sign" in msgs and "leakage canary FAILED" in msgs
    assert all(a.experiment_id in f.evidence for f in bt)  # every finding traceable
    me = ModelEvaluationAgent(paths).run()
    assert any("variants without" in f.message and b.experiment_id in f.evidence for f in me)
    assert any("independent backtest-auditor" in f.message for f in me)


def test_planning_agent_recommends_per_hypothesis(tmp_path):
    paths, *_ = _seed(tmp_path)
    plan = {f.subject: f.message for f in ResearchPlanningAgent(paths).run()}
    assert "replicate on held-out" in plan["H-XM-900"]
    assert "low power" in plan["H-XM-901"]


def test_report_written_without_llm(tmp_path, monkeypatch):
    monkeypatch.delenv("NEBULA_LLM_URL", raising=False)
    paths, a, _ = _seed(tmp_path)
    (tmp_path / "config").mkdir()
    out = write_report(paths)
    text = out.read_text(encoding="utf-8")
    assert a.experiment_id in text and "Narrative summary" not in text
    assert (paths.logs / "agent_activity.jsonl").exists()


def test_md_table_escapes_pipes():
    assert "a\\|b" in md_table(pd.DataFrame({"c": ["a|b"]}))


def test_llm_rejects_non_local_url(monkeypatch):
    monkeypatch.setenv("NEBULA_LLM_URL", "https://api.example.com")
    monkeypatch.setenv("NEBULA_LLM_MODEL", "m")
    with pytest.raises(ValueError):
        llm.llm_config()


def test_llm_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("NEBULA_LLM_URL", raising=False)
    assert llm.narrate("x") is None


def test_llm_client_against_fake_local_server(monkeypatch):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"response": "summary text"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("NEBULA_LLM_URL", f"http://127.0.0.1:{srv.server_port}")
        monkeypatch.setenv("NEBULA_LLM_MODEL", "fake")
        assert llm.narrate("report body") == "summary text"
        assert seen["body"]["model"] == "fake" and seen["body"]["options"]["temperature"] == 0
    finally:
        srv.shutdown()
