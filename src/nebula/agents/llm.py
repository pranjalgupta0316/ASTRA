"""Optional local LLM narration via an Ollama-compatible HTTP API.

Disabled unless NEBULA_LLM_URL (e.g. http://127.0.0.1:11434) and NEBULA_LLM_MODEL are set. Only
localhost URLs are accepted, so report contents never leave the machine. Returns None on any failure:
the platform never depends on it. Status: client tested against a fake local server only; no model
has been benchmarked on this machine (none installed).
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import requests

PROMPT = (
    "Summarize this quantitative research report in at most 8 sentences for a technical reader. "
    "Do not invent numbers; quote only numbers that appear in the report; keep every caveat.\n\n"
)


def llm_config() -> tuple[str, str] | None:
    url, model = os.environ.get("NEBULA_LLM_URL"), os.environ.get("NEBULA_LLM_MODEL")
    if not url or not model:
        return None
    host = urlparse(url).hostname or ""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("NEBULA_LLM_URL must point to a local server (data must not leave the machine)")
    return url.rstrip("/"), model


def narrate(report_markdown: str, timeout_s: float = 120.0) -> str | None:
    cfg = llm_config()
    if cfg is None:
        return None
    url, model = cfg
    try:
        r = requests.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": PROMPT + report_markdown[:20000],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=timeout_s,
        )
        r.raise_for_status()
        return str(r.json().get("response", "")).strip() or None
    except (requests.RequestException, ValueError):
        return None
