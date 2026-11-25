"""Optional local LLM narration via an Ollama-compatible HTTP API.

Disabled unless ASTRA_LLM_URL (e.g. http://127.0.0.1:11434) and ASTRA_LLM_MODEL are set. Only
localhost URLs are accepted, so report contents never leave the machine. Returns None on any failure:
the platform never depends on it. Status: client tested against a fake local server only; no model
has been benchmarked on this machine (none installed).
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import requests

NUM_CTX = 8192
PROMPT = (
    "Summarize this quantitative research report in at most 8 sentences for a technical reader. "
    "Do not invent numbers; quote only numbers that appear in the report; keep every caveat.\n\n"
)


def llm_config() -> tuple[str, str] | None:
    url, model = os.environ.get("ASTRA_LLM_URL"), os.environ.get("ASTRA_LLM_MODEL")
    if not url or not model:
        return None
    host = urlparse(url).hostname or ""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("ASTRA_LLM_URL must point to a local server (data must not leave the machine)")
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
                # Ollama's default context is 2048 tokens, which silently truncated reports (~6k tokens)
                # in the benchmark; 8192 fits the RTX 4060's 8 GB with a 7B q4 model.
                "options": {"temperature": 0, "seed": 1, "num_ctx": NUM_CTX},
            },
            timeout=timeout_s,
            allow_redirects=False,  # a local proxy must not be able to forward report contents off-machine
        )
        r.raise_for_status()
        return str(r.json().get("response", "")).strip() or None
    except (requests.RequestException, ValueError):
        return None
