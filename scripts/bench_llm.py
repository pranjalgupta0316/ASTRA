"""Benchmark the optional local LLM (Ollama) for NEBULA report narration.

Measures load time, prompt/eval throughput, VRAM (nvidia-smi) and a faithfulness check: every number
in the generated summary must appear in the source report (invented numbers = hallucinations).
Usage: uv run python scripts/bench_llm.py [MODEL]   (server must run on 127.0.0.1:11434)
Writes data/reports/llm_benchmark.json.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time

import requests

from nebula.agents.llm import NUM_CTX, PROMPT
from nebula.common.paths import get_paths

URL = "http://127.0.0.1:11434"
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")


def vram_used_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def numbers(text: str) -> set[str]:
    return {n.rstrip("%").lstrip("-").rstrip("0").rstrip(".") or "0" for n in NUM.findall(text)}


def main(model: str = "qwen2.5:7b") -> None:
    paths = get_paths()
    report = (paths.reports / "research" / "latest.md").read_text(encoding="utf-8")
    requests.post(f"{URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)  # unload
    time.sleep(2)
    base_vram = vram_used_mb()
    runs = []
    for i in range(3):
        t = time.perf_counter()
        r = requests.post(
            f"{URL}/api/generate",
            json={
                "model": model,
                "prompt": PROMPT + report[:20000],
                "stream": False,
                "options": {"temperature": 0, "seed": 1, "num_ctx": NUM_CTX},
            },
            timeout=600,
            allow_redirects=False,
        )
        r.raise_for_status()
        j = r.json()
        wall = time.perf_counter() - t
        summary = j.get("response", "")
        invented = sorted(numbers(summary) - numbers(report))
        runs.append(
            {
                "run": i,
                "wall_s": round(wall, 2),
                "load_s": round(j.get("load_duration", 0) / 1e9, 2),
                "prompt_tokens": j.get("prompt_eval_count"),
                "prompt_tok_per_s": round(
                    j.get("prompt_eval_count", 0) / max(j.get("prompt_eval_duration", 1), 1) * 1e9, 1
                ),
                "output_tokens": j.get("eval_count"),
                "output_tok_per_s": round(
                    j.get("eval_count", 0) / max(j.get("eval_duration", 1), 1) * 1e9, 1
                ),
                "vram_used_mb": vram_used_mb(),
                "numbers_in_summary": len(numbers(summary)),
                "numbers_not_in_report": invented,
                "summary": summary,
            }
        )
    ps = requests.get(f"{URL}/api/ps", timeout=10).json()
    out = {
        "model": model,
        "baseline_vram_mb": base_vram,
        "loaded": ps.get("models", []),
        "runs": runs,
        "deterministic": len({r["summary"] for r in runs}) == 1,
    }
    dest = paths.reports / "llm_benchmark.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in runs:
        print({k: v for k, v in r.items() if k != "summary"})
    print(
        "deterministic:",
        out["deterministic"],
        "| loaded:",
        [(m.get("name"), m.get("size_vram")) for m in out["loaded"]],
    )
    print("--- summary (run 0) ---\n" + runs[0]["summary"])


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
