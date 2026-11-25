"""Runtime research agents: deterministic Python roles over ASTRA's stored data and experiment records.

These are NOT Claude Code subagents and need no LLM. Every finding cites the file / experiment id it
came from. An optional local LLM (Ollama-compatible HTTP API) may only *narrate* finished reports; it
never computes numbers or decides conclusions.
"""
