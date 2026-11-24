# Agent workflows

## W1 - New research result
1. quant-researcher / ml-researcher registers hypothesis -> runs experiment -> logs experiment ID.
2. Handoff package: hypothesis ID, experiment ID, command to reproduce, claimed metrics.
3. backtest-auditor re-runs the command, completes the checklist, returns PASS/FAIL per item.
4. Orchestrator reconciles: FAIL items go back to the producer; result marked `validated` only when all
   applicable items PASS. Disagreements are written into the experiment's notes, not silently resolved.

## W2 - New data source
data-engineer verifies terms -> adapter + tests -> ingestion -> quality report ->
test-security-reviewer checks network hygiene (timeouts, rate limits, no secrets) -> orchestrator
updates DATA_SOURCES.md status.

## W3 - Milestone close
test-security-reviewer runs pytest/ruff/pyright -> orchestrator updates PROJECT_STATUS.md,
IMPLEMENTATION_LOG.md, KNOWN_ISSUES.md, NEXT_STEPS.md with exact commands and results.

## Handoff format (every agent's final message)
```
Files changed: ...
Commands run: ... (with exit status)
Results: ... (numbers, not adjectives)
Open issues: ...
Status: planned | designed | implemented | tested | validated
```
