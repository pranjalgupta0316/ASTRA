# Agent architecture (development time)

Two separate systems - do not confuse them:

1. **Development agents** (this folder): Claude Code subagents in `.claude/agents/` that help build
   NEBULA. They exist only while Claude Code runs.
2. **Runtime research workflows** (`src/nebula/agents/`): deterministic Python pipelines with
   optional local-LLM narration. NEBULA runs without Claude and without any LLM.

## Roster (6 subagents + orchestrator)
| Agent | Writes to | Reads | Skill |
|---|---|---|---|
| main session = Project Architect / orchestrator | anything; reconciles conflicts, owns docs/ADRs | all | all |
| data-engineer | src/nebula/data, config/markets.yaml, DATA_* docs | | financial-data-engineering |
| quant-researcher | src/nebula/research, src/nebula/stats, RESEARCH_HYPOTHESES | data | quantitative-research |
| ml-researcher | src/nebula/ml, src/nebula/dl | data, features | financial-ml-research |
| microstructure-specialist | src/nebula/microstructure | data | market-microstructure |
| portfolio-risk-analyst | src/nebula/portfolio, src/nebula/risk | | portfolio-risk-engineering |
| backtest-auditor | **nothing** (read + run only) | all | backtesting-audit |
| test-security-reviewer | **nothing** (read + run only) | all | - |

Recommended 8 roles were merged to 6: the Project Architect role is the main session (a subagent
cannot orchestrate other subagents), and "Backtesting Auditor" stays separate from producers to keep
validation independent.

## Concurrency rules
- Parallel only for non-overlapping write sets (table above). Two writers never share a directory.
- Shared files (`pyproject.toml`, `src/nebula/common/`, docs at repo root) are edited only by the
  orchestrator.
- Auditors are read-only, so they may run in parallel with anything.
