# NEBULA project skills

Project-level Claude Code skills (`.claude/skills/<name>/SKILL.md`). Claude Code discovers them
automatically when a session is opened in `C:\Projects\NEBULA`; each loads only when its
`description` matches the task (progressive disclosure: `SKILL.md` is short, detail lives in
`references/`).

| Skill | One line |
|---|---|
| quantitative-research | hypothesis-first alpha / stats / regimes / volatility research |
| financial-data-engineering | provenanced, validated, point-in-time data |
| backtesting-audit | event-driven engine, costs, leakage audit - independent validator |
| market-microstructure | L1/L2 semantics, Binance book reconstruction, OFI, microprice |
| financial-ml-research | baselines first, purged walk-forward, PyTorch on 8 GB GPU |
| portfolio-risk-engineering | CVXPY optimizers, shrinkage covariance, VaR/ES, annualization |
| research-paper-analysis | verified citations, honest reproductions |

See `SKILL_REGISTRY.md`, `SKILL_DEPENDENCIES.md`, `SKILL_VALIDATION_REPORT.md`.
These are **development-time** aids; NEBULA's runtime does not depend on them or on Claude.
