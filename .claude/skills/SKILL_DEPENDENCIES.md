# Skill dependencies

```
financial-data-engineering  ──► (contracts, calendars, availability timestamps) ──► all skills
research-paper-analysis     ──► methods ──► quantitative-research, financial-ml-research, market-microstructure
quantitative-research ─┐
financial-ml-research ─┼─► signals ──► portfolio-risk-engineering ──► weights ─┐
market-microstructure ─┘                                                       ▼
                                                          backtesting-audit (independent validator)
```

Rules
1. A result produced under quantitative-research / financial-ml-research / market-microstructure is
   only **validated** after a backtesting-audit pass by a different agent (or a separate, explicit
   audit step in the same session that re-runs the checks rather than re-reading the conclusion).
2. Any change to a data contract (financial-data-engineering) requires re-running dependent tests.
3. portfolio-risk-engineering owns annualization constants; others import them, never hard-code.
