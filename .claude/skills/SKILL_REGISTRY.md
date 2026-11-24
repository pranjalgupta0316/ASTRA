# Skill registry

| Name | Path | Activation (summary) | References | Owner code |
|---|---|---|---|---|
| quantitative-research | quantitative-research/SKILL.md | signal/factor/pair/regime/vol hypotheses, stat tests, multiple testing | hypothesis_template.md, methods.md | src/nebula/research, src/nebula/stats |
| financial-data-engineering | financial-data-engineering/SKILL.md | adapters, schemas, calendars, timezones, quality reports, storage | provider_notes.md | src/nebula/data |
| backtesting-audit | backtesting-audit/SKILL.md | engine, costs, fills, walk-forward, auditing any result | cost_and_cv.md | src/nebula/backtest |
| market-microstructure | market-microstructure/SKILL.md | order books, depth capture, trades, OFI, microprice | (inline) | src/nebula/microstructure |
| financial-ml-research | financial-ml-research/SKILL.md | ML/DL training & evaluation, ablations, GPU | (inline) | src/nebula/ml, src/nebula/dl |
| portfolio-risk-engineering | portfolio-risk-engineering/SKILL.md | optimizers, covariance, risk metrics | (inline) | src/nebula/portfolio, src/nebula/risk |
| research-paper-analysis | research-paper-analysis/SKILL.md | citations, derivations, reproductions | (inline) | docs/literature |

## Overlap with installed user/plugin skills (not part of this repo)
| External skill | Overlap | Resolution |
|---|---|---|
| quant-research-scientist (plugin) | general quant research | NEBULA skills take precedence inside this repo: they encode repo paths, contracts and cost tables |
| ml-experiment-reviewer (plugin) | experiment review | may be used as a second-opinion reviewer; NEBULA acceptance criteria still apply |
| deep-learning-research-mentor (plugin) | DL methodology | advisory only |
| superpowers:* (plugin) | process (TDD, debugging, planning) | complementary - process, not domain |
