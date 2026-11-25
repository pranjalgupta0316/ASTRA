# Skill registry

| Name | Path | Activation (summary) | References | Owner code |
|---|---|---|---|---|
| quantitative-research | quantitative-research/SKILL.md | signal/factor/pair/regime/vol hypotheses, stat tests, multiple testing | hypothesis_template.md, methods.md | src/astra/research, src/astra/stats |
| financial-data-engineering | financial-data-engineering/SKILL.md | adapters, schemas, calendars, timezones, quality reports, storage | provider_notes.md | src/astra/data |
| backtesting-audit | backtesting-audit/SKILL.md | engine, costs, fills, walk-forward, auditing any result | cost_and_cv.md | src/astra/backtest |
| market-microstructure | market-microstructure/SKILL.md | order books, depth capture, trades, OFI, microprice | (inline) | src/astra/microstructure |
| financial-ml-research | financial-ml-research/SKILL.md | ML/DL training & evaluation, ablations, GPU | (inline) | src/astra/ml, src/astra/dl |
| portfolio-risk-engineering | portfolio-risk-engineering/SKILL.md | optimizers, covariance, risk metrics | (inline) | src/astra/portfolio, src/astra/risk |
| research-paper-analysis | research-paper-analysis/SKILL.md | citations, derivations, reproductions | (inline) | docs/literature |

## Overlap with installed user/plugin skills (not part of this repo)
| External skill | Overlap | Resolution |
|---|---|---|
| quant-research-scientist (plugin) | general quant research | ASTRA skills take precedence inside this repo: they encode repo paths, contracts and cost tables |
| ml-experiment-reviewer (plugin) | experiment review | may be used as a second-opinion reviewer; ASTRA acceptance criteria still apply |
| deep-learning-research-mentor (plugin) | DL methodology | advisory only |
| superpowers:* (plugin) | process (TDD, debugging, planning) | complementary - process, not domain |
