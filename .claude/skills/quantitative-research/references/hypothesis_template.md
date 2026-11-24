# Hypothesis template

```
ID:              H-<US|IN|CR|XM>-<nnn>
Title:
Status:          proposed | running | supported | rejected | inconclusive
Registered:      YYYY-MM-DD (before results were seen)
1. Intuition:          why should this exist economically / behaviourally / structurally?
2. Formulation:        exact formula of signal s_t and target y_{t+h}
3. Data:               datasets, frequency, availability lag
4. Features:           list with definitions (link DATA_DICTIONARY.md)
5. Target:             horizon h, return type (log/simple), excess or raw
6. Mechanism:          who is on the other side of the trade, why it is not arbitraged away
7. H0:                 e.g. "IC(s_t, y_{t+h}) = 0" or "net Sharpe <= 0"
8. Validation design:  split dates, walk-forward scheme, purge/embargo length
9. Costs:              fee + half-spread + slippage assumption per market
10. Failure modes:     what would make the result spurious
Variants tried:        running count (feeds multiple-testing correction)
Result:                metrics with CIs, link to experiment IDs
Conclusion:
```
