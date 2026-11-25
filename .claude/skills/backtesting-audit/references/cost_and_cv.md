# Costs and cross-validation reference

## Default ASTRA cost assumptions (per side, editable in config/markets.yaml)
| Market | Commission | Half-spread | Slippage | Notes |
|---|---|---|---|---|
| US large-cap | 0 bps (retail zero-commission) + SEC/TAF fees ~0.3 bps on sells | 1 bp | 2 bps | conservative for AAPL/SPY |
| India NSE cash (delivery) | ~ 0 brokerage at discount brokers + STT 10 bps per side on delivery + exchange/stamp ~0.5-1.5 bps | 2 bps | 3 bps | STT dominates; intraday STT differs |
| Crypto Binance spot | 10 bps taker (VIP0, no BNB discount) | 1 bp (BTC) | 2 bps | 24/7 |
These are assumptions, not measurements; always run sensitivity.

## Purged K-fold with embargo (Lopez de Prado, Advances in Financial Machine Learning, 2018, ch. 7)
- Purge: drop training samples whose label interval [t, t+h] overlaps the test fold.
- Embargo: additionally drop training samples within `embargo` bars after the test fold.
- Implemented in `astra.ml.validation.PurgedWalkForward`.

## Annualization
periods_per_year: US/IN equities 252, crypto daily 365, crypto hourly 8760. Sharpe = mean/std * sqrt(P).
Sharpe standard error (Lo 2002, iid approx): sqrt((1 + 0.5 SR^2)/T) on per-period SR.
