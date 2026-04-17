---
title: "NFP Stop Configurations Comparison"
tags: [trading, forex, nfp, straddle, comparison]
date: 2026-04-16
source_count: 3
status: draft
---

## Summary

Comparison of different stop-loss and take-profit configurations for the NFP straddle strategy, based on backtest results across multiple NFP releases.

## Configurations Tested

| Config | Stop Type | Details | Source |
|--------|-----------|---------|--------|
| Fixed | Fixed SL/TP | Fixed pip distances for both stop and target | nfp-backtest-results.txt |
| Trailing | Trailing stop | Trailing stop activates after minimum profit | nfp-backtest-results-trailing.txt |
| Tighter | Tighter stops | Reduced entry distance and stop levels | nfp-backtest-results-tighter.txt |

## Key Findings

- Trailing stop captures more of large moves but gives back profit on reversals
- Tighter configurations increase fill rate but also increase whipsaw losses
- Fixed TP captures consistent profit but misses extended moves
- Optimal configuration depends on account risk tolerance and NFP volatility regime

## Open Questions

- Does the optimal configuration shift with VIX/volatility regime?
- Should different pairs (EUR/USD vs GBP/USD) use different configs?

## Sources

- `raw/trading/nfp-backtest-results` *(not yet ingested)* — baseline backtest
- `raw/trading/nfp-backtest-results-trailing` *(not yet ingested)* — trailing stop variant
- `raw/trading/nfp-backtest-results-tighter` *(not yet ingested)* — tighter stop variant

## Related

- [[../strategies/nfp-straddle-forex]] — parent strategy
- [[../concepts/straddle-strategy]] — general straddle mechanics
- [[../entities/oanda]] — execution broker
