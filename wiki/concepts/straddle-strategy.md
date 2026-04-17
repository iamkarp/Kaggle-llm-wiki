---
title: "Straddle Strategy"
tags: [trading, forex, options, straddle, nfp]
date: 2026-04-16
source_count: 1
status: draft
---

## What It Is

A straddle strategy places simultaneous long and short orders around a known volatility event (e.g., NFP release), profiting from large directional moves regardless of direction. In forex, this is implemented with pending buy-stop and sell-stop orders bracketing the current price.

## When To Use It

- High-impact scheduled economic releases (NFP, CPI, FOMC)
- Price is expected to move significantly but direction is uncertain
- Spread and slippage are manageable relative to expected move size

## Mechanics

1. Before event: place buy-stop above and sell-stop below current price
2. Event triggers one order; the other becomes the stop-loss (or is manually cancelled)
3. Use trailing stop or fixed TP to manage the winning position
4. Key parameter: distance from current price to entry orders (too tight = whipsaw, too wide = missed move)

## Hyperparameters

- **Entry distance**: pips above/below current price (Jason uses ~15-25 pips for NFP EUR/USD)
- **Stop-loss**: typically the opposing order's level or a fixed pip amount
- **Trailing stop**: activated after minimum profit threshold
- **Position size**: fixed lot size per account (never auto-scaled)

## Gotchas

- Whipsaw: price triggers one order, reverses, triggers the other — both stopped out
- Spread widening during releases can trigger orders prematurely
- Slippage on stop orders during fast markets
- Never modify position sizing without explicit approval

## In Jason's Work

Implemented as the NFP straddle forex strategy via OANDA API, running on cron across two accounts (006/007). Backtested across multiple stop configurations.

## Related

- [[../strategies/nfp-straddle-forex]] — Jason's specific NFP implementation
- [[../entities/oanda]] — broker used for execution
- [[../comparisons/nfp-stop-configurations]] — stop config comparison
