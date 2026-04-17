---
title: "NFP Straddle — Forex Volatility Strategy on Non-Farm Payroll"
tags: [forex, trading, straddle, nfp, oanda, usd-jpy, eur-usd, gbp-usd]
date: 2026-04-14
source_count: 2
status: active
---

## Summary
A simultaneous BUY + SELL straddle placed on USD_JPY, EUR_USD, and GBP_USD around Non-Farm Payroll (NFP) release events. One direction stops out; the other rides momentum. Backtested 2015–present, best results with 4-pip initial stop and 5-pip trailing stop. Live on OANDA with 25 units per trade (hardcoded — DO NOT CHANGE).

## Architecture

```
NFP Event
├── BUY straddle (account 006)  → rides if NFP bullish USD
└── SELL straddle (account 007) → rides if NFP bearish USD

Stop logic:
- Initial stop: 4 pips (cuts loser quickly)
- Trailing stop: 5 pips (locks in winner's gains)
```

## Live Accounts (OANDA, real money)
| Account ID | Role | Capital |
|-----------|------|---------|
| `001-001-160511-005` | Sniper | ~$418 |
| `001-001-160511-006` | Auto Trader 1 — BUY straddle | $50 |
| `001-001-160511-007` | Auto Trader 2 — SELL straddle | $50 |

Credentials: `trading/.oanda.env` (never commit this file).

## Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Instruments | USD_JPY, EUR_USD, GBP_USD | Highest NFP volatility, best liquidity |
| Position size | **25 units** | Hardcoded. Never change without explicit Jason instruction |
| Initial stop | 4 pips | Cuts losing leg; prevents whipsaw from taking both sides |
| Trailing stop | 5 pips | Locks in gains as winner extends |
| Entry timing | At NFP release | Straddle placed pre-release |

## Backtest Results (Best Configuration: 4-pip stop, 5-pip trail)

| Instrument | Net Pips | Avg Pips/Trade | Wins | Losses |
|-----------|----------|----------------|------|--------|
| EUR_USD | 2,478.4 | $13.85 | 114 | 65 |
| GBP_USD | 2,774.5 | $15.50 | 122 | 57 |
| USD_JPY | 3,387.0 | $16.77 | 120 | 56 |

Win rates approximately 65% across all three pairs with trailing stops.

## Comparison: Stop Configurations Tested
| Config | EUR_USD | GBP_USD | USD_JPY |
|--------|---------|---------|---------|
| 5-pip fixed | 1,324 | 1,748 | 2,869 |
| 4-pip fixed | 1,537 | — | 3,067 |
| 2-pip fixed | — | 1,918 | — |
| 4-pip + 5-pip trail | **2,478** | **2,775** | **3,387** |

Trailing stop is clearly superior — it turns a ~40% win rate into ~68% by letting winners run.

## Execution
```bash
cd /Users/admin/Documents/Openclaw/workspace/trading/
python live_straddle_v2.py
# Or use the cron script: bash run_nfp_straddle.sh
```

Log: `trading/nfp_cron.log`

## What Worked
- Trailing stop dramatically outperforms fixed stop (captures trend extensions after NFP surprise)
- USD_JPY is consistently the best performer (highest pip moves on USD data)
- Simultaneous BUY/SELL ensures participation regardless of NFP direction
- 4-pip initial stop is tight enough to kill the loser fast

## What Didn't Work / Risks
- Fixed stops leave 1000+ pips of potential on the table vs trailing
- Tighter initial stops (2-pip) hurt EUR_USD; pairs have different typical noise levels
- During low-surprise NFP (number near consensus), both legs may stop out
- Slippage risk at exact NFP release (illiquid moment); bot uses OANDA streaming

## Safety Rules
1. **Never change 25-unit position size** without explicit Jason instruction
2. Monitor `nfp_cron.log` after each release
3. Accounts 006/007 are separate — never combine or transfer between them
4. Credentials in `.oanda.env` — never commit to git

## Sources
- [[../../raw/trading/nfp-backtest-results-trailing.txt]] — trailing stop backtest results
- `raw/trading/nfp-backtest-results-tighter` *(not yet ingested)* — tighter stop backtest results

## Related
- [[../entities/oanda]] — broker API details
- [[../concepts/straddle-strategy]] — general straddle mechanics
- [[../comparisons/nfp-stop-configurations]] — detailed comparison of stop configs
