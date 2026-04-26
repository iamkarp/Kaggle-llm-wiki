---
id: strategy:nfp-straddle-forex
type: strategy
title: NFP Adaptive Breakout-Reversal Straddle
slug: nfp-straddle-forex
aliases:
- nfp-straddle
- nfp-sar
- nfp-breakout-reversal
tags:
- forex
- trading
- straddle
- nfp
- sar
- stop-and-reverse
- oanda
- eur-usd
- usd-jpy
- gbp-usd
- breakout
- trailing-stop
- ratchet
status: active
created: 2026-04-14
updated: 2026-04-26
---

# NFP Adaptive Breakout-Reversal Straddle

## Summary

A latency-aware volatility breakout engine for Non-Farm Payroll (NFP) releases. Places staged pending stop orders (buy-stop above, sell-stop below) before release. Whichever side fills first, the opposite is cancelled. If the winning side gets stopped out, immediately flip to the opposite direction using freed margin (SAR). Max 2 attempts per event. Uses spread filters, volatility-adjusted stops, and a 3-regime ratcheting trailing stop that tightens as profit grows.

---

## Architecture

```
NFP Event
├── Pre-release: place buy-stop + sell-stop (staged, not simultaneous)
├── On fill: cancel opposite order, apply dynamic stop
├── Trail: 3-regime ratchet (survival → lock-in → squeeze)
└── On stop-out: immediately reverse (SAR), attempt 2 of 2
```

**Key constraint:** Never open both sides simultaneously. One fills, one cancels. This avoids double spread, margin lockup, and broker hedging restrictions.

---

## Entry Logic

### Pair Selection

Start with **EUR/USD** — deepest liquidity, tightest spread under stress. USD/JPY also viable (strong NFP sensitivity). GBP/USD has larger moves but worse noise.

### Spread Filter (pre-trade gate)

```
normal_spread = median spread over last 10 minutes

Allow trade only if:
    current_spread <= 2.5 × normal_spread
    AND current_spread <= 3.0 pips (absolute max for EUR/USD)

If spread > threshold: no trade this release.
```

### Order Placement (T - 0.5 seconds)

```
mid = current midpoint

breakout_distance = max(
    4 pips,
    1.5 × current_spread,
    0.35 × pre_release_1min_ATR
)

buy_stop  = mid + breakout_distance
sell_stop = mid - breakout_distance
```

### On Fill

```
cancel opposite pending order

initial_stop = max(
    6 pips,
    2.0 × current_spread,
    0.40 × first_5sec_range
)

position_size = account_risk_per_attempt / initial_stop_distance
```

---

## Stop & Trailing Logic

### 3-Regime Ratcheting Trail

| Regime | Open Profit | Action |
|--------|-------------|--------|
| Survival | 0 – 6 pips | Hold initial stop. No trail yet. |
| Lock-in | 6 – 15 pips | Move stop to entry + 1 pip (breakeven+) |
| Squeeze | 15 – 30 pips | Trail at 45% giveback: `stop = price - 0.45 × profit` |
| Squeeze+ | 30+ pips | Trail at 25% giveback: `stop = price - 0.25 × profit` |

**The farther it goes, the tighter the leash.**

Example locked profit at each level:

| Open profit | Giveback allowed | Locked profit |
|-------------|-----------------|---------------|
| 10 pips | 5 pips | 5 pips |
| 20 pips | 9 pips | 11 pips |
| 40 pips | 10 pips | 30 pips |
| 70 pips | 15 pips | 55 pips |

---

## Reversal (SAR)

When the first trade is stopped out, **immediately open the opposite direction** using freed margin. Same sizing logic applies.

```
On stop-out (attempt 1):
    if attempts < 2 AND spread <= 4 × normal_spread:
        open opposite direction
        apply same dynamic stop + trail logic
        attempts += 1
    else:
        done for this release
```

**Max 2 total attempts per NFP event.** No third trade. No martingale.

---

## Kill Switch Conditions

Stop trading immediately if any of:

- Spread > 4× normal spread
- Total event loss > max_event_loss (e.g. 2% of account)
- Both attempts used
- Both sides triggered within 1 second (chaos/whipsaw detected)
- Platform latency spike detected
- Price gaps through stop by more than 2× allowed slippage

---

## Pseudocode

```python
def nfp_strategy():
    attempts = 0
    max_attempts = 2
    max_event_loss = account_equity * 0.02

    normal_spread = median_spread(last_minutes=10)
    pre_atr = range_pips(last_seconds=60)

    wait_until(release_time - 0.5)

    if current_spread_pips() > 2.5 * normal_spread:
        return "No trade: spread too wide"

    mid = current_mid_price()
    breakout_distance = max(4, 1.5 * current_spread_pips(), 0.35 * pre_atr)

    place_buy_stop(mid + pips(breakout_distance))
    place_sell_stop(mid - pips(breakout_distance))

    while attempts < max_attempts:
        fill = wait_for_fill(timeout_seconds=10)
        if not fill:
            cancel_all_orders()
            return "No trade: no breakout"

        attempts += 1
        cancel_opposite_order(fill)

        first_range = range_pips(last_seconds=5)
        stop_pips = max(6, 2 * current_spread_pips(), 0.40 * first_range)
        size = risk_to_units(account_equity * 0.005, stop_pips)

        set_position(fill, size, stop_pips)

        while position_is_open():
            profit = open_profit_pips()
            spread = current_spread_pips()

            if spread > 4 * normal_spread:
                close_position()
                return "Exit: spread explosion"

            if profit < 6:
                pass  # hold initial stop
            elif profit < 15:
                move_stop_to_breakeven_plus(1)
            elif profit < 30:
                trail_stop(price - 0.45 * profit)
            else:
                trail_stop(price - 0.25 * profit)

        # stopped out — SAR
        if attempts < max_attempts and current_spread_pips() <= 4 * normal_spread:
            enter_reverse_trade()
        else:
            return "Done"

    return "Done"
```

---

## Risk Parameters

| Parameter | Value |
|-----------|-------|
| Pair | EUR/USD (primary), USD/JPY, GBP/USD |
| Risk per attempt | 0.5% of account |
| Max event loss | 2% of account |
| Max attempts | 2 |
| Entry timing | T - 0.5 seconds |
| Spread gate | ≤ 2.5× normal |
| Kill spread | > 4× normal |

**Never size by fixed units. Size by % risk of account.**

---

## Live Accounts (OANDA)

| Account ID | Role |
|-----------|------|
| `001-001-160511-005` | Sniper |
| `001-001-160511-006` | Auto Trader 1 |
| `001-001-160511-007` | Auto Trader 2 |

Credentials: `trading/.oanda.env` — never commit.

## Execution

```bash
cd /Users/admin/Documents/Openclaw/workspace/trading/
python live_straddle_v2.py
# cron: bash run_nfp_straddle.sh
```

Log: `trading/nfp_cron.log`

---

## What Worked (prior version)

- Trailing stop dramatically outperforms fixed stop
- USD/JPY consistently highest pip moves on USD data
- 4-pip initial stop cuts loser fast

## What This Version Fixes

- Spread filter prevents entering into illiquid pre-release conditions
- Staged orders (not simultaneous) avoid margin lockup + hedging rule issues
- Dynamic stop sized to actual volatility, not fixed pip
- Ratcheting trail locks more profit on extended moves
- SAR reversal captures the "fake first spike → real move" pattern
- Max 2 attempts prevents revenge trading

---

## Sources

- `raw/trading/nfp-backtest-results-trailing.txt` — trailing stop backtest
- `raw/trading/nfp-backtest-results-tighter` — tighter stop variant

## Related

- [[../entities/oanda]] — broker API
- [[../concepts/straddle-strategy]] — straddle mechanics
- [[../comparisons/nfp-stop-configurations]] — stop config comparison

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _uses_ → [[entities/oanda|OANDA]]
- _requires_ → [[concepts/straddle-strategy|Straddle Strategy]]
- _compared_to_ → [[comparisons/nfp-stop-configurations|NFP Stop Configurations Comparison]]

**Incoming:**
- [[comparisons/nfp-stop-configurations|NFP Stop Configurations Comparison]] _compared_to_ → here
- [[entities/jason-profile|Jason — Human Collaborator Profile]] _works_with_ → here
- [[entities/oanda|OANDA]] _related_to_ → here
- [[index|Wiki Index]] _related_to_ → here

<!-- kg:end -->
