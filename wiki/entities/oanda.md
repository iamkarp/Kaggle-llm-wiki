---
title: "OANDA"
tags: [forex, broker, api, trading, tool]
date: 2026-04-16
source_count: 1
status: draft
---

## What It Is

OANDA is a forex broker providing a REST API (v20) for automated trading. Jason uses it for live execution of the NFP straddle strategy across two practice/live accounts.

## Typical Use in Jason's Work

- Live execution of NFP straddle orders via OANDA v20 API
- Accounts 006 (primary) and 007 (secondary) — never combine or transfer between them
- Automated order placement via `live_straddle_v2.py` running on cron

## Key Parameters Used

- API: `api-fxpractice.oanda.com` (practice) / `api-fxtrade.oanda.com` (live)
- Instruments: EUR_USD, GBP_USD (NFP-sensitive pairs)
- Credentials stored in `.oanda.env` — never committed to git

## Performance Notes

API is reliable but has rate limits. Order execution latency is typically <100ms for market orders. The v20 API supports streaming prices and trailing stops natively.

## Version / Installation

- API version: v20
- Python client: `oandapyV20` or raw REST via `requests`

## Related

- [[../strategies/nfp-straddle-forex]] — primary trading strategy using OANDA
- [[../concepts/straddle-strategy]] — general straddle mechanics
