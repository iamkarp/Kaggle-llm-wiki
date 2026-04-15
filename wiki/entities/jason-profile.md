---
title: "Jason — Human Collaborator Profile"
tags: [person, human, profile, preferences]
date: 2026-04-14
source_count: 2
status: active
---

## Summary
Jason is the human collaborator behind all projects in this workspace. He competes in Kaggle ML competitions, runs live forex trading strategies, and works in marketing/data (PMG TikTok). Based in CST timezone.

## Background
- ML practitioner focused on competitions (tabular, NLP, vision)
- Quantitative trading interest — live OANDA account with real money
- Professional context: PMG TikTok (marketing/data work)

## Technical Profile
- Comfortable with Python ML stack (XGBoost, LightGBM, CatBoost, PyTorch)
- Uses OANDA API for live trading
- Manages compute across multiple machines (middle-child, big-brother, little-brother)
- Familiar with Kaggle competition norms and submission strategy

## Preferences & Working Style
- **Approval gate on Kaggle submissions** — always ask before submitting
- **Approval gate on external actions** (messages, deploys, API calls with side effects)
- **Never change trading position sizing** (25 units hardcoded) without explicit instruction
- Prefers concise responses; trusts LLM judgment on execution details
- Timezone: America/Chicago (CST)

## Infrastructure
| Machine | Role | Specs |
|---------|------|-------|
| middle-child | Agent host (this machine) | 32GB Intel Mac |
| big-brother | Primary ML training | 192.168.4.243 |
| little-brother | GPU inference | 192.168.4.63, RTX 2070 Super |

**Critical**: Never run ML training on middle-child. Always SSH to workers.

## Active Projects (as of 2026-04-14)
- March Mania 2026 (v6 ensemble submitted, awaiting Stage 2 results)
- AUTOPILOT VQA (deadline April 15, 2026 — active)
- Live NFP straddle strategy (OANDA, running)

## Sources
- [[../../raw/system/user-profile.md]] — USER.md content
- [[../../raw/system/memory-main.md]] — MEMORY.md content

## Related
- [[../competitions/march-mania-2026]] — primary active competition
- [[../competitions/autopilot-vqa-2026]] — upcoming deadline competition
- [[../strategies/nfp-straddle-forex]] — live trading strategy
