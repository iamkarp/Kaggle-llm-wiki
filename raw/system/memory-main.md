# MEMORY.md - Long-Term Memory

## Identity
- My name is Middle (named after the host: Middle-Child)
- Took over from previous agent "Noman" 👻 on 2026-02-22
- Vibe: direct, dry, gets things done

## People
- **Jason** — my human. First contact 2026-02-22 via Telegram. Based in CST timezone.
- Previous assistant was named Noman — I absorbed all his context.

## Infrastructure
- **This machine**: middle-child (32GB RAM Intel Mac, user=admin, Tailscale 100.123.245.18)
- **Homebrew**: `/usr/local/` (Intel, not ARM)
- **Workers** (SSH from here):
  - big-brother: 192.168.4.243 (primary ML worker)
  - little-brother: 192.168.4.63 (GPU: RTX 2070 Super)
- **🚨 CRITICAL**: NEVER run ML/training on middle-child. Always SSH to big-brother or little-brother.
- **Workspace**: Previous workspace at `/Users/admin/Documents/clawd/` (rsync'd from Mac mini)
- **OpenClaw migrated** from Mac mini → middle-child as of 2026-02-21

## Forex Trading
- **OANDA LIVE accounts** (real money, not practice):
  - `001-001-160511-005` — Sniper account (~$418)
  - `001-001-160511-006` — Auto Trader 1 ($50, BUY straddle)
  - `001-001-160511-007` — Auto Trader 2 ($50, SELL straddle)
  - Token in `trading/.oanda.env`
- **Strategies**: News Sniper (RSS → surprise → trade) + Straddle (buy+sell at event time)
- **Rule**: Trade every high-impact event at 25 units. NEVER change position sizing without Jason's explicit instruction.
- **Data**: InvestingLive RSS (~10-80s delay)
- **Scripts**: `~/Documents/clawd/trading/news_sniper.py`, `straddle_sniper.py`, etc.

## Kaggle Competitions

### March Mania 2026 ($50K prize)
- **Best submission**: v6 = 0.02210 LB (Stage 1, overfit — honest MSE ~0.145)
- **Pipeline**: 35% v2.8 + 35% v2.9 + 30% v5_hybrid
- **Deadline**: Stage 2 = Mar 19, 2026
- **Files**: big-brother `~/march-mania-2026/`
- **Key rule**: v6 is the ceiling — new models need <0.05 honest MSE to help

### Harmonizing the Data of Your Data
- **Score**: 0.444 (7th place)
- **Jason's notebook**: Hardcoded SDRF extractions for all 15 test PXDs from actual papers
- **Rule**: MUST get Jason's approval before submitting (Sonnet burned 5 slots)
- **Deadline**: Mar 18, 2026

### PAML Author Prediction
- **#1 on LB** (0.361) — deadline Feb 22, 2026 (today)
- Jason submitting personally — hands off

### Bidding Predictions
- **#6** (0.290) — deadline Mar 2, 2026

## PMG TikTok POV
- v4 PDF delivered (`pmg_tiktok_pov_v4.pdf`)
- Target live date: 3/9/26 — awaiting Jason's feedback

## Key Lessons from Noman
- Gate Kaggle submissions behind Jason approval
- Never change position sizing in trading without explicit instruction
- Sub-agent chaos is real — always verify termination before spawning replacements
- ML on workers only, never on middle-child

## Notes
- Fresh takeover as of 2026-02-22. Still learning what Jason needs day-to-day.
