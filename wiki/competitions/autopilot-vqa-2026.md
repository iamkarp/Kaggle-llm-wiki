---
title: "AUTOPILOT VQA — Dashcam Visual Question Answering"
tags: [kaggle, cvpr, vqa, vision, multimodal, qwen, claude, ensemble, classification]
date: 2026-04-14
source_count: 2
status: active
---

## Summary
Kaggle/CVPR competition: classify 25 questions about dashcam video clips. Metric is mean per-question accuracy across all 25 columns. Jason's strategy uses a hierarchical VLM pipeline (Qwen2.5-VL-32B for bulk + Claude Sonnet for causal reasoning) with majority-vote ensemble. Deadline April 15, 2026.

## Competition Metadata
- **Platform**: Kaggle (CVPR 2026 challenge)
- **Prize**: $300
- **Metric**: Mean per-question accuracy (25 questions, averaged)
- **Deadline**: April 15, 2026
- **Team**: Jason (solo)
- **Submissions Used**: 0 (as of bootstrap — first-mover advantage)
- **Dataset**: 2COOOL benchmark (COOOL, DADA-2000, Nexar collision footage)
- **Reference Paper**: [arXiv:2510.12190](https://arxiv.org/abs/2510.12190) — 2COOOL CVPR 2nd-place solution

## Task Description
Each video clip has 25 columns of questions covering:
- Weather, Lighting, Traffic Environment, Road Configuration
- Incident entities and their behavior
- Prevention strategies, Impact points
- Traffic control, Road surface conditions

Answers are **integer codes** (varies by question). Special values:
- `-1` = unknown
- `-2` = N/A (entity/condition doesn't apply)

Submission format: same column headers as `sample_submission.csv` (⚠️ headers contain literal newlines — do not strip).

## Strategy Summary
3-stage hierarchical VLM pipeline per video. Full detail: [[../strategies/autopilot-vqa-pipeline]].

**Stage 1 — Scene Captioning**: Sample 5 keyframes using optical flow (motion boundary detection); describe weather, lighting, road, entities.

**Stage 2 — Incident Detection**: Identify 1–2 peak incident frames from captions.

**Stage 3 — Structured Q&A**: Ask all 25 questions with full label legend; output JSON with integer codes.

**Ensemble**: Majority vote per column across runs; ties → trust Claude Sonnet output.

Key insight from 2COOOL paper: 4–6 high-motion frames beat 64 uniform frames across benchmarks.

## N/A Auto-Fill Rules (Critical)
Correctly filling `-2` boosts score significantly:
- Vehicle-type questions → `-2` if entity is not a vehicle
- Impact-point questions → `-2` if no collision detected
- Road-surface questions → `-2` if indoor/parking

## Model Stack
| Model | Role | Hardware |
|-------|------|---------|
| Qwen2.5-VL-32B | Bulk inference (23 of 25 questions) | big-brother / little-brother via vLLM |
| Claude claude-sonnet-4-6 | Causal reasoning (Q5, Q6) + ensemble tiebreak | Anthropic API |

## Timeline (from strategy doc)
| Days | Milestone |
|------|-----------|
| 1–3 | Baseline inference, submission format verified |
| 4–8 | Full 3-stage pipeline |
| 9–12 | Ensemble across multiple runs |
| 13–16 | Calibration and N/A rule tuning |
| 17+ | Final push, submission |

## Key Dependencies
```
transformers, accelerate, qwen-vl-utils, opencv-python, pandas, tqdm, anthropic, vllm, gdown
```

## What Worked
*(to be filled after submissions)*

## What Didn't Work
*(to be filled after submissions)*

## Open Questions
- Which N/A auto-fill rules give the biggest score lift?
- Does the 5-frame optical-flow sampling outperform uniform sampling on this dataset?
- Does Claude provide meaningful uplift on Q5/Q6 vs Qwen alone?

## Key Files
- `raw/kaggle/autopilot-vqa-strategy.md` — complete winning strategy
- `raw/kaggle/autopilot-vqa-claude.md` — competition overview for Claude Code
- Scripts: `autopilot_vqa.py`, `run_vqa_local.py`, `run_vqa_sonnet.py`
- `sample_submission.csv` — exact submission format (25 columns with literal newlines)

## Sources
- [[../../raw/kaggle/autopilot-vqa-strategy.md]] — hierarchical VLM pipeline strategy
- [[../../raw/kaggle/autopilot-vqa-claude.md]] — competition structure and label legend

## Related
- [[../strategies/autopilot-vqa-pipeline]] — detailed pipeline page
- [[../entities/qwen-vl]] — Qwen2.5-VL model details
- [[../entities/claude-sonnet]] — Claude's role in the ensemble
- [[../concepts/multimodal-classification]] — broader technique context
