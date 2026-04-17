---
title: "AUTOPILOT VQA Pipeline"
tags: [kaggle, cvpr, vqa, vision, pipeline, qwen, claude]
date: 2026-04-16
source_count: 2
status: draft
---

## Summary

Three-stage VLM pipeline for the AUTOPILOT VQA dashcam classification competition. Uses Qwen-VL for visual understanding and Claude Sonnet for structured JSON extraction across 25 classification columns.

## Architecture

```
Stage 1: Qwen-VL scene description → raw text per image
Stage 2: Claude Sonnet JSON extraction → structured 25-column output
Stage 3: Post-processing → validation, fallback defaults, submission formatting
```

## When To Use

Vision-language competitions requiring structured multi-label output from images, especially when:
- Output is multi-column categorical (not single-label classification)
- Pre-trained VLMs outperform training from scratch
- Prompt engineering replaces model fine-tuning

## Sources

- [[../../raw/kaggle/autopilot-vqa-strategy]] — competition strategy document
- [[../../raw/kaggle/autopilot-vqa-claude]] — Claude integration details

## Related

- [[../competitions/autopilot-vqa-2026]] — competition page
- [[../entities/qwen-vl]] — primary VLM used
- [[../entities/claude-sonnet]] — JSON extraction model
- [[../concepts/multimodal-classification]] — general technique
