---
title: "Claude Sonnet"
tags: [llm, claude, anthropic, json-extraction, tool]
date: 2026-04-16
source_count: 1
status: draft
---

## What It Is

Claude Sonnet is Anthropic's mid-tier LLM, balancing capability with speed and cost. Used in Jason's work primarily for structured data extraction tasks where instruction-following precision matters more than raw generation quality.

## Typical Use in Jason's Work

- Stage 2 of the AUTOPILOT VQA pipeline: extracts structured 25-column JSON from Qwen-VL scene descriptions
- Chosen for reliable JSON output formatting and strong instruction adherence

## Key Parameters Used

- API-based inference via Anthropic SDK
- Temperature: 0 for deterministic extraction
- System prompt includes exact schema definition with valid values per column

## Performance Notes

Excels at structured extraction. Output format adherence is significantly better than open-source alternatives for complex multi-field JSON. Cost-effective for batch extraction at competition scale.

## Related

- [[../strategies/autopilot-vqa-pipeline]] — primary usage context
- [[../entities/qwen-vl]] — paired model for visual understanding
- [[../entities/machine-learning-advisor]] — another system using LLM synthesis
