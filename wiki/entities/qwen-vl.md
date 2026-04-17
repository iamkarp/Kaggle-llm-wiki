---
title: "Qwen-VL"
tags: [vlm, vision, qwen, multimodal, tool]
date: 2026-04-16
source_count: 1
status: draft
---

## What It Is

Qwen-VL is a vision-language model from Alibaba capable of understanding images and generating text descriptions. Used in Jason's work as the primary visual understanding component in VQA pipelines.

## Typical Use in Jason's Work

- Stage 1 of the AUTOPILOT VQA pipeline: generates scene descriptions from dashcam images
- Chosen for its strong zero-shot visual understanding and instruction-following capabilities

## Key Parameters Used

- Model variant: Qwen-VL-Chat or Qwen2-VL
- Inference: local GPU (little-brother) or API
- Prompt style: detailed scene description requests with domain-specific vocabulary

## Performance Notes

Good at generating detailed descriptions but can hallucinate scene elements. Structured extraction works better as a second stage (Claude Sonnet) rather than forcing Qwen-VL to output JSON directly.

## Related

- [[../strategies/autopilot-vqa-pipeline]] — primary usage context
- [[../entities/claude-sonnet]] — paired model for JSON extraction
- [[../concepts/multimodal-classification]] — underlying technique
