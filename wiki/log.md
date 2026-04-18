---
title: Wiki Log
---

# Changelog

## 2026-04-16 ingest | Santa 2024 Pancake Sorting

Added competition page and supporting technique/pattern/mistake pages from the Santa
2024 pancake sorting competition. Key findings: C beam search with 2-ply lookahead is
the workhorse; inverse permutation solving provides ~70% of improvements; NMCS is
unviable at high branching factors; neural beam search (GPU-trained ResNet heuristic)
is required to match top teams. New pages: `competitions/santa-2024-pancake`,
`techniques/beam-search`, `techniques/inverse-problem-solving`,
`patterns/c-vs-python-compute`, `mistakes/nmcs-large-search-spaces`.

## 2026-04-15 ingest | 3rd WEAR Dataset Challenge HASCA 2026

Initial wiki creation. Seeded with knowledge from competing in the 3rd WEAR Dataset
Challenge. Created pages for all techniques used, patterns discovered, mistakes made,
and tools learned. Key finding: PatchTST with sensor embedding is the only approach
that generalizes cross-subject; GBM with handcrafted features fails catastrophically
on unseen subjects despite strong OOF scores.
