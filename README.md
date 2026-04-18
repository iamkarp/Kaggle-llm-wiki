# Kaggle LLM Wiki

An LLM-maintained knowledge base for Kaggle competition insights, following the [llm-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern.

## What is this?

Instead of losing hard-won competition knowledge to notebook comments and forgotten notes, this wiki compounds ML insights across competitions. An LLM reads competition materials, extracts lessons, and maintains structured, interlinked markdown pages.

## Structure

```
raw/              # Immutable sources (writeups, papers, forum posts)
wiki/             # LLM-maintained pages
  index.md        # Content catalog
  log.md          # Changelog
  competitions/   # Per-competition pages
  techniques/     # ML methods and architectures
  patterns/       # Recurring cross-competition patterns
  tools/          # Platform and infrastructure knowledge
  mistakes/       # Anti-patterns and lessons learned
CLAUDE.md         # Schema and operational rules
```

## How to use

1. **Browse** — Read `wiki/index.md` for an overview, then follow links
2. **Ingest** — Drop competition materials into `raw/`, ask the LLM to process them
3. **Query** — Ask questions; valuable answers become new wiki pages
4. **Lint** — Periodically ask the LLM to check for contradictions and stale content

## Current Coverage

- **Santa 2024 — Pancake Sorting** — Combinatorial optimization with prefix reversals, 89,748 total moves
  - Beam search (C, 2-ply lookahead), inverse permutation solving, breakpoint heuristic
  - Why NMCS fails at high branching factors
  - C vs Python for compute-intensive search

- **3rd WEAR Dataset Challenge @ HASCA 2026** — HAR with wearable sensors, #1 on public LB
  - PatchTST, sensor embeddings, L-R swap augmentation, threshold optimization
  - Why GBM fails at cross-subject generalization
  - OOF vs LB divergence patterns

## Philosophy

> The tedious bookkeeping—updating cross-references, maintaining consistency, noting contradictions—is where humans abandon wikis. LLMs don't tire and can touch dozens of files in one pass.

The human curates sources and asks questions. The LLM handles maintenance.
