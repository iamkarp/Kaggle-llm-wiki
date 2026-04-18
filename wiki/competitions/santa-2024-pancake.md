---
title: Santa 2024 — Pancake Sorting
category: competitions
tags: [combinatorial-optimization, beam-search, permutation, pancake-sorting]
created: 2026-04-16
updated: 2026-04-16
---

# Santa 2024 — Pancake Sorting

**Platform:** Kaggle  
**Metric:** Total prefix reversal moves across all test cases (lower is better)  
**Result:** 89,748 total moves (best local), 89,831 submitted. Top teams: 89,573 (4 tied).  
**Problem:** Given 2,405 permutations of sizes n=5 to n=100, sort each to `[0,1,...,n-1]` using prefix reversals Rk (reverse the first k elements). Minimize total moves. UNSOLVED cases cost 100,000 each.

## Problem Structure

| n | Cases | Breakpoint sum | Our total | Moves/n |
|---|-------|---------------|-----------|---------|
| 5 | 5 | 18 | 18 | 0.72 |
| 12 | 200 | 2,200 | 2,167 | 0.90 |
| 15 | 200 | 2,881 | 2,825 | 0.94 |
| 16 | 200 | 3,160 | 3,094 | 0.97 |
| 20 | 200 | 3,485 | 3,422 | 0.86 |
| 25–50 | 800 | 21,024 | ~20,600 | ~0.91 |
| 75 | 200 | 14,851 | ~14,800 | ~0.99 |
| 100 | 200 | 19,682 | ~19,700 | ~0.99 |

Key insight: n=5 through n=16 are provably optimal via IDA*. All improvement effort goes into n≥20.

## What Worked

### 1. C beam search with 2-ply lookahead (`beam_search2.c`)
The workhorse solver. Written in C for speed (~100x faster than Python). Uses breakpoint count as the primary heuristic with 2-ply lookahead: for each candidate flip, simulate the best possible follow-up flip and score that state.

- Beam widths: 300–2000 depending on n
- Stochastic: uses seeded random noise for tie-breaking
- Run with many seeds (10–40 per case) to find diverse solutions

### 2. Inverse permutation direction
Solving the inverse permutation (where `inv[perm[i]] = i`) and then reversing the move sequence. Almost all improvements for n≥35 came from the INV direction. The forward and inverse search spaces have different local optima — running both doubles your chances of finding shorter solutions.

See [[techniques/inverse-problem-solving]].

### 3. Breakpoint heuristic
A breakpoint at position i means `|perm[i] - perm[i-1]| != 1`. Each prefix reversal can remove at most 1 breakpoint, so `breakpoints(perm)` is a lower bound. The admissible heuristic for IDA* is `ceil(bp/2)`.

In practice: small n solutions beat the breakpoint count (solved below the lower bound via multi-breakpoint removals), while n=100 solutions slightly exceed it.

### 4. Massive parallelization with seed determinism
beam_search2 is fully deterministic for a given seed. This enables:
- Running 40+ seeds per case across multiple solver scripts
- Reconstructing any improvement later by replaying the exact seed
- Partitioning seed ranges across concurrent solvers (0–460M, 700M, 800M, 900M, 1B+, 2B, 3B)

### 5. IDA* for small n
Iterative deepening A* with `ceil(bp/2)` admissible heuristic. Guarantees optimal solutions for n=5, 12, 15, 16. Not feasible for n≥20 due to exponential branching.

## What Failed

### 1. Nested Monte Carlo Search (NMCS)
Completely unviable for n=100. The branching factor (n-1=99 choices per step) makes even level-1 NMCS impossibly slow.

See [[mistakes/nmcs-large-search-spaces]].

### 2. CayleyPy library
- No pre-trained models for pancake sorting (only LRX puzzle models)
- Hamming distance heuristic gets stuck at local minima for n≥20
- 1-ply beam search finds zero improvements over our 2-ply C solver

### 3. Move replacement ("Replacing Moves" technique)
Scanning solutions for 3-move subsequences that can be replaced by 1 or 2 moves. Found zero improvements — solutions were already locally optimal.

### 4. Python-based solvers
Too slow for n=100. A single beam search evaluation at bw=300 takes ~5 minutes in C vs ~8+ hours in Python. C is non-negotiable for this problem.

See [[patterns/c-vs-python-compute]].

## Key Lessons

1. **Top teams used GPU neural beam search.** Pilgrim ResNet (1024→512, 4 residual blocks, 400 epochs) trained to predict optimal move count, then used as beam search heuristic with bw=4K–65K on T4 GPU. This is likely the only path below 89,500.

2. **Inverse direction is not optional.** For combinatorial problems where you can solve forward or backward, always try both. The improvement rate from INV was ~3x higher than FWD.

3. **Seed diversity matters more than beam width.** 20 runs at bw=300 found more improvements than 5 runs at bw=2000. Stochastic restarts explore more of the search space.

4. **Breakpoints are loose but useful.** The gap between breakpoint lower bound and actual solution length is where all the optimization happens. The top teams closed this gap completely on most cases.

5. **Score 89,000 is not achievable without neural heuristics.** Closing the 175-move gap to top teams requires ~574 below-breakpoint improvements — needs a fundamentally better heuristic, not more CPU time.

## Timeline

- Day 1: Python solver with BFS (n=5), IDA* (n≤16), basic beam search. Score ~91,000.
- Day 1-2: C beam search with 2-ply lookahead. Score ~90,000 → 89,852.
- Day 2-3: Inverse permutation, targeted solver, n100 focused. Score 89,852 → 89,748.
- Kaggle GPU notebook (Pilgrim model): attempted but competition ended before results.

## See Also
- [[techniques/beam-search]]
- [[techniques/inverse-problem-solving]]
- [[patterns/c-vs-python-compute]]
- [[mistakes/nmcs-large-search-spaces]]
