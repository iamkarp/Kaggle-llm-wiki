---
title: "Combinatorial Optimization in Kaggle Competitions"
tags: [optimization, santa, traveling-salesman, simulated-annealing, genetic-algorithm, lux-ai, kaggle]
date: 2026-04-18
source_count: 1
status: active
---

## Summary

Kaggle's annual Santa/optimization competitions (2015–2023) and traveling salesman variants require combinatorial optimization algorithms, not ML. The dominant approaches are: simulated annealing (most universal), genetic algorithms, or-tools / LKH for TSP, and Monte Carlo Tree Search for game-tree searches. GPU-parallelized annealing is increasingly the winning move for large search spaces.

## What It Is

A class of Kaggle competitions where the objective is to find the best arrangement/assignment/route rather than a predictive model. Solutions are scored on the quality of a constructed solution, not a trained model's predictions. Examples:
- **Traveling Santa** — TSP variants on grids with prime/non-prime constraints
- **Santa Gift Matching** — stable matching with happiness functions
- **Santa's Workshop Tour** — scheduling/routing under constraints
- **Packing Santa's Sleigh** — 3D bin packing
- **Santa 2020/2021** — combinatorial puzzle solving / movie scheduling
- **Lux AI** — real-time strategy game AI (connects to RL, see [[reinforcement-learning-games]])

## Key Approaches

### Simulated Annealing (SA)
The most robust starting point for TSP-style problems. Key parameters:
- **Initial temperature**: set so ~80% of moves are accepted initially
- **Cooling schedule**: geometric (T × α each iteration, α ~0.9999) or Lundy-Mees
- **Move operators**: 2-opt, 3-opt, Or-opt, node insertion — start with 2-opt
- Parallelizable across independent chains; take best result

### Lin-Kernighan / LKH
For pure TSP (Traveling Santa 2018 Prime Paths, Traveling Santa 2011):
- LKH-3 solver handles pure TSP to near-optimality
- Constraint-aware variants needed for prime-path problems (every 10th step must be prime)
- Hybridize: LKH initial solution → SA refinement

### Genetic Algorithms / Evolutionary Methods
- Better than SA when solution space has strong modularity (sub-routes, sub-assignments that can be combined)
- Tournament selection + order crossover (OX) for permutation problems
- **Santa 2021 (Movie Montage)**: sparrow search algorithm won 3rd place; customized evolutionary search

### Constraint Programming (OR-Tools)
- Google OR-Tools CP-SAT or routing library for vehicle routing / scheduling problems
- Useful when hard constraints make SA infeasible (tight time windows, exact matching)
- Santa's Workshop Tour 2019: exact scheduling of family visits under night-cap constraints

### GPU-Parallelized Annealing
- Run 1000s of independent SA chains on GPU simultaneously
- CuPy or custom CUDA kernels for neighbor evaluation
- 2021 Santa: 1st place used GPU annealing to evaluate millions of swap candidates/second

## Kaggle-Specific Patterns

- **Score is deterministic**: No train/test split, no overfitting concern. All effort goes to search quality and runtime efficiency.
- **Incremental improvement**: Submit often — leaderboard shows absolute score, not relative. Each improvement is verifiable.
- **Symmetry exploitation**: Many problems have symmetry (mirror solutions, rotational invariance) that can prune the search space.
- **Problem-specific move operators always win**: Generic 2-opt is a baseline. Problem-specific moves (e.g., "swap a present across two sleighs" for packing) dominate generic approaches.
- **Visualization matters**: Plotting the current solution during SA reveals stuck local optima and informs move design.

## Competition Results

| Competition | Year | Winner Approach | Votes |
|---|---|---|---|
| Traveling Santa 2018 — Prime Paths | 2018 | LKH-3 + prime-aware 2-opt | 239 |
| Santa's Workshop Tour 2019 | 2019 | CP-SAT constraint scheduling | — |
| Santa 2020 — The Candy Cane Contest | 2020 | Bandit + adaptive strategy | — |
| Santa 2021 — Merry Movie Montage | 2021 | GPU-parallel simulated annealing | 304 |
| Traveling Santa (2011) | 2012 | LKH-based TSP solver | — |
| Packing Santa's Sleigh | 2013 | 3D bin packing heuristics | — |
| Santa Gift Matching | 2016 | Stable matching + local search | — |
| Helping Santa's Helpers | 2014 | Greedy + SA refinement | — |

## In Jason's Work

Not yet applied. The Santa competitions run December-January each year; the March Mania competition (February-April) is Jason's active sports focus. If entering a future Santa challenge, start with: SA for TSP-like, OR-Tools for scheduling, GPU annealing for scale.

## Sources
- [[../../raw/kaggle/solutions/missing-batch-optimization-games.md]] — Santa 2020/2021, Traveling Santa 2018, Traveling Santa (original), Packing Santa's Sleigh, Santa Gift Matching, Santa Workshop Tour 2019, Helping Santa's Helpers

## Related
- [[../concepts/reinforcement-learning-games]] — Lux AI and game-AI competitions
- [[../concepts/validation-strategy]] — scoring is direct for optimization (no train/test split)
- [[../strategies/kaggle-meta-strategy]] — meta-strategy section on optimization competition differences
