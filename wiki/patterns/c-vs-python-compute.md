---
title: C vs Python for Compute-Intensive Search
category: patterns
tags: [performance, optimization, language-choice]
created: 2026-04-16
updated: 2026-04-16
---

# C vs Python for Compute-Intensive Search

When the inner loop of your solver involves tight numerical computation (array manipulation, heuristic evaluation, state expansion), Python is ~100x slower than C. For Kaggle competitions with compute time limits, this difference determines whether a solution is feasible.

## The Pattern

Split your system into two layers:

1. **C/C++/Rust binary** — The hot loop. Takes a problem instance, parameters (beam width, seed), and outputs a solution string. Single-purpose, no dependencies, compiles in seconds.

2. **Python orchestrator** — Manages parallel workers, tracks best solutions across runs, handles I/O, seeds, and result merging. Uses `subprocess.run()` to invoke the C binary.

```python
from concurrent.futures import ProcessPoolExecutor

def try_solve(args):
    cid, n, perm_csv, beam_width, seed = args
    result = subprocess.run(
        ["./solver", str(n), str(beam_width), perm_csv, str(seed)],
        capture_output=True, text=True, timeout=300
    )
    return parse(result.stdout)

with ProcessPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(try_solve, w): w for w in work_items}
    for f in as_completed(futures):
        # track improvements...
```

## When C is Non-Negotiable

- **Beam search at n≥50**: Each step evaluates `W × n` candidate states. At W=300, n=100, that's 30,000 evaluations per step × ~100 steps = 3M evaluations per run. In C: seconds. In Python: minutes.
- **IDA* at depth 15+**: The recursive search tree is too deep for Python's function call overhead.
- **Any solver you plan to run 10,000+ times**: The orchestrator amortizes startup cost; the inner loop must be fast.

## When Python is Fine

- One-shot computations (feature extraction, model training with NumPy/PyTorch)
- BFS on small state spaces (n=5: 120 states)
- Solution post-processing (verification, merging, formatting)
- Anything where I/O or library calls dominate compute

## Kaggle-Specific Notes

- Kaggle notebooks can compile C: `!gcc -O2 -o solver solver.c`
- Upload C source as a dataset or inline in the notebook
- P100 GPU notebooks have `nvcc` for CUDA if you need GPU compute
- CPU notebooks: 4 cores, use `ProcessPoolExecutor(max_workers=4)`

## Evidence

In Santa 2024 pancake sorting:
- C beam search (bw=300, 2-ply): ~3 minutes for n=100
- Python beam search (bw=300, 1-ply): ~30+ minutes for n=100, and finds worse solutions
- Running 40 C seeds per case in the time one Python run completes

## See Also
- [[competitions/santa-2024-pancake]] — C solver was critical for achieving competitive scores
- [[techniques/beam-search]] — the algorithm that benefits most from C implementation
