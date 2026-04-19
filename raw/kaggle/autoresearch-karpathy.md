# Karpathy's AutoResearch — Autonomous Agent Experimentation Framework

Source: https://github.com/karpathy/autoresearch (74K stars, MIT license, March 2026)

## What It Is

An autonomous AI agent research loop: the agent modifies training code, runs a 5-minute experiment, checks if the result improved, keeps or discards the change, and repeats indefinitely. ~12 experiments/hour, ~100 experiments overnight unattended.

## Architecture

Three files:
- **`prepare.py`** — immutable constants, one-time data prep (downloads data, trains BPE tokenizer), runtime utilities (dataloader, evaluation). Agent cannot modify.
- **`train.py`** — the single file the agent edits. Contains model, optimizer (Muon + AdamW), training loop. Everything is fair game: architecture, hyperparameters, batch size, etc.
- **`program.md`** — human-written instructions directing agent behavior. The "org code" that humans iterate on.

## The Experiment Loop (from program.md)

LOOP FOREVER:
1. Look at git state (current branch/commit)
2. Modify `train.py` with an experimental idea
3. git commit
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If crashed, read traceback, attempt fix or skip
7. Record in results.tsv (commit, val_bpb, memory_gb, status, description)
8. If improved → keep commit, advance branch
9. If worse → git reset back

**NEVER STOP**: Agent runs indefinitely until manually interrupted. No asking "should I continue?"

## Design Principles

- **Fixed time budget**: Always 5 minutes wall clock. Makes experiments directly comparable regardless of what agent changes. ~12/hour, ~100 overnight.
- **Single metric**: val_bpb (validation bits per byte) — lower is better, vocab-size-independent.
- **Self-contained**: One GPU, one file, one metric. No distributed training, no complex configs.
- **Simplicity criterion**: Small improvement + ugly complexity = not worth it. Improvement from deleting code = always keep.

## Output Format

```
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

## results.tsv Format

Tab-separated, 5 columns:
```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## Platform Support

Requires NVIDIA GPU (tested H100). Notable forks:
- MacOS: miolini/autoresearch-macos, trevin-creator/autoresearch-mlx
- Windows: jsegov/autoresearch-win-rtx
- AMD: andyluo7/autoresearch

For smaller compute: use TinyStories dataset, reduce vocab_size (down to 256), lower MAX_SEQ_LEN, decrease DEPTH, adjust TOTAL_BATCH_SIZE.

## Key Insight

"You're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the program.md Markdown files that provide context to the AI agents and set up your autonomous research org." — Karpathy
