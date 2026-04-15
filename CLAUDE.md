# LLM Wiki — Schema & Operating Instructions

This file is the authoritative schema for the `llm-wiki/` knowledge base. Every LLM session that operates on this wiki MUST read this file first. It defines the architecture, operations, conventions, and domain-specific rules.

---

## 1. Three-Layer Architecture

```
raw/        ← Immutable source material. Never edit these files.
wiki/       ← LLM-maintained markdown pages. This is the living knowledge base.
CLAUDE.md   ← This file. The schema that governs all operations.
```

### raw/
Drop source documents here: PDFs, articles, notebooks, CSV result files, strategy docs, backtest logs. Files are **immutable** — never edit raw sources; only summarize them into wiki pages.

Naming convention: `raw/<category>/<slug>.<ext>`
- `raw/kaggle/march-mania-v6-ensemble-doc.md`
- `raw/trading/nfp-straddle-backtest-results.txt`
- `raw/papers/2coool-cvpr-arxiv-2510-12190.pdf`

`raw/assets/` — downloaded images referenced by wiki pages.

### wiki/
Living markdown pages. Every page has YAML frontmatter (see §4). Pages cross-reference each other with `[[wikilinks]]`. The LLM is responsible for keeping pages accurate, linked, and indexed.

### CLAUDE.md
Never modify this file unless explicitly expanding the schema. Treat it as an immutable contract.

---

## 2. Operations

### INGEST
Add new knowledge to the wiki from a raw source or conversation.

Steps:
1. Copy or symlink source file into `raw/` with a descriptive slug.
2. Identify which wiki pages are affected (create new or update existing).
3. Write or update wiki page(s) with frontmatter, content, and citations.
4. Update `wiki/index.md` — add/update the entry for each changed page.
5. Append to `wiki/log.md`: `## [YYYY-MM-DD] ingest | <description>`

### QUERY
Answer a question from wiki content. Do NOT modify wiki files during a pure query. If the answer reveals a gap or stale content, log it in `wiki/log.md` and offer to run INGEST.

### LINT
Audit the wiki for consistency. Steps:
1. Check every page in `wiki/` has valid frontmatter (title, tags, date, status).
2. Check every `[[wikilink]]` resolves to an existing file.
3. Check `wiki/index.md` lists every page.
4. Check `wiki/log.md` has an entry for every recent change.
5. Report broken links, missing index entries, pages with `status: draft`.
6. Append to `wiki/log.md`: `## [YYYY-MM-DD] lint | <summary of issues found>`

---

## 3. Page Conventions

### File naming
- Lowercase, hyphen-separated slugs.
- Category prefix matches the subdirectory: `competitions/march-mania-2026.md`
- Entity files: `entities/jason-profile.md`, `entities/xgboost.md`

### YAML Frontmatter (required on every page)
```yaml
---
title: "Human-readable title"
tags: [kaggle, ensemble, xgboost]          # lowercase, relevant taxonomy
date: YYYY-MM-DD                            # date last meaningfully updated
source_count: 3                             # number of raw sources cited
status: active | draft | archived          # active = maintained, draft = incomplete, archived = historical
---
```

`status: active` — page is current and maintained.
`status: draft` — incomplete; needs more work before trusting.
`status: archived` — historical record; no longer updated.

### Body structure
```markdown
## Summary
1–3 sentence plain-English summary. What is this? Why does it matter?

## Key Facts / Details
Structured content. Use tables, bullet lists.

## What Worked / What Didn't
(for competitions and strategies only)

## Sources
- [[raw/category/filename]] — one-line description
- [External link](url) — description

## Related
- [[wiki/concepts/topic]] — why it's related
- [[wiki/entities/tool]] — how it was used
```

---

## 4. Citation Format

Always link back to raw sources. Use relative paths from the wiki root:
```markdown
## Sources
- [[../raw/kaggle/march-mania-v6-ensemble-doc.md]] — primary ensemble documentation
```

For external papers/URLs:
```markdown
- [2COOOL CVPR paper](https://arxiv.org/abs/2510.12190) — architecture basis for VQA pipeline
```

---

## 5. Cross-Referencing

Use `[[wikilinks]]` for internal links (Obsidian-compatible):
- Link to full path from wiki root: `[[wiki/concepts/xgboost-ensembles]]`
- Or relative from current file: `[[../entities/xgboost]]`

Every wiki page should have a `## Related` section with at least 2 links. This keeps the graph connected.

---

## 6. Index Maintenance

`wiki/index.md` is the **content catalog**. Update it every time you create or significantly update a page.

Format:
```markdown
| Page | Tags | Status | Summary |
|------|------|--------|---------|
| [[competitions/march-mania-2026]] | kaggle, ncaa, ensemble | active | March Mania 2026 — best score 0.02210 with v6 weighted ensemble |
```

The index is sorted by category, then alphabetically within category.

---

## 7. Log Format

`wiki/log.md` is a chronological audit trail.

Entry format:
```
## [YYYY-MM-DD] action | Description
```

Where `action` is one of: `ingest`, `query`, `lint`, `create`, `update`, `archive`

Example:
```
## [2026-04-14] ingest | Added March Mania v6 ensemble doc from raw source; created competition page and updated index
## [2026-04-14] create | Initial wiki bootstrap — seeded 8 pages from workspace artifacts
```

The format is grep-parseable:
```bash
grep "^\#\# \[2026" wiki/log.md          # all 2026 entries
grep "ingest" wiki/log.md                # all ingests
```

---

## 8. Domain-Specific Sections

### 8.1 Kaggle Competitions

Every competition page (`wiki/competitions/<slug>.md`) must include:

```markdown
## Competition Metadata
- **Platform**: Kaggle
- **Prize**: $X
- **Metric**: LogLoss / Accuracy / MAP@5 / etc.
- **Deadline**: YYYY-MM-DD
- **Team**: Jason (solo / + collaborators)
- **Best LB Score**: X.XXXXX (as of YYYY-MM-DD)
- **Best LB Rank**: #N of M
- **Submission Count**: N used / M total

## Strategy Summary
High-level approach. Link to strategy page if one exists.

## What Worked
- Bullet list of approaches that improved score

## What Didn't Work
- Bullet list of approaches that hurt or didn't help

## Submission History
| Version | Score | Notes |
|---------|-------|-------|
| v6 ensemble | 0.02210 | Best score; weighted XGBoost + meta-ensemble |

## Open Questions
- Outstanding uncertainties or things to try
```

### 8.2 ML Techniques

Every technique/concept page (`wiki/concepts/<slug>.md`) must include:

```markdown
## What It Is
Plain-English explanation. No assumed ML knowledge.

## When To Use It
Conditions under which this technique helps.

## Hyperparameters
Key parameters and their typical ranges / what they control.

## Gotchas
Common failure modes, leakage risks, overfitting patterns.

## In Jason's Work
Specific instances where this was used and what happened.
```

### 8.3 Ensembling Strategies

Pages in `wiki/strategies/` covering ensemble approaches should include:

```markdown
## Architecture
How models are combined (weighted average, stacking, voting, etc.)

## Component Models
| Model | Weight | Notes |
|-------|--------|-------|
| XGBoost v2.9 | 35% | Deep trees, high variance |

## Rationale
Why this specific combination? What does each model contribute?

## Calibration
How probabilities are calibrated, if at all. (Platt scaling, isotonic, etc.)

## Results
Scores achieved with this strategy.
```

### 8.4 Validation Approaches

```markdown
## Strategy
Hold-out / k-fold / time-series split / etc.

## Rationale
Why this split? Risk of leakage?

## Observed CV-to-LB Gap
How well does CV predict LB? Known discrepancies?
```

### 8.5 Feature Engineering

Feature engineering pages should document:

```markdown
## Feature Type
Tabular / text / time-series / image

## Features Engineered
| Feature | Description | Impact |
|---------|-------------|--------|
| Elo Rating | Team strength from historical games | +0.003 LB improvement |

## Leakage Risks
Any features that could leak future information.

## Preprocessing
Normalization, encoding, imputation choices.
```

### 8.6 Tools & Frameworks

Entity pages for tools (`wiki/entities/<tool>.md`):

```markdown
## What It Is
## Typical Use in Jason's Work
## Key Parameters Used
## Performance Notes
## Version / Installation
```

---

## 9. Infrastructure Notes

Jason's compute setup (DO NOT run ML on middle-child):
- **middle-child** (this machine): Agent host, 32GB Intel Mac. Code editing, orchestration only.
- **big-brother** (`192.168.4.243`): Primary ML worker. SSH for all training runs.
- **little-brother** (`192.168.4.63`): RTX 2070 Super. GPU inference.

When documenting experiments, always note which machine ran them.

---

## 10. Safety Rules

1. **Never edit files in `raw/`** — treat them as append-only source of truth.
2. **Never submit to Kaggle without Jason's explicit approval** — gate every submission.
3. **Never change trading position sizing** without explicit instruction.
4. **Prefer updating existing pages** over creating new ones (check index first).
5. **Keep log.md entries brief** — one line per action, scannable at a glance.

---

*Schema version: 1.0 — initialized 2026-04-14*
