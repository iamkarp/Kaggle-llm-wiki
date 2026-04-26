---
name: kaggle-llm-wiki
description: Kaggle competition knowledge base — 21 first-place solution writeups, 40 concept pages, and end-to-end playbooks covering tabular, CV, NLP, time-series, and medical imaging. Trigger when the user is planning or working on a Kaggle / ML competition, asks for competition strategy, validation/CV design, feature engineering, ensembling/stacking, pseudo-labeling, or cites a specific Kaggle competition or technique that may be covered here. Skip for ordinary ML/data-science work unrelated to competitions.
user-invocable: true
allowed-tools:
  - Read
  - Bash(ls *)
  - Bash(grep *)
  - Bash(find *)
  - Bash(rg *)
  - Bash(git -C ~/.claude/skills/kaggle-llm-wiki *)
---

# kaggle-llm-wiki — Kaggle competition knowledge base

A structured, LLM-queryable wiki of winning Kaggle techniques. Source of truth
lives in this skill directory:

```
~/.claude/skills/kaggle-llm-wiki/
├── CLAUDE.md          ← schema contract (read if editing the wiki)
├── wiki/
│   ├── index.md       ← START HERE: full content catalog with tags
│   ├── overview.md    ← high-level knowledge map
│   ├── log.md         ← chronological audit trail
│   ├── concepts/      ← 40 reusable technique pages
│   ├── competitions/  ← active & reference competition pages
│   ├── strategies/    ← end-to-end competition playbooks
│   ├── entities/      ← tools, frameworks, people
│   ├── patterns/ techniques/ tools/ comparisons/ mistakes/
└── raw/kaggle/        ← immutable primary sources (21 writeups + 12 reports)
```

## How to use this skill

When the skill is activated for a competition question, follow the README's
query recipe:

1. **Read `wiki/index.md` first.** It's the catalog — page titles, tags, and
   one-line summaries. Use it to pick which concept/strategy pages are
   relevant to the user's domain (tabular / CV / NLP / time-series / medical /
   finance).
2. **Pull the matching concept pages** from `wiki/concepts/` and any
   competition-specific page from `wiki/competitions/` or `wiki/strategies/`.
   Every page has YAML frontmatter (`tags`, `status`, `source_count`) plus
   `## Summary`, `## Key Facts / Details`, `## What Worked / What Didn't`,
   `## Sources`, `## Related`.
3. **Synthesize a domain-specific playbook.** Cite the specific techniques
   and measured gains from the pulled pages; link back to the `raw/kaggle/`
   source file(s) the page draws from when the user asks "why" or "show me
   the writeup."
4. **Cross-link.** The hub pages (`validation-strategy`,
   `gradient-boosting-advanced`, `feature-engineering-tabular`,
   `ensembling-strategies`, `pseudo-labeling`) are the most connected — if
   the user is broad/unsure, these are good starting points.

## Worked example (from the README)

User: *"I'm starting an X-ray cancer classification competition, what's the
plan?"* → pull `medical-imaging-patterns`, `loss-functions-cv`,
`imbalanced-data`, `external-data-leakage`, `validation-strategy`,
`image-augmentation`, `ensembling-strategies`, `pseudo-labeling-cv` and
synthesize a concrete, measurable plan citing specific techniques (CLAHE
preprocessing, FocalLoss γ=2, GroupKFold on patient_id, 8-way TTA, etc.).

## Editing the wiki

If the user wants to ingest a new source or add/update a page:

1. Drop the raw file into `raw/kaggle/<slug>.md` (never edit existing raw
   files — they are immutable primary sources).
2. Identify wiki pages to create or update.
3. Write/update wiki pages using the YAML frontmatter + section structure
   defined in `CLAUDE.md` (title, tags, date, source_count, status).
4. Add an entry to `wiki/index.md` and append an entry to `wiki/log.md`.

`CLAUDE.md` at the skill root is the authoritative schema — read it before
structural edits.

## Keeping the wiki up to date

It's a git repo. To pull upstream updates:

```
git -C ~/.claude/skills/kaggle-llm-wiki pull
```

Local additions the user makes to `raw/` or `wiki/` are their own working
copy; mention this before pulling if there are uncommitted changes.
