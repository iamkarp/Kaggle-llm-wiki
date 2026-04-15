# Kaggle ML Wiki — Schema

This is an LLM-maintained wiki for Kaggle competition knowledge. It follows the
[llm-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern:
the LLM incrementally builds and maintains a structured collection of markdown files
that compound knowledge across competitions.

## Three-Layer Architecture

1. **raw/** — Immutable source material. Competition descriptions, forum posts, papers,
   winning solution writeups, notebook exports. The LLM reads but never modifies these.

2. **wiki/** — LLM-generated and maintained pages organized by topic. Every page has
   YAML frontmatter. Cross-references use `[[wiki-links]]` syntax.

3. **This file (CLAUDE.md)** — The schema. Documents structure, conventions, and
   operational workflows.

## Wiki Organization

Pages are organized into these categories:

- **Techniques** (`techniques/`) — ML methods, architectures, tricks. One page per
  technique. Include: what it is, when to use it, gotchas, Kaggle-specific tips.

- **Competitions** (`competitions/`) — One page per competition entered. Include:
  problem description, what worked, what didn't, final standing, key lessons.

- **Patterns** (`patterns/`) — Recurring cross-competition patterns. Feature engineering
  patterns, validation strategies, ensemble methods, post-processing tricks.

- **Tools** (`tools/`) — Kaggle platform specifics, libraries, infrastructure knowledge.

- **Mistakes** (`mistakes/`) — Anti-patterns and lessons learned the hard way. These
  prevent repeating errors across competitions.

## Page Format

Every wiki page uses this template:

```markdown
---
title: Page Title
category: techniques|competitions|patterns|tools|mistakes
tags: [tag1, tag2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# Page Title

Content here. Use `[[other-page]]` for cross-references.

## See Also
- [[related-page-1]]
- [[related-page-2]]
```

## Core Operations

### Ingest
When new source material is added to `raw/`:
1. Read the source thoroughly
2. Extract key insights, techniques, and lessons
3. Create new wiki pages or update existing ones
4. Update `wiki/index.md` with new entries
5. Append to `wiki/log.md`

### Query
When answering questions:
1. Read `wiki/index.md` to find relevant pages
2. Read those pages for context
3. Synthesize an answer
4. If the answer reveals new knowledge worth preserving, write it to the wiki

### Lint
Periodically check for:
- Contradictions between pages
- Stale claims (outdated library versions, deprecated APIs)
- Orphaned pages not in the index
- Missing cross-references
- Pages that should be merged or split

## Conventions

- File names: lowercase, hyphens, no spaces (e.g., `patchtst-for-har.md`)
- Keep pages focused — one concept per page
- Prefer concrete examples over abstract advice
- Include code snippets where helpful
- Always note which competition a lesson came from
- Tag pages for discoverability
- When updating a page, update the `updated:` date in frontmatter
