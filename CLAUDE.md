# Kaggle ML Wiki — Schema

This is an LLM-maintained wiki for Kaggle competition knowledge. It follows the
[llm-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern:
the LLM incrementally builds and maintains a structured collection of markdown files
that compound knowledge across competitions.

## Four-Layer Architecture

1. **raw/** — Immutable source material. Competition descriptions, forum posts, papers,
   winning solution writeups, notebook exports. The LLM reads but never modifies these.

2. **wiki/** — LLM-generated and maintained pages organized by topic. **Source of truth.**
   Every page has YAML frontmatter. Cross-references use `[[wiki-links]]` syntax.

3. **kg/** — Typed JSON knowledge graph **derived from wiki/** by `tools/build_kg.py`.
   `kg/nodes/<type>/<slug>.json` is one file per node; `kg/edges.jsonl` is one typed
   edge per line; `kg/indexes/*.json` are derived lookups. **Never hand-edit kg/** —
   re-run the builder instead. Hand-only escape hatch: `kg/overlays/*.json` for facts
   the markdown can't carry; merged last.

4. **This file (CLAUDE.md)** — The schema. Documents structure, conventions, and
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
id: <type>:<slug>            # e.g. competition:march-mania-2026
type: competition|technique|concept|pattern|strategy|mistake|comparison|synthesis|tool|library|model|person|organization|api_service|dataset|metric|task_type|feature|paper|source
title: Page Title
slug: page-slug
aliases: []                  # alternate names the resolver should accept
tags: [tag1, tag2]
status: draft|active|stale|deprecated|archived
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# Page Title

Content here. Use `[[other-page]]` for cross-references.

## See Also
- [[related-page-1]]
- [[related-page-2]]
```

The 21 node types and their per-type attributes are defined in `kg/schema/*.schema.json`.
The base attributes every node carries are in `kg/schema/node.base.schema.json`.

## Knowledge Graph Build

```
python tools/build_kg.py    # wiki/ → kg/
python tools/validate.py    # JSON Schema + referential integrity
```

The builder parses frontmatter, sections, tables, and `[[wikilinks]]`. Wikilinks become
typed edges based on the section they appear in (`## Sources` → cites; `## Related` is
promoted to `uses` when a competition links a library, etc.). All edges are written to
`kg/edges.jsonl` with provenance.

Edge relations: `uses, applied_in, evaluated_by, measured_with, trained_on, derived_from,
improves_on, supersedes, contradicts, compared_to, requires, part_of, works_with,
failed_in, succeeded_in, caused, prevents, cites, authored_by, affiliated_with,
hosted_by, instance_of, mentioned_in, related_to`.

## Core Operations

### Ingest
When new source material is added to `raw/`:
1. Read the source thoroughly
2. Extract key insights, techniques, and lessons
3. Create new wiki pages or update existing ones (with full frontmatter, including `id` and `type`)
4. Update `wiki/index.md` with new entries
5. Append to `wiki/log.md`
6. Run `python tools/build_kg.py && python tools/validate.py` — commit only when validation is clean.

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
