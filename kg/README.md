# Knowledge Graph

This directory is **build output**. Do not hand-edit `nodes/`, `edges.jsonl`, or `indexes/`. Edit the markdown under `wiki/` instead and rerun the builder.

## Layout

- `schema/` — JSON Schema definitions for every node type and for edges
- `nodes/<type>/<slug>.json` — one file per node, validated against `schema/<type>.schema.json`
- `edges.jsonl` — one typed edge per line
- `indexes/` — derived lookup files (by-type, by-tag, by-category, backlinks, alias-map, orphans)
- `proposed/` — LLM-extracted nodes/edges with `confidence < 1.0` awaiting human accept
- `overlays/` — hand-authored facts the markdown cannot carry; merged last by builder
- `manifest.json` — build metadata (counts, schema version, hash)
- `build.log` — last build's log

## Build

```
python tools/build_kg.py
python tools/validate.py
```

See `/Users/macbook/.claude/plans/ok-now-i-want-zesty-sundae.md` for the full design.
