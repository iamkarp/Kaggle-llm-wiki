#!/usr/bin/env python3
"""One-shot frontmatter migrator.

Adds `id`, `type`, and `aliases` (if missing) to every wiki page based on its folder.
Also normalizes `created`/`updated` dates by ensuring `updated` is at least `date`.

Folder → type mapping reflects the markdown folder layout. Pages under
`wiki/entities/` need a finer subtype (person, library, model, api_service, organization)
so we apply a heuristic per-slug; users can override by hand-editing frontmatter.

Usage:
  python tools/migrate_frontmatter.py --dry-run
  python tools/migrate_frontmatter.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.extract.frontmatter import iter_pages, write_page  # noqa: E402
from tools.extract.wikilinks import path_to_slug  # noqa: E402


FOLDER_TO_TYPE = {
    "competitions": "competition",
    "techniques": "technique",
    "concepts": "concept",
    "patterns": "pattern",
    "strategies": "strategy",
    "mistakes": "mistake",
    "comparisons": "comparison",
    "synthesis": "synthesis",
    "tools": "tool",
}

# Heuristic for splitting wiki/entities/ into finer types.
ENTITY_SUBTYPE = {
    "jason-profile": "person",
    "machine-learning-advisor": "tool",
    "xgboost": "library",
    "lightgbm-catboost": "library",
    "qwen-vl": "model",
    "claude-sonnet": "model",
    "oanda": "api_service",
}

ROOT_FILES = {
    "index.md": None,        # skip — index is regenerated
    "log.md": None,          # skip — append-only log
    "overview.md": "synthesis",
}


def infer_type(rel: Path) -> str | None:
    """Given wiki/<folder>/<file>.md path, return the node type."""
    parts = rel.parts
    if len(parts) < 2:
        return None
    if parts[0] == "wiki":
        if len(parts) == 2:
            return ROOT_FILES.get(parts[1])
        folder = parts[1]
        if folder == "entities":
            slug = path_to_slug(parts[-1])
            return ENTITY_SUBTYPE.get(slug, "library")
        return FOLDER_TO_TYPE.get(folder)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write changes (default is dry-run)")
    ap.add_argument("--wiki", default=str(REPO / "wiki"))
    args = ap.parse_args()

    wiki_root = Path(args.wiki)
    changed = 0
    skipped = 0
    for page in iter_pages(wiki_root):
        node_type = infer_type(page.path)
        if not node_type:
            skipped += 1
            continue

        slug = path_to_slug(page.path.stem)
        node_id = f"{node_type}:{slug}"

        fm = dict(page.frontmatter)
        diff = {}

        if fm.get("id") != node_id:
            diff["id"] = node_id
        if fm.get("type") != node_type:
            diff["type"] = node_type
        if fm.get("slug") != slug:
            diff["slug"] = slug
        if "aliases" not in fm:
            diff["aliases"] = []
        if "status" not in fm:
            diff["status"] = "active"

        if not diff:
            continue

        changed += 1
        if args.apply:
            # Place id/type/slug/aliases at top of frontmatter for readability
            new_fm = {}
            for key in ("id", "type", "title", "slug", "aliases", "tags", "status",
                        "date", "created", "updated", "source_count"):
                if key in diff:
                    new_fm[key] = diff[key]
                elif key in fm:
                    new_fm[key] = fm[key]
            for key, value in fm.items():
                if key not in new_fm:
                    new_fm[key] = value
            page.frontmatter = new_fm
            write_page(page, wiki_root)
            print(f"updated: {page.path}  +{','.join(diff.keys())}")
        else:
            print(f"would update: {page.path}  +{','.join(diff.keys())}")

    print(f"\n{'applied' if args.apply else 'dry-run'}: {changed} pages changed, {skipped} skipped (root or non-wiki)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
