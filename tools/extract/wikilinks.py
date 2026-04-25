"""Find [[wikilinks]] in markdown and resolve them to KG node ids."""
from __future__ import annotations

import re
from pathlib import Path

WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]")


def find_wikilinks(text: str) -> list[tuple[str, str | None, int]]:
    """Return [(target, display_or_None, position), ...]."""
    out = []
    for m in WIKILINK_RE.finditer(text):
        target = m.group(1).strip()
        display = m.group(2).strip() if m.group(2) else None
        out.append((target, display, m.start()))
    return out


def resolve_target(target: str, page_path: Path, slug_to_id: dict[str, str]) -> str | None:
    """Resolve a wikilink target string to a canonical node id.

    Strategies, in order:
    1. If target is a path with a known slug at the end, look up slug → id.
    2. Strip any leading '../' and trailing '.md' and try slug match.
    3. Return None (caller may emit an unresolved warning).
    """
    # raw/ link → Source node
    norm = target.strip().lstrip("./")
    norm = re.sub(r"\.md$", "", norm)
    # strip leading ../ chunks
    while norm.startswith("../"):
        norm = norm[3:]

    if norm.startswith("raw/"):
        slug = path_to_slug(norm[len("raw/"):])
        return f"source:{slug}"

    # Try last path component as slug (most wikilinks point inside wiki/)
    parts = norm.split("/")
    candidate_slug = path_to_slug(parts[-1])
    if candidate_slug in slug_to_id:
        return slug_to_id[candidate_slug]
    # try whole normalized path
    full_slug = path_to_slug(norm)
    if full_slug in slug_to_id:
        return slug_to_id[full_slug]
    return None


def path_to_slug(path: str) -> str:
    """Convert a path like 'competitions/march-mania-2026' to 'march-mania-2026' slug."""
    s = path.rsplit("/", 1)[-1]
    s = re.sub(r"\.md$", "", s)
    s = re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")
    return s
