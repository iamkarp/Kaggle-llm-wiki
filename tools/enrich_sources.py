#!/usr/bin/env python3
"""Enrich Source nodes with title/year/origin from raw/ files, and auto-create
Paper nodes for arxiv-shaped references found inside any markdown source.

Run between `build_kg.py` and `validate.py`. Idempotent.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import date

REPO = Path(__file__).resolve().parents[1]

ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", re.IGNORECASE)
ARXIV_BARE_RE = re.compile(r"\barXiv:?\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
ARXIV_PAREN_RE = re.compile(r"\(arXiv\s+(\d{4}\.\d{4,5})\)", re.IGNORECASE)
DOI_RE = re.compile(r"10\.\d{4,9}/[\w./;()-]+", re.IGNORECASE)
TITLE_LINE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
URL_RE = re.compile(r"https?://[^\s\)>\]]+", re.IGNORECASE)


def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return re.sub(r"-+", "-", s)[:80]


def find_arxiv_ids(text: str) -> set[str]:
    found: set[str] = set()
    for rx in (ARXIV_RE, ARXIV_BARE_RE, ARXIV_PAREN_RE):
        for m in rx.finditer(text):
            found.add(m.group(1))
    return found


def find_dois(text: str) -> set[str]:
    return set(DOI_RE.findall(text))


def extract_title(text: str) -> str | None:
    m = TITLE_LINE_RE.search(text)
    if m:
        title = m.group(1).strip()
        if 3 <= len(title) <= 200:
            return title
    return None


def detect_year_from_text(text: str, fallback: int = 2026) -> int:
    """Pick the most-recent-looking 4-digit year (2010-2026) appearing early in the text."""
    snippet = text[:1500]
    years = [int(y) for y in re.findall(r"\b(20[12][0-9])\b", snippet)]
    return max(years) if years else fallback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=str(REPO / "kg"))
    ap.add_argument("--raw", default=str(REPO / "raw"))
    args = ap.parse_args()

    kg_root = Path(args.kg)
    raw_root = Path(args.raw)

    # Load nodes & edges
    nodes_dir = kg_root / "nodes"
    nodes: dict[str, dict] = {}
    for p in nodes_dir.rglob("*.json"):
        n = json.loads(p.read_text())
        nodes[n["id"]] = n

    edges_path = kg_root / "edges.jsonl"
    edges: list[dict] = []
    for line in edges_path.read_text().splitlines():
        if line.strip():
            edges.append(json.loads(line))

    # Map source nodes to raw paths
    sources_by_path: dict[str, dict] = {}
    for nid, n in nodes.items():
        if n["type"] == "source":
            wp = n.get("wiki_path") or n.get("raw_path")
            if wp:
                sources_by_path[str(REPO / wp)] = n

    # Also walk EVERY markdown/txt file under raw/ — sources not yet linked from wiki
    # pages should still be ingested so we catch all arxiv refs.
    for raw_path in sorted(raw_root.rglob("*")):
        if not raw_path.is_file() or raw_path.suffix not in {".md", ".txt"}:
            continue
        if raw_path.name == "CLAUDE.md":
            continue
        rel = raw_path.relative_to(REPO)
        key = str(REPO / rel)
        if key in sources_by_path:
            continue
        slug = re.sub(r"[^a-z0-9-]+", "-", raw_path.stem.lower()).strip("-")[:80]
        sid = f"source:{slug}"
        # Disambiguate if slug clashes
        if sid in nodes:
            slug = re.sub(r"[^a-z0-9-]+", "-", f"{raw_path.parent.name}-{raw_path.stem}".lower()).strip("-")[:80]
            sid = f"source:{slug}"
        if sid in nodes:
            continue
        new_src = {
            "id": sid, "type": "source",
            "title": raw_path.stem.replace("-", " ").replace("_", " ").title(),
            "slug": slug,
            "aliases": [], "tags": [], "status": "active",
            "wiki_path": str(rel),
            "raw_path": str(rel),
            "summary": f"Auto-ingested from {rel}.",
            "source_refs": [], "confidence": 0.85, "version": 1,
            "provenance": {
                "method": "regex", "extractor_version": "0.4.0",
                "ingested_from": [str(rel)],
            },
            "origin": "kaggle-writeup" if "kaggle" in str(rel) else "doc",
            "license_status": "internal",
            "extracted_nodes": [],
            "created": str(date.today()),
            "updated": str(date.today()),
        }
        nodes[sid] = new_src
        sources_by_path[key] = new_src

    enriched = 0
    paper_nodes: dict[str, dict] = {}
    cite_edges: list[dict] = []

    for raw_path_str, src in sources_by_path.items():
        raw_path = Path(raw_path_str)
        if not raw_path.exists() or raw_path.suffix not in {".md", ".txt"}:
            continue
        try:
            text = raw_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Enrich title (prefer first H1)
        title = extract_title(text)
        if title and title != src.get("title"):
            src["title"] = title
            enriched += 1
        # Year
        if "year" not in src:
            src["year"] = detect_year_from_text(text)
        # Bytes / line count
        src["size_bytes"] = raw_path.stat().st_size
        src["line_count"] = text.count("\n")

        # Find arxiv IDs → Paper nodes
        for axid in find_arxiv_ids(text):
            paper_id = f"paper:arxiv-{axid.replace('.', '-')}"
            if paper_id not in paper_nodes:
                paper_nodes[paper_id] = {
                    "id": paper_id, "type": "paper",
                    "title": f"arXiv:{axid}",
                    "slug": f"arxiv-{axid.replace('.', '-')}",
                    "aliases": [axid, f"arXiv:{axid}"],
                    "tags": ["paper", "arxiv"],
                    "status": "active",
                    "wiki_path": f"kg/canonical/paper/arxiv-{axid.replace('.', '-')}",
                    "summary": f"arXiv paper {axid} (auto-extracted from raw sources).",
                    "source_refs": [],
                    "confidence": 0.9,
                    "provenance": {
                        "method": "regex",
                        "extractor_version": "0.4.0",
                        "ingested_from": [str(raw_path.relative_to(REPO))],
                    },
                    "version": 1,
                    "year": int("20" + axid[:2]) if axid[:2].isdigit() else None,
                    "venue": "arXiv",
                    "arxiv_id": axid,
                    "url": f"https://arxiv.org/abs/{axid}",
                    "authors": [],
                    "created": str(date.today()),
                    "updated": str(date.today()),
                }
            else:
                paper_nodes[paper_id]["provenance"]["ingested_from"].append(
                    str(raw_path.relative_to(REPO)))
            cite_edges.append({
                "source": src["id"], "target": paper_id,
                "relation": "cites", "confidence": 0.95,
                "provenance": {
                    "method": "regex",
                    "extractor_version": "0.4.0",
                    "src_section": "raw_arxiv_match",
                },
            })

        # Find DOIs (lightweight — just stash on the source for future enrichment)
        dois = find_dois(text)
        if dois:
            src["dois"] = sorted(dois)[:5]

    # Persist enriched sources
    for nid, n in nodes.items():
        if n["type"] == "source":
            type_dir = nodes_dir / "source"
            (type_dir / f"{n['slug']}.json").write_text(
                json.dumps(n, indent=2, ensure_ascii=False, default=str) + "\n")

    # Persist Paper nodes
    paper_dir = nodes_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    for pid, pn in paper_nodes.items():
        out = paper_dir / f"{pn['slug']}.json"
        out.write_text(json.dumps(pn, indent=2, ensure_ascii=False, default=str) + "\n")

    # Append cite edges to edges.jsonl (dedup)
    seen = {(e["source"], e["target"], e["relation"]) for e in edges}
    new_edges = [e for e in cite_edges
                 if (e["source"], e["target"], e["relation"]) not in seen]
    if new_edges:
        with edges_path.open("a", encoding="utf-8") as f:
            for e in new_edges:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"enriched: {enriched} sources, +{len(paper_nodes)} papers, +{len(new_edges)} cite edges")

    # Refresh manifest with post-build totals
    manifest_path = kg_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        all_nodes = list((kg_root / "nodes").rglob("*.json"))
        manifest["node_count"] = len(all_nodes)
        manifest["edge_count"] = sum(
            1 for line in (kg_root / "edges.jsonl").read_text().splitlines() if line.strip())
        from collections import Counter
        type_counts = Counter()
        for p in all_nodes:
            type_counts[json.loads(p.read_text())["type"]] += 1
        manifest["node_count_by_type"] = dict(sorted(type_counts.items()))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
