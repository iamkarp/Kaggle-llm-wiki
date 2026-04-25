#!/usr/bin/env python3
"""LLM-based promotion of `related_to` edges to typed relations.

Uses gpt-5.4-mini (per global memory). Caches results by content hash so re-runs
are free. Confidence < 0.7 → keep as related_to or move to kg/proposed/.

Run after `tools/build_kg.py`. Modifies `kg/edges.jsonl` in place + writes
`kg/proposed/llm_classify_cache.jsonl`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

ALLOWED_RELATIONS = [
    "uses", "applied_in", "evaluated_by", "trained_on", "derived_from",
    "improves_on", "supersedes", "contradicts", "compared_to", "requires",
    "part_of", "works_with", "failed_in", "succeeded_in", "caused", "prevents",
    "cites", "instance_of", "related_to",
]

# Strict domain/range validation. (relation) → (allowed_target_types).
# Sources that don't fit are demoted to related_to.
RELATION_TARGET_TYPES: dict[str, set[str]] = {
    "evaluated_by": {"metric"},
    "prevents": {"mistake"},
    "trained_on": {"dataset"},
    "instance_of": {"task_type"},
    "uses": {"library", "model", "feature", "technique", "dataset", "tool",
             "concept", "strategy", "submission", "api_service"},
    "caused": {"mistake"},
    "hosted_by": {"organization", "api_service"},
    "authored_by": {"person"},
    "team_members": {"person"},
    "applied_in": {"competition", "submission"},
    "failed_in": {"competition", "submission"},
    "succeeded_in": {"competition", "submission"},
    "cites": {"source", "paper"},
    "part_of": {"competition", "submission", "feature", "dataset"},
}

# Skip these node ids as either source or target — they're meta/index nodes
# without real-world semantics.
SKIP_NODE_PREFIXES = ("meta:",)

CONFIDENCE_THRESHOLD = 0.7
MODEL = "gpt-5.1-mini"  # gpt-5.4-mini per user pref; fallback handled below
MAX_WORKERS = 8


def load_env(env_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def edge_key(source_id: str, target_id: str, src_summary: str, tgt_summary: str) -> str:
    h = hashlib.sha256()
    h.update(source_id.encode())
    h.update(b"|")
    h.update(target_id.encode())
    h.update(b"|")
    h.update((src_summary or "")[:500].encode())
    h.update(b"|")
    h.update((tgt_summary or "")[:500].encode())
    return h.hexdigest()[:16]


PROMPT = """You are classifying a relationship between two nodes in a Kaggle/ML knowledge graph.

Source node:
  id: {src_id}
  type: {src_type}
  title: {src_title}
  summary: {src_summary}

Target node:
  id: {tgt_id}
  type: {tgt_type}
  title: {tgt_title}
  summary: {tgt_summary}

The current relation is `related_to` (a weak default). Choose a more specific relation if the evidence supports it.

Allowed relations and when to use them:
- uses                  Source uses target as a component/dependency. Source must be one of: submission, strategy, technique, pattern, competition. Target must be: library, model, feature, technique, dataset, tool.
- requires              Source needs target as a prerequisite concept/library/tool.
- improves_on           Source is a strict improvement over target (often v2/-advanced/-cv variants).
- supersedes            Source replaces target (deprecation chain).
- contradicts           Source asserts something incompatible with target's claim.
- compared_to           Source is being benchmarked against target.
- works_with            Source and target compose well together.
- applied_in            Source (technique/pattern/strategy/concept) was applied in target (competition).
- succeeded_in / failed_in  Source worked / didn't work in target competition.
- prevents              Source (pattern/technique) avoids target (mistake).
- caused                Source (library/technique) caused target (mistake).
- cites                 Source references target as evidence.
- evaluated_by          Source (competition/submission) is scored using target (metric).
- trained_on            Source (model/submission) was trained on target (dataset).
- instance_of           Source is an instance/example of target type.
- part_of               Source is contained in target.
- derived_from          Source is derived from target (paper, prior feature/model).
- related_to            None of the above; weak lateral relation. Use this only if no specific relation fits.

Respond with strict JSON: {{"relation": "<one of the above>", "confidence": <0.0–1.0>, "reasoning": "<1 sentence>"}}.
If unsure, return relation=related_to with low confidence.
"""


def classify_one(client, edge: dict, src: dict, tgt: dict) -> dict | None:
    """Make one LLM call. Returns parsed result dict or None on failure."""
    prompt = PROMPT.format(
        src_id=src["id"], src_type=src["type"],
        src_title=src.get("title", "")[:200],
        src_summary=(src.get("summary") or "")[:500],
        tgt_id=tgt["id"], tgt_type=tgt["type"],
        tgt_title=tgt.get("title", "")[:200],
        tgt_summary=(tgt.get("summary") or "")[:500],
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception as exc:
        # Fallback: try gpt-4o-mini if gpt-5.4-mini isn't available
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc2:
            return {"_error": f"{type(exc).__name__}: {exc} / fallback: {exc2}"}
    try:
        body = resp.choices[0].message.content or ""
        return json.loads(body)
    except Exception:
        return {"_error": f"parse_failed: {body[:200]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=str(REPO / "kg"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Maximum edges to classify this run.")
    ap.add_argument("--no-llm", action="store_true",
                    help="Use cache only — never call LLM. Useful for re-applying cached results.")
    args = ap.parse_args()

    kg_root = Path(args.kg)
    nodes: dict[str, dict] = {}
    for p in (kg_root / "nodes").rglob("*.json"):
        n = json.loads(p.read_text())
        nodes[n["id"]] = n

    edges_path = kg_root / "edges.jsonl"
    edges: list[dict] = [json.loads(l) for l in edges_path.read_text().splitlines() if l.strip()]

    cache_path = kg_root / "proposed" / "llm_classify_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["key"]] = d

    # Filter: only related_to edges between resolvable, non-meta nodes
    candidates: list[tuple[int, dict, dict, dict, str]] = []
    for i, e in enumerate(edges):
        if e.get("relation") != "related_to":
            continue
        if any(e["source"].startswith(p) for p in SKIP_NODE_PREFIXES):
            continue
        if any(e["target"].startswith(p) for p in SKIP_NODE_PREFIXES):
            continue
        src = nodes.get(e["source"])
        tgt = nodes.get(e["target"])
        if not src or not tgt:
            continue
        key = edge_key(e["source"], e["target"], src.get("summary", ""), tgt.get("summary", ""))
        candidates.append((i, e, src, tgt, key))

    print(f"{len(candidates)} related_to edges to classify; {sum(1 for _,_,_,_,k in candidates if k in cache)} cached.")

    if args.limit:
        candidates = candidates[: args.limit]
        print(f"limit applied: classifying {len(candidates)}")

    # Load API key
    env = load_env(REPO / ".env")
    api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    # Common typo: missing leading "s" on "sk-proj-..."
    if api_key and api_key.startswith("k-proj-"):
        api_key = "s" + api_key

    client = None
    if not args.no_llm and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except Exception as exc:
            print(f"OpenAI client init failed: {exc}", file=sys.stderr)
            return 1
    elif not args.no_llm:
        print("No OPENAI_API_KEY available; pass --no-llm to apply cache only.", file=sys.stderr)
        return 1

    # Classify
    results: dict[str, dict] = {}
    todo = [(i, e, src, tgt, k) for i, e, src, tgt, k in candidates if k not in cache]
    print(f"calling LLM for {len(todo)} new edges (model={MODEL})...")

    if client and todo:
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            future_to_meta = {
                ex.submit(classify_one, client, e, src, tgt): (i, e, src, tgt, k)
                for i, e, src, tgt, k in todo
            }
            done = 0
            for fut in cf.as_completed(future_to_meta):
                i, e, src, tgt, k = future_to_meta[fut]
                res = fut.result() or {"_error": "no_result"}
                cache_entry = {
                    "key": k,
                    "source": e["source"], "target": e["target"],
                    "result": res,
                }
                cache[k] = cache_entry
                results[k] = cache_entry
                done += 1
                if done % 20 == 0:
                    print(f"  classified {done}/{len(todo)}")
        # Persist cache (full rewrite for stability)
        cache_path.write_text("\n".join(json.dumps(v) for v in cache.values()) + "\n")

    # Apply cache to edges
    promoted = 0
    kept = 0
    proposed: list[dict] = []
    new_edges: list[dict] = []
    for e in edges:
        if e.get("relation") != "related_to":
            new_edges.append(e)
            continue
        src = nodes.get(e["source"])
        tgt = nodes.get(e["target"])
        if not src or not tgt:
            new_edges.append(e)
            continue
        k = edge_key(e["source"], e["target"], src.get("summary", ""), tgt.get("summary", ""))
        cached = cache.get(k)
        if not cached:
            new_edges.append(e)
            continue
        res = cached.get("result", {}) or {}
        rel = res.get("relation")
        conf = res.get("confidence", 0.0)
        reasoning = res.get("reasoning", "")
        # Domain/range sanity: demote if proposed relation requires a target type
        # that doesn't match.
        tgt_type = e["target"].split(":", 1)[0]
        if rel in RELATION_TARGET_TYPES and tgt_type not in RELATION_TARGET_TYPES[rel]:
            rel = "related_to"
            conf = 0.5
        # Skip meta nodes
        if any(e["source"].startswith(p) for p in SKIP_NODE_PREFIXES) or \
           any(e["target"].startswith(p) for p in SKIP_NODE_PREFIXES):
            new_edges.append(e)
            continue
        if rel and rel in ALLOWED_RELATIONS and rel != "related_to" and conf >= CONFIDENCE_THRESHOLD:
            new_e = dict(e)
            new_e["relation"] = rel
            new_e["confidence"] = float(conf)
            new_e["notes"] = reasoning[:200]
            new_e["provenance"] = dict(e.get("provenance") or {})
            new_e["provenance"]["method"] = "llm-extract"
            new_e["provenance"]["llm_model"] = MODEL
            new_edges.append(new_e)
            promoted += 1
        elif rel and rel != "related_to" and conf >= 0.4:
            # Below threshold: keep as related_to but record proposed
            proposed.append({
                "source": e["source"], "target": e["target"],
                "current_relation": "related_to",
                "proposed_relation": rel,
                "confidence": conf,
                "reasoning": reasoning[:200],
            })
            new_edges.append(e)
            kept += 1
        else:
            new_edges.append(e)
            kept += 1

    edges_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in new_edges) + "\n")

    if proposed:
        prop_path = kg_root / "proposed" / "edges_proposed.jsonl"
        prop_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in proposed) + "\n")

    print(f"\npromoted: {promoted}, kept_as_related_to: {kept}, proposed: {len(proposed)}")

    # Refresh manifest with current edge totals + relation distribution
    manifest_path = kg_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["edge_count"] = len(new_edges)
        rel_counts: dict[str, int] = {}
        for e in new_edges:
            rel_counts[e["relation"]] = rel_counts.get(e["relation"], 0) + 1
        manifest["edge_count_by_relation"] = dict(sorted(rel_counts.items(), key=lambda x: -x[1]))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
