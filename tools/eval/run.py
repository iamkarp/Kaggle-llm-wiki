#!/usr/bin/env python3
"""Run KG eval queries from queries.yaml. Reports pass/fail per query.

Exit code 0 if all pass, 1 if any fail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def load_kg(kg_root: Path) -> tuple[dict[str, dict], list[dict]]:
    nodes: dict[str, dict] = {}
    for p in (kg_root / "nodes").rglob("*.json"):
        n = json.loads(p.read_text())
        nodes[n["id"]] = n
    edges: list[dict] = []
    for line in (kg_root / "edges.jsonl").read_text().splitlines():
        if line.strip():
            edges.append(json.loads(line))
    return nodes, edges


def normalise(value):
    """Normalise project results so list/tuple comparisons match YAML lists."""
    if isinstance(value, list):
        return [normalise(v) for v in value]
    if isinstance(value, tuple):
        return [normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    return value


def run_query(q: dict, nodes: dict, edges: list) -> tuple[bool, list, list]:
    """Returns (passed, actual_sorted, expected_sorted)."""
    kind = q["kind"]
    flt = q["filter"]
    proj = q.get("project", 'n["id"]' if kind != "edge-list" else None)
    expected = q["expected"]

    actual_set: list = []
    if kind == "node-list":
        for n in nodes.values():
            if eval(flt, {"n": n, "nodes": nodes, "edges": edges}):
                actual_set.append(eval(proj, {"n": n, "nodes": nodes, "edges": edges}))
        actual = sorted(actual_set)
    elif kind == "edge-list":
        for e in edges:
            if eval(flt, {"e": e, "nodes": nodes, "edges": edges}):
                actual_set.append(eval(proj, {"e": e, "nodes": nodes, "edges": edges}))
        actual = sorted(actual_set)
    elif kind == "lookup":
        # Run filter on every node; first match returns project.
        actual_val = None
        for n in nodes.values():
            if eval(flt, {"n": n, "nodes": nodes, "edges": edges}):
                actual_val = eval(proj, {"n": n, "nodes": nodes, "edges": edges})
                break
        actual = normalise(actual_val) if actual_val is not None else []
    else:
        return False, [f"unknown kind: {kind}"], expected

    if isinstance(actual, list):
        actual_norm = [normalise(x) for x in actual]
    else:
        actual_norm = actual

    return actual_norm == expected, actual_norm, expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=str(REPO / "kg"))
    ap.add_argument("--queries", default=str(REPO / "tools" / "eval" / "queries.yaml"))
    args = ap.parse_args()

    nodes, edges = load_kg(Path(args.kg))
    queries = yaml.safe_load(Path(args.queries).read_text())

    passed = 0
    failed = 0
    failures: list[str] = []
    for q in queries:
        ok, actual, expected = run_query(q, nodes, edges)
        if ok:
            print(f"  PASS  {q['name']}")
            passed += 1
        else:
            print(f"  FAIL  {q['name']}")
            print(f"        expected: {expected}")
            print(f"        actual:   {actual}")
            failed += 1
            failures.append(q["name"])

    print()
    print(f"{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
