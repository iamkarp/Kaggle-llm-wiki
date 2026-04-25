#!/usr/bin/env python3
"""Validate the KG: per-node JSON schema + referential integrity.

Exit code:
  0 — clean
  1 — schema violations
  2 — referential integrity violations
  3 — both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

REPO = Path(__file__).resolve().parents[1]


def load_schemas(schema_dir: Path) -> tuple[dict[str, Draft202012Validator], dict]:
    """Load every per-type schema, plus edge schema.

    Returns (validators_by_type, edge_validator_dict_unused_legacy).
    """
    schemas: dict[str, dict] = {}
    for p in schema_dir.glob("*.schema.json"):
        data = json.loads(p.read_text())
        schemas[p.name] = data

    # Build a Registry for $ref resolution. Schemas reference each other by
    # filename (e.g. "node.base.schema.json"), so register each under both
    # its declared $id and its filename.
    registry: Registry = Registry()
    for name, data in schemas.items():
        resource = Resource.from_contents(data)
        registry = registry.with_resource(uri=name, resource=resource)
        if "$id" in data:
            registry = registry.with_resource(uri=data["$id"], resource=resource)

    type_validators: dict[str, Draft202012Validator] = {}
    edge_validator = None
    for name, data in schemas.items():
        if name in ("node.base.schema.json", "edge.schema.json"):
            continue
        # Type from filename, e.g. "competition.schema.json" → "competition"
        type_name = name.split(".", 1)[0]
        type_validators[type_name] = Draft202012Validator(data, registry=registry)

    edge_validator = Draft202012Validator(schemas["edge.schema.json"], registry=registry)
    return type_validators, edge_validator


def validate_nodes(kg_root: Path, type_validators) -> list[str]:
    errors: list[str] = []
    for node_path in sorted((kg_root / "nodes").rglob("*.json")):
        node = json.loads(node_path.read_text())
        ntype = node.get("type")
        validator = type_validators.get(ntype)
        if not validator:
            errors.append(f"{node_path}: unknown type '{ntype}'")
            continue
        for err in validator.iter_errors(node):
            path = "/".join(str(p) for p in err.absolute_path) or "<root>"
            errors.append(f"{node_path}: {path}: {err.message}")
    return errors


def validate_edges(kg_root: Path, edge_validator, node_ids: set[str]) -> list[str]:
    errors: list[str] = []
    edges_path = kg_root / "edges.jsonl"
    if not edges_path.exists():
        return ["edges.jsonl missing"]
    for i, line in enumerate(edges_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        edge = json.loads(line)
        for err in edge_validator.iter_errors(edge):
            errors.append(f"edges.jsonl:{i}: {err.message}")
        # Referential integrity
        for endpoint in ("source", "target"):
            nid = edge.get(endpoint)
            if nid and nid not in node_ids:
                errors.append(f"edges.jsonl:{i}: dangling {endpoint} '{nid}'")
        # Domain/range type sanity for a subset of strict edges
        rel = edge.get("relation")
        src_type = edge.get("source", "").split(":", 1)[0]
        tgt_type = edge.get("target", "").split(":", 1)[0]
        if rel == "uses" and tgt_type not in {"library", "model", "feature", "technique", "dataset", "tool", "concept", "strategy", "submission", "api_service"}:
            errors.append(f"edges.jsonl:{i}: 'uses' target type '{tgt_type}' not allowed")
        if rel == "evaluated_by" and tgt_type != "metric":
            errors.append(f"edges.jsonl:{i}: 'evaluated_by' target must be metric, got '{tgt_type}'")
        if rel == "authored_by" and tgt_type != "person":
            errors.append(f"edges.jsonl:{i}: 'authored_by' target must be person, got '{tgt_type}'")
        if rel == "prevents" and tgt_type != "mistake":
            errors.append(f"edges.jsonl:{i}: 'prevents' target must be mistake, got '{tgt_type}'")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=str(REPO / "kg"))
    args = ap.parse_args()

    kg_root = Path(args.kg)
    type_validators, edge_validator = load_schemas(kg_root / "schema")

    node_errors = validate_nodes(kg_root, type_validators)
    node_ids = {json.loads(p.read_text())["id"] for p in (kg_root / "nodes").rglob("*.json")}
    edge_errors = validate_edges(kg_root, edge_validator, node_ids)

    code = 0
    if node_errors:
        print(f"\n— node schema errors ({len(node_errors)}) —")
        for e in node_errors[:50]:
            print(e)
        if len(node_errors) > 50:
            print(f"... and {len(node_errors) - 50} more")
        code |= 1
    if edge_errors:
        print(f"\n— edge errors ({len(edge_errors)}) —")
        for e in edge_errors[:50]:
            print(e)
        if len(edge_errors) > 50:
            print(f"... and {len(edge_errors) - 50} more")
        code |= 2
    if code == 0:
        print(f"OK: {len(node_ids)} nodes, edges checked, no errors")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
