#!/usr/bin/env python3
"""Main KG builder. Reads wiki/ markdown, writes kg/nodes/, kg/edges.jsonl, kg/indexes/.

Usage:
  python tools/build_kg.py
  python tools/build_kg.py --wiki ./wiki --kg ./kg
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.extract.frontmatter import iter_pages, Page, strip_kg_block  # noqa: E402
from tools.extract.sections import split_sections, section_aliases  # noqa: E402
from tools.extract.tables import parse_tables, strip_md, parse_number  # noqa: E402
from tools.extract.wikilinks import find_wikilinks, resolve_target, path_to_slug  # noqa: E402

EXTRACTOR_VERSION = "0.3.0"


# ────────────────────────────────────────────────────────────────────────────
# Inferred TaskType nodes.
# Selected when (modality, metric_family) pair is recognised.
# ────────────────────────────────────────────────────────────────────────────

CANONICAL_TASK_TYPES: dict[str, dict[str, Any]] = {
    "task_type:binary-classification": {
        "id": "task_type:binary-classification", "type": "task_type",
        "title": "Binary Classification", "slug": "binary-classification",
        "name": "binary_classification", "status": "active",
        "wiki_path": "kg/canonical/task_type/binary-classification",
        "summary": "Predict one of two classes."},
    "task_type:multiclass-classification": {
        "id": "task_type:multiclass-classification", "type": "task_type",
        "title": "Multiclass Classification", "slug": "multiclass-classification",
        "name": "multiclass_classification", "status": "active",
        "wiki_path": "kg/canonical/task_type/multiclass-classification",
        "summary": "Predict one of K>2 classes."},
    "task_type:regression": {
        "id": "task_type:regression", "type": "task_type",
        "title": "Regression", "slug": "regression",
        "name": "regression", "status": "active",
        "wiki_path": "kg/canonical/task_type/regression",
        "summary": "Predict a continuous value."},
    "task_type:time-series-forecasting": {
        "id": "task_type:time-series-forecasting", "type": "task_type",
        "title": "Time Series Forecasting", "slug": "time-series-forecasting",
        "name": "time_series_forecasting", "status": "active",
        "wiki_path": "kg/canonical/task_type/time-series-forecasting",
        "summary": "Predict future values from a temporal sequence."},
    "task_type:har": {
        "id": "task_type:har", "type": "task_type",
        "title": "Human Activity Recognition", "slug": "har",
        "name": "human_activity_recognition", "status": "active",
        "wiki_path": "kg/canonical/task_type/har",
        "summary": "Classify activities from sensor data, often cross-subject."},
    "task_type:vqa": {
        "id": "task_type:vqa", "type": "task_type",
        "title": "Visual Question Answering", "slug": "vqa",
        "name": "vqa", "status": "active",
        "wiki_path": "kg/canonical/task_type/vqa",
        "summary": "Answer questions about images."},
    "task_type:ranking-prediction": {
        "id": "task_type:ranking-prediction", "type": "task_type",
        "title": "Probability/Ranking Prediction", "slug": "ranking-prediction",
        "name": "probability_ranking", "status": "active",
        "wiki_path": "kg/canonical/task_type/ranking-prediction",
        "summary": "Output calibrated probabilities or rankings (e.g. tournament predictions)."},
    "task_type:combinatorial-optimization": {
        "id": "task_type:combinatorial-optimization", "type": "task_type",
        "title": "Combinatorial Optimization", "slug": "combinatorial-optimization",
        "name": "combinatorial_optimization", "status": "active",
        "wiki_path": "kg/canonical/task_type/combinatorial-optimization",
        "summary": "Find a discrete sequence or arrangement that minimises an objective."},
}


# ────────────────────────────────────────────────────────────────────────────
# Heuristics for feature-name extraction
# ────────────────────────────────────────────────────────────────────────────

# Tokens after which a bolded name is more likely a Feature (not a Technique).
FEATURE_SUFFIXES = {
    "features", "feature", "ratings", "rating", "ordinals", "encoding",
    "embeddings", "embedding", "decomposition", "%", "indices", "indicators",
    "averaging", "smoothing", "averaging within each model",
}


# ────────────────────────────────────────────────────────────────────────────
# Implicit / canonical nodes
# Auto-created when referenced by metadata or tags so edges resolve cleanly.
# ────────────────────────────────────────────────────────────────────────────

CANONICAL_NODES: dict[str, dict[str, Any]] = {
    "organization:kaggle": {
        "id": "organization:kaggle",
        "type": "organization",
        "title": "Kaggle",
        "slug": "kaggle",
        "name": "Kaggle",
        "url": "https://kaggle.com",
        "org_type": "platform",
        "status": "active",
        "wiki_path": "kg/canonical/organization/kaggle",
        "summary": "Data-science competition platform owned by Google.",
    },
    "organization:google": {
        "id": "organization:google",
        "type": "organization",
        "title": "Google",
        "slug": "google",
        "name": "Google",
        "url": "https://google.com",
        "org_type": "company",
        "status": "active",
        "wiki_path": "kg/canonical/organization/google",
        "summary": "Parent company of Kaggle, Anthropic-adjacent in some ML infra.",
    },
    "organization:anthropic": {
        "id": "organization:anthropic",
        "type": "organization",
        "title": "Anthropic",
        "slug": "anthropic",
        "name": "Anthropic",
        "url": "https://anthropic.com",
        "org_type": "company",
        "status": "active",
        "wiki_path": "kg/canonical/organization/anthropic",
        "summary": "AI lab; provider of Claude.",
    },
    # Common metrics
    "metric:mse": {"id": "metric:mse", "type": "metric", "title": "Mean Squared Error", "slug": "mse",
                    "name": "MSE", "direction": "lower_better", "status": "active",
                    "wiki_path": "kg/canonical/metric/mse", "summary": "Mean of squared errors."},
    "metric:rmse": {"id": "metric:rmse", "type": "metric", "title": "Root Mean Squared Error", "slug": "rmse",
                    "name": "RMSE", "direction": "lower_better", "status": "active",
                    "wiki_path": "kg/canonical/metric/rmse", "summary": "Square root of MSE."},
    "metric:logloss": {"id": "metric:logloss", "type": "metric", "title": "Log Loss", "slug": "logloss",
                        "name": "LogLoss", "direction": "lower_better", "status": "active",
                        "wiki_path": "kg/canonical/metric/logloss", "summary": "Cross-entropy of predicted probabilities."},
    "metric:f1": {"id": "metric:f1", "type": "metric", "title": "F1 Score", "slug": "f1",
                    "name": "F1", "direction": "higher_better", "range": {"min": 0, "max": 1},
                    "status": "active", "wiki_path": "kg/canonical/metric/f1",
                    "summary": "Harmonic mean of precision and recall."},
    "metric:micro-f1": {"id": "metric:micro-f1", "type": "metric", "title": "Micro F1", "slug": "micro-f1",
                        "name": "Micro F1", "direction": "higher_better", "range": {"min": 0, "max": 1},
                        "status": "active", "wiki_path": "kg/canonical/metric/micro-f1",
                        "summary": "F1 computed by aggregating TP/FP/FN globally."},
    "metric:accuracy": {"id": "metric:accuracy", "type": "metric", "title": "Accuracy", "slug": "accuracy",
                        "name": "Accuracy", "direction": "higher_better", "range": {"min": 0, "max": 1},
                        "status": "active", "wiki_path": "kg/canonical/metric/accuracy",
                        "summary": "Fraction of correct predictions."},
    "metric:auc": {"id": "metric:auc", "type": "metric", "title": "ROC AUC", "slug": "auc",
                    "name": "AUC", "direction": "higher_better", "range": {"min": 0, "max": 1},
                    "status": "active", "wiki_path": "kg/canonical/metric/auc",
                    "summary": "Area under the ROC curve."},
    "metric:map-at-k": {"id": "metric:map-at-k", "type": "metric", "title": "Mean Average Precision @K",
                        "slug": "map-at-k", "name": "MAP@K", "direction": "higher_better",
                        "status": "active", "wiki_path": "kg/canonical/metric/map-at-k",
                        "summary": "Ranking quality across top-K predictions."},
    "metric:correlation": {"id": "metric:correlation", "type": "metric", "title": "Pearson Correlation",
                            "slug": "correlation", "name": "Correlation", "direction": "higher_better",
                            "status": "active", "wiki_path": "kg/canonical/metric/correlation",
                            "summary": "Linear correlation between predicted and actual."},
}

# Map normalized metric strings → metric id
METRIC_ALIASES: dict[str, str] = {
    "mse": "metric:mse",
    "mean squared error": "metric:mse",
    "rmse": "metric:rmse",
    "root mean squared error": "metric:rmse",
    "logloss": "metric:logloss",
    "log loss": "metric:logloss",
    "log-loss": "metric:logloss",
    "cross entropy": "metric:logloss",
    "cross-entropy": "metric:logloss",
    "binary log loss": "metric:logloss",
    "f1": "metric:f1",
    "f1 score": "metric:f1",
    "micro f1": "metric:micro-f1",
    "micro-f1": "metric:micro-f1",
    "accuracy": "metric:accuracy",
    "auc": "metric:auc",
    "roc auc": "metric:auc",
    "roc-auc": "metric:auc",
    "map@k": "metric:map-at-k",
    "map@5": "metric:map-at-k",
    "map@10": "metric:map-at-k",
    "mean average precision": "metric:map-at-k",
    "correlation": "metric:correlation",
    "pearson correlation": "metric:correlation",
    "pearson": "metric:correlation",
}

# Tag → node id. Used to emit tag-implied edges.
TAG_TO_NODE: dict[str, str] = {
    "kaggle": "organization:kaggle",
    "xgboost": "library:xgboost",
    "lightgbm": "library:lightgbm-catboost",
    "catboost": "library:lightgbm-catboost",
    "patchtst": "technique:patchtst",
    "qwen-vl": "model:qwen-vl",
    "claude": "model:claude-sonnet",
    "claude-sonnet": "model:claude-sonnet",
    "oanda": "api_service:oanda",
    "videomae": "model:videomae",  # may not exist; will just skip if unresolved
    "beam-search": "technique:beam-search",
    "calibration": "concept:calibration",
}

PLATFORM_TO_ORG: dict[str, str] = {
    "kaggle": "organization:kaggle",
    "drivendata": "organization:drivendata",
    "aicrowd": "organization:aicrowd",
}


# ────────────────────────────────────────────────────────────────────────────
# Edge classification
# ────────────────────────────────────────────────────────────────────────────

def classify_edge(src_type: str, tgt_type: str, section: str) -> str:
    """Pick a typed edge given source type, target type, and surrounding section.

    Section names should already be canonicalized by section_aliases().
    Return '_flip:<rel>' to swap source and target before emitting.
    """
    sec = section.lower().strip()

    # Source-typed targets are always CITES from the page that links to them
    if tgt_type == "source":
        return "cites"

    # Sources/Key Files always cite
    if sec in {"sources", "key_files"}:
        return "cites"

    # "What worked" on a competition page: linked thing succeeded in this competition.
    # Source = competition, target = technique/library/etc. Flip so technique → succeeded_in → competition.
    if sec in {"what_worked", "what_works"}:
        if src_type == "competition" and tgt_type in {
            "technique", "library", "model", "feature", "concept", "pattern", "strategy"
        }:
            return "_flip:succeeded_in"
        # Pattern's "what works" section lists techniques that compose this pattern.
        if src_type == "pattern" and tgt_type in {"technique", "library", "model", "feature"}:
            return "uses"

    # "What didn't work" / "What failed" on competition page: linked thing failed in competition.
    if sec in {"what_didnt_work", "what_failed", "what_fails", "anti_patterns"}:
        if src_type == "competition":
            if tgt_type == "mistake":
                return "caused"  # competition caused this mistake
            if tgt_type in {"technique", "library", "model", "feature", "concept", "pattern", "strategy"}:
                return "_flip:failed_in"
        if src_type == "pattern" and tgt_type == "mistake":
            return "prevents"  # pattern's what_fails section: things this pattern prevents

    # Typical Use / In Jason's Work — page links out to competitions where it was used.
    # Source page is library/model/concept/technique etc; target is competition.
    if sec in {"typical_use", "in_jasons_work", "kaggle_specific_patterns"}:
        if tgt_type == "competition":
            if src_type in {"technique", "pattern", "strategy"}:
                return "applied_in"
            if src_type in {"library", "model", "feature"}:
                return "_flip:uses"  # flip: competition uses library
            if src_type == "concept":
                return "applied_in"

    # Architecture / Components / Implementation Pattern / Final Submission Selection
    # imply uses/part_of edges
    if sec in {"architecture", "components", "implementation_pattern",
                "final_submission_selection", "key_approaches", "model_stack"}:
        if tgt_type in {"library", "model", "feature", "technique", "submission"}:
            return "uses"
        if src_type == "strategy" and tgt_type == "strategy":
            return "uses"

    # Strategy summary on a competition page → strategy applied_in this competition (flip).
    if sec == "strategy_summary":
        if src_type == "competition" and tgt_type == "strategy":
            return "_flip:applied_in"

    # The fix / Detection / Prevention sections inside Mistake pages.
    # Mistake page links a technique that fixes it: flip to technique → prevents → mistake.
    if sec in {"the_fix", "fix", "prevention"}:
        if src_type == "mistake" and tgt_type in {"technique", "pattern", "library", "model", "concept", "strategy"}:
            return "_flip:prevents"

    # Pairs Well With → works_with
    if sec == "pairs_well_with":
        if {src_type, tgt_type} & {"technique", "library", "model", "pattern"}:
            return "works_with"

    # Submission History rows often cite related strategies/concepts
    if sec == "submission_history":
        if tgt_type in {"library", "model", "feature", "technique", "strategy"}:
            return "uses"

    # Detection / Evidence sections — supporting material
    if sec in {"detection", "evidence", "diagnostic"}:
        return "mentioned_in"

    # Anti-patterns referenced from Pattern pages
    if sec == "anti_patterns" and src_type == "pattern" and tgt_type == "mistake":
        return "_flip:prevents"  # pattern prevents mistake

    # Default for See Also / Related — type-pair promotion
    if sec in {"see_also", "related"}:
        if src_type == "competition" and tgt_type in {"library", "model", "feature", "technique"}:
            return "uses"
        if src_type == "competition" and tgt_type == "concept":
            return "_flip:applied_in"  # concept applied in competition
        if src_type == "competition" and tgt_type == "pattern":
            return "_flip:applied_in"
        if src_type == "competition" and tgt_type == "strategy":
            return "_flip:applied_in"
        if src_type == "competition" and tgt_type == "mistake":
            return "caused"
        if src_type == "submission" and tgt_type in {"library", "model", "feature", "technique"}:
            return "uses"
        if src_type in {"technique", "pattern", "strategy"} and tgt_type == "competition":
            return "applied_in"
        if src_type == "mistake" and tgt_type in {"technique", "pattern", "library", "model"}:
            return "_flip:prevents"
        if src_type == "strategy" and tgt_type == "concept":
            return "requires"
        if src_type == "strategy" and tgt_type in {"library", "model", "technique"}:
            return "uses"
        if src_type == "strategy" and tgt_type == "strategy":
            return "uses"
        if src_type == "concept" and tgt_type == "strategy":
            return "_flip:requires"  # strategy requires this concept
        if src_type == "concept" and tgt_type == "concept":
            return "related_to"
        if src_type == "technique" and tgt_type == "technique":
            return "works_with"
        if src_type == "tool" and tgt_type == "concept":
            return "requires"
        if src_type == "tool" and tgt_type in {"strategy", "competition"}:
            return "applied_in"
        if src_type == "pattern" and tgt_type in {"technique", "library", "model"}:
            return "uses"
        if src_type == "pattern" and tgt_type == "mistake":
            return "prevents"
        if src_type == "library" and tgt_type in {"library", "concept"}:
            return "related_to"
        if src_type == "library" and tgt_type in {"strategy", "competition"}:
            return "_flip:uses"  # competition/strategy uses library
        if src_type == "model" and tgt_type in {"strategy", "competition"}:
            return "_flip:uses"
        if src_type == "person" and tgt_type == "competition":
            return "_flip:team_members"
        if src_type == "competition" and tgt_type == "metric":
            return "evaluated_by"
        if src_type == "competition" and tgt_type == "task_type":
            return "instance_of"
        if tgt_type == "person":
            return "authored_by"
        if tgt_type == "organization":
            return "affiliated_with"
        return "related_to"

    # Sections we treat as content-only (no implicit edge unless in default Related/See Also)
    if sec in {"installation", "key_parameters", "performance_notes", "hyperparameters",
                "summary", "what_it_is", "how_it_works", "when_to_use_it", "when_to_use",
                "open_questions", "caveats", "leakage_risks", "key_facts_/_details",
                "key_facts", "competition_results", "key_lessons", "post-cutoff_cv",
                "post_cutoff_cv", "gotchas"}:
        # Inline links here still get a weak edge — typed if obvious, else related_to.
        if tgt_type == "competition" and src_type in {"library", "model", "feature", "technique", "concept", "pattern", "strategy"}:
            if src_type in {"library", "model", "feature"}:
                return "_flip:uses"
            return "applied_in"
        if src_type == "competition" and tgt_type in {"library", "model", "feature", "technique"}:
            return "uses"
        return "related_to"

    # Phase / Step / Pipeline subsections (e.g. "phase_7:_models", "step_1", "stage_2")
    if re.match(r"phase[_\s\-]\d+|step[_\s\-]\d+|stage[_\s\-]\d+", sec):
        if tgt_type in {"library", "model", "technique", "feature"}:
            return "uses"
        if tgt_type == "concept":
            return "requires"
        if tgt_type == "competition":
            return "applied_in"

    # Comparison page subjects → compared_to
    if src_type == "comparison":
        return "compared_to"

    return "related_to"


# ────────────────────────────────────────────────────────────────────────────
# Per-type extractors
# ────────────────────────────────────────────────────────────────────────────

def extract_competition(page: Page, sections: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Two formats are accepted:
    # 1. Bullet list under `## Competition Metadata`:  `- **Platform**: Kaggle`
    # 2. Single-line pipe-separated:  `**Platform:** Kaggle | **Metric:** ... | ...`
    cm = sections.get("competition_metadata", "")
    metadata_pairs: list[tuple[str, str]] = []
    if cm:
        for line in cm.splitlines():
            m = re.match(r"^\s*-\s*\*\*([^*]+)\*\*:?\s*(.+)$", line)
            if m:
                metadata_pairs.append((strip_md(m.group(1)).lower(), strip_md(m.group(2))))
    # Pipe-separated header line — anywhere in the body of the page
    pipe_header = re.search(
        r"^\*\*[^*\n]+:\*\*\s*[^|\n]+(?:\s*\|\s*\*\*[^*\n]+:\*\*\s*[^|\n]+)+\s*$",
        page.body, re.MULTILINE)
    if pipe_header:
        for chunk in pipe_header.group(0).split("|"):
            m = re.match(r"\s*\*\*([^*]+):\*\*\s*(.+?)\s*$", chunk)
            if m:
                metadata_pairs.append((strip_md(m.group(1)).lower(), strip_md(m.group(2))))
    for key, value in metadata_pairs:
        if key == "platform":
            out["platform"] = value.lower().strip()
        elif key == "prize":
            out["prize_usd"] = parse_number(value)
        elif key == "metric":
            out["_metric_text"] = value
        elif "deadline" in key:
            out.setdefault("_deadlines", {})[key] = value
        elif key in {"team", "team size"}:
            if "solo" in value.lower():
                out["solo"] = True
                out["team_size"] = 1
            else:
                out["_team_text"] = value
        elif "lb score" in key or "best lb" in key or "result" in key:
            n = parse_number(value)
            if n is not None:
                out["best_lb_score"] = n
                # If value looks like "#1 (0.60855)" extract rank
                rank_m = re.search(r"#(\d+)", value)
                if rank_m:
                    out["final_rank"] = int(rank_m.group(1))
        elif "stage 2 strategy" in key:
            out["_stage2_strategy"] = value

    sh = sections.get("submission_history", "")
    submissions: list[dict[str, Any]] = []
    if sh:
        for table in parse_tables(sh):
            for row in table:
                version = strip_md(row.get("Version", row.get("Submission", "")))
                if not version:
                    continue
                score_cell = next((v for k, v in row.items() if "score" in k.lower()), "")
                submissions.append({
                    "version_label": version,
                    "lb_score": parse_number(score_cell),
                    "notes": strip_md(row.get("Notes", "")),
                })
    if submissions:
        out["_submissions_table"] = submissions

    if "what_worked" in sections:
        out["what_worked"] = [strip_md(l[2:]) for l in sections["what_worked"].splitlines() if l.startswith("- ")]
    if "what_didnt_work" in sections:
        out["what_didnt_work"] = [strip_md(l[2:]) for l in sections["what_didnt_work"].splitlines() if l.startswith("- ")]
    if "open_questions" in sections:
        out["open_questions"] = [strip_md(l[2:]) for l in sections["open_questions"].splitlines() if l.startswith("- ")]
    return out


def extract_mistake(page: Page, sections: dict[str, str], body: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # "What happened:" / "Why it's wrong:" / "The fix:" / "General rule:" — bolded inline
    patterns = {
        "what_happened": r"\*\*What happened:\*\*\s*([^\n]+(?:\n(?!\*\*)[^\n]+)*)",
        "why_wrong": r"\*\*Why it's wrong:\*\*\s*([^\n]+(?:\n(?!\*\*)[^\n]+)*)",
        "fix_text": r"\*\*The fix:\*\*\s*([^\n]+(?:\n(?!\*\*)[^\n]+)*)",
        "general_rule": r"\*\*General rule:\*\*\s*([^\n]+(?:\n(?!\*\*)[^\n]+)*)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, body, re.MULTILINE)
        if m:
            out[key] = strip_md(m.group(1).strip())

    # Evidence table
    ev = sections.get("evidence", "")
    if not ev:
        # Sometimes 'Evidence from XXX:' is inline; tables follow
        m = re.search(r"\*\*Evidence(?: from [^*]+)?:\*\*", body)
        if m:
            ev = body[m.end():]
    if ev:
        for table in parse_tables(ev):
            rows = []
            for row in table:
                cleaned = {k.lower().replace(" ", "_"): strip_md(v) for k, v in row.items()}
                # Convert numeric columns
                for nk in ("oof", "lb", "oof_f1", "lb_f1", "delta"):
                    if nk in cleaned:
                        cleaned[nk] = parse_number(cleaned[nk])
                rows.append(cleaned)
            if rows:
                out["evidence_rows"] = rows
                break

    if "detection" in sections:
        out["detection_heuristic"] = sections["detection"].strip()

    return out


def extract_library(page: Page, sections: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    install = sections.get("installation", "")
    if install:
        m = re.search(r"pip install\s+(\S+)", install)
        if m:
            out["pypi_name"] = m.group(1)
            out["language"] = "python"
    kp = sections.get("key_parameters", "")
    if kp:
        for table in parse_tables(kp):
            params: dict[str, Any] = {}
            for row in table:
                # Try common header names
                name_keys = ["Parameter", "parameter", "Param", "Name"]
                value_keys = ["Value", "value", "Default"]
                notes_keys = ["Notes", "notes", "Description"]
                name = next((strip_md(row[k]) for k in name_keys if k in row and row[k]), None)
                if not name:
                    continue
                params[name.strip("`")] = {
                    "default": strip_md(next((row[k] for k in value_keys if k in row), "")),
                    "notes": strip_md(next((row[k] for k in notes_keys if k in row), "")),
                }
            if params:
                out["typical_hyperparams"] = params
                break
    return out


def extract_technique(page: Page, sections: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "when_to_use" in sections:
        out["when_to_use"] = [strip_md(l[2:]) for l in sections["when_to_use"].splitlines() if l.startswith("- ")]
    if "when_not_to_use" in sections:
        out["when_not_to_use"] = [strip_md(l[2:]) for l in sections["when_not_to_use"].splitlines() if l.startswith("- ")]
    if "gotchas" in sections:
        out["weaknesses"] = [strip_md(l[2:]) for l in sections["gotchas"].splitlines() if l.startswith("- ")]
    return out


def extract_summary(sections: dict[str, str], body: str) -> str:
    """Pull the summary section, or first paragraph as fallback."""
    s = sections.get("summary", "").strip()
    if s:
        # take first paragraph
        return s.split("\n\n", 1)[0].strip()
    # otherwise first non-heading paragraph in body
    for para in body.split("\n\n"):
        para = para.strip()
        if para and not para.startswith("#"):
            return para
    return ""


# ────────────────────────────────────────────────────────────────────────────
# Build pipeline
# ────────────────────────────────────────────────────────────────────────────

def build(wiki_root: Path, kg_root: Path, raw_root: Path) -> tuple[int, int]:
    pages = list(iter_pages(wiki_root))

    # First pass: build slug → id map for wikilink resolution
    slug_to_id: dict[str, str] = {}
    for page in pages:
        fm = page.frontmatter
        if "id" in fm and "slug" in fm:
            slug_to_id[fm["slug"]] = fm["id"]
            for alias in fm.get("aliases", []):
                slug_to_id[path_to_slug(alias)] = fm["id"]

    # Source nodes from raw/
    source_nodes: dict[str, dict[str, Any]] = {}
    if raw_root.exists():
        for raw_path in sorted(raw_root.rglob("*")):
            if not raw_path.is_file():
                continue
            rel = raw_path.relative_to(raw_root.parent)
            slug = path_to_slug(raw_path.stem)
            sid = f"source:{slug}"
            if sid in source_nodes:
                # de-dup by appending parent folder
                slug = path_to_slug(f"{raw_path.parent.name}-{raw_path.stem}")
                sid = f"source:{slug}"
            sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            source_nodes[sid] = {
                "id": sid,
                "type": "source",
                "title": raw_path.stem.replace("-", " ").replace("_", " ").title(),
                "slug": slug,
                "status": "active",
                "wiki_path": str(rel),
                "raw_path": str(rel),
                "sha256": sha,
                "ingested_at": str(date.today()),
                "origin": _classify_origin(rel),
                "license_status": "internal",
                "extracted_nodes": [],
            }
            slug_to_id[slug] = sid

    # Second pass: extract nodes + edges
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    issues: list[str] = []
    submission_nodes: list[dict[str, Any]] = []
    inferred_aux_nodes: dict[str, dict[str, Any]] = {}

    for page in pages:
        fm = page.frontmatter
        if "id" not in fm:
            issues.append(f"missing id: {page.path}")
            continue
        nid = fm["id"]
        ntype = fm["type"]
        sections = split_sections(page.body)
        sections_canon = {section_aliases(k): v for k, v in sections.items() if k}

        node = {
            "id": nid,
            "type": ntype,
            "title": fm.get("title", ""),
            "slug": fm["slug"],
            "aliases": fm.get("aliases", []),
            "tags": [str(t) for t in (fm.get("tags") or [])],
            "status": fm.get("status", "active"),
            "wiki_path": str(page.path),
            "summary": extract_summary(sections_canon, page.body),
            "source_refs": [],
            "confidence": 1.0,
            "provenance": {
                "ingested_from": [str(page.path)],
                "method": "parser",
                "extractor_version": EXTRACTOR_VERSION,
            },
            "version": fm.get("version", 1),
        }
        for date_key in ("date", "created", "updated"):
            if date_key in fm and fm[date_key]:
                d = fm[date_key]
                node[date_key if date_key != "date" else "created"] = (
                    d.isoformat() if hasattr(d, "isoformat") else str(d)
                )
        if "created" not in node and "date" in fm:
            node["created"] = str(fm["date"])
        if "updated" not in node:
            node["updated"] = node.get("created", str(date.today()))

        # Per-type extraction
        if ntype == "competition":
            extra = extract_competition(page, sections_canon)
            for k in ("platform", "prize_usd", "best_lb_score", "what_worked",
                      "what_didnt_work", "open_questions", "solo", "team_size"):
                if k in extra and extra[k] is not None:
                    node[k] = extra[k]
            # Promote bold metadata fields to typed edges
            md_edges = _competition_metadata_edges(nid, extra)
            edges.extend(md_edges)
            # Determine inferred metric (if any) for task-type inference
            metric_id = next((e["target"] for e in md_edges if e["relation"] == "evaluated_by"), None)
            data_modality = node.get("data_modality")
            tt_id = _infer_task_type(data_modality, metric_id,
                                      node.get("tags", []) or [], node.get("title", ""))
            if tt_id:
                node["task_type"] = tt_id
                edges.append({
                    "source": nid, "target": tt_id, "relation": "instance_of",
                    "confidence": 0.85,
                    "provenance": {"method": "regex", "extractor_version": EXTRACTOR_VERSION,
                                    "src_section": "task_type_inference"},
                })
            # Dataset inference
            for ds in _infer_datasets(node, page.body):
                ds_id = ds["id"]
                if ds_id not in inferred_aux_nodes and ds_id not in nodes:
                    ds_node = dict(ds)
                    ds_node.update({
                        "aliases": [], "tags": [], "status": "active",
                        "wiki_path": str(page.path), "source_refs": [],
                        "confidence": 0.85, "version": 1,
                        "provenance": {"method": "regex",
                                        "extractor_version": EXTRACTOR_VERSION,
                                        "ingested_from": [str(page.path)]},
                        "created": node.get("created", str(date.today())),
                        "updated": str(date.today()),
                        "source_competition": nid,
                    })
                    inferred_aux_nodes[ds_id] = ds_node
                edges.append({
                    "source": nid, "target": ds_id, "relation": "uses",
                    "confidence": 0.85,
                    "provenance": {"method": "regex", "extractor_version": EXTRACTOR_VERSION,
                                    "src_section": "dataset_inference"},
                })
            # Extract Feature nodes from "What Worked" bullets and add edges
            ww_text = sections_canon.get("what_worked", "")
            for feat in _extract_features_from_what_worked(ww_text):
                feat_id = f"{feat['kind']}:{feat['slug']}"
                # Avoid clobbering existing nodes
                if feat_id in nodes:
                    edges.append({
                        "source": nid, "target": feat_id, "relation": "uses",
                        "confidence": 0.85,
                        "provenance": {"method": "regex", "extractor_version": EXTRACTOR_VERSION,
                                        "src_section": "what_worked"},
                    })
                    continue
                # Auto-create node
                feat_node = {
                    "id": feat_id, "type": feat["kind"],
                    "title": feat["name"], "slug": feat["slug"],
                    "aliases": [], "tags": [], "status": "draft",
                    "wiki_path": str(page.path),
                    "summary": feat["description"][:240],
                    "source_refs": [], "confidence": 0.7,
                    "provenance": {"method": "regex",
                                    "extractor_version": EXTRACTOR_VERSION,
                                    "ingested_from": [str(page.path)]},
                    "version": 1,
                    "created": node.get("created", str(date.today())),
                    "updated": str(date.today()),
                }
                if feat["kind"] == "feature":
                    feat_node["used_in"] = []
                    feat_node["leakage_risk"] = "none"
                # Stash for post-loop merge
                inferred_aux_nodes[feat_id] = feat_node
                edges.append({
                    "source": nid, "target": feat_id, "relation": "uses",
                    "confidence": 0.8,
                    "provenance": {"method": "regex", "extractor_version": EXTRACTOR_VERSION,
                                    "src_section": "what_worked"},
                })
            # Build submission nodes from the table
            for sub in extra.get("_submissions_table", []):
                sub_slug = path_to_slug(f"{node['slug']}--{sub['version_label']}")
                sub_id = f"submission:{sub_slug}"
                sub_node = {
                    "id": sub_id,
                    "type": "submission",
                    "title": f"{node['title']} — {sub['version_label']}",
                    "slug": sub_slug,
                    "status": "active",
                    "wiki_path": str(page.path),
                    "competition_id": nid,
                    "version_label": sub["version_label"],
                    "notes": sub.get("notes", ""),
                    "summary": f"Submission {sub['version_label']} for {node['title']}",
                    "provenance": {
                        "ingested_from": [str(page.path)],
                        "method": "parser",
                        "extractor_version": EXTRACTOR_VERSION,
                    },
                    "created": node.get("created", str(date.today())),
                    "updated": node.get("updated", str(date.today())),
                }
                if sub.get("lb_score") is not None:
                    sub_node["lb_score"] = sub["lb_score"]
                # Parse blend recipe from notes
                weights = _parse_blend(sub.get("notes") or "")
                if weights:
                    sub_node["weights"] = weights
                submission_nodes.append(sub_node)
                edges.append({
                    "source": sub_id,
                    "target": nid,
                    "relation": "part_of",
                    "confidence": 1.0,
                    "provenance": {"method": "table", "extractor_version": EXTRACTOR_VERSION,
                                    "src_section": "submission_history"},
                })
        elif ntype == "mistake":
            extra = extract_mistake(page, sections_canon, page.body)
            for k in ("what_happened", "why_wrong", "general_rule",
                      "evidence_rows", "detection_heuristic"):
                if k in extra:
                    node[k] = extra[k]
        elif ntype == "library":
            node.update(extract_library(page, sections_canon))
        elif ntype == "technique":
            node.update(extract_technique(page, sections_canon))

        # Wikilink → edges, scoped by current section
        for sec_name, sec_body in sections_canon.items():
            for target, _display, _pos in find_wikilinks(sec_body):
                tid = resolve_target(target, page.path, slug_to_id)
                if tid is None:
                    issues.append(f"unresolved link: {page.path}: [[{target}]]")
                    continue
                if tid == nid:
                    continue
                tgt_type = tid.split(":", 1)[0]
                rel = classify_edge(ntype, tgt_type, sec_name)
                if rel.startswith("_flip:"):
                    rel = rel.split(":", 1)[1]
                    edge_src, edge_tgt = tid, nid
                else:
                    edge_src, edge_tgt = nid, tid
                edges.append({
                    "source": edge_src,
                    "target": edge_tgt,
                    "relation": rel,
                    "confidence": 0.9,  # link-derived edges are high but not perfect
                    "provenance": {
                        "method": "wikilink",
                        "extractor_version": EXTRACTOR_VERSION,
                        "src_section": sec_name,
                    },
                })
                # If target is a Source, record on source node too
                if tid.startswith("source:") and tid in source_nodes:
                    source_nodes[tid]["extracted_nodes"].append(nid)

        # Tag-implied edges
        edges.extend(_tag_edges(node))

        nodes[nid] = node

    # Build name → node id map for entity-mention pass (libraries + models + api_services)
    name_to_node: dict[str, str] = {}
    for nid_, n_ in nodes.items():
        if n_["type"] in {"library", "model", "api_service"}:
            # Canonical names: extract proper-noun-y title before any em-dash, plus aliases
            title = (n_.get("title") or "").split("—")[0].strip()
            for nm in [title] + (n_.get("aliases") or []):
                if nm and len(nm) > 2 and nm.lower() not in {"the", "and", "for"}:
                    name_to_node[nm] = nid_
    # Curated additional names that appear in prose without exactly matching wiki titles
    for nm, nid_ in (
        ("XGBoost", "library:xgboost"),
        ("LightGBM", "library:lightgbm-catboost"),
        ("CatBoost", "library:lightgbm-catboost"),
        ("PatchTST", "technique:patchtst"),
        ("Qwen-VL", "model:qwen-vl"),
        ("Qwen2.5-VL", "model:qwen-vl"),
        ("Claude Sonnet", "model:claude-sonnet"),
        ("OANDA", "api_service:oanda"),
        ("Kaggle", "organization:kaggle"),
    ):
        if nid_ in nodes:
            name_to_node[nm] = nid_

    # Entity-mention pass
    for nid_, n_ in list(nodes.items()):
        page_path = Path(n_["wiki_path"])
        if not (Path(__file__).resolve().parents[1] / page_path).exists():
            continue
        body = (Path(__file__).resolve().parents[1] / page_path).read_text(encoding="utf-8")
        edges.extend(_entity_mention_edges(n_, strip_kg_block(body), name_to_node))

    # Slug-pattern improves_on: <base>-advanced / <base>-cv / <base>-deep → <base>
    slug_index: dict[tuple[str, str], str] = {}
    for nid, n in nodes.items():
        slug_index[(n["type"], n["slug"])] = nid
    for (ntype, slug), nid in list(slug_index.items()):
        for suffix in ("-advanced", "-cv", "-deep", "-v2"):
            if slug.endswith(suffix):
                base = slug[: -len(suffix)]
                base_id = slug_index.get((ntype, base))
                if base_id and base_id != nid:
                    edges.append({
                        "source": nid, "target": base_id,
                        "relation": "improves_on", "confidence": 0.9,
                        "provenance": {
                            "method": "regex",
                            "extractor_version": EXTRACTOR_VERSION,
                            "src_section": "slug_pattern",
                        },
                    })

    # Submission lineage: group by competition, emit supersedes from best → others
    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sn in submission_nodes:
        by_comp[sn["competition_id"]].append(sn)
    for comp_id, subs in by_comp.items():
        edges.extend(_submission_lineage_edges(subs, comp_id))

    # Add submission nodes
    for sn in submission_nodes:
        nodes[sn["id"]] = sn

    # Add only sources actually referenced
    referenced_sources = {
        sid for sid in source_nodes if source_nodes[sid]["extracted_nodes"]
    }
    for sid in referenced_sources:
        nodes[sid] = source_nodes[sid]

    # Add canonical nodes referenced by edges (Kaggle org, common metrics, task types).
    canon_pool = {**CANONICAL_NODES, **CANONICAL_TASK_TYPES}
    referenced_canon = {
        e["target"] for e in edges if e["target"] in canon_pool
    } | {
        e["source"] for e in edges if e["source"] in canon_pool
    }
    for cid in referenced_canon:
        canon = dict(canon_pool[cid])
        canon.setdefault("created", str(date.today()))
        canon.setdefault("updated", str(date.today()))
        canon.setdefault("aliases", [])
        canon.setdefault("tags", [])
        canon.setdefault("source_refs", [])
        canon.setdefault("confidence", 1.0)
        canon.setdefault("provenance", {
            "method": "manual",
            "extractor_version": EXTRACTOR_VERSION,
            "ingested_from": ["kg/canonical/"],
        })
        canon.setdefault("version", 1)
        nodes[cid] = canon

    # Add auto-created Feature/Technique/Dataset nodes inferred from prose
    for aid, anode in inferred_aux_nodes.items():
        if aid in nodes:
            continue
        nodes[aid] = anode

    # Drop edges whose endpoints don't resolve (canonical not referenced, etc.)
    valid_ids = set(nodes.keys())
    edges = [e for e in edges if e["source"] in valid_ids and e["target"] in valid_ids]

    # Dedupe edges
    edges_dedup = _dedupe_edges(edges)

    # Write nodes and edges
    nodes_dir = kg_root / "nodes"
    if nodes_dir.exists():
        for old in nodes_dir.rglob("*.json"):
            old.unlink()
    for nid, node in nodes.items():
        type_dir = nodes_dir / node["type"]
        type_dir.mkdir(parents=True, exist_ok=True)
        out_path = type_dir / f"{node['slug']}.json"
        out_path.write_text(json.dumps(node, indent=2, ensure_ascii=False, default=str) + "\n")

    edges_path = kg_root / "edges.jsonl"
    edges_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in edges_dedup) + "\n")

    # Build indexes
    _write_indexes(kg_root, nodes, edges_dedup)

    # Manifest
    manifest = {
        "schema_version": "0.1.0",
        "extractor_version": EXTRACTOR_VERSION,
        "built_at": str(date.today()),
        "node_count": len(nodes),
        "edge_count": len(edges_dedup),
        "node_count_by_type": {
            t: sum(1 for n in nodes.values() if n["type"] == t)
            for t in sorted({n["type"] for n in nodes.values()})
        },
        "issue_count": len(issues),
    }
    (kg_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (kg_root / "build.log").write_text("\n".join(issues) + "\n")

    print(f"built {len(nodes)} nodes, {len(edges_dedup)} edges, {len(issues)} issues")
    print(f"by type: {manifest['node_count_by_type']}")
    return len(nodes), len(edges_dedup)


def _infer_datasets(node: dict[str, Any], body: str) -> list[dict[str, Any]]:
    """Auto-detect named datasets referenced by a competition page.

    Returns a list of Dataset node dicts (not yet persisted) with appropriate edges
    captured separately by the caller.
    """
    title = (node.get("title") or "").lower()
    text = body.lower()
    datasets: list[dict[str, Any]] = []
    rules: list[tuple[str, dict[str, Any]]] = [
        ("wear", {
            "id": "dataset:wear", "type": "dataset", "title": "WEAR — Wearable Activity Recognition",
            "slug": "wear", "name": "WEAR", "modality": "timeseries",
            "subjects_or_entities": "subject (cross-subject HAR)",
            "split_protocol": "leave-one-subject-out / group-kfold-by-subject",
            "summary": "Wearable inertial sensor dataset for cross-subject activity recognition.",
        }),
        ("horse colic", {
            "id": "dataset:horse-colic", "type": "dataset", "title": "UCI Horse Colic",
            "slug": "horse-colic", "name": "horse-colic", "modality": "tabular",
            "size_rows": 988, "subjects_or_entities": "horse",
            "summary": "Tabular 3-class classification of horse colic outcome (UCI).",
        }),
        ("massey ordinals", {
            "id": "dataset:massey-ordinals", "type": "dataset", "title": "Massey Ordinals",
            "slug": "massey-ordinals", "name": "Massey Ordinals", "modality": "tabular",
            "subjects_or_entities": "team season",
            "summary": "Strength-of-schedule rankings for NCAA basketball teams across systems.",
        }),
        ("march madness", {
            "id": "dataset:march-madness", "type": "dataset", "title": "Kaggle March Madness Data",
            "slug": "march-madness", "name": "Kaggle March Madness Data", "modality": "tabular",
            "subjects_or_entities": "team-game",
            "summary": "Historical NCAA tournament games, Massey Ordinals, regular-season results.",
        }),
        ("ncaa", {
            "id": "dataset:march-madness", "type": "dataset", "title": "Kaggle March Madness Data",
            "slug": "march-madness", "name": "Kaggle March Madness Data", "modality": "tabular",
            "subjects_or_entities": "team-game",
            "summary": "Historical NCAA tournament games, Massey Ordinals, regular-season results.",
        }),
    ]
    seen: set[str] = set()
    for keyword, ds in rules:
        if (keyword in title or keyword in text) and ds["id"] not in seen:
            datasets.append(ds)
            seen.add(ds["id"])
    return datasets


def _infer_task_type(data_modality: str | None, metric_id: str | None,
                      tags: list[str], title: str) -> str | None:
    """Pick the best canonical task_type id from competition signals."""
    title_l = title.lower()
    tags_l = [t.lower() for t in tags]
    # Strong overrides from title/tags
    if "har" in tags_l or "human activity" in title_l or "har" in title_l:
        return "task_type:har"
    if "vqa" in tags_l or "visual question" in title_l or "vqa" in title_l:
        return "task_type:vqa"
    if "tournament" in title_l or "ncaa" in title_l or "march mania" in title_l:
        return "task_type:ranking-prediction"
    if "pancake" in title_l or "santa" in title_l or "sorting" in title_l:
        return "task_type:combinatorial-optimization"
    if data_modality == "timeseries":
        if metric_id in {"metric:rmse", "metric:mse"}:
            return "task_type:time-series-forecasting"
        return "task_type:time-series-forecasting"
    if metric_id == "metric:logloss" or metric_id == "metric:auc":
        return "task_type:binary-classification"
    if metric_id in {"metric:f1", "metric:micro-f1", "metric:accuracy"}:
        return "task_type:multiclass-classification"
    if metric_id == "metric:mse" or metric_id == "metric:rmse":
        return "task_type:regression"
    return None


def _extract_features_from_what_worked(text: str) -> list[dict[str, str]]:
    """Pull bold-prefixed bullets from a 'What Worked' / 'What Did Work' section.

    Returns list of {name, slug, description, kind} where kind is 'feature' or 'technique'.
    """
    out: list[dict[str, str]] = []
    for line in text.splitlines():
        line_strip = line.strip()
        # Pattern A: bold-prefixed bullet — `- **<name>** ...`
        m = re.match(r"^\s*-\s*\*\*([^*]+?)\*\*\s*[—\-:]?\s*(.*)$", line_strip)
        if not m:
            # Pattern B: plain bullet whose lead is a Feature-shaped noun phrase ending
            # with a known feature suffix. E.g. "- Four Factors features (eFG%, ...)",
            # "- Elo ratings as a baseline", "- Massey Ordinals for strength-of-schedule".
            m2 = re.match(
                r"^\s*-\s*([A-Z][A-Za-z0-9 \-/]*?\s(?:features?|ratings?|ordinals?|"
                r"encoding|embeddings?|%|win\s*%|win-percent|win\s*percentage))"
                r"(?:\s|\b)(.*)$",
                line_strip,
            )
            if not m2:
                continue
            name, desc = m2.group(1).strip(), m2.group(2).strip(" -—:.")
        else:
            name = m.group(1).strip().rstrip(".:")
            desc = m.group(2).strip()
        if len(name) < 3 or len(name) > 60:
            continue
        # Reject lines that look like measurements / observations / colon-prefixed labels
        # ("Runtime: 345 min on CPU", "Best score: 0.221", "Blend > best individual").
        if re.search(r"\d|[<>=]|min|MB|GB|hours?|fold|seed", name):
            continue
        words = name.split()
        if len(words) > 5:
            continue
        if any(c in name for c in {":", ";", "(", ")"}):
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        slug = re.sub(r"-+", "-", slug)
        if not slug or len(slug) < 3:
            continue
        # Decide kind: heuristic on suffix words
        last_word = words[-1].lower().rstrip(".")
        kind = "feature" if (last_word in FEATURE_SUFFIXES or
                              any(s in name.lower() for s in
                                  ["features", "ratings", "ordinals", "encoding", "embeddings"])) else "technique"
        out.append({"name": name, "slug": slug, "description": desc, "kind": kind})
    return out


BLEND_RE = re.compile(r"(\d*\.?\d+)\s*[×x*]\s*([vV]\d+(?:\.\d+)?)")


def _parse_blend(notes: str) -> dict[str, float]:
    """Parse blend recipes like '0.35×v2.9 + 0.35×v2.8 + 0.30×v5' from submission notes."""
    out: dict[str, float] = {}
    for m in BLEND_RE.finditer(notes):
        try:
            out[m.group(2).lower()] = float(m.group(1))
        except ValueError:
            continue
    return out


def _submission_lineage_edges(submissions_for_comp: list[dict[str, Any]],
                                competition_id: str) -> list[dict[str, Any]]:
    """Emit improves_on / supersedes edges between submissions of one competition.

    Strategy: take submissions with a numeric lb_score; for the BEST one, emit
    `supersedes` from best → all others. Also emit `improves_on` for any submission
    with a higher score than its predecessor by version order.
    """
    out: list[dict[str, Any]] = []
    scored = [s for s in submissions_for_comp if s.get("lb_score") is not None]
    if not scored:
        return out

    # Determine direction: pick highest-magnitude version as "best" (lower-better
    # for many Kaggle metrics; pages mark best explicitly so use notes contains 'Best').
    best = None
    for s in scored:
        if "best" in (s.get("notes") or "").lower():
            best = s
            break
    if best is None:
        # fallback: pick lowest score (most Kaggle metrics on competitions here are
        # lower-better — MSE, LogLoss).
        best = min(scored, key=lambda x: x["lb_score"])

    best["kept_for_final"] = True

    prov = {"method": "regex", "extractor_version": EXTRACTOR_VERSION,
            "src_section": "submission_lineage"}
    for s in scored:
        if s["id"] == best["id"]:
            continue
        out.append({
            "source": best["id"], "target": s["id"],
            "relation": "supersedes", "confidence": 0.85, "provenance": prov,
        })
    return out


def _competition_metadata_edges(nid: str, extra: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert parsed Competition Metadata bold fields into typed edges."""
    out: list[dict[str, Any]] = []
    prov = {"method": "regex", "extractor_version": EXTRACTOR_VERSION,
            "src_section": "competition_metadata"}
    plat = (extra.get("platform") or "").lower().strip()
    if plat in PLATFORM_TO_ORG:
        out.append({"source": nid, "target": PLATFORM_TO_ORG[plat],
                    "relation": "hosted_by", "confidence": 1.0, "provenance": prov})
    metric_text = (extra.get("_metric_text") or "").lower()
    if metric_text:
        for alias in sorted(METRIC_ALIASES.keys(), key=len, reverse=True):
            if re.search(r"\b" + re.escape(alias) + r"\b", metric_text):
                out.append({"source": nid, "target": METRIC_ALIASES[alias],
                            "relation": "evaluated_by", "confidence": 0.95, "provenance": prov})
                break
    team_text = (extra.get("_team_text") or "").lower()
    if "jason" in team_text:
        out.append({"source": nid, "target": "person:jason-profile",
                    "relation": "team_members", "confidence": 1.0, "provenance": prov})
    return out


def _tag_edges(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Emit edges implied by the node's tags."""
    out: list[dict[str, Any]] = []
    nid = node["id"]
    ntype = node["type"]
    prov = {"method": "tag", "extractor_version": EXTRACTOR_VERSION, "src_section": "frontmatter.tags"}
    for tag in node.get("tags") or []:
        target = TAG_TO_NODE.get(tag.lower())
        if not target or target == nid:
            continue
        tgt_type = target.split(":", 1)[0]
        # Skip: a concept page tagged "kaggle" doesn't mean concept→hosted_by→Kaggle.
        # Only competition/submission pages get hosted_by from a platform tag.
        if tgt_type == "organization" and ntype not in {"competition", "submission"}:
            continue
        # Decide relation based on type pair.
        if tgt_type == "organization" and ntype in {"competition", "submission"}:
            rel = "hosted_by"
        elif ntype in {"competition", "submission", "strategy", "pattern", "technique"} \
                and tgt_type in {"library", "model", "technique", "feature"}:
            rel = "uses"
        elif ntype == "mistake" and tgt_type in {"library", "model", "technique"}:
            # Library/technique caused this mistake. Flip the edge.
            out.append({"source": target, "target": nid, "relation": "caused",
                        "confidence": 0.7, "provenance": prov})
            continue
        elif ntype == "concept" and tgt_type in {"library", "model", "technique"}:
            # Concept tagged with implementation: weak — prefer related_to.
            rel = "related_to"
        else:
            rel = "related_to"
        out.append({"source": nid, "target": target, "relation": rel,
                    "confidence": 0.7, "provenance": prov})
    return out


def _entity_mention_edges(node: dict[str, Any], body: str,
                          name_to_node: dict[str, str]) -> list[dict[str, Any]]:
    """Detect named-entity mentions in prose (libraries, models). Word-boundary regex.

    Skips fenced code blocks. Emits a single edge per (source, target) pair regardless
    of how many mentions appear.
    """
    out: list[dict[str, Any]] = []
    nid = node["id"]
    ntype = node["type"]
    # Skip entire pass for Mistake pages — prose mentions there are mixed (cause + remedy
    # + comparison rows in evidence tables); tags + explicit wikilinks are reliable enough.
    if ntype == "mistake":
        return out
    # Strip fenced code blocks so we don't pick up `import xgboost as xgb`
    text = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", "", text)
    seen: set[str] = set()
    prov = {"method": "regex", "extractor_version": EXTRACTOR_VERSION,
            "src_section": "prose_mention"}
    for name, target in name_to_node.items():
        if target == nid or target in seen:
            continue
        if re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
            tgt_type = target.split(":", 1)[0]
            if ntype in {"competition", "submission"} and tgt_type in {"library", "model", "technique", "feature"}:
                rel, src, tgt = "uses", nid, target
            elif ntype in {"strategy", "pattern"} and tgt_type in {"library", "model", "technique"}:
                rel, src, tgt = "uses", nid, target
            elif ntype == "mistake" and tgt_type in {"library", "model", "technique"}:
                # The library/technique mentioned in a mistake's body is what caused the mistake.
                rel, src, tgt = "caused", target, nid
            elif tgt_type == "organization" and ntype in {"competition", "submission"}:
                rel, src, tgt = "hosted_by", nid, target
            elif ntype == "concept" and tgt_type == "competition":
                rel, src, tgt = "applied_in", nid, target
            else:
                continue  # too weak to assert
            seen.add(target)
            out.append({"source": src, "target": tgt, "relation": rel,
                        "confidence": 0.6, "provenance": prov})
    return out


def _classify_origin(path: Path) -> str:
    parts = path.parts
    if "kaggle" in parts:
        s = path.name.lower()
        if "writeup" in s or "1st" in s or "2nd" in s or "3rd" in s:
            return "kaggle-writeup"
        if "notebook" in s:
            return "kaggle-notebook"
        if "memory" in s or "session" in s:
            return "internal-note"
        return "kaggle-writeup"
    if "trading" in parts:
        return "internal-note"
    if "system" in parts:
        return "doc"
    return "other"


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for e in edges:
        key = (e["source"], e["target"], e["relation"])
        if key in seen:
            seen[key]["confidence"] = max(seen[key].get("confidence", 0), e.get("confidence", 0))
        else:
            seen[key] = e
    return list(seen.values())


def _write_indexes(kg_root: Path, nodes: dict, edges: list) -> None:
    idx_dir = kg_root / "indexes"
    idx_dir.mkdir(exist_ok=True)

    by_type: dict[str, list[str]] = defaultdict(list)
    by_tag: dict[str, list[str]] = defaultdict(list)
    by_category: dict[str, list[str]] = defaultdict(list)
    alias_map: dict[str, str] = {}

    for nid, node in nodes.items():
        by_type[node["type"]].append(nid)
        for tag in node.get("tags", []):
            by_tag[tag].append(nid)
        # category derived from wiki folder
        wp = node.get("wiki_path", "")
        m = re.match(r"wiki/([^/]+)/", wp)
        if m:
            by_category[m.group(1)].append(nid)
        else:
            by_category["root"].append(nid)
        for alias in node.get("aliases", []):
            alias_map[alias] = nid

    backlinks: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_competition: dict[str, list[str]] = defaultdict(list)
    out_count: dict[str, int] = defaultdict(int)
    for e in edges:
        backlinks[e["target"]].append({"from": e["source"], "relation": e["relation"]})
        out_count[e["source"]] += 1
        if e["source"].startswith("competition:"):
            by_competition[e["source"]].append(e["target"])
        if e["target"].startswith("competition:"):
            by_competition[e["target"]].append(e["source"])

    orphans = sorted(nid for nid in nodes if out_count[nid] == 0 and not backlinks[nid])

    (idx_dir / "by-type.json").write_text(json.dumps({k: sorted(v) for k, v in by_type.items()}, indent=2) + "\n")
    (idx_dir / "by-tag.json").write_text(json.dumps({k: sorted(v) for k, v in sorted(by_tag.items())}, indent=2) + "\n")
    (idx_dir / "by-category.json").write_text(json.dumps({k: sorted(v) for k, v in by_category.items()}, indent=2) + "\n")
    (idx_dir / "by-competition.json").write_text(json.dumps({k: sorted(set(v)) for k, v in by_competition.items()}, indent=2) + "\n")
    (idx_dir / "backlinks.json").write_text(json.dumps(backlinks, indent=2) + "\n")
    (idx_dir / "alias-map.json").write_text(json.dumps(alias_map, indent=2) + "\n")
    (idx_dir / "orphans.json").write_text(json.dumps(orphans, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default=str(REPO / "wiki"))
    ap.add_argument("--raw", default=str(REPO / "raw"))
    ap.add_argument("--kg", default=str(REPO / "kg"))
    args = ap.parse_args()
    build(Path(args.wiki), Path(args.kg), Path(args.raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
