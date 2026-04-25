"""Split a markdown body into H2/H3 sections keyed by heading text."""
from __future__ import annotations

import re
from typing import OrderedDict

HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)


def split_sections(body: str) -> "OrderedDict[str, str]":
    """Return ordered dict of {heading_text -> content}.

    Top-level (before any H2) lives under key '' (empty string).
    Nested H3s are folded into their parent H2 — content includes the H3 line itself.
    """
    sections: "OrderedDict[str, str]" = OrderedDict()
    sections[""] = ""

    matches = list(HEADING_RE.finditer(body))
    if not matches:
        sections[""] = body
        return sections

    sections[""] = body[: matches[0].start()].strip()

    current_key = ""
    for i, m in enumerate(matches):
        level, heading = m.group(1), m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[m.end() : end].strip()
        if level == "##":
            current_key = heading
            sections[current_key] = content
        else:
            # H3 — fold into the current H2 with its own line preserved
            existing = sections.get(current_key, "")
            sections[current_key] = (existing + f"\n\n### {heading}\n\n{content}").strip()
    return sections


def section_aliases(name: str) -> str:
    """Normalize section heading to a canonical key."""
    n = name.lower().strip()
    aliases = {
        "summary": "summary",
        "what it is": "summary",
        "competition metadata": "competition_metadata",
        "submission history": "submission_history",
        "what worked": "what_worked",
        "what didn't work": "what_didnt_work",
        "what didnt work": "what_didnt_work",
        "open questions": "open_questions",
        "key files": "key_files",
        "sources": "sources",
        "related": "related",
        "see also": "see_also",
        "evidence": "evidence",
        "the fix": "the_fix",
        "general rule": "general_rule",
        "detection": "detection",
        "key parameters used": "key_parameters",
        "typical use in jason's work": "typical_use",
        "performance notes": "performance_notes",
        "in jason's work": "typical_use",
        "when to use": "when_to_use",
        "when not to use": "when_not_to_use",
        "gotchas": "gotchas",
        "strategy summary": "strategy_summary",
        "installation": "installation",
    }
    return aliases.get(n, n.replace(" ", "_").replace("'", ""))
