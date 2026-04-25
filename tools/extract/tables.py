"""Parse markdown tables into list[dict[str,str]]."""
from __future__ import annotations

import re

ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}.*$")


def parse_tables(text: str) -> list[list[dict[str, str]]]:
    """Find every markdown table in text. Return list of tables; each table is list of row dicts."""
    lines = text.splitlines()
    tables: list[list[dict[str, str]]] = []
    i = 0
    while i < len(lines):
        m = ROW_RE.match(lines[i])
        # need header + sep + at least 1 row
        if m and i + 1 < len(lines) and SEP_RE.match(lines[i + 1]):
            headers = [c.strip() for c in m.group(1).split("|")]
            rows: list[dict[str, str]] = []
            j = i + 2
            while j < len(lines):
                rm = ROW_RE.match(lines[j])
                if not rm:
                    break
                cells = [c.strip() for c in rm.group(1).split("|")]
                # pad/truncate to header length
                cells = (cells + [""] * len(headers))[: len(headers)]
                rows.append(dict(zip(headers, cells)))
                j += 1
            if rows:
                tables.append(rows)
            i = j
        else:
            i += 1
    return tables


def strip_md(s: str) -> str:
    """Strip bold/italic/wikilink markup, keeping just text."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"\[\[(.+?)\]\]", r"\1", s)
    s = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", s)
    return s.strip()


def parse_number(s: str) -> float | int | None:
    s = strip_md(s)
    # match optional $/£/€, digits with thousands separators, optional decimal, optional K/M/B suffix
    m = re.search(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        val: float | int = float(raw) if "." in raw else int(raw)
    except ValueError:
        return None
    # apply trailing magnitude suffix immediately after the number
    suffix_m = re.search(re.escape(m.group(0)) + r"\s*([kKmMbB])", s)
    if suffix_m:
        mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[suffix_m.group(1).lower()]
        val = val * mult
        if isinstance(val, float) and val.is_integer():
            val = int(val)
    return val
