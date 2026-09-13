"""Renumber the bibliography into citation order, and rewrite every citation.

MDPI numbers references by first mention. Keeping that true by hand means that
adding one reference to the Introduction renumbers most of the list and every
citation after it, which is why references get appended at the end instead and
the rule quietly breaks. This does it mechanically, in both documents.

It reads the citations in the manuscript in order -- the supplement cites the
manuscript's list and does not extend it -- assigns numbers by first
appearance, and rewrites the reference entries into that order. A citation
with no entry, or an entry nothing cites, is an error rather than a silent
renumber, because either means a reference was lost.

    python scripts/renumber_refs.py [--check]

--check reports what would change and exits non-zero if anything would,
which is what a build gate wants.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

DRAFT = DRIVE_ROOT / "PAPER_DRAFT.md"
SUPP = DRIVE_ROOT / "SUPPLEMENTARY.md"
# "[4]", "[10, 11]", "[1-3]" and the en-dash form of a range
CITE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
ENTRY = re.compile(r"(?m)^(\d+)\.\s")


def numbers(group: str) -> list[int]:
    """The numbers a citation names, ranges expanded."""
    out: list[int] = []
    for part in re.split(r"\s*,\s*", group.strip()):
        m = re.match(r"^(\d+)\s*[–-]\s*(\d+)$", part)
        if m:
            out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            out.append(int(part))
    return out


def split_entries(block: str) -> dict[int, str]:
    """The reference list as {number: text without its number}."""
    entries: dict[int, str] = {}
    cur, buf = None, []
    for line in block.split("\n"):
        m = ENTRY.match(line)
        if m:
            if cur is not None:
                entries[cur] = "\n".join(buf).strip()
            cur, buf = int(m.group(1)), [ENTRY.sub("", line, count=1)]
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        entries[cur] = "\n".join(buf).strip()
    return entries


def render(group: str, mapping: dict[int, int]) -> str:
    """One citation, renumbered, ranges rebuilt where they stay contiguous."""
    new = sorted({mapping[n] for n in numbers(group)})
    out, i = [], 0
    while i < len(new):
        j = i
        while j + 1 < len(new) and new[j + 1] == new[j] + 1:
            j += 1
        out.append(str(new[i]) if j - i < 2 else f"{new[i]}-{new[j]}")
        if j - i == 1:
            out[-1] = f"{new[i]}, {new[j]}"
        i = j + 1
    return "[" + ", ".join(out) + "]"


def main() -> None:
    check = "--check" in sys.argv
    draft = io.open(DRAFT, encoding="utf-8", newline="").read()
    supp = io.open(SUPP, encoding="utf-8", newline="").read()
    body, refs = draft.split("## References", 1)
    appendix = ""
    if "## Appendix A" in refs:
        refs, appendix = refs.split("## Appendix A", 1)
        appendix = "## Appendix A" + appendix
    entries = split_entries(refs)

    order: list[int] = []
    for m in CITE.finditer(body):
        for n in numbers(m.group(1)):
            if n not in order:
                order.append(n)
    for m in CITE.finditer(supp):
        for n in numbers(m.group(1)):
            if n not in order:
                order.append(n)

    missing = [n for n in order if n not in entries]
    if missing:
        raise SystemExit(f"cited with no entry in the list: {missing}")
    uncited = sorted(set(entries) - set(order))
    if uncited:
        raise SystemExit(f"in the list and never cited: {uncited}")

    mapping = {old: i for i, old in enumerate(order, 1)}
    moved = {o: n for o, n in mapping.items() if o != n}
    if not moved:
        print(f"{len(order)} references, already in citation order")
        return
    if check:
        print(f"{len(moved)} of {len(order)} references are out of citation order")
        for o in sorted(moved)[:12]:
            print(f"  {o} -> {moved[o]}")
        raise SystemExit(1)

    body = CITE.sub(lambda m: render(m.group(1), mapping), body)
    supp = CITE.sub(lambda m: render(m.group(1), mapping), supp)
    lines = ["## References", ""]
    for old in order:
        lines.append(f"{mapping[old]}. {entries[old]}")
        lines.append("")
    out = body + "\n".join(lines).rstrip() + "\n"
    if appendix:
        out += "\n" + appendix
    io.open(DRAFT, "w", encoding="utf-8", newline="").write(out)
    io.open(SUPP, "w", encoding="utf-8", newline="").write(supp)
    # these are keyed by the reference's number, so they move with it
    for name in ("ref_authors.json", "ref_events.json"):
        p = DRIVE_ROOT / "reports" / name
        if not p.exists():
            continue
        try:
            d = json.loads(io.open(p, encoding="utf-8").read())
        except ValueError:
            continue
        moved_side = {str(mapping[int(k)]): v for k, v in d.items()
                      if k.isdigit() and int(k) in mapping}
        kept = {k: v for k, v in d.items() if not k.isdigit()}
        io.open(p, "w", encoding="utf-8", newline="").write(
            json.dumps({**kept, **moved_side}, ensure_ascii=False, indent=1))
        print(f"  {name}: {len(moved_side)} entries follow the new numbering")
    print(f"{len(order)} references renumbered into citation order; "
          f"{len(moved)} moved")


if __name__ == "__main__":
    main()
