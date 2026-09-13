"""Renumber the supplement's sections and rewrite every pointer to them.

The supplement is cited by section number from both documents. Merging two
sections, or dropping one, therefore means editing pointers scattered across
two files, which is why sections accumulate instead: it is cheaper to add S21
than to fold S13 into S12 and fix the forty pointers that move.

This renumbers S1..Sn in document order and rewrites every "S<k>" pointer in
the manuscript and the supplement to match. Table and figure numbers are left
alone: they are numbered in their own sequence and do not follow the section.

    python scripts/renumber_supp.py [--check]
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

DRAFT = DRIVE_ROOT / "PAPER_DRAFT.md"
SUPP = DRIVE_ROOT / "SUPPLEMENTARY.md"
HEAD = re.compile(r"(?m)^##\s+S(\d+)\.\s")
# A bare S12 is a section pointer. "Table S12" is not, and neither is any
# number inside a table or figure group: a lookbehind only guards the first
# number of "Tables S1-S13", which is how a table range came to be renumbered
# as though its upper bound were a section.
GROUP = re.compile(
    r"\b(?:Tables?|Figures?)\s+S?\d+(?:\s*(?:,|and|\u2013|-)\s*S?\d+)*")
POINTER = re.compile(r"\bS(\d+)\b")


def _map_outside_groups(text, fn):
    """Apply `fn` to section pointers, skipping table and figure groups."""
    out, at = [], 0
    for m in GROUP.finditer(text):
        out.append(POINTER.sub(fn, text[at:m.start()]))
        out.append(m.group(0))
        at = m.end()
    out.append(POINTER.sub(fn, text[at:]))
    return "".join(out)


def main() -> None:
    check = "--check" in sys.argv
    supp = io.open(SUPP, encoding="utf-8", newline="").read()
    draft = io.open(DRAFT, encoding="utf-8", newline="").read()

    order = [int(m.group(1)) for m in HEAD.finditer(supp)]
    if len(order) != len(set(order)):
        raise SystemExit(f"the supplement numbers a section twice: {order}")
    mapping = {old: i for i, old in enumerate(order, 1)}

    both = draft + "\n" + supp
    seen = []
    _map_outside_groups(both, lambda m: seen.append(int(m.group(1))) or m.group(0))
    cited = set(seen)
    dangling = sorted(cited - set(mapping))
    if dangling:
        raise SystemExit(
            "pointers with no section: " + str(["S%d" % d for d in dangling]))

    moved = {o: n for o, n in mapping.items() if o != n}
    if not moved:
        print(f"{len(order)} supplement sections, already in order")
        return
    if check:
        print(f"{len(moved)} of {len(order)} sections would move")
        for o in sorted(moved):
            print(f"  S{o} -> S{moved[o]}")
        raise SystemExit(1)

    def sub(t: str) -> str:
        return _map_outside_groups(
            t, lambda m: "S" + str(mapping[int(m.group(1))]))

    io.open(SUPP, "w", encoding="utf-8", newline="").write(sub(supp))
    io.open(DRAFT, "w", encoding="utf-8", newline="").write(sub(draft))
    print(f"{len(order)} supplement sections renumbered; {len(moved)} moved")


if __name__ == "__main__":
    main()
