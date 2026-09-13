"""Resolve [AuthorYear] citation keys in the draft to numbered references.

Numbering is derived, never typed. A numbered bibliography has to follow the
order the references are first cited in, so any edit that adds a paragraph
renumbers everything after it; doing that by hand guarantees an error that a
reader will find. This scans the document, assigns numbers, rewrites the markers
and regenerates the reference list in one pass, and it can be re-run after every
edit.

Two orderings are supported because the target journal is not yet fixed:

    appearance   numbered in order of first citation (Vancouver, IEEE, most
                 medical imaging journals) -- the default
    alphabetical numbered after sorting by first author surname (some Elsevier
                 and APA-numeric styles)

The bibliography itself comes from reports/litreview/bibliography.md, which
holds only entries verified against a fetched primary source. A key with no
entry there is left as-is and reported: an unresolved marker is loud and
fixable, a silently dropped citation is neither, and a plausible-looking
invented reference is worse than both.

    python scripts/number_citations.py                 # check, report, no write
    python scripts/number_citations.py --write
    python scripts/number_citations.py --write --order alphabetical
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

DRAFT = DRIVE_ROOT / "PAPER_DRAFT.md"
BIB = DRIVE_ROOT / "reports" / "litreview" / "bibliography.md"
HEADING = "## References"

# A marker is [Key] or [Key1; Key2; Key3]. Keys look like Menze2015 or Zhang2023a.
KEY = r"[A-Za-z][A-Za-z&]*\d{4}[a-z]?"
MARKER = re.compile(rf"\[({KEY}(?:\s*;\s*{KEY})*)\]")

# Bibliography lines look like:  12. Menze BH, Jakab A, ... DOI: 10.1109/...
BIB_LINE = re.compile(r"^\s*\d+\.\s+(.*\S)\s*$")
REF_SECTION = "## 1. Reference list"
# and the mapping table gives  Key | 12 | one-line identification
MAP_ROW = re.compile(rf"^\|\s*({KEY})\s*\|\s*(\d+)\s*\|")


def load_bibliography() -> dict:
    """key -> formatted reference text, from the verified bibliography file."""
    if not BIB.exists():
        raise SystemExit(f"{BIB} missing -- build it before numbering")
    text = BIB.read_text(encoding="utf-8")

    # Only the reference list itself. The file also carries a numbered list of
    # corrections and several numbered caveat sections; parsing the whole file
    # pulled those in as references 1-4 and silently displaced the real ones.
    start = text.index(REF_SECTION) + len(REF_SECTION)
    rest = text[start:]
    nxt = re.search(r"^## ", rest, flags=re.M)
    section = rest[: nxt.start()] if nxt else rest

    by_number: dict[int, str] = {}
    for line in section.splitlines():
        m = BIB_LINE.match(line)
        if m and not line.lstrip().startswith("|"):
            n = int(line.strip().split(".", 1)[0])
            by_number[n] = m.group(1)

    out: dict[str, str] = {}
    for line in text.splitlines():
        m = MAP_ROW.match(line)
        if m and int(m.group(2)) in by_number:
            out[m.group(1)] = by_number[int(m.group(2))]
    if not out:
        raise SystemExit("no Key|number|... mapping rows found in bibliography.md")
    return out


def body_before_references(text: str) -> str:
    """Everything above the reference list -- markers inside it must not count."""
    return text.split(HEADING, 1)[0]


def cited_keys(text: str) -> list:
    """Keys in first-appearance order, scanning the body only."""
    seen, order = set(), []
    for m in MARKER.finditer(body_before_references(text)):
        for k in (x.strip() for x in m.group(1).split(";")):
            if k not in seen:
                seen.add(k)
                order.append(k)
    return order


def surname(ref: str) -> str:
    return ref.split(",")[0].split()[0].lower() if ref else "~"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite the draft in place")
    ap.add_argument("--order", choices=("appearance", "alphabetical"),
                    default="appearance")
    a = ap.parse_args()

    bib = load_bibliography()
    text = DRAFT.read_text(encoding="utf-8", newline="")
    keys = cited_keys(text)

    missing = [k for k in keys if k not in bib]
    unused = [k for k in bib if k not in keys]

    resolved = [k for k in keys if k in bib]
    if a.order == "alphabetical":
        resolved.sort(key=lambda k: (surname(bib[k]), k))
    number = {k: i + 1 for i, k in enumerate(resolved)}

    def renumber(m):
        parts = [x.strip() for x in m.group(1).split(";")]
        if any(p not in number for p in parts):
            return m.group(0)            # leave unresolved markers visible
        nums = sorted(number[p] for p in parts)
        # collapse runs: [3,4,5] -> 3-5, as numbered styles do
        out, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            out.append(f"{nums[i]}" if j == i else
                       f"{nums[i]},{nums[j]}" if j == i + 1 else
                       f"{nums[i]}-{nums[j]}")
            i = j + 1
        return "[" + ",".join(out) + "]"

    head = body_before_references(text)
    new_head = MARKER.sub(renumber, head)

    lines = [HEADING, ""]
    for k in resolved:
        lines.append(f"{number[k]}. {bib[k]}")
    refs = "\n".join(lines) + "\n"

    new_text = new_head.rstrip() + "\n\n---\n\n" + refs

    print(f"cited keys      : {len(keys)}")
    print(f"resolved        : {len(resolved)}")
    print(f"ordering        : {a.order}")
    if missing:
        print(f"UNRESOLVED ({len(missing)}) -- markers left untouched:")
        for k in missing:
            print(f"    [{k}]")
    if unused:
        print(f"in bibliography but never cited ({len(unused)}): {', '.join(sorted(unused))}")

    if a.write:
        DRAFT.write_text(new_text, encoding="utf-8", newline="")
        print(f"\nwrote {DRAFT.name}: {len(resolved)} numbered references")
    else:
        print("\ndry run -- pass --write to apply")


if __name__ == "__main__":
    main()
