"""Structural checks on a built .docx.

Numbers are checked by audit_numbers.py; this is about the things a reader
notices instead: leftover markdown, figures and tables numbered out of order,
a float that is never referred to, a reference that appears before the thing it
points at.

It also looks for what should not reach a reviewer -- working notes, unfilled
placeholders, paths into the repository -- and for the back matter that most
journals require. Those are reported separately, because they are open items
rather than errors.

    python scripts/check_docx.py ../../manuscript_en.docx
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def flow(doc):
    """Paragraphs and tables in the order they appear in the body."""
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def _cited(bad, label, caps, refs):
    """Every float referred to before it appears, and none of them orphaned."""
    where = {n: p for n, p in caps}
    cited = {n for n, _ in refs}
    for n, _ in caps:
        if n not in cited:
            bad.append(f"{label} {n} is never referred to in the text")
    for n in cited:
        if n not in where:
            bad.append(f"reference to {label} {n}, which has no caption")
            continue
        first = min(p for k, p in refs if k == n)
        if first > where[n]:
            bad.append(f"{label} {n} appears before it is first cited")


def main(path: Path) -> int:
    doc = Document(path)
    bad = []

    fig_cap, tbl_cap = [], []      # (number, position)
    fig_ref, tbl_ref = [], []
    n_tables = 0
    pos = 0
    for item in flow(doc):
        pos += 1
        if isinstance(item, Table):
            n_tables += 1
            continue
        t = item.text
        if not t.strip():
            continue

        if "**" in t or re.search(r"(?<!\w)_[A-Za-z]", t) or "](" in t:
            bad.append(f"leftover markdown at position {pos}: {t[:80]}")

        own = None
        m = re.match(r"^(Figure|Şekil) ([A-Z]?\d+)\.", t)
        if m:
            own = ("fig", m.group(2))
            fig_cap.append((own[1], pos))
        else:
            m = re.match(r"^(Table|Tablo) ([A-Z]?\d+)\.", t)
            if m:
                own = ("tbl", m.group(2))
                tbl_cap.append((own[1], pos))
        # a caption may still point at another float, so keep reading it --
        # only its own number is skipped
        for m in re.finditer(r"\b(?:Figure|Şekil) ([A-Z]?\d+)", t):
            n = m.group(1)
            if own != ("fig", n):
                fig_ref.append((n, pos))
        for m in re.finditer(r"\b(?:Table|Tablo) ([A-Z]?\d+)", t):
            n = m.group(1)
            if own != ("tbl", n):
                tbl_ref.append((n, pos))

    # A document can run two series at once: the body numbers its floats 1, 2,
    # ... and an appendix numbers its own A1, A2, ... Each is checked on its own.
    for kind, caps, refs in (("figure", fig_cap, fig_ref),
                             ("table", tbl_cap, tbl_ref)):
        series = {}
        for n, p in caps:
            series.setdefault(n.rstrip("0123456789"), []).append((n, p))
        for prefix, group in series.items():
            nums = [n for n, _ in group]
            mine = [(n, p) for n, p in refs
                    if n.rstrip("0123456789") == prefix]
            label = f"{prefix}{kind}" if prefix else kind
            seq = [prefix + str(i) for i in range(1, len(nums) + 1)]
            if nums != seq:
                bad.append(f"{label} numbering is not 1..n in order: {nums}")
            if len(set(nums)) != len(nums):
                bad.append(f"{label} number used twice: {nums}")
            _cited(bad, label, group, mine)
        continue

    for kind, caps, refs in ():
        nums = []

        pass

    # what a reviewer should not be shown, and what they will look for
    LEAK = [
        ("working note left in", r"drafting note|taslak notu|^Source:|^Kaynak:"),
        ("placeholder left in", r"\[[A-Z][A-Z0-9_]{2,}\]|\bAYKUT\b|\bTODO\b|\bFIXME\b"),
        ("path into the repository", r"demov2|RUNS_ANALYSIS|PAPER_DRAFT|"
                                     r"scripts/[a-z_]+\.py|reports/[a-z_/]+"),
    ]
    open_items = []
    for item in flow(doc):
        if isinstance(item, Table):
            continue
        t = item.text.strip()
        for label, pat in LEAK:
            if re.search(pat, t, re.M):
                open_items.append(f"{label}: {t[:90]}")

    # Diagnostics (MDPI) wants these headings verbatim, and wants Informed
    # Consent as its own section even when the answer is "not applicable".
    REQUIRED = ["Author Contributions", "Funding",
                "Institutional Review Board Statement",
                "Informed Consent Statement", "Data Availability Statement",
                "Conflicts of Interest"]
    # the MDPI template names its heading styles MDPI21heading1 and sets the
    # back matter in MDPI62backmatter, so matching on "Heading" alone reported
    # every required section missing from a file that had all of them
    # style names differ between the plain and the MDPI template -- "Heading 1"
    # against "MDPI_6.2_back_matter" -- so the name is compared with its
    # punctuation removed rather than matched literally
    def _sty(p):
        return "".join(ch for ch in p.style.name.lower() if ch.isalnum())
    heads = [p.text.strip() for p in doc.paragraphs
             if "heading" in _sty(p) or "backmatter" in _sty(p)]
    for want in REQUIRED:
        if not any(want.lower() in h.lower() for h in heads):
            open_items.append(f"no '{want}' section")

    # MDPI sets the abbreviations list as a table, and it takes no "Table N."
    # caption, so it is one table that is not meant to have one
    n_captioned = n_tables - (1 if any("abbreviation" in h.lower()
                                       for h in heads) else 0)
    if len(tbl_cap) != n_captioned:
        bad.append(f"{n_captioned} captionable tables but "
                   f"{len(tbl_cap)} table captions")

    print(f"{path.name}: {len(fig_cap)} figures, {n_tables} tables, "
          f"{len(doc.paragraphs)} paragraphs")
    if bad:
        print("\nPROBLEMS")
        for b in bad:
            print("  -", b)
    else:
        print("figures and tables numbered in order, each cited before it appears")

    if open_items:
        print("\nBEFORE SUBMISSION")
        for o in open_items:
            print("  -", o)
    else:
        print("nothing left that a reviewer should not see")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
