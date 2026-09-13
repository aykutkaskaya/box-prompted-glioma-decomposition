"""Numbering, caption placement and citation order, read from the built document.

Everything here is checked against the .docx that goes to the journal rather
than the Markdown source, because placement is a property of the built file:
a caption sits above or below its object, and a citation comes before or after
the thing it points at, only once the document exists.

MDPI sets table captions above the table and figure captions below the figure.

    python scripts/check_layout.py [path/to.docx]
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

CITE = re.compile(r"\[((?:\d+\s*[-–]\s*\d+|\d+)(?:\s*,\s*(?:\d+\s*[-–]\s*\d+|\d+))*)\]")
CAPTION = re.compile(r"^(Table|Figure)\s+(S?\d+)\.\s*(.*)$", re.S)
NAMED = re.compile(r"\b(Table|Figure)\s+(S?\d+)\b")


def body_items(doc):
    """Paragraphs and tables in the order they appear in the document."""
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def numbers(group: str) -> list[int]:
    out = []
    for part in re.split(r"\s*,\s*", group):
        m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part.strip())
        out += (list(range(int(m.group(1)), int(m.group(2)) + 1)) if m
                else [int(part)])
    return out


def _companion_count(path: Path) -> int:
    """How many references the manuscript beside this supplement carries."""
    for name in ("manuscript_en_mdpi.docx", "manuscript_en.docx"):
        p = path.with_name(name)
        if p.exists() and p != path:
            return sum(1 for q in Document(str(p)).paragraphs
                       if q.style.name == "MDPI_8.1_references" and q.text.strip())
    return 0


def main() -> None:
    """Check every built document, or the one named on the command line.

    The default used to be the manuscript alone. The supplement carries its
    own tables and figures and is the file a renumbering touches, so a round
    that moved four of its captions was checked only against the document it
    had not changed.
    """
    if len(sys.argv) > 1:
        raise SystemExit(check_one(Path(sys.argv[1])))
    worst = 0
    for name in ("manuscript_en_mdpi.docx", "supplementary_en.docx"):
        p = DRIVE_ROOT / name
        if not p.exists():
            continue
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        worst = max(worst, check_one(p))
    raise SystemExit(worst)


def check_one(path: Path) -> int:
    doc = Document(str(path))
    items = list(body_items(doc))
    problems: list[str] = []

    # ---------------------------------------------- captions and placement
    seq: list[tuple[int, str, str, str]] = []   # index, kind, number, where
    for i, it in enumerate(items):
        if not isinstance(it, Paragraph):
            continue
        m = CAPTION.match(it.text.strip())
        if not m:
            continue
        kind, num = m.group(1), m.group(2)
        nxt = items[i + 1] if i + 1 < len(items) else None
        prv = items[i - 1] if i else None
        if kind == "Table":
            where = ("above" if isinstance(nxt, Table)
                     else "below" if isinstance(prv, Table) else "none")
        else:
            has = lambda x: (isinstance(x, Paragraph)
                             and bool(x._p.findall(
                                 ".//{http://schemas.openxmlformats.org/"
                                 "drawingml/2006/main}blip")))
            where = ("below" if has(prv) else "above" if has(nxt) else "none")
        seq.append((i, kind, num, where))
        if not m.group(3).strip():
            problems.append(f"{kind} {num}: caption has no text")

    print(f"{path.name}\n")
    print("CAPTIONS, in the order they appear")
    for _, kind, num, where in seq:
        want = "above" if kind == "Table" else "below"
        mark = "ok" if where == want else f"** {where} **"
        print(f"  {kind:6s} {num:4s} caption {where:6s} its object   {mark}")
        if where != want:
            problems.append(
                f"{kind} {num}: caption is {where} the object; MDPI puts a "
                f"{kind.lower()} caption {want}")

    # ---------------------------------------------- numbering runs in order
    for kind in ("Table", "Figure"):
        got = [int(n) for _, k, n, _ in seq if k == kind and not n.startswith("S")]
        want = list(range(1, len(got) + 1))
        print(f"\n{kind.upper()} NUMBERING: {got}")
        if got != want:
            problems.append(f"{kind}s are numbered {got}, not {want}")
        else:
            print("  sequential from 1, no gaps")

    # ------------------------------- every object named in the text, first
    # mention before the object itself
    text_pos: dict[tuple[str, str], int] = {}
    for i, it in enumerate(items):
        if not isinstance(it, Paragraph):
            continue
        if CAPTION.match(it.text.strip()):
            continue
        for kind, num in NAMED.findall(it.text):
            text_pos.setdefault((kind, num), i)

    print("\nEACH OBJECT CITED BEFORE IT APPEARS")
    for i, kind, num, _ in seq:
        at = text_pos.get((kind, num))
        if at is None:
            print(f"  {kind} {num}: never named in the text   ** check **")
            problems.append(f"{kind} {num} is never referred to in the text")
        elif at > i:
            print(f"  {kind} {num}: first named after it appears   ** check **")
            problems.append(f"{kind} {num} appears before the text names it")
        else:
            print(f"  {kind} {num}: named first at paragraph {at}   ok")

    # ---------------------------------------------------------- references
    refs = [p.text.strip() for p in doc.paragraphs
            if p.style.name == "MDPI_8.1_references" and p.text.strip()]
    print(f"\nREFERENCE LIST: {len(refs)} entries")

    body_text = []
    for it in items:
        if isinstance(it, Paragraph):
            if it.style.name == "MDPI_8.1_references":
                break
            body_text.append(it.text)
    body = "\n".join(body_text)

    seen: list[int] = []
    for m in CITE.finditer(body):
        for v in numbers(m.group(1)):
            if 1 <= v <= len(refs) and v not in seen:
                seen.append(v)

    missing = [n for n in range(1, len(refs) + 1) if n not in seen]
    out_of_order = [(i, v) for i, v in enumerate(seen, 1) if i != v]
    print(f"  cited in the text: {len(seen)} of {len(refs)}")
    if missing:
        print(f"  never cited: {missing}   ** check **")
        problems.append(f"references never cited: {missing}")
    else:
        print("  every entry is cited")
    if out_of_order:
        first = out_of_order[0]
        print(f"  first citation order breaks at position {first[0]} "
              f"(found {first[1]})   ** check **")
        problems.append(
            f"{len(out_of_order)} references are not in order of first "
            f"citation; the first break is at position {first[0]}")
    else:
        print("  numbered in order of first citation")

    # A supplement carries no list of its own and cites the manuscript's, so
    # its citations are resolved against that file rather than reported as
    # pointing past the end of an empty list.
    companion = len(refs) or _companion_count(path)
    high = [v for m in CITE.finditer(body) for v in numbers(m.group(1))
            if v > companion]
    if not refs and companion:
        used = {v for m in CITE.finditer(body) for v in numbers(m.group(1))}
        print(f"  no list of its own; {len(used)} citations resolved against "
              f"the manuscript's {companion}")
    if high:
        print(f"  citations past the end of the list: {sorted(set(high))}")
        problems.append(f"citations to non-existent references: {sorted(set(high))}")

    listed = [int(re.match(r"(\d+)[.\s]", r).group(1)) if re.match(r"(\d+)[.\s]", r)
              else None for r in refs]
    if any(listed):
        print("  entries carry their own numbers as text as well as the "
              "list's   ** check **")
        problems.append("reference entries are numbered twice")


    # ------------------------------------------------ paragraphs cut in half
    # A continuation line that lost its indent ends its list item early and
    # the remainder is emitted as a paragraph of its own, starting mid
    # sentence. Captions, list items and equations legitimately do not start
    # with a capital, so only plain body paragraphs are examined.
    for it in items:
        if not isinstance(it, Paragraph):
            continue
        t = it.text.strip()
        style = (it.style.name or "").lower()
        # headings carry their own number, a reference may open on a
        # lower-case surname and an affiliation on a marker; only ordinary
        # body prose is examined
        if any(k in style for k in ("bullet", "list", "equation", "heading",
                                    "reference", "affiliation", "title",
                                    "author", "keyword")):
            continue
        if not t or CAPTION.match(t) or t.startswith(("http", "\u00a7")):
            continue
        if re.match(r"^\d+(?:\.\d+)*\.?\s", t) or "\t" in t[:4]:
            continue
        broken = (t[0].islower() and not t.startswith(("van ", "de ", "von "))) \
            or re.match(r"^\d+,\s", t) is not None
        if broken:
            print(f"  paragraph begins mid-sentence: {t[:60]!r}   ** check **")
            problems.append(f"paragraph begins mid-sentence: {t[:40]!r}")


    # ------------------------------------------------------- sections present
    # A heading that loses its "## " -- an indented line, a stray rule swallowing
    # the blank line before it -- becomes body text, and every other check still
    # passes. Compare the source's numbered sections against the document's.
    src = path.parent / ("PAPER_DRAFT.md" if "mdpi" in path.name else "SUPPLEMENTARY.md")
    if src.exists():
        want = re.findall(r"(?m)^#{2,3}\s+(\d+(?:\.\d+)*)\.?\s+\S",
                          io.open(src, encoding="utf-8").read())
        got = set()
        for it in items:
            if isinstance(it, Paragraph) and "heading" in (it.style.name or "").lower():
                m = re.match(r"^(\d+(?:\.\d+)*)\.", it.text.strip())
                if m:
                    got.add(m.group(1))
        missing = [w for w in want if w not in got]
        print(f"\nSECTIONS: {len(want)} numbered in the source, "
              f"{len(got)} as headings in the document")
        if missing:
            print(f"  missing from the document: {missing}   ** check **")
            problems.append(f"sections lost their heading: {missing}")


    # ------------------------------------------------ cross-reference targets
    # "§3.1" and "S9" are pointers; a pointer whose target does not exist sends
    # the reader looking for something that was moved or never written.
    draft = path.parent / "PAPER_DRAFT.md"
    supp = path.parent / "SUPPLEMENTARY.md"
    if draft.exists() and supp.exists():
        dtext = io.open(draft, encoding="utf-8").read()
        stext = io.open(supp, encoding="utf-8").read()
        have_sec = set(re.findall(r"(?m)^#{2,3}\s+(\d+(?:\.\d+)*)\.?\s", dtext))
        have_sup = set(re.findall(r"(?m)^##\s+(S\d+)\.", stext))
        have_tab = set(re.findall(r"(?m)^\*\*Table (S\d+)\.", stext))
        have_fig = set(re.findall(r"(?m)^\*\*Figure (S\d+)\.", stext))
        both = dtext + chr(10) + stext
        dangling = set()
        # "§3.1" was the old form and "Section 3.1" is the current one; a
        # plural — "Sections 3.1 and 3.4", "Sections 2.1-2.3" — names several
        for m in re.finditer(r"\u00a7\s?(\d+(?:\.\d+)*)"
                             r"|\bSections?\s+(\d+(?:\.\d+)*"
                             r"(?:\s*(?:,|and|\u2013|-)\s*\d+(?:\.\d+)*)*)", both):
            group = m.group(1) or m.group(2)
            for ref in re.findall(r"\d+(?:\.\d+)*", group):
                if ref not in have_sec:
                    dangling.add("Section " + ref)
        # a bare S12 is a supplement section; its tables and figures are named
        for m in re.finditer(r"(?<!Table )(?<!Figure )\b(S\d+)\b", both):
            if m.group(1) not in have_sup:
                dangling.add(m.group(1))
        for kind, have in (("Table", have_tab), ("Figure", have_fig)):
            for m in re.finditer(rf"\b{kind}s?\s+((?:S\d+)"
                                 rf"(?:\s*(?:,|and|\u2013|-)\s*S?\d+)*)", both):
                for ref in re.findall(r"S?\d+", m.group(1)):
                    ref = ref if ref.startswith("S") else "S" + ref
                    if ref not in have:
                        dangling.add(f"{kind} {ref}")
        print(f"\nCROSS-REFERENCES: {len(have_sec)} numbered sections, "
              f"{len(have_sup)} supplement sections, "
              f"{len(have_tab)} supplement tables, {len(have_fig)} figures")
        if dangling:
            print(f"  pointing at nothing: {sorted(dangling)}   ** check **")
            problems.append(f"cross-references with no target: {sorted(dangling)}")


    # ------------------------------------- supplement numbering and pointers
    # The supplement numbers its own tables and figures, and the docx check
    # below skips anything with an S in it. Merging two sections moves their
    # captions, so the order is a property of the source and is checked there.
    if draft.exists() and supp.exists():
        for kind in ("Table", "Figure"):
            seen = [int(m.group(1)) for m in
                    re.finditer(rf"(?m)^\*\*{kind} S(\d+)\.", stext)]
            # ascending is not enough: a gap or a repeat is also ascending,
            # and both are what inserting a table in the middle produces
            if seen and seen != list(range(1, len(seen) + 1)):
                out = ", ".join(f"S{n}" for n in seen)
                print(f"  {kind}s are numbered {out}, not S1..S{len(seen)}"
                      f"   ** check **")
                problems.append(
                    f"supplement {kind.lower()}s are numbered {out}, which is "
                    f"not S1 to S{len(seen)} in document order")
        # "(S12, S12;)" is what a merge leaves when two sections collapse
        for m in re.finditer(r"\bS(\d+)((?:\s*,\s*S\d+)+)", both):
            group = [m.group(1)] + re.findall(r"S(\d+)", m.group(2))
            if len(group) != len(set(group)):
                print(f"  a pointer names the same section twice: "
                      f"{m.group(0)}   ** check **")
                problems.append(f"repeated section in one pointer: {m.group(0)}")

    # ------------------------------------------------ table and figure order
    # MDPI cites tables and figures in numerical order. A pointer added to an
    # earlier section can put a high number first without breaking anything
    # else, so first-citation order is checked the way the reference list is.
    for kind in ("Table", "Figure"):
        first: dict[int, int] = {}
        for i, it in enumerate(items):
            if not isinstance(it, Paragraph):
                continue
            if CAPTION.match(it.text.strip()):
                continue          # a caption names its own object
            for m in re.finditer(rf"\b{kind}\s+(\d+)\b", it.text):
                first.setdefault(int(m.group(1)), i)
        order = [n for n, _ in sorted(first.items(), key=lambda kv: kv[1])]
        if order != sorted(order):
            print(f"  {kind}s first cited out of order: {order}   ** check **")
            problems.append(f"{kind.lower()}s cited out of numerical order: {order}")
        else:
            print(f"\n{kind.upper()}S: first cited in order {order}")


    # --------------------------------------------------- captions arrive whole
    # A caption is written as one block in the source. If the builder keeps only
    # its first line, the rest becomes a body paragraph and the caption ends
    # mid-sentence, which no other check notices.
    src = path.parent / ("PAPER_DRAFT.md" if "mdpi" in path.name else "SUPPLEMENTARY.md")
    if src.exists():
        raw = io.open(src, encoding="utf-8").read().split(chr(10))
        want = {}
        for j, line in enumerate(raw):
            m = re.match(r"^\*\*((?:Table|Figure)\s+S?\d+)\.\*\*", line.strip())
            if not m:
                continue
            block = [line.strip()]
            k = j + 1
            while (k < len(raw) and raw[k].strip()
                   and not raw[k].lstrip().startswith(("|", "!["))):
                block.append(raw[k].strip())
                k += 1
            want[m.group(1)] = len(" ".join(block).replace("**", "").split())
        for it in items:
            if not isinstance(it, Paragraph):
                continue
            m = CAPTION.match(it.text.strip())
            if not m:
                continue
            key = f"{m.group(1)} {m.group(2)}"
            if key in want:
                got = len(it.text.split())
                if got < want[key] - 2:          # a word or two of markup may go
                    print(f"  {key}: caption truncated, {got} words of "
                          f"{want[key]}   ** check **")
                    problems.append(f"{key} caption arrived truncated "
                                    f"({got} of {want[key]} words)")


    # ------------------------------------------------------------ placeholders
    # An address that does not exist yet is held open in the highlighted form
    # the builder understands. Nothing may be submitted while one is still in.
    for it in items:
        if not isinstance(it, Paragraph):
            continue
        for m in re.finditer(r"\[[^\]]*AYKUT\]", it.text):
            print(f"  placeholder still in the document: {m.group(0)}"
                  f"   ** check **")
            problems.append(f"placeholder not filled in: {m.group(0)}")

    # --------------------------------------------------------- export is fresh
    # The PDF is exported from Word by hand, so it can fall behind the
    # document it came from without anything saying so, and the version
    # that reaches the journal is then not the version that was checked.
    pdf = path.with_suffix(".pdf")
    if pdf.exists():
        behind = path.stat().st_mtime - pdf.stat().st_mtime
        if behind > 0:
            mins = int(behind // 60)
            print(f"  {pdf.name} is {mins} min older than {path.name}"
                  f"   ** check **")
            problems.append(f"{pdf.name} predates {path.name} by {mins} "
                            f"min; re-export it before submitting")

    print("\n" + ("-" * 60))
    if problems:
        print(f"{len(problems)} to look at")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0
    print("numbering, caption placement and citation order are all consistent")


if __name__ == "__main__":
    main()
