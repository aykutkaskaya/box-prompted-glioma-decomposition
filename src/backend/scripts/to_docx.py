"""PAPER_DRAFT.md -> a .docx to edit and share on Google Drive.

Word headings, real tables and embedded PNGs, so Google Docs keeps the outline
and the advisor can comment inline. Journal-independent: when the venue is
picked, this is still the draft everyone has been editing.

LaTeX maths is rendered to Unicode rather than to Word equation objects.
Nothing here survives a round trip through Google Docs' equation editor, and
alpha, IoU and a couple of derivatives read fine as text.

Open items stay visible: anything in square brackets ending AYKUT, and the
pending ORCIDs, get a yellow highlight.

    python scripts/to_docx.py
    python scripts/to_docx.py --lang tr
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

DRAFT = DRIVE_ROOT / "PAPER_DRAFT.md"
GREY = RGBColor(0x60, 0x66, 0x6C)
INK = RGBColor(0x1A, 0x1D, 0x21)
BODY_FONT = "Times New Roman"
HEAD_FONT = "Calibri"
SHADE = "F2F2F2"          # table header fill


def _font(style, name, size=None, bold=None, colour=None):
    style.font.name = name
    # python-docx only writes w:ascii; without w:eastAsia Word falls back for
    # anything outside Latin-1, which here means every Turkish character
    style.element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if colour is not None:
        style.font.color.rgb = colour


def _page_numbers(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            fld = OxmlElement("w:fldChar")
            fld.set(qn("w:fldCharType"), tag)
            r._r.append(fld)
        else:
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = txt
            r._r.append(instr)
    r.font.size = Pt(9)
    r.font.name = BODY_FONT


def _keep_together(table):
    """Stop Word breaking a table across a page.

    A caption with keep_with_next only glues the caption to the first row, so a
    table could still leave its header stranded at the foot of a page. Rows are
    marked unsplittable, the header row is marked as one so it repeats if a
    long table does break, and every row but the last keeps with the next.
    """
    for i, row in enumerate(table.rows):
        pr = row._tr.get_or_add_trPr()
        pr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            pr.append(OxmlElement("w:tblHeader"))
        if i < len(table.rows) - 1:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True


def _shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:fill"), fill)
    tcPr.append(sh)

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "lambda": "λ", "mu": "μ", "sigma": "σ",
    "tau": "τ", "phi": "φ", "theta": "θ", "Delta": "Δ",
    "partial": "∂", "cap": "∩", "cup": "∪", "ge": "≥",
    "le": "≤", "in": "∈", "times": "×", "pm": "±",
    "cdot": "·", "approx": "≈", "neq": "≠", "to": "→",
    "leftarrow": "←", "rightarrow": "→", "infty": "∞",
    "sum": "Σ", "sqrt": "√", "circ": "°", "ldots": "…",
    # LaTeX accepts both spellings of the relations; only the short ones were
    # here, so the long form lost its backslash and printed as a word
    "geq": "≥", "leq": "≤", "gg": "≫", "ll": "≪", "equiv": "≡",
    "propto": "∝", "subset": "⊂", "subseteq": "⊆", "supset": "⊃",
    "setminus": "∖", "emptyset": "∅", "forall": "∀", "exists": "∃",
    "prod": "Π", "int": "∫", "nabla": "∇", "perp": "⊥", "angle": "∠",
    "notin": "∉", "cong": "≅", "sim": "∼", "prime": "′",
}
SUP = str.maketrans("0123456789+-()n", "⁰¹²³⁴⁵"
                                       "⁶⁷⁸⁹⁺⁻"
                                       "⁽⁾ⁿ")
SUB = str.maketrans("0123456789+-()", "₀₁₂₃₄₅"
                                      "₆₇₈₉₊₋"
                                      "₍₎")


def braced(s: str, i: int):
    """Return (contents, index after) for a {...} group starting at s[i]."""
    while i < len(s) and s[i] in " \t":   # a fraction split
        i += 1                            # across lines joins with a space
    if i >= len(s) or s[i] != "{":
        return (s[i] if i < len(s) else ""), i + 1
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def math(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if not m:
                out.append(" " if s[i + 1:i + 2] in (",", ";", " ")
                           else "" if s[i + 1:i + 2] == "!" else s[i + 1])
                i += 2
                continue
            name = m.group(1)
            i += m.end()
            if name == "frac":
                a, i = braced(s, i)
                b, i = braced(s, i)
                a, b = math(a), math(b)
                wrap = lambda t: t if len(t) <= 3 or t.startswith("(") else "(" + t + ")"
                out.append(wrap(a) + "/" + wrap(b))
            elif name in ("mathrm", "mathcal", "mathbf", "text", "operatorname",
                          "mathit", "boldsymbol"):
                a, i = braced(s, i)
                out.append(math(a))
            elif name == "underbrace":
                a, i = braced(s, i)
                out.append(math(a))
                if s[i:i + 1] == "_":
                    lbl, i = braced(s, i + 1)
                    out.append(" [" + math(lbl) + "]")
            elif name in ("qquad", "quad"):
                out.append("    ")
            elif name in ("bigl", "bigr", "Bigl", "Bigr", "left", "right",
                          "displaystyle", "nonumber", "limits"):
                pass
            elif name in GREEK:
                out.append(GREEK[name])
            else:
                out.append(name)
        elif c in "_^":
            grp, i = braced(s, i + 1)
            g = math(grp)
            tbl = SUB if c == "_" else SUP
            if g and all(ord(ch) in tbl for ch in g):
                out.append(g.translate(tbl))
            elif g.isalnum():                 # L_ov reads better than L_(ov)
                out.append(c + g)
            else:
                out.append(c + ("(" + g + ")" if len(g) > 1 else g))
            continue
        elif c == "$":
            i += 1
            continue
        else:
            out.append(c)
            i += 1
    return "".join(out)


INLINE = re.compile(
    r"(\*\*.+?\*\*|\*[^*\n]+?\*|`[^`]+?`|\$[^$]+?\$|\[[^\]\n]*?AYKUT\])")


def runs(par, text: str, base_italic=False, colour=None, size=None):
    # Markdown carries no apostrophe, en dash or power of ten; the journal
    # sets all three. Applied here so the source stays plain and every
    # caller gets it.
    try:
        import typeset
        text = typeset.inline(text)
    except Exception:
        pass

    def add(t, **kw):
        if not t:
            return
        # a name like d_free in running prose is set the way the equations
        # set it, as an italic letter with a real subscript
        try:
            import typeset
            if typeset.SUBSCRIPT_RE.search(t):
                typeset.subscript_runs(par, t,
                                       italic=kw.get("italic", base_italic),
                                       bold=kw.get("bold", False))
                return
        except Exception:
            pass
        r = par.add_run(t)
        # None inherits from the paragraph style, False overrides it: the
        # MDPI heading, abstract, keyword and back-matter styles are bold
        # or italic, and writing False stripped all of them
        r.italic = True if kw.get("italic", base_italic) else None
        r.bold = True if kw.get("bold", False) else None
        if kw.get("mono"):
            r.font.name = "Consolas"
        if kw.get("hl"):
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if colour is not None:
            r.font.color.rgb = colour
        if size is not None:
            r.font.size = Pt(size)

    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            # a bold span may itself contain maths or code; render those rather
            # than printing their markers
            for sub in INLINE.split(part[2:-2]):
                if not sub:
                    continue
                if sub.startswith("$") and sub.endswith("$"):
                    add(math(sub[1:-1]), bold=True, italic=True)
                elif sub.startswith("`") and sub.endswith("`"):
                    add(sub[1:-1], bold=True, mono=True)
                else:
                    add(sub.replace("**", ""), bold=True)
        elif part.startswith("$") and part.endswith("$"):
            add(math(part[1:-1]), italic=True)
        elif part.startswith("`") and part.endswith("`"):
            body = part[1:-1]
            add(body, mono=True, hl="ORCID" in body or "FORMAT" in body)
        elif part.startswith("[") and part.endswith("AYKUT]"):
            add(part, hl=True)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            add(part[1:-1], italic=True)
        else:
            # anything still carrying markers means an unbalanced span; show
            # the words rather than the syntax
            add(part.replace("**", ""))


def para(doc, text, style=None, **kw):
    p = doc.add_paragraph(style=style)
    runs(p, text, **kw)
    return p


def add_table(doc, block, caption, num):
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block]
    header, body = rows[0], rows[2:]
    n = len(header)
    # every table gets a numbered caption; the ones the draft never wrote are
    # left as a visible blank rather than invented here
    # captions are numbered in the source and check_docx.py verifies the
    # sequence, so an existing one is passed through untouched -- renumbering
    # here mangled the S-series of the supplementary
    if not caption:
        caption = f"**Table {num}.** [caption AYKUT]"
    c = doc.add_paragraph()
    c.paragraph_format.space_before = Pt(12)
    c.paragraph_format.space_after = Pt(3)
    c.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    c.paragraph_format.keep_with_next = True
    c.paragraph_format.keep_together = True
    runs(c, caption, size=9)

    t = doc.add_table(rows=1, cols=n)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    def fill(cell, text, bold=False, first=False):
        p = cell.paragraphs[0]
        p.text = ""
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        # the label column reads as text, the rest as figures
        p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if first
                       else WD_ALIGN_PARAGRAPH.CENTER)
        runs(p, text, size=9)
        if bold:
            for r in p.runs:
                r.bold = True

    for k, (cell, h) in enumerate(zip(t.rows[0].cells, header)):
        fill(cell, h, bold=True, first=k == 0)
        _shade(cell, SHADE)
    for row in body:
        row = (row + [""] * n)[:n]
        cells = t.add_row().cells
        for k, (cell, v) in enumerate(zip(cells, row)):
            fill(cell, v, first=k == 0)

    _keep_together(t)          # after every row exists, not before

    tail = doc.add_paragraph()
    tail.paragraph_format.space_after = Pt(6)


def convert(md: str, lang: str, authors: bool = True) -> Document:
    doc = Document()

    sec = doc.sections[0]
    # python-docx defaults to US Letter; the manuscript is A4 and the two
    # documents are submitted together
    # justified text without hyphenation stretches any line carrying a long
    # identifier; the manuscript template sets this and the supplement did not
    el = doc.settings.element
    if el.find(qn("w:autoHyphenation")) is None:
        h = OxmlElement("w:autoHyphenation")
        h.set(qn("w:val"), "true")
        el.append(h)
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    sec.left_margin = sec.right_margin = Cm(2.5)
    _page_numbers(sec)

    st = doc.styles["Normal"]
    _font(st, BODY_FONT, 11, colour=INK)
    pf = st.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Word's built-in headings are blue and sans; a manuscript wants neither
    for name, size, before, after in (("Title", 17, 0, 6),
                                      ("Heading 1", 13.5, 18, 6),
                                      ("Heading 2", 11.5, 12, 4)):
        h = doc.styles[name]
        _font(h, HEAD_FONT, size, bold=True, colour=INK)
        h.paragraph_format.space_before = Pt(before)
        h.paragraph_format.space_after = Pt(after)
        h.paragraph_format.keep_with_next = True
        h.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        h.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    lines = md.split("\n")
    i, fig_pending = 0, None
    table_caption, tbl_n = "", 0
    front = authors              # authors and affiliations keep their line
                                 # breaks; body text, and every line of a
                                 # document without an author block, reflows

    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if s.startswith("# "):
            h = doc.add_heading(s[2:], level=0)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            h.paragraph_format.space_after = Pt(14)
            i += 1
            continue
        if s.startswith("## "):
            front = False
            doc.add_heading(s[3:], level=1)
            i += 1
            continue
        if s.startswith("### "):
            doc.add_heading(s[4:], level=2)
            i += 1
            continue

        if s == "---":
            i += 1
            continue

        if s.startswith("> "):                       # drafting note
            note = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                note.append(lines[i].strip().lstrip("> ").rstrip())
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            tag = "[drafting note] " if lang == "en" else "[taslak notu] "
            runs(p, tag + " ".join(note), base_italic=True,
                 colour=GREY, size=9.5)
            continue

        # "Figure S1" as well as "Figure 1": the supplement numbers its
        # figures with an S, and requiring a bare digit meant its two
        # figures were silently dropped from the built document.
        m = re.match(r"^!\[(?:Figure|\u015eekil) (S?\d+)\]\((.+?)\)$", s)
        if m:
            path = m.group(2).replace("${FIG_LANG}", lang).replace(".pdf", ".png")
            fig_pending = DRIVE_ROOT / path
            i += 1
            continue

        # the Turkish version captions its figures "Şekil n."; without this
        # the image is parsed, never captioned, and silently dropped
        m = re.match(r"^\*\*(?:Figure|Şekil) (S?\d+)\.\*\*\s*(.*)$", s)
        if m and fig_pending:
            cap = [m.group(2)]
            i += 1
            while i < len(lines) and lines[i].strip():
                cap.append(lines[i].strip())
                i += 1
            if fig_pending.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.keep_with_next = True
                p.add_run().add_picture(str(fig_pending), width=Inches(6.0))
            else:
                para(doc, f"[missing figure: {fig_pending.name}]")
            c = doc.add_paragraph()
            c.paragraph_format.space_after = Pt(12)
            c.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            label = "Figure" if lang == "en" else "Şekil"
            runs(c, f"**{label} " + m.group(1) + ".** " + " ".join(cap), size=9.5)
            fig_pending = None
            continue

        if s.startswith(("**Table", "**Tablo")):
            # a caption runs on until a blank line or the table itself; taking
            # only the first line dropped the rest into its own paragraph,
            # printed ahead of the caption and cut mid-sentence
            cap = [s]
            i += 1
            while (i < len(lines) and lines[i].strip()
                   and not lines[i].lstrip().startswith("|")):
                cap.append(lines[i].strip())
                i += 1
            table_caption = " ".join(cap)
            continue

        if s.startswith("$$"):                        # display equation
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("$$"):
                block.append(lines[i].strip())
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(10)
            r = p.add_run(math(" ".join(block)))
            r.italic = True
            continue

        if s.startswith("|"):                         # markdown table
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            tbl_n += 1
            add_table(doc, block, table_caption, tbl_n)
            table_caption = ""
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", s)          # reference list
        if m and int(m.group(1)) > 0 and any(
                h in _last_heading(doc) for h in ("References", "Kaynaklar")):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.first_line_indent = Inches(-0.35)
            p.paragraph_format.space_after = Pt(3)
            runs(p, m.group(1) + ". " + m.group(2), size=9.5)
            i += 1
            continue

        if s.startswith("- "):
            # a list item runs on across indented continuation lines; taking
            # only the first left the rest as a stray paragraph and, when the
            # bold markers straddled the break, printed the asterisks
            item = [s[2:]]
            i += 1
            while (i < len(lines) and lines[i].strip()
                   and lines[i].startswith(("  ", "	"))
                   and not lines[i].strip().startswith("- ")):
                item.append(lines[i].strip())
                i += 1
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(4)
            runs(p, " ".join(item))
            continue

        if not s:                                     # blank -> paragraph break
            i += 1
            continue

        if front:                                     # one line, one paragraph
            p = para(doc, s)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            i += 1
            continue

        block = []                                    # ordinary paragraph
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#|>|\||\$\$|!\[|- |\*\*Figure|\*\*\u015eekil|\*\*Table"
                r"|\*\*Tablo|---$)",
                lines[i].strip()):
            block.append(lines[i].strip())
            i += 1
        if not block:            # nothing matched and nothing consumed
            block = [s]
            i += 1
        para(doc, " ".join(block))

    return doc


_HEADS: list = []


def _last_heading(doc) -> str:
    for p in reversed(doc.paragraphs):
        if p.style.name.startswith("Heading"):
            return p.text
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=("en", "tr"))
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # --lang picked the figure language but not the text, so "--lang tr"
    # wrote the English manuscript under the Turkish name
    src = (Path(a.src) if a.src
           else DRAFT if a.lang == "en"
           else DRIVE_ROOT / f"PAPER_DRAFT_{a.lang.upper()}.md")
    md = io.open(src, encoding="utf-8").read()
    # only a document with a correspondence line has an author block to keep
    # unreflowed; the supplement's cover is ordinary prose
    head = md.split(chr(10) + "## ", 1)[0]
    authors = any(l.startswith("* ") and "orrespond" in l
                  for l in head.split(chr(10)))
    doc = convert(md, a.lang, authors=authors)
    cp = doc.core_properties
    cp.author = "Aykut Kaşkaya; Baha Şen"
    cp.comments = ""
    cp.title = "Supplementary Materials"
    out = Path(a.out) if a.out else DRIVE_ROOT / f"manuscript_{a.lang}.docx"
    doc.save(out)
    print(f"{out.name}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
