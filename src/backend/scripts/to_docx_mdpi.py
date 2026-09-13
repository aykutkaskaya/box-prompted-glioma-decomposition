"""Render the manuscript into the MDPI (Diagnostics) template.

The plain build in to_docx.py formats everything itself. This one opens
template.docx, empties its body and rewrites it using the template's own named
styles (MDPI12title, MDPI21heading1, MDPI41tablecaption and so on), so the
result carries the journal's formatting rather than an imitation of it.

The section structure is left alone. MDPI's template lists the sections that
*can* be used rather than requiring them, and renumbering would move all 59
in-text cross-references, so Related Work and Limitations keep their own
numbers. What changes is presentation: heading numbers take MDPI's trailing
period, the back matter is emitted in MDPI's order under its own style, an
abbreviations block is added, and the reference list is converted from the
Vancouver form the draft uses to MDPI's.

    python scripts/to_docx_mdpi.py
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_TAB_ALIGNMENT
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import to_docx as base  # noqa: E402

DRAFT = DRIVE_ROOT / "PAPER_DRAFT.md"
TEMPLATE = DRIVE_ROOT / "template.docx"
# MDPI sets the journal name in ISO-4 abbreviated form. Crossref's own short
# title is publisher house style rather than ISO-4, so the map is explicit.
JOURNALS = {
    "IEEE Transactions on Medical Imaging": "IEEE Trans. Med. Imaging",
    "Scientific Data": "Sci. Data",
    "Scientific Reports": "Sci. Rep.",
    "Medical Image Analysis": "Med. Image Anal.",
    "Korean Journal of Radiology": "Korean J. Radiol.",
    "Radiology: Artificial Intelligence": "Radiol. Artif. Intell.",
    "Medical Physics": "Med. Phys.",
    "PLOS Medicine": "PLoS Med.",
    "Data in Brief": "Data Brief",
    "Neuro-Oncology Advances": "Neuro-Oncol. Adv.",
    "Computers in Biology and Medicine": "Comput. Biol. Med.",
    "Nature Communications": "Nat. Commun.",
    "Nature Methods": "Nat. Methods",
    "Patterns": "Patterns",
    "Molecular Psychiatry": "Mol. Psychiatry",
    "Psychological Review": "Psychol. Rev.",
    "Neural Computing and Applications": "Neural Comput. Appl.",
    "Journal of Clinical Oncology": "J. Clin. Oncol.",
}
import os  # noqa: E402
OUT = DRIVE_ROOT / os.environ.get("MDPI_OUT", "manuscript_en_mdpi.docx")

# the back matter, in the order and wording MDPI expects
# MDPI's order for the back matter. A section not named here is dropped
# silently, which is how an Acknowledgments block went missing.
BACK_ORDER = ["Supplementary Materials", "Author Contributions", "Funding",
              "Institutional Review Board Statement",
              "Informed Consent Statement", "Data Availability Statement",
              "Acknowledgments", "Conflicts of Interest"]

ABBREVIATIONS = [
    ("WT", "whole tumour"),
    ("TC", "tumour core"),
    ("ET", "enhancing tumour"),
    ("HD95", "95th percentile Hausdorff distance"),
    ("NSD", "normalised surface Dice"),
    ("IoU", "intersection over union"),
    ("LoRA", "low-rank adaptation"),
    ("MRI", "magnetic resonance imaging"),
    ("FLAIR", "fluid-attenuated inversion recovery"),
    ("BraTS", "Brain Tumour Segmentation (benchmark)"),
    ("SAM", "Segment Anything Model"),
    ("RT-DETR", "real-time detection transformer"),
    ("CI", "confidence interval"),
    ("RHUH-GBM", "Río Hortega University Hospital Glioblastoma dataset"),
    ("TCIA", "The Cancer Imaging Archive"),
    ("RANO", "Response Assessment in Neuro-Oncology"),
    ("IDH", "isocitrate dehydrogenase"),
]


def clear_body(doc: Document) -> None:
    """Drop the template's placeholder content, keeping its styles and layout."""
    body = doc.element.body
    for child in list(body):
        if child.tag.endswith("}sectPr"):
            continue
        body.remove(child)


def style_or_none(doc: Document, name: str):
    try:
        return doc.styles[name]
    except KeyError:
        return None


def para(doc, text, style=None, italic=False, size=None):
    p = doc.add_paragraph()
    st = style_or_none(doc, style) if style else None
    if st is not None:
        p.style = st
    base.runs(p, text, base_italic=italic, size=size)
    return p


def mdpi_heading_number(text: str) -> str:
    """`4.4 External validation` -> `4.4. External Validation`.

    MDPI numbers subsections with a trailing period and sets headings in title
    case. Section words that are part of a name keep their own capitalisation.
    """
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$", text)
    if not m:
        return title_case(text)
    num, rest = m.groups()
    return f"{num}. {title_case(rest)}"


MINOR = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "nor",
         "of", "on", "or", "the", "to", "up", "via", "with", "under"}


def title_case(t: str) -> str:
    """Headline capitalisation, leaving anything already capitalised alone.

    A word with an internal capital is an acronym or a name (BraTS, RT-DETR,
    HD95) and is left exactly as written.
    """
    out = []
    words = t.split()
    for i, w in enumerate(words):
        if any(c.isupper() for c in w[1:]) or not w[:1].isalpha():
            out.append(w)
        elif i and w.lower() in MINOR:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


EVENTS = DRIVE_ROOT / "reports" / "ref_events.json"
_EVENTS = None


def event_location(num: str):
    """The conference location, where the record carries one."""
    global _EVENTS
    if _EVENTS is None:
        try:
            _EVENTS = json.loads(io.open(EVENTS, encoding="utf-8").read())
        except (OSError, ValueError):
            _EVENTS = {}
    return (_EVENTS.get(num) or {}).get("location")


AUTHORS = DRIVE_ROOT / "reports" / "ref_authors.json"
_AUTHORS = None


def _initials(given: str) -> str:
    """"Bjoern H." -> "B.H."; "Y.-T." is kept as written."""
    out = []
    for part in (given or "").replace(".", " ").split():
        if "-" in part:
            out.append("-".join(x[0].upper() + "." for x in part.split("-") if x))
        else:
            out.append(part[0].upper() + ".")
    return "".join(out)


def full_authors(num: str) -> str | None:
    """The reference's author list, all of them up to ten, then et al."""
    global _AUTHORS
    if _AUTHORS is None:
        try:
            _AUTHORS = json.loads(io.open(AUTHORS, encoding="utf-8").read())
        except (OSError, ValueError):
            _AUTHORS = {}
    names = _AUTHORS.get(num)
    if not names:
        return None
    out = [f"{fam}, {_initials(giv)}" if giv else str(fam) for fam, giv in names[:10]]
    return "; ".join(out) + ("; et al." if len(names) > 10 else "")


def _fix_author(a: str) -> str:
    a = a.strip()
    if a in ("et al", "et al."):
        return "et al."
    # a trailing generational suffix belongs after the initials
    suffix = ""
    sm = re.match(r"^(.*?)\s+(Jr|Sr|II|III|IV)$", a)
    if sm:
        a, suffix = sm.group(1), ", " + sm.group(2) + "."
    # surname may be several words and carry diacritics; initials may be
    # hyphenated, as in "Hu Y-T"
    am = re.match(r"^(.+?)\s+([A-Z](?:[–—-][A-Z])?[A-Z]*)$", a)
    if am:
        surname, initials = am.groups()
        parts = re.split(r"[–—-]", initials)
        init = ("-".join(x + "." for x in parts) if len(parts) > 1
                else ".".join(initials) + ".")
        return surname + ", " + init + suffix
    return a + suffix


def convert_reference(entry: str) -> str:
    """Vancouver -> MDPI for the three shapes this bibliography uses.

    MDPI abbreviates the venue and sets it in italic, the year in bold and the
    volume in italic, and keeps the DOI as a link. The draft stays in Vancouver
    form because that is what check_refs.py verifies against Crossref; the
    conversion happens here, on the way into the document.

        Menze BH, Jakab A, et al. Title. *IEEE Transactions on Medical
        Imaging*, 34(10):1993-2024, 2015. DOI: 10.1109/TMI.2014.2377694.

    becomes

        Menze, B.H.; Jakab, A.; et al. Title. *IEEE Trans. Med. Imaging*
        **2015**, *34*, 1993-2024. https://doi.org/10.1109/TMI.2014.2377694

    Anything that does not match is returned unchanged rather than mangled; the
    caller reports how many were left alone.
    """
    m = re.match(r"^(?P<num>\d+)\.\s+(?P<auth>.+?)\.\s+(?P<rest>.+)$", entry.strip())
    if not m:
        return entry
    num, rest = m.group("num"), m.group("rest")
    # the stored list where there is one, the draft's own three otherwise
    authors = full_authors(num) or "; ".join(
        _fix_author(a) for a in m.group("auth").split(", "))

    # the apparatus the draft carries after the venue
    doi = re.search(r"DOI:\s*(10\.\S+?)\.?(?:\s|$)", rest)
    arx = re.search(r"arXiv:(\d{4}\.\d{4,5})", rest)
    url = re.search(r"URL:\s*(\S+?)\.?$", rest)
    rest = re.split(r"\s+(?:DOI|arXiv|URL):", rest)[0].strip()

    vm = re.match(r"^(?P<title>.+?)\.\s+\*(?P<venue>[^*]+)\*(?P<tail>.*)$", rest)
    if not vm:
        return num + ". " + authors + " " + rest
    title = vm.group("title").strip().rstrip(".")
    venue, tail = vm.group("venue").strip(), vm.group("tail")
    ym = re.search(r"((?:19|20)\d\d)(?!.*(?:19|20)\d\d)", tail)
    year = ym.group(1) if ym else ""

    # --- a preprint: the archive, the year and the identifier
    if arx and venue.lower().startswith("arxiv"):
        return (num + ". " + authors + " " + title + ". *arXiv* **" + year
                + "**, arXiv:" + arx.group(1) + ".")

    # --- a journal article: abbreviated venue, bold year, italic volume
    PROC = ("proceedings", "advances in neural", "medical image computing",
            "machine learning in", "bildverarbeitung", "international conference")
    pm = re.search(r",\s*(?P<vol>\d+)(?:\((?P<iss>[^)]*)\))?:(?P<pages>[^,]+),", tail)
    if pm and not venue.lower().startswith(PROC):
        jour = JOURNALS.get(venue, venue)
        out = (num + ". " + authors + " " + title + ". *" + jour + "* **" + year
               + "**, *" + pm.group("vol") + "*, " + pm.group("pages") + ".")
        if doi:
            out += " https://doi.org/" + doi.group(1)
        return out

    # --- a conference paper: MDPI puts it in the proceedings, pages last
    pp = re.search(r"pp\.\s*([\d–—-]+)", tail)
    lead = venue if venue.lower().startswith("proceedings") else "Proceedings of " + venue
    out = num + ". " + authors + " " + title + ". In " + lead
    where = event_location(num)
    if where:
        out += ", " + where
    if year:
        out += ", " + year
    if pp:
        out += "; pp. " + pp.group(1)
    out += "."
    if doi:
        out += " https://doi.org/" + doi.group(1)
    elif arx:
        out += " arXiv:" + arx.group(1) + "."
    elif url:
        out += (" Available online: " + url.group(1)
                + " (accessed on 12 September 2026).")
    return out


def load_authors() -> dict:
    """Front matter from an untracked authors.json, or placeholders."""
    p = DRIVE_ROOT / "authors.json"
    if p.exists():
        return json.loads(io.open(p, encoding="utf-8").read())
    return {
        "authors": [
            {"name": "First Author", "mark": "1,*",
             "affiliation": "Affiliation 1",
             "email": "first.author@example.org",
             "orcid": "0000-0000-0000-0000"},
            {"name": "Second Author", "mark": "2",
             "affiliation": "Affiliation 2",
             "email": "second.author@example.org",
             "orcid": "0000-0000-0000-0000"},
        ],
        "correspondence": "first.author@example.org",
    }


def main() -> None:
    md = io.open(DRAFT, encoding="utf-8").read()
    doc = Document(str(TEMPLATE))
    clear_body(doc)

    title = next(l[2:].strip() for l in md.split("\n") if l.startswith("# "))
    para(doc, "Article", "MDPI11articletype")
    para(doc, title, "MDPI12title")
    # The repository carries no personal address; a local, untracked
    # authors.json supplies the front matter, and a fresh checkout prints
    # placeholders, which is visible rather than silent.
    people = load_authors()
    authorline = para(doc, "", "MDPI13authornames")
    for k, a in enumerate(people["authors"]):
        if k:
            authorline.add_run(" and ")
        authorline.add_run(a["name"] + " ")
        authorline.add_run(a["mark"]).font.superscript = True
    for k, a in enumerate(people["authors"], 1):
        para(doc, str(k) + chr(9) + a['affiliation'] + '; ' + a['email']
             + '; ORCID ' + a['orcid'], "MDPI16affiliation")
    para(doc, '*' + chr(9) + 'Correspondence: ' + people['correspondence'],
         "MDPI16affiliation")

    abstract = md.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0]
    abstract = " ".join(" ".join(abstract.split("\n")).split())
    para(doc, "**Abstract:** " + abstract, "MDPI17abstract")
    kw = md.split("**Keywords:**", 1)[1].split("\n", 1)[0].strip()
    para(doc, "**Keywords:** " + kw, "MDPI18keywords")

    body = md.split("**Keywords:**", 1)[1].split("\n", 1)[1]
    body = body.split("## References", 1)[0]
    # render_back_matter emits these under MDPI's own style; without this they
    # would appear a second time as ordinary headings
    body = body.split(f"## {BACK_ORDER[0]}", 1)[0]
    refs_block = md.split("## References", 1)[1]
    appendix = ""
    if "## Appendix A" in refs_block:
        refs_block, appendix = refs_block.split("## Appendix A", 1)

    stats = render_body(doc, body)
    render_back_matter(doc, md)
    render_abbreviations(doc)
    if appendix:
        para(doc, "Appendix A", "MDPI21heading1")
        render_body(doc, appendix.split("\n", 1)[1], stats)
    render_references(doc, refs_block)
    stats["ref_unconverted"] = render_references.unconverted

    # the template arrives with its own placeholder title, and Word shows
    # these in the file's properties and in the exported PDF
    cp = doc.core_properties
    cp.title = title
    cp.author = "; ".join(a["name"] for a in people["authors"])
    cp.comments = ""
    cp.category = ""
    cp.subject = ""
    doc.save(OUT)
    print(f"{OUT.name}  ({OUT.stat().st_size // 1024} KB)")
    print(f"  {stats['tables']} tables, {stats['figures']} figures, "
          f"{stats['headings']} headings, {stats['equations']} equations")
    if stats["ref_unconverted"]:
        print(f"  {stats['ref_unconverted']} references left in their original "
              f"form (shape not recognised)")


def render_body(doc, body: str, stats=None) -> dict:
    sec = doc.sections[0]
    if stats is None:
        stats = {"tables": 0, "figures": 0, "headings": 0, "ref_unconverted": 0,
                 "equations": 0}
    lines = body.split("\n")
    i = 0
    fig_pending = None
    while i < len(lines):
        s = lines[i].rstrip()
        if not s.strip():
            i += 1
            continue

        # A Markdown horizontal rule separates sections in the source and has
        # no place in the built document. Six of them were being emitted as
        # the literal characters, one of them directly above "3. Results".
        if set(s.strip()) in ({"-"}, {"*"}, {"_"}) and len(s.strip()) >= 3:
            i += 1
            continue

        if s.strip() == "$$":
            i += 1
            eq = []
            while i < len(lines) and lines[i].strip() != "$$":
                eq.append(lines[i].strip())
                i += 1
            i += 1
            stats["equations"] += 1
            p = doc.add_paragraph()
            st = style_or_none(doc, "MDPI39equation")
            if st is not None:
                p.style = st
            import typeset
            txt = typeset.minus(base.math(" ".join(eq)))
            # LaTeX's \, before the closing period leaves " ." and the equation
            # number then collides with it when the tab stop is already past
            # the end of the line
            txt = re.sub(r"\s+\.\s*$", ".", txt)
            # without stops of its own the formula began wherever the style
            # left it and the number sat against the formula's last character,
            # so the three equations lined up three different ways
            usable = (sec.page_width - sec.left_margin - sec.right_margin)
            p.paragraph_format.tab_stops.add_tab_stop(
                int(usable / 2), WD_TAB_ALIGNMENT.CENTER)
            p.paragraph_format.tab_stops.add_tab_stop(
                int(usable), WD_TAB_ALIGNMENT.RIGHT)
            p.add_run("\t")
            typeset.subscript_runs(p, txt)
            # MDPI numbers display equations; the number is a tabbed run in the
            # same paragraph so it right-aligns against the style's tab stop
            # A long equation leaves the tab stop behind, and the number
            # then prints against the closing period. Two spaces keep
            # them apart whether or not the tab has room.
            p.add_run("\t(" + str(stats["equations"]) + ")")
            continue

        m = re.match(r"^(#{2,4})\s+(.*)$", s)
        if m:
            level = len(m.group(1))
            style = {2: "MDPI21heading1", 3: "MDPI22heading2",
                     4: "MDPI23heading3"}.get(level, "MDPI23heading3")
            h = para(doc, mdpi_heading_number(m.group(2).strip()), style)
            # a heading left alone at the foot of a page sends the reader over
            # the break for its own first sentence
            h.paragraph_format.keep_with_next = True
            stats["headings"] += 1
            i += 1
            continue

        m = re.match(r"^!\[(?:Figure|Şekil) (\d+)\]\((.+?)\)$", s)
        if m:
            path = m.group(2).replace("${FIG_LANG}", "en").replace(".pdf", ".png")
            fig_pending = DRIVE_ROOT / path
            i += 1
            continue

        m = re.match(r"^\*\*Figure (\d+)\.\*\*\s*(.*)$", s)
        if m and fig_pending:
            cap = [m.group(2)]
            i += 1
            while i < len(lines) and lines[i].strip():
                cap.append(lines[i].strip())
                i += 1
            if not fig_pending.exists():
                # Silence here put a caption with no figure into the file that
                # goes to the journal, and nothing in the source could show it.
                raise SystemExit(
                    f"figure file missing: {fig_pending}; the caption "
                    f"for Figure {m.group(1)} has no figure, and a "
                    f"document is not built with one missing")
            p = doc.add_paragraph()
            st = style_or_none(doc, "MDPI52figure")
            if st is not None:
                p.style = st
            p.paragraph_format.keep_with_next = True
            # the style centres on the full page width, which puts a figure
            # 1.9 cm left of the text column it fits inside
            p.paragraph_format.left_indent = Cm(4.6)
            p.add_run().add_picture(str(fig_pending), width=Pt(370))
            c = doc.add_paragraph()
            st = style_or_none(doc, "MDPI51figurecaption")
            if st is not None:
                c.style = st
            base.runs(c, f"**Figure {m.group(1)}.** " + " ".join(cap))
            stats["figures"] += 1
            fig_pending = None
            continue

        if s.startswith("**Table"):
            m = re.match(r"^\*\*Table ([\dA]\d*)\.\*\*\s*(.*)$", s)
            cap = [m.group(2)] if m else [s]
            i += 1
            while i < len(lines) and lines[i].strip():
                cap.append(lines[i].strip())
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            c = doc.add_paragraph()
            # The 4.6 cm indent this style carries is the template's own: the
            # figure caption style has it too, and the template applies it to
            # its full-width table as well. The one-line caption style that
            # avoids the indent is centred, which set a 187-word caption
            # centred line by line.
            st = style_or_none(doc, "MDPI41tablecaption")
            if st is not None:
                c.style = st
            base.runs(c, f"**Table {m.group(1) if m else ''}.** " + " ".join(cap))
            # A caption that falls at the foot of a page while its table starts
            # the next one makes the reader turn back; two of them did.
            c.paragraph_format.keep_with_next = True
            c.paragraph_format.keep_together = True
            if block:
                add_mdpi_table(doc, block)
                _no_hyphenation(doc.tables[-1])
                _repeat_header(doc.tables[-1])
                stats["tables"] += 1
            continue

        bullet = s.startswith(("- ", "* "))
        numbered = bool(re.match(r"^\d+\.\s", s))
        if bullet or numbered:
            # MDPI37itemize numbers the list itself; keeping "1. " in the
            # text printed each contribution as "1.  1. An accounting...".
            item = [s[2:].strip() if bullet
                    else re.sub(r"^\d+\.\s+", "", s.strip())]
            i += 1
            # continuation lines are indented and start no new block
            while (i < len(lines) and lines[i].strip()
                   and lines[i][:1] in " \t"
                   and not re.match(r"^\s*[-*]\s", lines[i])
                   and not re.match(r"^\s*\d+\.\s", lines[i])):
                item.append(lines[i].strip())
                i += 1
            para(doc, " ".join(item),
                 "MDPI38bullet" if bullet else "MDPI37itemize")
            continue

        chunk = [s]
        i += 1
        while (i < len(lines) and lines[i].strip()
               and not lines[i].startswith(
                   ("#", "|", "!", "- ", "* ", "**Table", "**Figure"))
               and not re.match(r"^\d+\.\s", lines[i])):
            chunk.append(lines[i].strip())
            i += 1
        para(doc, " ".join(chunk), "MDPI31text")
    return stats


def _border(kind: str, size: int):
    e = OxmlElement("w:" + kind)
    e.set(qn("w:val"), "single")
    e.set(qn("w:sz"), str(size))
    e.set(qn("w:space"), "0")
    e.set(qn("w:color"), "auto")
    return e


def three_line(t) -> None:
    """Rules above and below the table, and one under the header row."""
    pr = t._tbl.tblPr
    for tag in ("w:tblBorders", "w:tblCellMar"):
        old = pr.find(qn(tag))
        if old is not None:
            pr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "bottom"):
        borders.append(_border(side, 8))
    pr.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        m = OxmlElement("w:" + side)
        m.set(qn("w:w"), "0")
        m.set(qn("w:type"), "dxa")
        margins.append(m)
    pr.append(margins)
    for cell in t.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        old = tcPr.find(qn("w:tcBorders"))
        if old is not None:
            tcPr.remove(old)
        tb = OxmlElement("w:tcBorders")
        tb.append(_border("bottom", 4))
        tcPr.append(tb)


def add_mdpi_table(doc, block) -> None:
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block]
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    if not rows:
        return
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    # no style: MDPI's own tables carry none and set their rules directly
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row[:len(rows[0])]):
            p = t.cell(ri, ci).paragraphs[0]
            s2 = style_or_none(doc, "MDPI42tablebody")
            if s2 is not None:
                p.style = s2
            base.runs(p, cell)
            if ri == 0:
                for r in p.runs:
                    r.bold = True
    three_line(t)


def render_back_matter(doc, md: str) -> None:
    for head in BACK_ORDER:
        if f"## {head}" not in md:
            continue
        seg = md.split(f"## {head}", 1)[1].split("\n## ", 1)[0]
        seg = seg.split("\n---", 1)[0]
        text = " ".join(" ".join(seg.split("\n")).split())
        # the style is not bold; the template bolds the label run itself
        para(doc, f"**{head}:** {text}", "MDPI62backmatter")


def render_abbreviations(doc) -> None:
    para(doc, "Abbreviations", "MDPI21heading1")
    para(doc, "The following abbreviations are used in this manuscript:",
         "MDPI31text")
    # this was the only table built without a header row, and so the only
    # one Word was free to split: it broke across two pages and the second
    # opened with seven bare abbreviations and nothing naming the columns
    head = ("Abbreviation", "Definition")
    t = doc.add_table(rows=len(ABBREVIATIONS) + 1, cols=2)
    for r, (short, long) in enumerate([head] + ABBREVIATIONS):
        for c, v in enumerate((short, long)):
            p = t.cell(r, c).paragraphs[0]
            s2 = style_or_none(doc, "MDPI42tablebody")
            if s2 is not None:
                p.style = s2
            base.runs(p, f"**{v}**" if r == 0 else v)
    three_line(t)
    _repeat_header(t)


def render_references(doc, block: str) -> None:
    para(doc, "References", "MDPI21heading1")
    render_references.unconverted = 0
    for line in block.split("\n"):
        line = line.strip()
        if not re.match(r"^\d+\.\s", line):
            continue
        # convert_reference anchors on the number, so it has to see it. The
        # MDPI reference style then numbers the list itself, so the number
        # comes off afterwards; stripping it first silently disabled the
        # whole conversion.
        entry = re.sub(r"^\d+\.\s+", "", convert_reference(line))
        para(doc, entry, "MDPI81references")
        if entry == re.sub(r"^\d+\.\s+", "", line):
            render_references.unconverted += 1


def _repeat_header(table) -> None:
    """Mark the first row as a header so Word repeats it after a page break.

    Table 1 split across two pages and the second page opened with two data
    rows and no column names: seven numbers with nothing to say what they
    were. A table that cannot fit one page has to carry its header over.
    """
    from docx.oxml.ns import qn as _qn
    from docx.oxml import OxmlElement as _El
    tr = table.rows[0]._tr
    trPr = tr.get_or_add_trPr()
    for tag in ("tblHeader", "cantSplit"):
        el = _El("w:" + tag)
        el.set(_qn("w:val"), "true")
        trPr.append(el)
    # and no row of its own may break in half
    for row in table.rows:
        p = row._tr.get_or_add_trPr()
        p.append(_El("w:cantSplit"))


def _no_hyphenation(table) -> None:
    """Word hyphenates inside narrow cells, so "cohort detection F1" printed as
    "cohort detec-tion F1". A table cell is too narrow for hyphenation to help."""
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.widow_control = True
                pPr = p._p.get_or_add_pPr()
                el = pPr.makeelement(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                    "suppressAutoHyphens", {})
                pPr.append(el)


if __name__ == "__main__":
    main()
