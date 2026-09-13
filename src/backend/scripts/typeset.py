"""Typographic conventions the journal sets and Markdown cannot carry.

The source is plain text, so it writes d_free for a subscript, a hyphen for a
minus sign, 6.0e-07 for a power of ten and ' for an apostrophe. Word can set
all four properly, and a referee reading the built document sees the difference.
Applied at build time, so the source stays diffable and the manuscript text is
never rewritten.
"""
from __future__ import annotations

import re

SUB = re.compile(r"\b([A-Za-z](?:_\{)?)_?([a-z]{1,4})\b")
# names that really are subscripted in this manuscript
NAMES = {"d_orc", "d_pipe", "d_free", "d_box", "A_z", "B_z", "L_ov"}
SCI = re.compile(r"\b(\d(?:\.\d+)?)e-0*(\d+)\b")
SUPS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
NB = " "   # non-breaking space


def scientific(s: str) -> str:
    """6.0e-07 is computer output; a manuscript sets 6.0 × 10⁻⁷.

    The spaces around the multiplication sign are non-breaking: a mantissa at
    the end of one line and its power at the start of the next is not a number
    any more.
    """
    return SCI.sub(
        lambda m: f"{m.group(1)}{NB}×{NB}10⁻{m.group(2).translate(SUPS)}",
        s)


def units(s: str) -> str:
    """A value and its unit stay together."""
    return re.sub(r"(?<=[0-9]) (?=(?:mm|MiB|GB|ms|s)(?![A-Za-z]))", NB, s)


def cite_dashes(s: str) -> str:
    """A citation range takes an en dash, as the rest of the numbers do."""
    return re.sub(r"(?<=\[)(\d+)-(\d+)(?=[,\]])", r"\1–\2", s)


def quotes(s: str) -> str:
    """Typographic quotes and apostrophes, left alone inside code and URLs."""
    if "http" in s:
        return s
    s = re.sub(r'"([^"]*)"', "“\\1”", s)
    return s.replace("'", "’")


def minus(s: str) -> str:
    """A minus sign between terms is U+2212, not a hyphen."""
    return re.sub(r"(?<=[\w\)\}\]]) - (?=[\(\{\w])", " − ", s)


def inline(s: str) -> str:
    """Everything that applies to a run of ordinary body text."""
    return units(quotes(cite_dashes(scientific(s))))


# the named quantities, plus the parenthesised form math() emits for a sum's
# index, which Word sets as a subscript rather than as literal parentheses
SUBSCRIPT_RE = re.compile(
    "|".join(re.escape(n) for n in sorted(NAMES, key=len, reverse=True))
    + r"|(?<=[A-ZΣ])_\([^)]{1,12}\)")


def subscript_runs(paragraph, text: str, italic: bool = False,
                   bold: bool = False) -> None:
    """Write `text` into `paragraph`, setting X_y as a real subscript.

    python-docx has no markup for this, so the string is split at each name
    and the subscripted part written as its own run with the attribute set.
    """
    def plain(t):
        if not t:
            return
        r = paragraph.add_run(t)
        # see to_docx.runs: False overrides the style, None inherits it
        r.italic = True if italic else None
        r.bold = True if bold else None

    pos = 0
    for m in SUBSCRIPT_RE.finditer(text):
        plain(text[pos:m.start()])
        tok = m.group(0)
        if tok.startswith("_("):          # a bare subscript, stem already out
            r = paragraph.add_run(tok[2:-1].replace(" ", ""))
            r.font.subscript = True
            r.italic = True
            r.bold = bold
            pos = m.end()
            continue
        stem, _, sub = tok.partition("_")
        r = paragraph.add_run(stem)
        r.italic = True
        r.bold = bold
        r = paragraph.add_run(sub)
        r.font.subscript = True
        r.italic = True
        r.bold = bold
        pos = m.end()
    plain(text[pos:])
