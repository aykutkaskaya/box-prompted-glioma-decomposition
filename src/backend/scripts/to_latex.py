"""Convert the manuscript from Markdown to a LaTeX body.

Doing this by hand once would be quicker than writing the converter; doing it
after every edit would not, and the numbers in the text are regenerated often
enough that a hand-converted copy would drift within a day.

Plain article class, no publisher template -- the journal is not settled and
the body is the part that does not change when it is. Back matter is emitted
as visible TODO markers, because a funding or ethics statement written by
guesswork is worse than an obvious blank.

    python scripts/to_latex.py               # English body
    python scripts/to_latex.py --lang tr     # Turkish body, when it exists
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

SRC = {"en": DRIVE_ROOT / "PAPER_DRAFT.md",
       "tr": DRIVE_ROOT / "PAPER_DRAFT_TR.md"}
KEYMAP = DRIVE_ROOT / "reports" / "citation_number_to_key.json"
OUT_DIR = DRIVE_ROOT / "reports"

# LaTeX-special characters that appear in ordinary prose. Backslash is handled
# separately because escaping it after the others would double-escape them.
SPECIALS = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}", "$": r"\$"}

# Characters pdfLaTeX cannot set unaided. Explicit rather than clever: a
# silent pass-through becomes a missing glyph in a file nobody rereads
# before submitting.
UNICODE_MAP = {
    '—': '---',
    '–': '--',
    '−': '$-$',
    '×': '$\\times$',
    '≥': '$\\geq$',
    '≤': '$\\leq$',
    '→': '$\\rightarrow$',
    '⊂': '$\\subset$',
    '∪': '$\\cup$',
    '∩': '$\\cap$',
    'α': '$\\alpha$',
    'β': '$\\beta$',
    '±': '$\\pm$',
    '≈': '$\\approx$',
    '≠': '$\\neq$',
    '·': '$\\cdot$',
    '§': '\\S ',
    '…': '\\ldots{}',
    'í': "\\'i",
    '“': '``',
    '”': "''",
    '’': "'",
    '‘': '`',
}


def protect(text: str) -> tuple:
    """Pull out spans that must pass through untouched, leaving placeholders."""
    stash: list[str] = []

    def keep(m):
        stash.append(m.group(0))
        return f"\x00{len(stash) - 1}\x00"

    # display maths, inline maths, then inline code -- longest first
    text = re.sub(r"\$\$.*?\$\$", keep, text, flags=re.S)
    text = re.sub(r"\$[^$\n]+\$", keep, text)
    text = re.sub(r"`[^`\n]+`", keep, text)
    return text, stash


def restore(text: str, stash: list) -> str:
    def put(m):
        raw = stash[int(m.group(1))]
        if raw.startswith("`"):
            inner = raw.strip("`")
            for ch, esc in SPECIALS.items():
                inner = inner.replace(ch, esc)
            # verbatim-ish text still has to be settable: these are the
            # characters that actually reach \texttt from the manuscript
            inner = (inner.replace("−", "-")
                          .replace("…", r"\\ldots{}")
                          .replace("—", "---"))
            return r"\texttt{" + inner + "}"
        return raw
    return re.sub(r"\x00(\d+)\x00", put, text)


def inline(text: str, keys: dict) -> str:
    """Markdown inline markup to LaTeX, maths and code left alone."""
    text, stash = protect(text)

    for ch, esc in SPECIALS.items():
        text = text.replace(ch, esc)

    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text, flags=re.S)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\\emph{\1}", text)

    # [12] / [3,5] / [8-11] -> \cite{a,b,c}
    def cite(m):
        out = []
        for part in m.group(1).split(","):
            part = part.strip()
            if "-" in part:
                a, b = (int(x) for x in part.split("-"))
                out += [keys[str(n)] for n in range(a, b + 1)]
            else:
                out.append(keys[part])
        return r"\cite{" + ",".join(out) + "}"

    text = re.sub(r"\[(\d+(?:\s*[,\-]\s*\d+)*)\]", cite, text)

    for src, dst in UNICODE_MAP.items():
        text = text.replace(src, dst)
    return restore(text, stash)


def table(block: list, keys: dict, caption: str) -> str:
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block]
    header, body = rows[0], rows[2:]          # rows[1] is the --- separator
    n = len(header)
    # L and C are the X-type columns declared in the preamble. A tabularx
    # with no X column has no way to reach its target width and spins on it,
    # which is what the first Overleaf run hit as a compile timeout.
    spec = "L" + "C" * (n - 1) if n > 1 else "L"
    out = [r"\begin{table}[H]",
           r"\caption{" + caption + "}",
           r"\centering",
           r"\begin{tabularx}{\textwidth}{" + spec + "}",
           r"\toprule",
           " & ".join(r"\textbf{" + inline(c, keys) + "}" for c in header) + r" \\",
           r"\midrule"]
    for r in body:
        r = (r + [""] * n)[:n]
        out.append(" & ".join(inline(c, keys) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabularx}",
            r"\end{table}"]
    return "\n".join(out)


def convert(md: str, keys: dict, lang: str) -> str:
    md = md[: md.index("## References")]           # bibliography comes from .bib
    md = md[md.index("\n## 1. "):]                 # front matter is in the template

    out: list[str] = []
    lines = md.split("\n")
    i = 0
    fig_pending = None
    section = ""

    while i < len(lines):
        line = lines[i]

        if line.startswith("> "):                  # drafting note
            i += 1
            continue

        m = re.match(r"^## (?:\d+\.\s*)?(.+)$", line)
        if m:
            section = inline(m.group(1), keys)
            out += ["", r"\section{" + section + "}"]
            i += 1
            continue
        m = re.match(r"^### (?:[\d.]+\s*)?(.+)$", line)
        if m:
            out += ["", r"\subsection{" + inline(m.group(1), keys) + "}"]
            i += 1
            continue

        if line.strip() == "---":
            i += 1
            continue

        m = re.match(r"^!\[Figure (\d+)\]\((.+?)\)$", line.strip())
        if m:
            fig_pending = (m.group(1), m.group(2).replace("${FIG_LANG}", lang))
            i += 1
            continue

        m = re.match(r"^\*\*Figure (\d+)\.\*\*\s*(.*)$", line)
        if m and fig_pending:
            cap = [m.group(2)]
            i += 1
            while i < len(lines) and lines[i].strip():
                cap.append(lines[i])
                i += 1
            num, path = fig_pending
            out += ["", r"\begin{figure}[H]", r"\centering",
                    r"\includegraphics[width=\linewidth]{" + path + "}",
                    r"\caption{" + inline(" ".join(cap), keys) + "}",
                    r"\label{fig:" + num + "}", r"\end{figure}"]
            fig_pending = None
            continue

        if line.strip().startswith("$$"):
            block = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().endswith("$$"):
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
                i += 1
            body = "\n".join(block).strip().strip("$").strip()
            out += ["", r"\begin{equation}", body, r"\end{equation}"]
            continue

        if line.strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            if len(block) >= 2:
                lead = ""
                # walk back over the blank separators the emitter inserts
                for prev in reversed([x.strip() for x in out[-6:] if x.strip()][-3:]):
                    if prev.startswith(r"\textbf{") and len(prev) < 240:
                        lead = re.sub(r"\\textbf\{(.*?)\}",
                                      r"\1", prev).strip().rstrip(".:;,") + "."
                        break
                cap = lead or "[caption AYKUT]"
                out += ["", table(block, keys, cap)]
            continue

        if re.match(r"^[-*] ", line):
            out += ["", r"\begin{itemize}"]
            while i < len(lines) and (re.match(r"^[-*] ", lines[i])
                                      or lines[i].startswith("  ")):
                if re.match(r"^[-*] ", lines[i]):
                    item = [re.sub(r"^[-*] ", "", lines[i])]
                    i += 1
                    while i < len(lines) and lines[i].startswith("  "):
                        item.append(lines[i].strip())
                        i += 1
                    out.append(r"\item " + inline(" ".join(item), keys))
                else:
                    i += 1
            out.append(r"\end{itemize}")
            continue

        if re.match(r"^\d+\. ", line):
            out += ["", r"\begin{enumerate}"]
            while i < len(lines) and (re.match(r"^\d+\. ", lines[i])
                                      or lines[i].startswith("   ")):
                if re.match(r"^\d+\. ", lines[i]):
                    item = [re.sub(r"^\d+\. ", "", lines[i])]
                    i += 1
                    while i < len(lines) and lines[i].startswith("   "):
                        item.append(lines[i].strip())
                        i += 1
                    out.append(r"\item " + inline(" ".join(item), keys))
                else:
                    i += 1
            out.append(r"\end{enumerate}")
            continue

        if not line.strip():
            i += 1
            continue

        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{2,3} |\||[-*] |\d+\. |> |!\[|\$\$|---$)", lines[i]):
            para.append(lines[i])
            i += 1
        out += ["", inline(" ".join(para), keys)]

    return "\n".join(out).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("en", "tr"), default="en")
    a = ap.parse_args()

    src = SRC[a.lang]
    if not src.exists():
        raise SystemExit(f"{src.name} does not exist yet")
    keys = json.loads(KEYMAP.read_text(encoding="utf-8"))
    body = convert(io.open(src, encoding="utf-8").read(), keys, a.lang)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"body_{a.lang}.tex"
    io.open(out, "w", encoding="utf-8", newline="\n").write(body)
    n_cite = len(re.findall(r"\\cite\{", body))
    print(f"wrote {out.relative_to(DRIVE_ROOT)}  "
          f"({len(body.splitlines())} lines, {n_cite} \\cite calls, "
          f"{body.count(chr(92) + 'section')} sections, "
          f"{body.count(chr(92) + 'begin{table}')} tables, "
          f"{body.count(chr(92) + 'begin{figure}')} figures)")


if __name__ == "__main__":
    main()
