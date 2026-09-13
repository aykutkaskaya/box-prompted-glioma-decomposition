"""Assemble the manuscript folder and zip from the sources.

Not tied to a journal. main.tex is a plain article-class document, so it
compiles on Overleaf, in TeX Live, in tectonic -- anywhere. When the journal is
settled, the body drops into that publisher's template unchanged; only the
front and back matter move.

Everything here is derived, so the folder can be deleted and rebuilt.

    python scripts/to_latex.py
    python scripts/build_submission.py
"""
from __future__ import annotations

import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

DRAFT = DRIVE_ROOT / "PAPER_DRAFT.md"
OUT = DRIVE_ROOT / "submission"
FIGS = DRIVE_ROOT / "reports" / "figures"
BIB = DRIVE_ROOT / "reports" / "references.bib"
MAIN_TPL = DRIVE_ROOT / "reports" / "main_template.tex"


def abstract_text(md: str) -> str:
    body = md[md.index("## Abstract") + len("## Abstract"):]
    body = body[: body.index("\n## ")]
    paras = [" ".join(p.split()) for p in body.strip().split("\n\n") if p.strip()]
    text = " ".join(paras)
    # the abstract runs as one paragraph; markdown emphasis has to go
    text = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", text)
    for a, b in (("\u2014", "---"), ("\u2212", "$-$"), ("\u2019", "'")):
        text = text.replace(a, b)
    return text


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "Figures").mkdir(exist_ok=True)
    # stale Definitions/ from the MDPI-specific builds
    shutil.rmtree(OUT / "Definitions", ignore_errors=True)

    for pdf in sorted(FIGS.glob("*_en.pdf")):
        shutil.copy2(pdf, OUT / "Figures" / pdf.name)
    shutil.copy2(BIB, OUT / "references.bib")

    # the body was written with drive-relative figure paths
    body_path = DRIVE_ROOT / "reports" / "body_en.tex"
    body = io.open(body_path, encoding="utf-8").read()
    body = body.replace("reports/figures/", "Figures/")
    io.open(body_path, "w", encoding="utf-8", newline="\n").write(body)

    md = io.open(DRAFT, encoding="utf-8").read()
    tex = io.open(MAIN_TPL, encoding="utf-8").read()
    # the title lives in the draft, not in the template: they drifted apart
    # once and the bundle went out under a title the manuscript had dropped
    title = next(l[2:].strip() for l in md.split("\n") if l.startswith("# "))
    for a, b in (("—", "---"), ("−", "$-$"), ("’", "'")):
        title = title.replace(a, b)
    tex = tex.replace("@TITLE@", title)
    tex = tex.replace("@ABSTRACT@", abstract_text(md))
    # one file, not a wrapper plus an include: the author edits this, and a
    # second file only makes sense while the body is untouched
    tex = tex.replace("@BODY@", body.rstrip())
    io.open(OUT / "main.tex", "w", encoding="utf-8", newline="\n").write(tex)
    io.open(OUT / "README.md", "w", encoding="utf-8", newline="\n").write(README)

    zip_path = DRIVE_ROOT / "manuscript.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT).as_posix())

    figs = len(list((OUT / "Figures").glob("*.pdf")))
    print(f"submission/: main.tex, references.bib, {figs} figures, README.md")
    print(f"-> {zip_path.name}")


README = r"""# Manuscript

Plain article class, no publisher template. Upload the zip to Overleaf
(New Project -> Upload Project) or run `pdflatex main; bibtex main; pdflatex
main; pdflatex main` locally.

    main.tex          the whole manuscript
    references.bib    71 entries
    Figures/          7 figures

Open items are tagged AYKUT in square brackets so they show up in the PDF:
the table captions, the back matter, and two ORCIDs.

When the journal is picked, the body sections paste into their template as-is.
Title, authors, abstract, keywords and back matter are the only parts that have
to be re-done in the publisher's own macros.

main.tex is generated from PAPER_DRAFT.md. Once you start editing it in
Overleaf, keep editing it there -- regenerating overwrites it:

    python scripts/to_latex.py
    python scripts/build_submission.py
"""


if __name__ == "__main__":
    main()
