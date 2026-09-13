"""Run every maths span in the manuscript through the converter and flag leftovers."""
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
import to_docx as td  # noqa: E402

ENC = sys.stdout.encoding or "utf-8"


def show(s):
    print(s.encode(ENC, "replace").decode(ENC))


src = io.open(DRIVE_ROOT / "PAPER_DRAFT.md", encoding="utf-8").read()
body = src.split("## References")[0]
lines = body.split("\n")

spans = set(re.findall(r"\$([^$\n]+)\$", body))

# display blocks: the fenced lines between a lone `$$` and the next lone `$$`
inside, buf = False, []
for l in lines:
    if l.strip() == "$$":
        if inside:
            spans.add(" ".join(" ".join(buf).split()))
            buf = []
        inside = not inside
    elif inside:
        buf.append(l)

bad = []
for s in sorted(spans):
    out = td.math(s)
    leftover = sorted(set(re.findall(r"\\[a-zA-Z]+", out)))
    # a command whose backslash was eaten but whose name survived verbatim
    naked = sorted({w for w in re.findall(r"[a-zA-Z]{2,}", out)
                    if w in {"geq", "leq", "neq", "approx", "times", "cdot",
                             "subset", "cup", "cap", "frac", "sqrt", "mathcal",
                             "mathrm", "text", "bigl", "bigr", "underbrace",
                             "qquad", "quad", "partial", "infty", "pm", "ge",
                             "le", "alpha", "beta", "lambda", "sum", "prod"}})
    if leftover or naked:
        bad.append((s, out, leftover + naked))

show(f"{len(spans)} maths spans, {len(bad)} with unconverted commands\n")
for s, out, lo in bad:
    show(f"  source  : {s[:100]}")
    show(f"  output  : {out[:100]}")
    show(f"  leftover: {lo}\n")
