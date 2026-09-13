"""Check the claims a number-by-number audit cannot see.

audit_numbers.py verifies that every recomputed value appears in the text. That
missed four errors a referee found: a count taken from the wrong segmenter, a
spread taken from the wrong cohort, a table of medians described as if it were
the paper's usual mean, and the word "monotonically" applied to a series that
reverses twice. None of those is a missing number. Two were numbers that were
present and wrong; two were words.

So this checks the shape of claims rather than their digits: arithmetic that
must add up, words that assert a property the data can be asked about, and
counts that must agree with the file they came from.

    python scripts/check_claims.py
"""
from __future__ import annotations

import glob
import ast
import builtins
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import FIX, corrected  # noqa: E402

DRAFT = DRIVE_ROOT / "PAPER_DRAFT.md"
SUPP = DRIVE_ROOT / "SUPPLEMENTARY.md"
V = DRIVE_ROOT / "reports" / "validation"
SEG = "sam2.1_l"





def _fix(p):
    from pathlib import Path
    p = Path(p)
    return corrected(p)


def read(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                out[r["patient"]] = r
    return out


def main() -> None:
    if not DRAFT.exists():
        print(f"{DRAFT.name} is not in this checkout; claim-level checks "
              f"need the manuscript text and are skipped.")
        return
    text = io.open(DRAFT, encoding="utf-8").read()
    bad = []

    def claim(ok: bool, msg: str) -> None:
        if not ok:
            bad.append(msg)

    # ------------------------------------------- section cross-references
    # Splitting a long section renumbered its neighbours; a pointer to a
    # section that no longer exists is silent in Markdown and in the docx.
    _heads = set(re.findall(r"^#{2,4}\s+([0-9]+(?:[.][0-9]+)?)\s", text, re.M))
    for _ref in sorted(set(re.findall("§([0-9]+[.][0-9]+)", text))):
        claim(_ref in _heads,
              f"cross-reference to section {_ref}, which has no heading")

    # ------------------------------------------------ table and figure names
    # Every table and figure the text names must exist, here or in the
    # supplement. A pointer to "Table A1" -- an appendix numbering this paper
    # does not use -- survived a section move and read as a real table to
    # everything that checked only the Table S<n> form.
    _sup = io.open(SUPP, encoding="utf-8").read() if SUPP.exists() else ""
    _have = set()
    for _src in (text, _sup):
        _have |= set(re.findall(r"^\*\*((?:Table|Figure)\s+S?[0-9]+)[.]",
                                _src, re.M))
    _have = {re.sub(r"\s+", " ", h) for h in _have}
    for _name in sorted(set(re.findall(
            r"\b((?:Table|Figure)\s+[A-Z]?[0-9]+)\b",
            text.split("## References")[0]))):
        claim(re.sub(r"\s+", " ", _name) in _have,
              f"the text points to {_name}, which is defined nowhere")

    # ------------------------------------------- reference numbering order
    # Vancouver style, which both target journals use, numbers references in
    # order of first citation. 62 of 69 were out of order in an earlier draft.
    _body = text.split("## References")[0]
    _cit = re.compile(r"\[((?:[0-9]+\s*-\s*[0-9]+|[0-9]+)"
                      r"(?:\s*,\s*(?:[0-9]+\s*-\s*[0-9]+|[0-9]+))*)\]")
    _n = len(re.findall(r"^[0-9]+[.]\s", text.split("## References")[-1], re.M))
    _seen = []
    for _m in _cit.finditer(_body):
        _v = []
        for _t in re.split(r"\s*,\s*", _m.group(1)):
            _r = re.fullmatch(r"([0-9]+)\s*-\s*([0-9]+)", _t.strip())
            _v += (list(range(int(_r.group(1)), int(_r.group(2)) + 1))
                   if _r else [int(_t)])
        if not all(1 <= _x <= _n for _x in _v):
            continue
        for _x in _v:
            if _x not in _seen:
                _seen.append(_x)
    claim(_seen == list(range(1, _n + 1)),
          "references are not numbered in order of first citation: "
          f"first {sum(1 for i, v in enumerate(_seen, 1) if i != v)} of {_n} "
          "appear out of sequence")

    # --------------------------------------------- raw control characters
    # Writing the draft through shell heredocs turns an unescaped backslash
    # sequence into the character it names: \text became a literal tab and
    # the macro rendered as "L_{ ext{ov}}" in the built docx. Three times so
    # far, silently each time. No legitimate use in this text, so any hit is
    # the bug.
    for _i, _ln in enumerate(text.split("\n"), 1):
        for _code, _esc in {7: "a", 8: "b", 9: "t", 11: "v", 12: "f"}.items():
            claim(chr(_code) not in _ln,
                  f"line {_i}: raw {hex(_code)} where a backslash-{_esc} "
                  f"escape was meant -- {_ln.strip()[:70]!r}")

    # The same heredoc accident inside a checking script is worse than it is
    # in the draft: a regex whose word-boundary escape became a raw 0x08 matches
    # nothing, so the check it belongs to passes on every input and reports
    # no problem at all.
    # That is how a dangling "Table A1" survived a clean run of this file.
    # Five released scripts did not execute as distributed: one did not parse
    # and four used a name at module level before importing it, the wreckage
    # of a bulk path-anonymising edit. The data availability statement calls
    # the archive "the snapshot every figure in this paper was computed from",
    # so a script that cannot start falsifies it. Parsing catches the first;
    # the other four need the order of module-level statements, the failure
    # being a NameError at import time rather than a syntax error.
    def _module_scope(node):
        """Child nodes of `node` that run at import time, in source order.

        A comprehension and a lambda bind their own variables and a function
        body does not run on import, so a name used inside one of those is
        not evidence of anything.
        """
        for _c in ast.iter_child_nodes(node):
            if isinstance(_c, (ast.Lambda, ast.ListComp, ast.SetComp,
                               ast.DictComp, ast.GeneratorExp,
                               ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
                continue
            yield _c
            yield from _module_scope(_c)

    for _p in sorted(Path(__file__).parent.glob("*.py")):
        try:
            _tree = ast.parse(io.open(_p, encoding="utf-8").read(), str(_p))
        except SyntaxError as _e:
            claim(False, f"{_p.name}: does not parse -- line {_e.lineno}, "
                         f"{_e.msg}")
            continue
        _bound = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
        for _top in _tree.body:
            if isinstance(_top, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                _bound.add(_top.name)
                continue
            for _sub in _module_scope(_top):
                if isinstance(_sub, ast.alias):
                    _bound.add((_sub.asname or _sub.name).split(".")[0])
                elif isinstance(_sub, ast.ExceptHandler) and _sub.name:
                    _bound.add(_sub.name)   # `except ... as e` binds a string
                elif isinstance(_sub, ast.Name):
                    if isinstance(_sub.ctx, ast.Load):
                        if _sub.id not in _bound:
                            claim(False, f"{_p.name}: line {_sub.lineno} uses "
                                         f"'{_sub.id}' before it is defined or "
                                         f"imported at module level")
                            _bound.add(_sub.id)   # report each name once
                    else:
                        _bound.add(_sub.id)

    for _p in sorted(Path(__file__).parent.glob("*.py")):
        _src = io.open(_p, encoding="utf-8").read()
        for _code in (7, 8, 11, 12):
            claim(chr(_code) not in _src,
                  f"{_p.name}: raw {hex(_code)} in the source, which makes "
                  f"any pattern holding it match nothing")

    # ---------------------------------------------------- monotone claims
    # The word asserts something a series either does or does not do.
    rows = []
    for d in sorted(glob.glob(str(DRIVE_ROOT / "experiments" / "run_*"))):
        c = Path(d) / "config" / "requested_config.json"
        s_ = Path(d) / "summary" / "run_summary.json"
        if not (c.exists() and s_.exists()):
            continue
        cfg = json.loads(c.read_text(encoding="utf-8"))
        sm = json.loads(s_.read_text(encoding="utf-8"))
        p = cfg.get("loss_params") or {}
        if cfg.get("loss_name") == "icarb" and "alpha" in p \
                and cfg.get("overlap_loss_weight") == 3.0:
            rows.append((p["alpha"], sm["geometry"]["pred_coverage"],
                         sm["geometry"]["target_coverage"]))
    rows.sort()
    if rows:
        cp = [r[1] for r in rows]
        strictly = all(b < a for a, b in zip(cp, cp[1:]))
        says_monotone = bool(re.search(
            r"coverage[^.]{0,120}monotonic|monotonic[^.]{0,120}coverage",
            text, re.I | re.S))
        claim(strictly or not says_monotone,
              "the text calls the coverage series monotonic; it reverses "
              f"{sum(1 for a, b in zip(cp, cp[1:]) if b > a)} times "
              f"(Spearman {stats.spearmanr([r[0] for r in rows], cp)[0]:+.2f})")

    # ------------------------------------- identical-mask count, per segmenter
    recs = read(_fix(V / "rhuh_oracle_regions.jsonl"))
    for sid in ("sam2.1_l", "sam1_vit_b"):
        k = f"oracle:{sid}"
        n = sum(1 for r in recs.values()
                if k in r.get("arms", {})
                and r["arms"][k]["regions"].get("TC", {}).get("pred_voxels")
                == r["arms"][k]["regions"].get("ET", {}).get("pred_voxels"))
        m = re.search(r"the same mask in (\d+) of the 39", text)
        if m and sid == SEG:
            claim(int(m.group(1)) == n,
                  f"text says the ET and TC boxes give the same mask in "
                  f"{m.group(1)} of 39; on {SEG} it is {n} "
                  f"(the other segmenter would give a different count)")

    # -------------------------------- a spread quoted next to a cohort name
    tc = json.loads((DRIVE_ROOT / "reports" / "tc_detector_summary.json")
                    .read_text(encoding="utf-8"))
    spreads = {}
    for c, e in tc["cohorts"].items():
        v = [p[SEG]["pipeline"] for p in e["seeds"].values()]
        spreads[c] = max(v) - min(v)
    for c, sp in spreads.items():
        others = {k: v for k, v in spreads.items() if k != c}
        for para in re.split(r"\n\n", text):
            # only paragraphs actually discussing spreads: the same decimal
            # can legitimately be another quantity elsewhere, and after the
            # RHUH rerun one cohort's term equals another's seed spread
            if "spread" not in para.lower():
                continue
            if "BraTS-Africa" in para and c == "brats_africa":
                for k, v in others.items():
                    if f"{v:.4f}" in para and f"{sp:.4f}" not in para:
                        bad.append(f"a paragraph about BraTS-Africa quotes "
                                   f"{v:.4f}, which is {k}'s spread; "
                                   f"BraTS-Africa's is {sp:.4f}")

    # ------------------------------------ median presented without saying so
    for c in ("clean", "rhuh", "brats_africa"):
        rr = read(_fix(V / f"{c}_oracle_regions.jsonl"))
        k = f"oracle:{SEG}"
        for reg in ("TC", "ET"):
            v = [r["arms"][k]["regions"][reg]["rel_vol_diff"] for r in rr.values()
                 if k in r.get("arms", {})
                 and reg in r["arms"][k]["regions"]
                 and r["arms"][k]["regions"][reg].get("rel_vol_diff") is not None]
            if not v:
                continue
            med, mean = np.median(v) * 100, np.mean(v) * 100
            if abs(med - mean) < 2:
                continue
            if f"{med:+.1f}%" in text and "median" not in text.lower():
                bad.append(f"{c} {reg} volume error is quoted as {med:+.1f}%, "
                           f"which is the median; the mean is {mean:+.1f}% and "
                           f"the text does not say which it reports")

    # -------------------------------------------------- arithmetic in the text
    for m in re.finditer(r"(\d+)/(\d+)/(\d+) subsets|into (\d+)\s*\n?"
                         r"training, (\d+) validation and (\d+) test", text):
        g = [int(x) for x in m.groups() if x]
        if len(g) == 3:
            total = sum(g)
            claim(f"{total}" in text,
                  f"the split {g[0]}/{g[1]}/{g[2]} sums to {total}, which does "
                  f"not appear in the text")

    # ------------------------------- every arm mean matches its own file
    for c in ("clean", "rhuh", "brats_africa"):
        box = read(_fix(V / f"{c}_box.jsonl"))
        for arm in ("oracle", "pipeline"):
            k = f"{arm}:{SEG}"
            v = [r["arms"][k]["vol_dice"] for r in box.values()
                 if k in r.get("arms", {})]
            if v:
                claim(f"{np.mean(v):.4f}" in text,
                      f"{c} {arm} mean {np.mean(v):.4f} is not in the text")

    # --------------------------- the supporting information's own pointers
    # It cites the manuscript's tables and sections, and nothing checked those
    # until a referee followed one and landed on the wrong table. A pointer
    # that resolves to a real table is not enough; it has to resolve to the
    # table the citing sentence is about, which only a person can judge -- but
    # a pointer to a table that does not exist is checkable here.
    supp_p = DRAFT.parent / "SUPPLEMENTARY.md"
    if supp_p.exists():
        supp = io.open(supp_p, encoding="utf-8").read()
        m_tables = {int(m.group(1)) for m in
                    re.finditer(r"^\*\*Table ([0-9]+)[.]\*\*", text, re.M)}
        m_figs = {int(m.group(1)) for m in
                  re.finditer(r"^!\[Figure ([0-9]+)\]", text, re.M)}
        m_heads = set(re.findall(r"^#{2,4}\s+([0-9]+(?:[.][0-9]+)?)\s",
                                 text, re.M))
        s_tables = {m.group(1) for m in
                    re.finditer(r"^\*\*Table (S[0-9]+)[.]\*\*", supp, re.M)}
        s_figs = {m.group(1) for m in
                  re.finditer(r"^!\[Figure (S[0-9]+)\]", supp, re.M)}
        s_heads = {m.group(1) for m in
                   re.finditer(r"^## (S[0-9]+)[.]", supp, re.M)}
        flat = " ".join(supp.split())
        for n in sorted({int(x) for x in re.findall(r"Tables? ([0-9]+)",
                                                    flat)}):
            claim(n in m_tables,
                  f"the supplement cites Table {n}, which the manuscript "
                  f"does not have")
        for n in sorted({int(x) for x in re.findall(r"Figure ([0-9]+)",
                                                    flat)}):
            claim(n in m_figs,
                  f"the supplement cites Figure {n}, which the manuscript "
                  f"does not have")
        for r in sorted(set(re.findall(r"§([0-9]+[.][0-9]+)", supp))):
            claim(r in m_heads,
                  f"the supplement cites section {r}, which has no heading")
        # and the manuscript's pointers into the supplement
        mflat = " ".join(text.split())
        for t in sorted(set(re.findall(r"Table (S[0-9]+)", mflat))):
            claim(t in s_tables,
                  f"the manuscript cites {t}, which the supplement does not "
                  f"have")
        for f in sorted(set(re.findall(r"Figure (S[0-9]+)", mflat))):
            claim(f in s_figs,
                  f"the manuscript cites Figure {f}, which the supplement "
                  f"does not have")
        for sec in sorted(set(re.findall(r"\((S[0-9]+)[,)]", mflat))):
            claim(sec in s_heads,
                  f"the manuscript cites {sec}, which the supplement does "
                  f"not have")
        # a gap in the supplement's own table numbering
        if s_tables:
            nums = sorted(int(t[1:]) for t in s_tables)
            missing = [n for n in range(1, max(nums) + 1) if n not in nums]
            claim(not missing,
                  f"the supplement's table numbering skips {missing}")

    print(f"{DRAFT.name}: {len(bad)} claim-level problems")
    for b in bad:
        print("  -", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
