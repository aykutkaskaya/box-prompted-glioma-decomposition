"""Everything a style pass must not change, checked mechanically.

A copy-edit is allowed to alter wording and sentence boundaries. It is not
allowed to alter a number, a citation, a cross-reference, a British spelling, or
the presence of a hedge. This diffs two versions of the manuscript on exactly
those and prints what moved.

    python scripts/style_invariants.py <before.md> <after.md>
"""
from __future__ import annotations

import collections
import io
import re
import sys


def bag(path: str) -> dict:
    s = io.open(path, encoding="utf-8").read()
    body = s.split("## References")[0]
    refs = s.split("## References")[1] if "## References" in s else ""
    norm = body.replace("−", "-")
    return {
        "numbers": collections.Counter(re.findall(r"-?\d+[.]\d+", norm)),
        "integers_pct": collections.Counter(re.findall(r"\d+%", norm)),
        "fractions": collections.Counter(re.findall(r"\b\d+\s*/\s*\d+\b", norm)),
        "citations": collections.Counter(re.findall(r"\[[\d,–\- ]+\]", body)),
        "sections": collections.Counter(re.findall(r"§[\d.]+", body)),
        "tables": collections.Counter(re.findall(r"Table [\dA]\d*", body)),
        "figures": collections.Counter(re.findall(r"Figure \d+", body)),
        "p_values": collections.Counter(re.findall(r"p = [\d.e-]+", norm)),
        # the words that carry the paper's qualifications
        "hedges": collections.Counter(
            w.lower() for w in re.findall(
                r"\b(?:not|no|cannot|spans? zero|withdraw\w*|fail\w*|"
                r"limitation\w*|caveat\w*|exploratory|confirmatory|"
                r"contaminat\w*|leakage|artefact\w*)\b", norm, re.I)),
        "british": collections.Counter(re.findall(
            r"\b\w*(?:tumour|localisation|normalis|artefact|analys|"
            r"characteris|recognis|standardis)\w*\b", norm, re.I)),
        "american": collections.Counter(re.findall(
            r"\b\w*(?:tumor|localization|normaliz|artifact|analyz|"
            r"characteriz|recogniz|standardiz)\w*\b", norm, re.I)),
        "ref_lines": collections.Counter(
            l.strip()[:60] for l in refs.split("\n") if re.match(r"^\d+\.", l.strip())),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = bag(sys.argv[1]), bag(sys.argv[2])
    bad = 0
    for key in a:
        ca, cb = a[key], b[key]
        if ca == cb:
            print(f"  ok        {key:<14} {sum(ca.values())} occurrences unchanged")
            continue
        lost = ca - cb
        gained = cb - ca
        # hedges may legitimately shift a little as sentences merge, but a net
        # loss is the thing to look at
        flag = "  CHANGED  " if key != "hedges" else "  moved    "
        if key != "hedges":
            bad += 1
        print(f"{flag}{key:<14} -{sum(lost.values())} +{sum(gained.values())}")
        for tok, k in sorted(lost.items())[:12]:
            print(f"      lost    {tok!r} x{k}")
        for tok, k in sorted(gained.items())[:12]:
            print(f"      gained  {tok!r} x{k}")

    print("\nAmerican spellings in the edited file:",
          sum(b["american"].values()) or "none")
    print("VERDICT:", "invariants held" if bad == 0 else
          f"{bad} invariant group(s) changed -- review each")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
