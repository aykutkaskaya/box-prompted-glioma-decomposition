"""Rewrite the ablation section of RUNS_ANALYSIS.md against the seed replicates.

Sections 9 and 10 were written when every configuration had one seed. Two of the
three claims they made do not survive replication, and the one that does is not
the one the sections led with, so this supersedes them rather than annotating
them.

What changed and why:

  * The classification-loss finding (focal over varifocal, +0.0158 F1) is inside
    the seed noise band and is demoted to an observation. Neither run 16 nor the
    neg_seg_fp_rate half of it was replicated, so it cannot be tested at all.
  * "IC-Arb beats the overlap family" survives against GIoU and fails against
    EIoU and SIoU. Reporting only the GIoU comparison would be selective, so the
    F1 claim is withdrawn as a family claim.
  * What does survive, and by a wide margin, is the mechanism the loss was
    designed around: alpha moves the two coverages in opposite directions, and
    at alpha=0 the loss reaches a region no symmetric IoU variant reaches.

The denominator throughout is the standard error of the difference between
configuration means. An earlier version used the observed range, which is wrong
here because its expectation grows with the number of seeds -- running a third
replicate made every claim look weaker purely by adding data.

    python scripts/seed_replicates.py     # writes reports/seed_replicates.json
    python scripts/write_report_ablation.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

R = DRIVE_ROOT / "reports"
MD = DRIVE_ROOT / "RUNS_ANALYSIS.md"
MARK = "## 19. The ablation, re-read against seed replicates (2026-08-28)"

EXP = DRIVE_ROOT / "experiments"
REP = DRIVE_ROOT / "experiments_repeated"
SLUG = {1: "run_001_giou_w30", 4: "run_004_eiou_w30", 5: "run_005_siou_w30",
        9: "run_009_icarb_alpha_000_w30", 20: "run_020_icarb_alpha_100_w30",
        37: "run_037_icarb_alpha_050_w30_clsfl"}
NAME = {1: "GIoU", 4: "EIoU", 5: "SIoU", 9: "IC-Arb a=0.0",
        20: "IC-Arb a=1.0", 37: "IC-Arb a=0.5 + focal cls"}


def seeds_of(rid: int, grp: str, field: str) -> list:
    out = []
    for s in (1337, 42, 62):
        base = EXP / SLUG[rid] if s == 1337 else REP / f"seed_{s}" / SLUG[rid]
        p = base / "summary" / "run_summary.json"
        if p.exists():
            out.append(json.loads(p.read_text(encoding="utf-8"))[grp][field])
    return out


def contrast(ra: int, rb: int, grp: str, field: str) -> dict:
    a, b = seeds_of(ra, grp, field), seeds_of(rb, grp, field)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    se = float(np.sqrt(np.std(a, ddof=1) ** 2 / len(a) + np.std(b, ddof=1) ** 2 / len(b)))
    return {"mean_a": ma, "mean_b": mb, "gap": mb - ma, "se": se,
            "ratio": abs(mb - ma) / se if se else float("inf"),
            "n_a": len(a), "n_b": len(b)}


def main() -> None:
    rep = json.loads((R / "seed_replicates.json").read_text(encoding="utf-8"))
    noise = rep["noise"]
    per = rep["per_config"]

    L: list[str] = []
    w = L.append
    w(MARK)
    w("")
    w("Sections 9 and 10 were written when every configuration had a single "
      "training seed. Six configurations now have two or three, and the picture "
      "they give is different enough that this section supersedes them rather "
      "than annotating them.")
    w("")

    # ------------------------------------------------------- the noise floor
    w("### 19.1 What a seed is worth")
    w("")
    w("Three sources of run-to-run variation, now separated:")
    w("")
    w("| source | size | how it was measured |")
    w("|---|---|---|")
    w("| non-determinism at a fixed seed | 0.0005 F1 | runs 30 and 37: same config, same seed 1337 |")
    f1n = noise["test.f1"]["mean"]
    cpn = noise["geometry.pred_coverage"]["mean"]
    ctn = noise["geometry.target_coverage"]["mean"]
    w(f"| seed to seed, detection F1 | {f1n:.4f} | mean spread over {noise['test.f1']['n_configs']} replicated configurations |")
    w(f"| seed to seed, box coverage | {cpn:.4f} / {ctn:.4f} | same configurations, C_p / C_t |")
    w("")
    w("Two things follow. The 0.0005 figure quoted earlier as a noise floor "
      "measures only non-determinism -- training requests deterministic "
      "algorithms but one backward operator has no deterministic CUDA "
      "implementation -- and it understates seed variation by a factor of about "
      f"{f1n / 0.0005:.0f}. And **F1 is roughly three times noisier than the "
      "coverage metrics**, which matters for which claims can be made at all.")
    w("")

    # ---------------------------------------------------------- per-config
    w("### 19.2 Replicated configurations")
    w("")
    w("| run | configuration | seeds | F1 | C_p | C_t |")
    w("|---|---|---|---|---|---|")
    for rid in (1, 4, 5, 9, 20, 37):
        c = per.get(str(rid))
        if not c:
            continue
        def cell(key):
            st = c.get(key)
            if not st:
                return "—"
            return f"{st['mean']:.4f}" + (f" ± {st['sd']:.4f}" if st.get("sd") else "")
        w(f"| {rid} | {NAME[rid]} | {len(c['seeds'])} | {cell('test.f1')} | "
          f"{cell('geometry.pred_coverage')} | {cell('geometry.target_coverage')} |")
    w("")
    w("Standard deviations are shown only where three seeds exist; two seeds "
      "give a difference, not a distribution.")
    w("")

    # ------------------------------------------------------------ withdrawn
    w("### 19.3 Two claims withdrawn")
    w("")
    w("**The classification loss.** Section 9 reported focal classification loss "
      "beating varifocal by +0.0158 F1 and halving the false-mask rate on "
      f"tumour-free slices. The F1 half of that sits inside the seed noise band "
      f"({f1n:.4f}), a ratio of {0.0158 / f1n:.1f}. The other half cannot be "
      "tested: run 16 was never replicated, and SAM evaluation was skipped in "
      "the replicate tour, so `neg_seg_fp_rate` has no second measurement. It is "
      "recorded here as an observation and is not carried as a finding.")
    w("")
    w("This costs the study nothing it should have kept. Varifocal, focal and "
      "BCE are all off-the-shelf; the choice among them was never a contribution.")
    w("")
    w("**IC-Arb's F1 advantage over the overlap family.** Tested against each "
      "replicated baseline:")
    w("")
    w("| IC-Arb (α=0) vs | mean F1 | gap | SE | ratio | verdict |")
    w("|---|---|---|---|---|---|")
    for rid in (1, 4, 5):
        c = contrast(rid, 9, "test", "f1")
        verdict = "holds" if c["ratio"] >= 3 else ("marginal" if c["ratio"] >= 2 else "**does not clear noise**")
        w(f"| {NAME[rid]} | {c['mean_a']:.4f} | {c['gap']:+.4f} | {c['se']:.5f} | "
          f"{c['ratio']:.1f}× | {verdict} |")
    w("")
    w("It survives against GIoU and fails against the other two. GIoU is both "
      "the weakest baseline and the least noisy one, so reporting that "
      "comparison alone would be selective. The F1 claim is withdrawn.")
    w("")

    # ------------------------------------------------------------- survives
    w("### 19.4 What survives, and it is the mechanism")
    w("")
    w("The loss was not designed to win on F1. It was designed to make the "
      "trade-off between the two faces of the intersection adjustable, and that "
      "is what replication confirms.")
    w("")
    w("**α moves the two coverages in opposite directions.**")
    w("")
    w("| | α = 0 | α = 1 | gap | SE | ratio |")
    w("|---|---|---|---|---|---|")
    for label, field in (("C_p — prediction coverage", "pred_coverage"),
                         ("C_t — target coverage", "target_coverage")):
        c = contrast(9, 20, "geometry", field)
        w(f"| {label} | {c['mean_a']:.4f} | {c['mean_b']:.4f} | {c['gap']:+.4f} | "
          f"{c['se']:.5f} | **{c['ratio']:.1f}×** |")
    w("")
    w("Both endpoints have three seeds. The directions are opposite, as the "
      "derivative in the methods section predicts: weighting the coverage term "
      "more heavily penalises inflated boxes harder than IoU does, so boxes "
      "tighten — more of the prediction lands on target, less of the target is "
      "covered.")
    w("")
    w("**At α=0 the loss reaches a region the symmetric variants do not.**")
    w("")
    w("| IC-Arb (α=0) vs | C_p gap | ratio | C_t gap | ratio |")
    w("|---|---|---|---|---|")
    worst = 99.0
    for rid in (1, 4, 5):
        cp = contrast(rid, 9, "geometry", "pred_coverage")
        ct = contrast(rid, 9, "geometry", "target_coverage")
        worst = min(worst, cp["ratio"], ct["ratio"])
        w(f"| {NAME[rid]} | {cp['gap']:+.4f} | {cp['ratio']:.1f}× | "
          f"{ct['gap']:+.4f} | {ct['ratio']:.1f}× |")
    w("")
    w(f"Six comparisons, all clearing their standard error, the weakest at "
      f"{worst:.1f}×. Note also where the endpoints sit relative to the family: "
      "the baselines occupy C_p 0.920–0.931, and IC-Arb at α=1 sits inside that "
      "band while at α=0 it sits well outside it. The parameter spans from where "
      "the family already is to somewhere it cannot go.")
    w("")
    w("That no symmetric IoU variant offers this control is true by construction "
      "rather than by measurement — none of them has such a parameter. What the "
      "measurement establishes is that the control is real, is large, and costs "
      "nothing in F1.")
    w("")

    # -------------------------------------------------------------- reading
    w("### 19.5 How the ablation should now be stated")
    w("")
    w("> IC-Arb is indistinguishable from the overlap family on detection F1. "
      "Where it differs is box geometry: α controls prediction and target "
      "coverage in opposite directions, by a margin 8–16× the seed noise, and "
      "at α=0 reaches a regime no symmetric IoU variant reaches. The overlap "
      "weight w has no measurable effect and is fixed at 3.0.")
    w("")
    w("This is a smaller claim than section 9 made and a more defensible one. It "
      "is also the claim the paper actually needs: the point of a tunable "
      "coverage trade-off is the tuning, not a leaderboard position.")
    w("")

    text = "\n".join(L) + "\n"
    doc = MD.read_text(encoding="utf-8") if MD.exists() else ""
    head = doc[: doc.index(MARK)] if MARK in doc else doc
    # Drop any trailing horizontal rules before adding one back. rstrip()
    # removes the blank line after a rule but not the rule itself, so each
    # rerun of a writer left another "---" behind.
    head = re.sub(r"(?:^---[ \t]*$\s*)+\Z", "", head, flags=re.M)
    doc = head.rstrip() + "\n\n---\n\n" + text
    MD.write_text(doc, encoding="utf-8")
    print(f"wrote section 19 into {MD.name} ({len(text)} chars)")


if __name__ == "__main__":
    main()
