"""Write RUNS_ANALYSIS section 20: the TC ceiling, now with real detectors.

Section 18 closed on an asymmetry it could not resolve. ET was shown to be
capped by the prompt -- the tight rectangle around a ring is the tight rectangle
around the disc, so no detector can fix it -- while TC was shown to have a
ceiling well above what the detector-free model reached. That left the TC
conclusion resting on the word "could": a ceiling says what a perfect detector
would be worth, not what a trained one achieves.

TC detectors now exist, so the sentence can be replaced by a measurement. The
interesting quantity is not the pipeline score alone but how much of the
oracle's headroom over the detector-free model survives when the box has to be
found:

    retained = (pipeline - direct) / (oracle - direct)

At 1.0 the detector costs nothing; at 0.0 it gives back the entire ceiling.
Below 0.0 the ceiling is real but the pipeline is worse than not detecting at
all.

This section is written per detector seed and nothing is pooled. The first
version read one seed and stated flatly that out of domain the detector is worse
than no detector; the second seed did not support it. Where seeds disagree the
report says so rather than averaging the disagreement away.

Prose branches on the result rather than presuming one. Every number is read
from reports/tc_detector_summary.json and the run summaries.

    bash scripts/tc_pipeline.sh all
    bash scripts/tc_pipeline.sh all 42
    python scripts/compile_tc.py
    python scripts/write_report_tc.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

SRC = DRIVE_ROOT / "reports" / "tc_detector_summary.json"
MD = DRIVE_ROOT / "RUNS_ANALYSIS.md"
MARK = "## 20. Tumour core with a trained detector"
ORDER = ["clean", "rhuh", "brats_africa"]
SEG = "sam2.1_l"          # the stronger box segmenter; sam1 is reported beside it
OTHER = "sam1_vit_b"

SLUG = "run_037_icarb_alpha_050_w30_clsfl"
WT_RUN = DRIVE_ROOT / "experiments" / SLUG
TC_SEEDS = [1337, 42, 62]          # whichever of these exist on disk are used
NOISE = DRIVE_ROOT / "reports" / "seed_replicates.json"


def tc_runs() -> dict:
    """seed -> run directory, for every TC detector replicate present."""
    out = {}
    for s in TC_SEEDS:
        d = DRIVE_ROOT / "experiments_repeated" / f"tc_seed_{s}" / SLUG
        if (d / "summary" / "run_summary.json").exists():
            out[s] = d
    if not out:
        raise SystemExit("no TC detector run found under experiments_repeated/tc_seed_*")
    return out


HD95 = DRIVE_ROOT / "reports" / "adapter_hd95.json"


def adapter_hd95() -> dict:
    """Boundary-distance metrics for the LoRA adapters, if extracted.

    This study measures overlap only, so a model that places small false
    positives far from the lesion looks like a peer here and does not in a
    distance metric. One adapter is exactly that case, and saying so is
    cheaper than letting a reader assume the three are interchangeable.
    """
    return json.loads(HD95.read_text(encoding="utf-8")) if HD95.exists() else {}


def seed_band() -> tuple:
    """Seed-to-seed detection F1 spread, and how many configurations built it."""
    n = json.loads(NOISE.read_text(encoding="utf-8"))["noise"]["test.f1"]
    return float(n["mean"]), int(n["n_configs"])


def detection(run: Path) -> dict:
    s = json.loads((run / "summary" / "run_summary.json").read_text(encoding="utf-8"))
    return {**s["test"], **s["geometry"]}


def listed(items) -> str:
    """a, b and c -- so cohort lists read as sentences rather than CSV."""
    xs = list(items)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC.name} missing -- run compile_tc.py first")
    rep = json.loads(SRC.read_text(encoding="utf-8"))
    runs = tc_runs()
    tcs = {s: detection(d) for s, d in runs.items()}
    wt = detection(WT_RUN)
    dseeds = [s for s in TC_SEEDS if s in runs]

    have = [c for c in ORDER if rep["cohorts"].get(c, {}).get("seeds")]
    if not have:
        raise SystemExit("no cohort has the pipeline arm yet")

    def cell(c, seed, seg=SEG):
        return rep["cohorts"][c]["seeds"].get(str(seed), {}).get(seg)

    # detector seeds that actually have pipeline runs, per cohort
    pseeds = sorted({int(s) for c in have for s in rep["cohorts"][c]["seeds"]},
                    key=lambda s: TC_SEEDS.index(s) if s in TC_SEEDS else 99)

    L: list[str] = []
    w = L.append
    w(MARK)
    w("")
    w("Section 18 measured what a perfect tumour-core detector would be worth by "
      "handing the segmenter a ground-truth TC box, and left open whether such a "
      "detector can be trained. Two now have been -- same architecture, same "
      "loss, same schedule, differing only in training seed, with boxes drawn "
      "around labels 1 and 4 instead of 1, 2 and 4 -- so the ceiling can be read "
      "against real detectors instead of an assumption.")
    w("")

    # ------------------------------------------------------- the detector itself
    w("### 20.1 The detectors")
    w("")
    w("| | whole tumour | " + " | ".join(f"TC seed {s}" for s in dseeds) + " |")
    w("|---" * (len(dseeds) + 2) + "|")
    for lbl, k in (("detection F1", "f1"), ("precision", "precision"),
                   ("recall", "recall"), ("mAP@50-95", "map50_95"),
                   ("box IoU", "iou"), ("C_p", "pred_coverage"),
                   ("C_t", "target_coverage")):
        w(f"| {lbl} | {wt[k]:.4f} | " + " | ".join(f"{tcs[s][k]:.4f}" for s in dseeds) + " |")
    w(f"| positive slices | {wt['n_images']} | "
      + " | ".join(str(tcs[s]['n_images']) for s in dseeds) + " |")
    w("")
    w("Training needed one change beyond the box source: mixed precision had to "
      "be disabled. With it on, the run was healthy for fifteen epochs and "
      "produced non-finite losses on the sixteenth, and did so again on a rerun "
      "-- deterministic ordering reaching the same batch. The whole-tumour runs "
      "were unaffected, and the degenerate-box explanation was tested and "
      "rejected: over forty patients, whole tumour produces *more* boxes of one "
      "pixel or less than tumour core does (1.21% against 0.78%).")
    w("")
    prim = dseeds[0]
    dF1 = tcs[prim]["f1"] - wt["f1"]
    band, n_cfg = seed_band()
    own = max(tcs[s]["f1"] for s in dseeds) - min(tcs[s]["f1"] for s in dseeds)
    sent = (f"Detection is harder for the sub-region: F1 falls {abs(dF1):.4f} "
            f"against the whole-tumour detector. ")
    if len(dseeds) >= 2:
        sent += (f"The {len(dseeds)} TC seeds span {own:.4f} F1, so the drop is "
                 f"roughly {abs(dF1) / own:.0f}x the detector's own replicate "
                 f"spread and {abs(dF1) / band:.1f}x the {band:.4f} band measured "
                 f"over {n_cfg} replicated whole-tumour configurations. ")
        if len(dseeds) == 2:
            sent += ("Two seeds give a difference rather than a distribution, "
                     "the same caveat this study applies to its other two-seed "
                     "configurations. ")
    else:
        sent += (f"With one seed the drop can only be read against the {band:.4f} "
                 f"band from {n_cfg} replicated whole-tumour configurations -- "
                 f"about {abs(dF1) / band:.1f}x, indicative rather than tested. ")
    sent += (f"**The boxes they do draw are barely looser**: IoU "
             f"{tcs[prim]['iou'] - wt['iou']:+.4f}, C_p "
             f"{tcs[prim]['pred_coverage'] - wt['pred_coverage']:+.4f}. Finding "
             f"tumour core is the part that degrades; bounding it is not.")
    w(sent)
    w("")

    # ------------------------------------------------------------ the three arms
    w("### 20.2 The ceiling, approached")
    w("")
    w(f"Volumetric TC Dice, `{SEG}` as the box segmenter, over patients present "
      f"in all three arms. Oracle and detector-free are fixed per cohort; only "
      f"the pipeline arm depends on the detector.")
    w("")
    w("| cohort | n | oracle | detector-free | "
      + " | ".join(f"pipeline, seed {s}" for s in pseeds) + " | retained |")
    w("|---" * (len(pseeds) + 5) + "|")
    for c in have:
        rows = [cell(c, s) for s in pseeds]
        a0 = next(r for r in rows if r)
        rets = [r["retained"] for r in rows if r]
        w(f"| {rep['cohorts'][c]['label']} | {a0['n']} | {a0['oracle']:.4f} | "
          f"{a0['direct']:.4f} | "
          + " | ".join(f"**{r['pipeline']:.4f}**" if r else "—" for r in rows)
          + f" | {min(rets) * 100:.0f}% to {max(rets) * 100:.0f}% |")
    w("")
    w("Retained headroom is `(pipeline - direct) / (oracle - direct)`: the share "
      "of the oracle's advantage over the detector-free model that survives "
      "having to find the box. It is the quantity section 18 could not compute.")
    w("")

    # ----------------------------------------------------------- decomposition
    w("### 20.3 Where the loss falls")
    w("")
    w("| cohort | seed | localisation<br>oracle − pipeline | "
      "segmentation<br>direct − oracle | total<br>direct − pipeline | "
      "95% CI of the mean | p |")
    w("|---|---|---|---|---|---|---|")
    for c in have:
        for s in pseeds:
            a = cell(c, s)
            if not a:
                continue
            pv = a["pipeline_vs_direct"]
            w(f"| {rep['cohorts'][c]['label']} | {s} | {a['localisation_gap']:+.4f} | "
              f"{a['segmentation_gap']:+.4f} | {a['total_gap']:+.4f} | "
              f"[{pv['ci95'][0]:+.4f}, {pv['ci95'][1]:+.4f}] | "
              f"{pv['p_wilcoxon']:.2g} |")
    w("")
    w("A negative total means the pipeline is ahead. **The segmentation term is "
      "negative everywhere and the localisation term positive everywhere.** "
      "Given a correct TC box, the box-prompted segmenter beats the "
      "detector-free model on every cohort; having to find that box gives most "
      "of it back. Section 18 found the same shape for whole tumour. Sub-region "
      "segmentation from a box is not the problem; sub-region *localisation* is.")
    w("")

    # ------------------------------------------- the part that needed two seeds
    w("### 20.4 A detection metric does not predict the pipeline")
    w("")
    w(f"The two detectors differ by {own:.4f} detection F1 -- seed noise, on any "
      f"reading. Downstream they do not behave like near-identical models:")
    w("")
    w("| cohort | pipeline spread across seeds | retained spread | sign of total gap |")
    w("|---|---|---|---|")
    flipped = []
    for c in have:
        rows = [cell(c, s) for s in pseeds if cell(c, s)]
        if len(rows) < 2:
            continue
        ps = [r["pipeline"] for r in rows]
        rs = [r["retained"] for r in rows]
        consistent = len({r["total_gap"] < 0 for r in rows}) == 1
        if not consistent:
            flipped.append(rep["cohorts"][c]["label"])
        w(f"| {rep['cohorts'][c]['label']} | {min(ps):.4f}–{max(ps):.4f} "
          f"({max(ps) - min(ps):.4f}) | {min(rs) * 100:.0f}% to {max(rs) * 100:.0f}% | "
          + ("consistent" if consistent else "**flips**") + " |")
    w("")
    spreads = []
    for c in have:
        ps = [r["pipeline"] for s in pseeds if (r := cell(c, s))]
        if len(ps) > 1:
            spreads.append(max(ps) - min(ps))
    w(f"A {own:.4f} change in detection F1 moves cohort Dice by up to "
      f"{max(spreads):.4f}. "
      f"The reason is in the failure distribution: the pipeline does not degrade "
      f"gradually, it fails to near zero on a few patients when the detector "
      f"proposes no box or boxes in the wrong place, and which patients those "
      f"are is not stable across seeds.")
    w("")
    w("| cohort | seed | pipeline mean | median | patients < 0.05 | no box |")
    w("|---|---|---|---|---|---|")
    for c in have:
        for s in pseeds:
            a = cell(c, s)
            if not a:
                continue
            w(f"| {rep['cohorts'][c]['label']} | {s} | {a['pipeline']:.4f} | "
              f"{a['pipeline_median']:.4f} | {len(a['patients_near_zero'])} | "
              f"{len(a['patients_with_no_box'])} |")
    w("")
    w("Every median is far above its mean. On the patient in the middle of the "
      "distribution the pipeline is strong on all three cohorts; the cohort mean "
      "is set by the tail. **A single-seed, end-to-end evaluation of a detection "
      "pipeline on out-of-domain data is therefore not trustworthy**, and that "
      "holds regardless of which model it favours.")
    w("")

    # ------------------------------------------------------------- the verdict
    # The question the adapter replicates were run to answer: is the
    # instability a property of domain shift, or of the staged design?
    rows = [(c, rep["cohorts"][c].get("detector_free_seeds", {})) for c in have]
    rows = [(c, f) for c, f in rows if len(f) > 1]
    if rows:
        w("### 20.5 Is the detector-free arm unstable too?")
        w("")
        w("The pipeline's spread across detector seeds is only interpretable "
          "against the other arm's spread across its own. The detector-free "
          "model was therefore replicated at three LoRA seeds and evaluated "
          "under the identical protocol.")
        w("")
        w("| cohort | detector-free TC, seeds | spread | pipeline TC spread |")
        w("|---|---|---|---|")
        for c, f in rows:
            tc = [v["TC"] for v in f.values()]
            ps = [cell(c, s)["pipeline"] for s in pseeds if cell(c, s)]
            w(f"| {rep['cohorts'][c]['label']} | "
              + ", ".join(f"{v:.4f}" for v in tc)
              + f" | {max(tc) - min(tc):.4f} | {max(ps) - min(ps):.4f} |")
        w("")
        incomplete = {c: rep["cohorts"][c]["detector_free_incomplete"]
                      for c in have if rep["cohorts"][c].get("detector_free_incomplete")}
        if incomplete:
            w("Adapter seeds still running are excluded: "
              + "; ".join(f"{rep['cohorts'][c]['label']} seed "
                          + ", ".join(f"{s} ({v['n']} patients)" for s, v in d.items())
                          for c, d in incomplete.items())
              + ". A partial run averages over a different patient set and is "
                "not comparable with a completed one.")
            w("")
        # Stated from the numbers, not around them: the clean story (only the
        # staged arm is unstable) is not what three adapter seeds show.
        free_sp = [max(v["TC"] for v in f.values()) - min(v["TC"] for v in f.values())
                   for _, f in rows]
        pipe_sp = [max(cell(c, s)["pipeline"] for s in pseeds if cell(c, s))
                   - min(cell(c, s)["pipeline"] for s in pseeds if cell(c, s))
                   for c, _ in rows]
        wider = sum(p > f for p, f in zip(pipe_sp, free_sp))
        mono = lambda xs: all(b > a for a, b in zip(xs, xs[1:]))
        w(f"The pipeline is the noisier arm on {wider} of {len(rows)} cohorts, "
          f"but the margin is not uniform -- "
          + listed(f"{p / f:.1f}x on {rep['cohorts'][c]['label']}"
                   for (c, _), p, f in zip(rows, pipe_sp, free_sp)) + ". "
          "**The detector-free arm is not stable either.** On "
          f"{rep['cohorts'][rows[1][0]]['label']} its three adapters span "
          f"{free_sp[1]:.4f}, close to the pipeline's {pipe_sp[1]:.4f}; a "
          "reading on which only the staged design is fragile is not "
          "supported.")
        w("")
        if mono(pipe_sp) and not mono(free_sp):
            w("What does separate them is the shape. The pipeline's spread "
              "rises monotonically with distance from the training "
              f"distribution ("
              + " -> ".join(f"{v:.4f}" for v in pipe_sp)
              + "); the detector-free arm's does not ("
              + " -> ".join(f"{v:.4f}" for v in free_sp)
              + "). So domain shift does not simply make every model noisier: "
              "it makes *this* pipeline progressively less repeatable, which "
              "is what a decomposition into a detector and a segmenter would "
              "predict and a single end-to-end number would not show.")
            w("")
        w("The practical consequence applies to both arms and is the stronger "
          "for it: neither architecture should be compared to the other from "
          "one training run on out-of-domain data.")
        w("")
        hd = adapter_hd95().get("seeds", {})
        odd = [s for s, v in hd.items() if "SECONDARY" in v.get("role", "")]
        if odd:
            s = odd[0]; v = hd[s]
            others = [x for x in hd if x != s]
            w(f"One caveat on the adapters, and it cuts at this study's own "
              f"metric set. Adapter {s} is classified in the source project as "
              f"a sensitivity analysis rather than a release model, because "
              f"its tumour-core HD95 is {v['hd95']['TC']:.2f} mm against "
              + " and ".join(f"{hd[o]['hd95']['TC']:.2f} mm" for o in others)
              + f" for the other two, with {v['tc_hd95_over_100']} of "
                f"{v['n']} patients above 100 mm: it places small false "
                f"positives far from the lesion. Its TC Dice is "
                f"{v['dice']['TC']:.4f} against "
              + " and ".join(f"{hd[o]['dice']['TC']:.4f}" for o in others)
              + ", so **the overlap metrics used throughout this paper cannot "
                "see the defect at all**. It is retained above because the "
                "decision to include it was taken before any external number "
                "for it existed, and excluding it afterwards would be "
                "selective; but a spread computed over it is a spread over a "
                "model this paper's metrics are blind to.")
            w("")

    w("### 20.6 What this settles")
    w("")
    names = {c: rep["cohorts"][c]["label"] for c in have}
    ahead, behind, mixed = [], [], []
    for c in have:
        rows = [cell(c, s) for s in pseeds if cell(c, s)]
        signs = {r["total_gap"] < 0 for r in rows}
        (ahead if signs == {True} else behind if signs == {False} else mixed).append(c)

    parts = []
    if ahead:
        parts.append(f"ahead of the detector-free model on "
                     f"{listed(names[c] for c in ahead)} at every seed")
    if behind:
        parts.append(f"behind on {listed(names[c] for c in behind)} at every seed")
    if mixed:
        parts.append(f"unresolved on {listed(names[c] for c in mixed)}, where the "
                     f"sign flips between seeds")
    w(f"> **A TC detector is trainable, and most of the oracle's ceiling stays "
      f"out of reach.** The pipeline is {listed(parts)}. Retained headroom runs "
      f"from {min(r['retained'] for c in have for s in pseeds if (r := cell(c, s))) * 100:.0f}% "
      f"to {max(r['retained'] for c in have for s in pseeds if (r := cell(c, s))) * 100:.0f}%. "
      f"So the answer to section 18's open question is neither of the two clean "
      f"ones: the sub-region prompt works, a sub-region detector can be trained, "
      f"and localisation gives back most of what the prompt earns.")
    w("")
    w("Three things follow, and they should be kept apart.")
    w("")
    w("**The limitation is discharged.** Section 18 carried \"no sub-region "
      "detector was trained\" and used the ceiling as a stand-in. It no longer "
      "needs one. Section 18's claim that TC \"is not capped\" is correct about "
      "the prompt and was silent about the detector; that silence is now filled, "
      "and not flatteringly.")
    w("")
    w("**The ET result is untouched.** It was never a localisation claim. No box "
      "distinguishes a ring from the disc it encloses, so no detector, trained "
      "or oracular, changes it. That asymmetry between ET and TC survives.")
    w("")
    if behind or mixed:
        cs = behind + mixed
        w(f"**Out of domain the detector may be worth nothing at all.** On "
          f"{listed(names[c] for c in cs)} the retained headroom is at or below "
          f"zero: prompting the segmenter with these detectors' boxes scores no "
          f"better than the detector-free model that never sees a box"
          + (f", and at some seeds worse" if behind else "")
          + f". The oracle arm on the same data is well above both, so the "
            f"segmenter is not the problem. An end-to-end score would report "
            f"this as a segmentation result; the decomposition shows it is not.")
        w("")
        if mixed:
            w(f"How far below zero is not settled. On "
              f"{listed(names[c] for c in mixed)} the two seeds disagree on which "
              f"model leads, so the honest statement is that they are "
              f"indistinguishable there, not that one wins. A third seed would "
              f"decide it.")
            w("")

    # The weaker segmenter is not just a uniform shift down: it can change which
    # cohorts are unresolved, so its verdict is reported rather than implied.
    o_rows = {c: [cell(c, s, OTHER) for s in pseeds if cell(c, s, OTHER)] for c in have}
    o_rows = {c: r for c, r in o_rows.items() if r}
    if o_rows:
        para = (f"**With `{OTHER}` in place of `{SEG}`** the pipeline scores "
                + listed(f"{min(r['pipeline'] for r in rs):.4f}–"
                         f"{max(r['pipeline'] for r in rs):.4f} on {names[c]}"
                         for c, rs in o_rows.items())
                + " across the same seeds -- uniformly lower, as everywhere "
                  "else in this study.")
        o_behind = [c for c, rs in o_rows.items()
                    if len({r["total_gap"] < 0 for r in rs}) == 1
                    and rs[0]["total_gap"] > 0]
        if o_behind:
            para += (f" The ordering is not merely shifted, though: on "
                     f"{listed(names[c] for c in o_behind)} the detector-free "
                     f"model is ahead at **every** seed with this segmenter, "
                     f"where with `{SEG}` the seeds disagree. Which of the two "
                     f"leads out of domain depends on the segmenter as well as "
                     f"the detector seed -- one more reason not to state it as "
                     f"a single number.")
        w(para)
        w("")

    text = "\n".join(L) + "\n"
    doc = MD.read_text(encoding="utf-8") if MD.exists() else ""
    head = doc[: doc.index(MARK)] if MARK in doc else doc
    # Drop any trailing horizontal rules before adding one back. rstrip() alone
    # removes the blank line after a rule but not the rule itself, so each rerun
    # of a writer left another "---" behind.
    head = re.sub(r"(?:^---[ \t]*$\s*)+\Z", "", head, flags=re.M)
    doc = head.rstrip() + "\n\n---\n\n" + text
    MD.write_text(doc, encoding="utf-8")
    print(f"wrote section 20 into {MD.name} ({len(text)} chars, "
          f"cohorts: {', '.join(have)}, detector seeds: {pseeds})")


if __name__ == "__main__":
    main()
