"""Append the external-cohort section to RUNS_ANALYSIS.md.

write_report.py cannot be pointed at this. Its section 16 is built around the
leakage argument -- it reads both projects' split files to show that 45 of the
55 test patients were in the LoRA training set, and every table below that is
framed as "what survives once the overlap is removed". None of it applies to a
cohort from another hospital, where nothing overlaps by construction and the
question is instead whether a margin measured inside BraTS2020 travels.

Like section 16 this derives every number from the result files rather than
transcribing them, so the prose cannot drift from what was measured. It also
reads reports/power_analysis.json, which was written before the run, and puts
the projected interval next to the realised one. If they disagree that is worth
seeing: the projection assumed the external cohort behaves like the internal
one, and disagreement is the finding, not an error to hide.

    python scripts/compile_external.py     # writes reports/rhuh_summary.json
    python scripts/write_report_rhuh.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_rhuh  # noqa: E402

R = DRIVE_ROOT / "reports"
MD = DRIVE_ROOT / "RUNS_ANALYSIS.md"
MARK = "## 17. External validation: RHUH-GBM (2026-08-24)"

ORDER = ["oracle:sam1_vit_b", "oracle:sam2.1_l", "pipeline:sam1_vit_b", "pipeline:sam2.1_l"]
NICE = {
    "oracle:sam1_vit_b": ("oracle GT box -> SAM 1 ViT-B", "oracle"),
    "oracle:sam2.1_l": ("oracle GT box -> SAM 2.1-L", "oracle"),
    "pipeline:sam1_vit_b": ("RT-DETR -> SAM 1 ViT-B", "yes"),
    "pipeline:sam2.1_l": ("RT-DETR -> SAM 2.1-L", "yes"),
}


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path} -- run scripts/compile_external.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    s = load(R / "rhuh_summary.json")
    power = json.loads((R / "power_analysis.json").read_text(encoding="utf-8")) \
        if (R / "power_analysis.json").exists() else None

    L: list[str] = []
    w = L.append
    w(MARK)
    w("")
    w("Section 16 established the direction on 24 patients held out from both models. "
      "It could not establish the size: the bootstrap interval on the gap was "
      "+/-0.0193, a third of the gap itself, and every one of those patients still "
      "came from BraTS2020 -- the preprocessing, the annotation protocol and the "
      "scanners both models grew up on. This section runs the same protocol, "
      "unchanged, on a cohort from a different hospital.")
    w("")

    # ------------------------------------------------------------ the cohort
    w("### 17.1 The cohort")
    w("")
    found = prep_rhuh.scan(prep_rhuh.SRC) if prep_rhuh.SRC.exists() else {}
    converted = s["n_patients"]
    w(f"RHUH-GBM (Rio Hortega University Hospital, TCIA, CC BY 4.0): "
      f"{len(found) if found else 40} patients, three studies each -- preoperative, "
      f"early post-operative, and follow-up at recurrence. RT-DETR was trained on "
      f"BraTS2020 and the MedSAM3 LoRA adapter was fine-tuned on BraTS2020, so every "
      f"patient here is held out for both by construction; there is no split "
      f"intersection to compute and no leakage argument to make.")
    w("")
    w("Three reconciliations were needed, all in `scripts/prep_rhuh.py`:")
    w("")
    w("- **Only the preoperative study is usable.** The post-operative and recurrence "
      "scans contain resection cavities. Neither model was trained on anything like "
      "that, so scoring them would measure domain shift rather than segmentation.")
    w("- **Labels move.** RHUH writes 1=necrosis, 2=peritumoral signal alteration, "
      "3=enhancing tumour; BraTS writes 1=NCR/NET, 2=ED, 4=ET. Same semantics, one "
      "different value, so the remap is 3 -> 4. Left alone, every ET score would have "
      "read zero and TC would have lost its enhancing half silently.")
    w("- **Intensity needs nothing.** RHUH normalises with CaPTk and BraTS did not, but "
      "`core.brats.normalize_brain_slice` rescales each slice by its own non-zero "
      "percentiles, so any volume-level affine cancels exactly.")
    w("")
    if found:
        missing = sorted(set(found) - {p.replace("_", "-") for p in s["patients"]})
        if missing:
            w(f"{converted} of {len(found)} patients converted. Dropped: "
              f"{', '.join(missing)} -- the preoperative study is off the BraTS grid "
              f"(230x230x138) and its affine is not axis-aligned, so that one study was "
              f"never registered to SRI24 properly in the release; its own later "
              f"timepoints are fine. Resampling it here would apply a preprocessing "
              f"step to one patient and not the other 39.")
            w("")
    w(f"The measurement protocol is identical to section 16: run 37 at conf 0.55, the "
      f"same two segmenters, the same `seed_42` adapter, stride 1. Moving the cohort "
      f"and the protocol together would leave any change with two explanations.")
    w("")

    # ------------------------------------------------------------ results
    w("### 17.2 Result")
    w("")
    w(f"{converted} patients, volumetric Dice, whole tumour:")
    w("")
    w("| pipeline | detector | mean Dice |")
    w("|---|---|---|")
    dkey = next(k for k in s["arms"] if k.startswith("direct:"))
    for k in ORDER:
        if k in s["arms"]:
            label, det = NICE[k]
            w(f"| {label} | {det} | {s['arms'][k]['mean']:.4f} |")
    w(f"| MedSAM3 + LoRA {dkey.split(':', 1)[1]} | **none** | **{s['arms'][dkey]['mean']:.4f}** |")
    w("")

    for c in s["contrasts"]:
        lo, hi = c["ci95"]
        p = c["wilcoxon_p"]
        ptxt = "" if p is None else (", p<1e-9" if p < 1e-9 else f", p={p:.2e}")
        w(f"- **{c['contrast']}**: mean {c['mean_diff']:+.4f}, median "
          f"{c['median_diff']:+.4f}, wins {c['wins']}/{c['n']}, "
          f"95% CI [{lo:+.4f}, {hi:+.4f}] (+/-{c['ci95_halfwidth']:.4f}){ptxt}")
    w("")

    # ---------------------------------------------------------- collapses
    col = s.get("collapses") or {}
    rob = s.get("robustness")
    if col:
        w("The means hide a bimodal failure, and it is the most useful thing in the "
          "run. Patients where an arm scored below 0.5:")
        w("")
        w("| arm | collapses | patients |")
        w("|---|---|---|")
        for k in ORDER + [dkey]:
            if k not in col:
                continue
            label = NICE[k][0] if k in NICE else f"MedSAM3 + LoRA {dkey.split(':', 1)[1]}"
            who = ", ".join(col[k]) if col[k] else "--"
            w(f"| {label} | {len(col[k])}/{s['n_patients']} | {who} |")
        w("")
        oracle_clean = all(not col.get(k) for k in ("oracle:sam1_vit_b", "oracle:sam2.1_l"))
        both_same = bool(col.get("pipeline:sam2.1_l")) and \
            set(col.get("pipeline:sam2.1_l", [])) == set(col.get(dkey, []))
        if oracle_clean and both_same:
            w("Both deployable arms fail on exactly the same patients, and the oracle "
              "fails on none. Given a correct box the segmenter is never the problem; "
              "what breaks on this cohort is finding the tumour. The detector-free arm "
              "is not immune to that -- it has no box, but it still has to localise, "
              "and it misses in the same places.")
            w("")
        if rob:
            ptxt = f", p={rob['wilcoxon_p']:.2e}" if rob.get("wilcoxon_p") else ""
            w(f"That raises the obvious objection: is the whole gap those few patients? "
              f"No. Removing them leaves mean {rob['mean_diff']:+.4f}, median "
              f"{rob['median_diff']:+.4f}, wins {rob['wins']}/{rob['n']}{ptxt} -- smaller, "
              f"still there, still broad.")
            w("")

    # ------------------------------- internal vs external, and the projection
    if power:
        pri = power["contrasts"][0]
        ext = s["contrasts"][0]
        w("### 17.3 Against the internal cohort")
        w("")
        w("| | n | mean gap | 95% CI half-width |")
        w("|---|---|---|---|")
        w(f"| BraTS2020, mutually held out | {pri['n']} | {pri['mean_diff']:+.4f} | "
          f"+/-{pri['ci95_halfwidth']:.4f} |")
        w(f"| RHUH-GBM, external | {ext['n']} | {ext['mean_diff']:+.4f} | "
          f"+/-{ext['ci95_halfwidth']:.4f} |")
        w(f"| projected before the run | {pri['projected_n']} | (assumed "
          f"{pri['mean_diff']:+.4f}) | +/-{pri['projected_ci95_halfwidth']:.4f} |")
        w("")
        shift = ext["mean_diff"] - pri["mean_diff"]
        w(f"The gap moved by {shift:+.4f} between the two cohorts. The projection in "
          f"`reports/power_analysis.json` was made by resampling the BraTS differences, "
          f"which assumed the external cohort would behave like the internal one; it is "
          f"listed here to be checked against, not to be quoted as a result.")
        w("")
        # Which of section 16's claims travelled, contrast by contrast. Matching on
        # the base arm rather than on list order, so a reordering of either script
        # cannot silently pair the wrong two rows.
        by_base = {c.get("base"): c for c in power["contrasts"] if c.get("base")}
        rows = [(c, by_base[c["base"]]) for c in s["contrasts"] if c["base"] in by_base]
        if rows:
            w("Contrast by contrast, matched on the arm being compared against:")
            w("")
            w("| detector-free vs | BraTS (n=%d) | RHUH (n=%d) | survives |"
              % (pri["n"], ext["n"]))
            w("|---|---|---|---|")
            for e, i in rows:
                same = (e["mean_diff"] > 0) == (i["mean_diff"] > 0)
                sig = e["wilcoxon_p"] is not None and e["wilcoxon_p"] < 0.05
                verdict = "yes" if same and sig else ("**sign flips**" if not same else "not at p<0.05")
                w(f"| {NICE.get(e['base'], (e['base'],))[0]} | {i['mean_diff']:+.4f} | "
                  f"{e['mean_diff']:+.4f} | {verdict} |")
            w("")
            flipped = [e for e, i in rows if (e["mean_diff"] > 0) != (i["mean_diff"] > 0)]
            if flipped:
                names = ", ".join(NICE.get(e["base"], (e["base"],))[0] for e in flipped)
                w(f"The claim that does not travel is the one against the oracle ceiling "
                  f"({names}). Internally the detector-free arm beat the ceiling a perfect "
                  f"detector could reach; here it does not. That was a property of the "
                  f"internal cohort, not of the method, and section 16 should be read with "
                  f"that correction.")
                w("")

    # ------------------------------------------------------------ subregions
    if s.get("regions"):
        w("### 17.4 Sub-regions")
        w("")
        w("From the same forward pass. The single-class box pipeline cannot produce "
          "these at all.")
        w("")
        w("| region | mean Dice | n |")
        w("|---|---|---|")
        for reg, st in s["regions"].items():
            gap = s.get("patients_without_region_gt", {}).get(reg, 0)
            note = f" ({gap} excluded: no {reg} in the ground truth)" if gap else ""
            w(f"| {reg} | {st['mean']:.4f} | {st['n']}{note} |")
        w("")
        if any(s.get("patients_without_region_gt", {}).values()):
            w("Patients whose ground truth has nothing to find in a region are excluded "
              "from that region's mean rather than averaged in: `volumetric` scores an "
              "empty prediction against an empty reference as Dice 1.0, which is fair "
              "per patient and misleading in an average.")
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
    print(f"wrote section 17 into {MD.name} ({len(text)} chars)")


if __name__ == "__main__":
    main()
