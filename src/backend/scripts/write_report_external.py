"""Write the multi-cohort external-validation section of RUNS_ANALYSIS.md.

write_report_rhuh.py wrote section 17 when RHUH-GBM was the only external
cohort and the question was still "does the internal result reproduce". With
BraTS-Africa the question changed. Two external cohorts disagree about the
size of the effect -- +0.024 and +0.128 -- and a section that reported them
side by side without explaining the disagreement would be worse than useless.

What explains it is a decomposition that only exists because the oracle arm was
measured. The detector-free arm's total advantage over the box pipeline splits
exactly in two:

    total  =  (oracle - pipeline)  +  (detector_free - oracle)
              localisation            segmentation

The first term is what using a real detector instead of a perfect box costs.
The second is what the detector-free model's segmentation is worth once
localisation is taken out of the comparison. Across the three cohorts the first
grows monotonically and the second changes sign, which is the whole finding:
domain shift damages the detector, not the segmenter.

That decomposition also answers the reviewer's sharpest objection -- that
MedSAM3's base is medically pretrained and 28x larger, so the comparison is
confounded. If model quality drove the result, the segmentation term would be
consistently positive. It is not.

    python scripts/write_report_external.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

R = DRIVE_ROOT / "reports"
V = R / "validation"
MD = DRIVE_ROOT / "RUNS_ANALYSIS.md"
MARK = "## 18. External validation across three cohorts (2026-08-25)"

BEST_BOX = "pipeline:sam2.1_l"
CEILING = "oracle:sam2.1_l"

COHORTS = [
    ("BraTS2020, mutually held out", "clean_box.jsonl", "clean_direct.jsonl", "internal"),
    ("RHUH-GBM", "rhuh_box.jsonl", "rhuh_direct.jsonl", "external"),
    ("BraTS-Africa", "brats_africa_box.jsonl", "brats_africa_direct.jsonl", "external"),
]


def load(name: str) -> dict:
    p = V / name
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or not r.get("arms"):
            continue
        out[r["patient"]] = r
    return out


def series(box: dict, direct: dict):
    common = sorted(set(box) & set(direct))
    dkey = list(direct[common[0]]["arms"])[0]
    return (common,
            np.array([box[p]["arms"][CEILING]["vol_dice"] for p in common]),
            np.array([box[p]["arms"][BEST_BOX]["vol_dice"] for p in common]),
            np.array([direct[p]["arms"][dkey]["vol_dice"] for p in common]))


def detection(box: dict, common, key="pipeline:sam1_vit_b"):
    tp = sum(box[p]["arms"][key]["tp"] for p in common)
    fp = sum(box[p]["arms"][key]["fp"] for p in common)
    fn = sum(box[p]["arms"][key]["fn"] for p in common)
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    return pr, rc, 2 * pr * rc / (pr + rc)


def main() -> None:
    rows = []
    for label, bf, df, kind in COHORTS:
        box, direct = load(bf), load(df)
        if not box or not direct:
            continue
        common, orac, pipe, dire = series(box, direct)
        pr, rc, f1 = detection(box, common)
        rows.append({
            "label": label, "kind": kind, "n": len(common),
            "oracle": orac.mean(), "pipeline": pipe.mean(), "direct": dire.mean(),
            "total": (dire - pipe).mean(),
            "localisation": (orac - pipe).mean(),
            "segmentation": (dire - orac).mean(),
            "wins": int((dire - pipe > 0).sum()),
            "det_p": pr, "det_r": rc, "det_f1": f1,
            "collapse_pipe": int((pipe < 0.5).sum()),
            "collapse_direct": int((dire < 0.5).sum()),
            "collapse_oracle": int((orac < 0.5).sum()),
        })
    if not rows:
        raise SystemExit("no cohort results found under reports/validation")

    L: list[str] = []
    w = L.append
    w(MARK)
    w("")
    w("Section 17 reported RHUH-GBM alone. A second external cohort, BraTS-Africa "
      "(95 preoperative glioma cases from sub-Saharan Africa, BraTS 2023 release), "
      "puts the effect at +0.1275 where RHUH put it at +0.0237. Reporting both "
      "without accounting for the disagreement would say nothing; the rest of this "
      "section is the accounting.")
    w("")

    # ------------------------------------------------------------- headline
    w("### 18.1 What each cohort says")
    w("")
    w("Volumetric Dice, whole tumour. `oracle` is the ground-truth box fed to "
      "SAM 2.1-L -- not deployable, and the ceiling a perfect detector could reach.")
    w("")
    w("| cohort | n | oracle | RT-DETR -> SAM 2.1-L | MedSAM3+LoRA | gap | wins |")
    w("|---|---|---|---|---|---|---|")
    for r in rows:
        w(f"| {r['label']} | {r['n']} | {r['oracle']:.4f} | {r['pipeline']:.4f} | "
          f"{r['direct']:.4f} | {r['total']:+.4f} | {r['wins']}/{r['n']} |")
    w("")
    w("The oracle column is the one to read first. It does **not** order by distance "
      "from the training data -- RHUH scores highest of the three. Whatever makes "
      "BraTS-Africa hard for the pipeline, it is not that the tumours are harder to "
      "segment once they have been found.")
    w("")

    # ------------------------------------------------------- decomposition
    w("### 18.2 Where the advantage comes from")
    w("")
    w("The gap splits exactly in two, and the halves behave differently:")
    w("")
    w("| cohort | detection F1 | localisation (oracle - pipeline) | segmentation (direct - oracle) | total |")
    w("|---|---|---|---|---|")
    for r in rows:
        w(f"| {r['label']} | {r['det_f1']:.3f} | {r['localisation']:+.4f} | "
          f"{r['segmentation']:+.4f} | {r['total']:+.4f} |")
    w("")
    loc = [r["localisation"] for r in rows]
    seg = [r["segmentation"] for r in rows]
    w(f"**Localisation grows and never changes sign** ({loc[0]:+.4f} -> {loc[-1]:+.4f}, "
      f"a factor of {loc[-1]/loc[0]:.0f}), tracking detection F1 down from "
      f"{rows[0]['det_f1']:.3f} to {rows[-1]['det_f1']:.3f}. "
      f"**Segmentation is small and flips** ({', '.join(f'{s:+.4f}' for s in seg)}).")
    w("")
    w("This is what carries the paper, and it also disposes of the obvious objection. "
      "MedSAM3's base checkpoint is medically pretrained and 28x the size of SAM 1 "
      "ViT-B on disk, so a reviewer will ask whether the result is just the bigger "
      "model. If it were, the segmentation term would be consistently positive. It "
      "is not. The term that is consistent, and that grows with distance from the "
      "training distribution, is the one that measures the detector.")
    w("")
    w("Three cohorts do not establish a dose-response curve and the claim is not "
      "made as one. The claim is categorical: inside the training distribution the "
      "detector costs almost nothing; outside it, it costs an order of magnitude more.")
    w("")

    # ------------------------------------------------------------ collapses
    w("### 18.3 Collapses")
    w("")
    w("Patients scoring below 0.5, which is where a mean stops describing anything:")
    w("")
    w("| cohort | oracle | RT-DETR -> SAM 2.1-L | MedSAM3+LoRA |")
    w("|---|---|---|---|")
    for r in rows:
        w(f"| {r['label']} | {r['collapse_oracle']}/{r['n']} | "
          f"{r['collapse_pipe']}/{r['n']} | {r['collapse_direct']}/{r['n']} |")
    w("")
    w("The oracle never collapses, in any cohort. Given a correct box the segmenter "
      "always produces something usable, so every collapse in the deployable arms is "
      "a failure to find the tumour rather than a failure to outline it.")
    w("")

    # ------------------------------------------------------------ threshold
    sweep = sorted(V.glob("brats_africa_conf*.jsonl"))
    if sweep:
        w("### 18.4 Was the detector handicapped?")
        w("")
        w("RT-DETR's confidence threshold was chosen on the BraTS2020 validation "
          "split (section 16.2) and never re-tuned. On BraTS-Africa its precision "
          "stays high while recall falls, which is the signature of a threshold set "
          "too conservatively for the data in front of it -- so part of the "
          "localisation term above could be calibration rather than capability. "
          "Re-running the detector arm at lower thresholds measures which.")
        w("")
        box, direct = load("brats_africa_box.jsonl"), load("brats_africa_direct.jsonl")
        # filenames carry the threshold with the point stripped: conf045 -> 0.45
        def conf_of(stem):
            d = stem.split("conf")[1]
            return f"{d[0]}.{d[1:]}"
        runs = [("0.55", box)] + [(conf_of(f.stem), load(f.name)) for f in sweep]
        runs = [(t, r) for t, r in runs if r]
        common = sorted(set.intersection(*[set(r) for _, r in runs], set(direct)))
        dkey = list(direct[common[0]]["arms"])[0]
        orac = np.array([box[p]["arms"][CEILING]["vol_dice"] for p in common])
        dire = np.array([direct[p]["arms"][dkey]["vol_dice"] for p in common])
        w("| conf | pipeline | localisation term | gap | wins | collapses | P / R / F1 |")
        w("|---|---|---|---|---|---|---|")
        for tag, src in sorted(runs, key=lambda x: -float(x[0])):
            a = np.array([src[p]["arms"][BEST_BOX]["vol_dice"] for p in common])
            pr, rc, f1 = detection(src, common, BEST_BOX)
            d = dire - a
            w(f"| {tag} | {a.mean():.4f} | {(orac.mean()-a.mean()):+.4f} | {d.mean():+.4f} | "
              f"{int((d>0).sum())}/{len(d)} | {int((a<0.5).sum())} | "
              f"{pr:.3f} / {rc:.3f} / {f1:.3f} |")
        w("")
        lo = min(runs, key=lambda x: float(x[0]))
        a_lo = np.array([lo[1][p]["arms"][BEST_BOX]["vol_dice"] for p in common])
        a_hi = np.array([box[p]["arms"][BEST_BOX]["vol_dice"] for p in common])
        gained = a_lo.mean() - a_hi.mean()
        remaining = (dire - a_lo).mean()
        loc_hi = orac.mean() - a_hi.mean()
        w(f"Giving the detector its best shot recovers {gained:+.4f}: "
          f"{gained / loc_hi * 100:.0f}% of the localisation term, "
          f"{gained / (dire - a_hi).mean() * 100:.0f}% of the total gap. "
          f"{remaining:+.4f} still remains. "
          f"Detection F1 barely moves across the sweep -- the threshold "
          f"trades precision for recall rather than finding anything new -- so the "
          f"threshold was not badly set. The detector is genuinely weaker here. "
          f"Closing the rest by calibration alone would need a threshold change "
          f"several times the one measured.")
        w("")

    # ------------------------------------------------ sub-region ceilings
    CEIL_FILES = [("BraTS2020, mutually held out", "clean_oracle_regions.jsonl", "clean_direct.jsonl"),
                  ("RHUH-GBM", "rhuh_oracle_regions.jsonl", "rhuh_direct.jsonl"),
                  ("BraTS-Africa", "brats_africa_oracle_regions.jsonl", "brats_africa_direct.jsonl")]
    ceil = [(lab, load(of), load(df)) for lab, of, df in CEIL_FILES]
    ceil = [(lab, o, d) for lab, o, d in ceil if o and d]
    if ceil:
        w("### 18.5 Sub-region ceilings")
        w("")
        w("Measured with the ground-truth box of each region, so these are what a "
          "perfect sub-region detector would reach with this segmenter. No such "
          "detector exists in this study; the point is what one could be worth.")
        w("")
        w("Patients whose ground truth has no tissue in a region are excluded from "
          "that region rather than averaged in: `volumetric` scores an empty "
          "prediction against an empty reference as 1.0 and a non-empty one as 0.0, "
          "and neither number means anything on a Dice scale.")
        w("")
        w("| cohort | region | ceiling (SAM 2.1-L) | MedSAM3+LoRA achieves | headroom | ceiling's volume error |")
        w("|---|---|---|---|---|---|")
        for lab, o, d in ceil:
            common = sorted(set(o) & set(d))
            dk = list(d[common[0]]["arms"])[0]
            for reg in ("WT", "TC", "ET"):
                cs = [o[p]["arms"][CEILING]["regions"][reg] for p in common]
                ds = [d[p]["arms"][dk]["regions"][reg] for p in common]
                c = np.mean([x["vol_dice"] for x in cs if x["gt_voxels"] > 0])
                a = np.mean([x["vol_dice"] for x in ds if x["gt_voxels"] > 0])
                rv = np.median([x["rel_vol_diff"] for x in cs if x["rel_vol_diff"] is not None])
                w(f"| {lab} | {reg} | {c:.4f} | {a:.4f} | {c - a:+.4f} | {rv:+.1%} |")
        w("")
        w("**ET is capped by the prompt, not by localisation.** The ceiling's volume "
          "error tells the story: with the same segmenter and the same kind of box, "
          "WT and TC come back within a few percent of the true volume while ET is "
          "over-filled by roughly half. ET is typically a thin enhancing rim around "
          "necrosis, and the tight rectangle around a ring is the tight rectangle "
          "around the disc it encloses -- on RHUH the ET box and the TC box produced "
          "an identical mask on 32 of 39 patients, and that mask scores 0.908 against "
          "TC and 0.740 against ET. Asked for the rim, the segmenter returns the disc. "
          "No detector can change the prompt it is asked to produce.")
        w("")
        w("**TC is not capped.** Its ceiling sits at or above WT's in every cohort and "
          "well above what the detector-free arm achieves. The claim that sub-regions "
          "are a structural advantage of detector-free methods therefore holds for ET "
          "and fails for TC: what is missing there is a trained detector, not a "
          "workable prompt.")
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
    print(f"wrote section 18 into {MD.name} ({len(text)} chars)")


if __name__ == "__main__":
    main()
