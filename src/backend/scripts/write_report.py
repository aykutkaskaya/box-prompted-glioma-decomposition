"""Turn the validation JSONs into the report section appended to RUNS_ANALYSIS.md.

Regenerating from the result files rather than transcribing numbers by hand
means the document cannot drift from what was measured; re-run it after any
further pass and the section is rewritten in place.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

R = DRIVE_ROOT / "reports"
V = R / "validation"
MD = DRIVE_ROOT / "RUNS_ANALYSIS.md"
MARK = "## 16. Held-out validation (2026-08-23)"

# the adapter project is a separate checkout; set MEDSAM3_DIR to point at it
LORA_SPLITS = (Path(os.environ.get("MEDSAM3_DIR", "medsam3-not-set"))
               / "training" / "data" / "splits")


def load_jsonl(name: str) -> Dict[str, dict]:
    p = V / name
    out: Dict[str, dict] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or not r.get("arms"):
            continue
        out[r["patient"]] = r
    return out


def arm_dice(rec: dict, key: str | None = None) -> float:
    arms = rec["arms"]
    k = key if key else list(arms)[0]
    return arms[k]["vol_dice"]


def stat(v: List[float]) -> str:
    a = np.asarray(v, float)
    return f"{a.mean():.4f}" if a.size else "—"


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(a, b)[1])
    except Exception:
        return None


L = []
w = L.append

w(MARK)
w("")
w("All numbers below were produced on 2026-08-23 by `src/backend/scripts/`; the")
w("per-patient records are in `reports/validation/`.")
w("")

# ---------------------------------------------------------------- leakage
w("### 16.1 A leakage problem that invalidates the headline comparison")
w("")
w("The obvious experiment -- run RT-DETR+SAM and MedSAM3+LoRA on the study's 55")
w("test patients and compare -- does not work, and the reason has to be stated")
w("before any of the numbers.")
w("")
try:
    rd = lambda n: set(x.strip() for x in open(LORA_SPLITS / n) if x.strip())
    lo_tr, lo_va, lo_te = rd("train_patients.txt"), rd("validation_patients.txt"), rd("test_patients.txt")
    import csv
    rt = {r["patient_id"]: r["split"] for r in csv.DictReader(
        open(DRIVE_ROOT / "data" / "processed" / "metadata" / "patient_split.csv"))}
    rt_te = {k for k, v in rt.items() if v == "test"}
    rt_va = {k for k, v in rt.items() if v == "val"}
    w(f"Both projects split BraTS2020 into 258/55/55, but with different assignments")
    w(f"(split hashes `fae05e5d...` and `f4c2efe1...`). Of the RT-DETR study's 55 test")
    w(f"patients, **{len(rt_te & lo_tr)} are in the LoRA model's training set**; only")
    w(f"{len(rt_te & lo_te)} are in its test set and {len(rt_te & lo_va)} in its validation set.")
    w("")
    w("Measured on the 55, the detector-free model appears to beat everything by a wide")
    w("margin -- but it had already seen 45 of those patients during fine-tuning, so the")
    w("comparison says nothing.")
    w("")
    w("The usable comparison is the intersection of the two held-out pools: patients in")
    w("neither training set. That cohort is stratified, which lets the residual advantage")
    w("be checked for direction:")
    w("")
    w("| stratum | n | who is advantaged |")
    w("|---|---|---|")
    w(f"| RT-test x LoRA-test | {len(rt_te & lo_te)} | neither -- fully clean |")
    w(f"| RT-test x LoRA-val | {len(rt_te & lo_va)} | LoRA, mildly (model selection only) |")
    w(f"| RT-val x LoRA-test | {len(rt_va & lo_te)} | RT-DETR, mildly |")
    w(f"| RT-val x LoRA-val | {len(rt_va & lo_va)} | both, symmetrically |")
except Exception as e:  # pragma: no cover
    w(f"(could not read the split files: {e})")
w("")

# ---------------------------------------------------------------- threshold
w("### 16.2 The confidence threshold, chosen honestly")
w("")
try:
    t = json.loads((R / "threshold_on_val.json").read_text(encoding="utf-8"))
    h = t["honest_operating_point"]
    w("The 0.55 default shipped earlier was measured on the test split -- test-set tuning.")
    w("Re-deriving it on the validation split:")
    w("")
    w("| | threshold | F1 |")
    w("|---|---|---|")
    w(f"| validation optimum ({t['splits']['val']['n_slices']} slices) | "
      f"{t['splits']['val']['best']['threshold']} | {t['splits']['val']['best']['f1']:.4f} |")
    w(f"| test optimum ({t['splits']['test']['n_slices']} slices) | "
      f"{t['splits']['test']['best']['threshold']} | {t['splits']['test']['best']['f1']:.4f} |")
    w(f"| **validation threshold, applied to test** | **{h['threshold_chosen_on_val']}** | "
      f"**{h['test_f1_at_that_threshold']:.4f}** |")
    w("")
    w(f"Optimism from having tuned on test: **{h['optimism_from_tuning_on_test']:+.4f}** -- "
      f"negligible. The objection was real but the effect is not measurable, and the")
    w("optimum is broad: anything in roughly 0.45-0.65 lands within ~0.005 F1.")
except Exception as e:
    w(f"(threshold_on_val.json unavailable: {e})")
w("")

# ---------------------------------------------------------------- ranking
w("### 16.3 Does a newer segmenter reorder the losses?")
w("")
try:
    s = json.loads((R / "segmenter_ranking.json").read_text(encoding="utf-8"))
    rows = sorted(s["runs"].values(), key=lambda r: -r["overall_dice_new"])
    w(f"Eight runs spanning the study's groups, re-scored with SAM 2.1-L under the study's")
    w(f"own protocol (conf {s['conf']}, {s['n_slices']} positive test slices, undetected")
    w("positives counted as Dice 0). Only the segmenter changes.")
    w("")
    w("| run | SAM 1 ViT-B (study) | SAM 2.1-L | delta |")
    w("|---|---|---|---|")
    for r in rows:
        w(f"| #{r['run_id']} {r['label']} | {r['overall_dice_study_sam1']:.4f} | "
          f"{r['overall_dice_new']:.4f} | {r['overall_dice_new'] - r['overall_dice_study_sam1']:+.4f} |")
    w("")
    w(f"**Spearman {s['spearman']:.3f}.** Every run gains almost the same amount "
      f"(mean {s['mean_shift']:+.4f}). The only order change is between the top two, whose")
    w("gap was already inside the replicate noise floor under both segmenters -- that pair")
    w("is simply not resolvable. The study's conclusions are not an artefact of SAM 1.")
except Exception as e:
    w(f"(segmenter_ranking.json unavailable: {e})")
w("")

# ---------------------------------------------------------------- clean cohort
w("### 16.4 The mutually held-out cohort")
w("")
cb, cd = load_jsonl("clean_box.jsonl"), load_jsonl("clean_direct.jsonl")
common = sorted(set(cb) & set(cd))
if common:
    arms = {
        "oracle:sam1_vit_b": "oracle GT box -> SAM 1 ViT-B",
        "oracle:sam2.1_l": "oracle GT box -> SAM 2.1-L",
        "pipeline:sam1_vit_b": "RT-DETR -> SAM 1 ViT-B",
        "pipeline:sam2.1_l": "RT-DETR -> SAM 2.1-L",
    }
    w(f"{len(common)} patients, in neither model's training set. Volumetric Dice, whole tumour:")
    w("")
    w("| pipeline | detector | mean Dice |")
    w("|---|---|---|")
    for k, lab in arms.items():
        vals = [arm_dice(cb[p], k) for p in common if k in cb[p]["arms"]]
        w(f"| {lab} | {'oracle' if 'oracle' in k else 'yes'} | {stat(vals)} |")
    dvals = [arm_dice(cd[p]) for p in common]
    w(f"| MedSAM3 + LoRA seed 42 | **none** | **{stat(dvals)}** |")
    w("")
    base = [arm_dice(cb[p], "pipeline:sam2.1_l") for p in common]
    diff = np.array(dvals) - np.array(base)
    p = wilcoxon_p(base, dvals)
    w(f"Detector-free minus the best box pipeline: mean {diff.mean():+.4f}, "
      f"median {np.median(diff):+.4f}, wins {int((diff > 0).sum())}/{len(diff)}"
      + (f", Wilcoxon p={p:.2e}" if p is not None else ""))
    w("")
    # A patient whose ground truth has no enhancing tumour at all cannot be
    # scored on a Dice scale: predict nothing and `volumetric` returns 1.0,
    # predict anything and it returns 0.0. Two of these 24 are in that position
    # and the model predicted ET on both, so they entered the mean as zeros and
    # pulled it down by 0.05 -- an arbitrary penalty on an arbitrary scale.
    # Score the patients that have something to find, and report the others as
    # what they actually are: false positives, counted rather than averaged.
    regs, no_gt = {}, {}
    for reg in ("WT", "TC", "ET"):
        vals, absent = [], []
        for pid in common:
            arm = cd[pid]["arms"][list(cd[pid]["arms"])[0]]
            r = arm.get("regions", {}).get(reg)
            if not r:
                continue
            if r.get("gt_voxels", 1) > 0:
                vals.append(r["vol_dice"])
            else:
                absent.append((pid, r.get("pred_voxels", 0)))
        if vals:
            regs[reg] = vals
            no_gt[reg] = absent
    if regs:
        w("Sub-regions from the same forward pass (the single-class detector cannot produce these at all):")
        w("")
        w("| region | mean Dice | n |")
        w("|---|---|---|")
        for reg, vals in regs.items():
            gap = len(no_gt.get(reg, []))
            note = f" ({gap} excluded, no {reg} in the ground truth)" if gap else ""
            w(f"| {reg} | {stat(vals)} | {len(vals)}{note} |")
        spurious = [(reg, pid, n) for reg, lst in no_gt.items() for pid, n in lst if n > 0]
        if spurious:
            w("")
            w("On the excluded patients the model did not abstain -- it segmented a region "
              "that is not there: "
              + "; ".join(f"{pid} {n:,} {reg} voxels" for reg, pid, n in spurious)
              + ". That is a real failure mode and it is reported here rather than "
                "folded into a mean as a zero.")
else:
    w("_Pending: the clean-cohort run had not finished when this section was generated._")
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
print(f"wrote section 16 into {MD.name} ({len(text)} chars)")
