"""Can the box-prompted pipeline do BraTS sub-regions at all?

The deployable box arm is single-class: RT-DETR emits one tumour box per slice
and SAM turns it into one mask, whole tumour. The detector-free arm also returns
TC and ET. A reviewer will ask whether a multi-class detector would close that
gap, and the answer is not a matter of opinion -- it is the oracle ceiling.
full_validation.py --oracle-regions WT,TC,ET skips the detector, takes the tight
box of the ground-truth region itself and scores what SAM returns from it. No
detector, however good, can beat that.

This reports the ceiling next to what the detector-free arm actually achieved,
because the ceiling alone says nothing: TC at 0.95 and ET at 0.33 mean very
different things about whether the detector is worth building.

Empty ground truth is excluded, not averaged. full_validation.volumetric scores
an empty prediction against an empty ground truth as Dice 1.0, and a patient
with no enhancing component would post a perfect ET score for segmenting
nothing. Same rule and same reporting as compile_external.py.

    python scripts/compile_oracle_regions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

R = DRIVE_ROOT / "reports"
V = R / "validation"
OUT = R / "oracle_regions_summary.json"

REGIONS = ("WT", "TC", "ET")
SEGMENTERS = ["sam1_vit_b", "sam2.1_l"]
SEG_LABEL = {"sam1_vit_b": "SAM 1 ViT-B", "sam2.1_l": "SAM 2.1-L"}

# The detector-free regions are derived here for every cohort, from the file
# named in "direct". RHUH-GBM used to read rhuh_summary.json instead, on the
# reasoning that quoting the section's own file beats a second derivation that
# might disagree by a rounding. That file predates the preprocessing
# correction, so the entry ended up declaring the corrected direct file and
# reporting the superseded numbers beside it -- WT 0.8508 where the corrected
# run gives 0.9314, which flips the sign of the ET headroom for anyone reading
# the released summary. A derivation that follows the declared file cannot
# drift from it.
COHORTS = [
    {"key": "rhuh", "name": "RHUH-GBM (external, another hospital)",
     "oracle": "rhuh_oracle_regions_normfix.jsonl",
     "direct": "rhuh_direct_seed42_bgfix.jsonl",
     "summary": None},
    {"key": "brats_africa", "name": "BraTS-Africa (external, another continent)",
     "oracle": "brats_africa_oracle_regions.jsonl",
     "direct": "brats_africa_direct.jsonl",
     "summary": None},
    {"key": "clean", "name": "BraTS2020 mutually held-out (internal)",
     "oracle": "clean_oracle_regions.jsonl", "direct": "clean_direct.jsonl",
     "summary": None},
]


def load(name: str) -> Dict[str, dict]:
    p = V / name
    if not p.exists():
        raise SystemExit(f"missing {p}")
    out: Dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or not r.get("arms"):
            continue
        out[r["patient"]] = r
    return out


def describe(vals: List[float]) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"n": int(v.size), "mean": round(float(v.mean()), 4),
            "median": round(float(np.median(v)), 4),
            "sd": round(float(v.std(ddof=1)), 4) if v.size > 1 else 0.0,
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}


def score(records: Dict[str, dict], patients: List[str], arm_key: str, region: str):
    """Per-patient results for one region, keeping only the patients that have it.

    Returns pairs rather than bare numbers because the ET/TC diagnostic below has
    to line the two regions up patient by patient: TC and ET are scored on
    different patients whenever someone has no enhancing component, so matching
    them by list position would be wrong exactly when it mattered.
    """
    got = [(p, records[p]["arms"][arm_key]["regions"][region]) for p in patients
           if region in records[p]["arms"].get(arm_key, {}).get("regions", {})]
    scored = [(p, r) for p, r in got if r.get("gt_voxels", 1) > 0]
    return scored, len(got) - len(scored)


def et_returns_tc(oracle: Dict[str, dict], patients: List[str], arm_key: str) -> dict:
    """Is the ET box being answered with the filled disc, i.e. with TC?

    This is the check that separates the two readings of a low ET ceiling. The
    masks were not kept -- only their scores -- so Dice(ET-box prediction, TC)
    cannot be computed directly. What can be computed is stronger than nothing
    and nearly as good: when the ET box and the TC box produce the same predicted
    volume on the same slices, they produced the same mask, and that mask's Dice
    against TC and against ET are both already recorded. Comparing those two on
    exactly those patients answers the question without re-running anything.
    """
    et = dict(score(oracle, patients, arm_key, "ET")[0])
    tc = dict(score(oracle, patients, arm_key, "TC")[0])
    both = [p for p in et if p in tc]
    same = [p for p in both
            if et[p]["pred_voxels"] == tc[p]["pred_voxels"]
            and et[p]["n_slices"] == tc[p]["n_slices"]]
    out = {"n": len(both), "et_prediction_is_the_tc_prediction": len(same),
           "mean_rel_vol_diff_et": round(float(np.mean(
               [et[p]["rel_vol_diff"] for p in both if et[p]["rel_vol_diff"] is not None])), 4)}
    if same:
        out.update({
            "on_those_patients": {
                "dice_vs_ET": round(float(np.mean([et[p]["vol_dice"] for p in same])), 4),
                "dice_vs_TC": round(float(np.mean([tc[p]["vol_dice"] for p in same])), 4)}})
    return out


def main() -> None:
    summary = {}
    for c in COHORTS:
        oracle, direct = load(c["oracle"]), load(c["direct"])
        patients = sorted(set(oracle) & set(direct))
        if not patients:
            raise SystemExit(f"{c['key']}: no patient carries both arms")
        print(f"\n=== {c['name']} ===")
        print(f"{c['oracle']}: {len(oracle)} patients | {c['direct']}: {len(direct)} "
              f"-> {len(patients)} paired")

        rows, no_gt, collapse = {}, {}, {}
        for sid in SEGMENTERS:
            key = f"oracle:{sid}"
            rows[sid] = {}
            for region in REGIONS:
                scored, missing = score(oracle, patients, key, region)
                if not scored:
                    continue
                rows[sid][region] = describe([r["vol_dice"] for _, r in scored])
                # Volume error is what makes a low Dice legible. WT and TC come
                # back near zero; if ET is the same segmenter over-filling the
                # same kind of box, ET is where it goes positive and stays there.
                rows[sid][region]["mean_rel_vol_diff"] = round(float(np.mean(
                    [r["rel_vol_diff"] for _, r in scored
                     if r["rel_vol_diff"] is not None])), 4)
                no_gt[region] = missing
            collapse[sid] = et_returns_tc(oracle, patients, key)

        # The detector-free arm is keyed by adapter, which may vary by run.
        dkey = list(direct[patients[0]]["arms"])[0]
        if c["summary"]:
            free = json.loads((R / c["summary"]).read_text(encoding="utf-8"))["regions"]
            free_src = c["summary"]
        else:
            free = {}
            for region in REGIONS:
                scored, missing = score(direct, patients, dkey, region)
                if scored:
                    free[region] = describe([r["vol_dice"] for _, r in scored])
            free_src = c["direct"]

        print(f"\n{'region':<8}{'arm':<34}{'mean':>8}{'median':>9}{'sd':>8}{'n':>5}"
              f"{'no GT':>7}{'vol err':>9}")
        for region in REGIONS:
            for sid in SEGMENTERS:
                s = rows[sid].get(region)
                if not s:
                    continue
                print(f"{region:<8}{'oracle GT box -> ' + SEG_LABEL[sid]:<34}"
                      f"{s['mean']:>8.4f}{s['median']:>9.4f}{s['sd']:>8.4f}{s['n']:>5}"
                      f"{no_gt.get(region, 0):>7}{s['mean_rel_vol_diff']:>+9.2f}")
            f = free.get(region)
            if f:
                print(f"{region:<8}{'MedSAM3+LoRA (no detector)':<34}"
                      f"{f['mean']:>8.4f}{f['median']:>9.4f}{f['sd']:>8.4f}{f['n']:>5}"
                      f"{'':>7}")

        print("\nwhat the ET box actually returns:")
        for sid in SEGMENTERS:
            d = collapse[sid]
            print(f"  {SEG_LABEL[sid]:<12}{d['et_prediction_is_the_tc_prediction']}/{d['n']} "
                  f"patients where the ET box produced the same mask as the TC box; "
                  f"ET volume error {d['mean_rel_vol_diff_et']:+.2f}")
            t = d.get("on_those_patients")
            if t:
                print(f"{'':<14}on those patients that mask scores "
                      f"{t['dice_vs_TC']:.4f} against TC and {t['dice_vs_ET']:.4f} against ET")

        summary[c["key"]] = {
            "name": c["name"], "oracle_file": c["oracle"], "direct_file": c["direct"],
            "n_patients": len(patients), "patients": patients,
            "oracle": rows, "patients_without_region_gt": no_gt,
            "detector_free": {"source": free_src, "arm": dkey, "regions": free},
            "et_box_returns_tc": collapse,
        }

    OUT.write_text(json.dumps({"cohorts": summary, "regions": list(REGIONS),
                               "segmenters": SEGMENTERS}, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
