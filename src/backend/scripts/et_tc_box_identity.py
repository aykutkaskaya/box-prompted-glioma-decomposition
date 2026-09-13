"""How often the tight box around ET equals the tight box around TC.

The enhancing-tumour cap is argued from geometry: a tight rectangle around a
rim is the tight rectangle around the disc it encloses. That argument was first
supported indirectly, by counting patients whose ET and TC predictions had equal
volume, and a referee pointed out that the reference segmentations answer it
directly and that the per-patient criterion was hiding the mechanism in the
held-out cohort. This measures it on the labels, with no model involved.

Writes reports/et_tc_box_identity.json.

    python scripts/et_tc_box_identity.py
"""
from pathlib import Path

_HERE = Path(__file__).resolve()
import sys, os, io
import numpy as np
B = str(_HERE.parents[1])   # <root>/src/backend
sys.path.insert(0, B)
sys.path.insert(0, B + "/scripts")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import full_validation as FV   # noqa
from full_validation import load_patient, region_mask   # noqa
from core.metrics import tight_bbox                      # noqa
D = _HERE.parents[3]        # <root>

OUT = {}


def run(cohort, listfile, raw=None):
    # full_validation binds RAW at import, so the env var alone does nothing
    FV.RAW = Path(raw) if raw else Path(D / "data/raw")
    pats = [x.strip() for x in io.open(D / listfile, encoding="utf-8") if x.strip()]
    sl_tot = sl_eq = 0
    iou = []
    pat_all = pat_n = 0
    for pid in pats:
        vols, seg = load_patient(pid)
        et, tc = region_mask(seg, "ET"), region_mask(seg, "TC")
        if not et.any():
            continue
        pat_n += 1
        every = True
        had = False
        for z in range(et.shape[2]):
            e, t = et[:, :, z], tc[:, :, z]
            if not (e.any() and t.any()):
                continue
            had = True
            sl_tot += 1
            be, bt = tight_bbox(e), tight_bbox(t)
            if be == bt:
                sl_eq += 1
            else:
                every = False
            xa, ya = max(be[0], bt[0]), max(be[1], bt[1])
            xb, yb = min(be[2], bt[2]), min(be[3], bt[3])
            inter = max(0, xb - xa) * max(0, yb - ya)
            ae = (be[2] - be[0]) * (be[3] - be[1])
            at = (bt[2] - bt[0]) * (bt[3] - bt[1])
            u = ae + at - inter
            iou.append(inter / u if u > 0 else 0.0)
        if had and every:
            pat_all += 1
    print(f"  {cohort:14s} slices {sl_tot:5d}  boxes equal {sl_eq:5d} "
          f"({100*sl_eq/sl_tot:5.1f}%)  mean IoU {np.mean(iou):.3f}   "
          f"equal on every slice {pat_all}/{pat_n}")
    OUT[cohort] = {"slices": sl_tot, "equal": sl_eq,
                   "pct": round(100 * sl_eq / sl_tot, 1),
                   "mean_box_iou": round(float(np.mean(iou)), 3),
                   "patients_all_slices": pat_all, "patients": pat_n}

run("BraTS2020", "reports/clean_cohort.txt")
run("RHUH-GBM", "data/external/rhuh/patients.txt", D / "data/external/rhuh")
run("BraTS-Africa", "data/external/brats_africa/patients.txt", D / "data/external/brats_africa")

import json
io.open(D / "reports/et_tc_box_identity.json", "w", encoding="utf-8").write(
    json.dumps(OUT, indent=1))
print("-> reports/et_tc_box_identity.json")
