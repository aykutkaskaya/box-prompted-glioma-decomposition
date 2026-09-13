"""What shape is enhancing tumour, measured on the reference segmentations.

Section 4.9 used to explain the enhancing-tumour cap with a ring picture: an
enhancing rim around a necrotic core, so that filling the ring gives tumour
core. A referee pointed out that this is testable on the labels, and it does not
hold -- the enhancing mask fully encloses the non-enhancing core on 3% of the
held-out cohort's enhancing slices. What does hold is that enhancing tumour is
most of the core by volume, and that it is frequently not one connected object,
which no single axis-aligned box can express regardless of placement.

Prints, per cohort: slices carrying enhancing tumour, the share on which it
fully encloses the non-enhancing core, the share on which it has more than one
and at least three connected components, the median fill of its own tight box,
and the median enhancing-to-core volume ratio.

    python scripts/et_geometry.py
"""
from pathlib import Path

_HERE = Path(__file__).resolve()
import sys, os, io
import numpy as np
from scipy import ndimage
B = str(_HERE.parents[1])   # <root>/demov2/backend
sys.path.insert(0, B); sys.path.insert(0, B + "/scripts")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import full_validation as FV
from full_validation import load_patient, region_mask
D = _HERE.parents[3]        # <root>
ST = np.ones((3, 3), int)

def run(lab, lf, raw=None):
    FV.RAW = Path(raw) if raw else Path(D / "data/raw")
    pats = [x.strip() for x in io.open(D / lf, encoding="utf-8") if x.strip()]
    encl = tot = multi = ge3 = 0
    fills, ratios = [], []
    for pid in pats:
        _, seg = load_patient(pid)
        et, tc = region_mask(seg, "ET"), region_mask(seg, "TC")
        ncr = tc & ~et
        if et.sum() and tc.sum():
            ratios.append(et.sum() / tc.sum())
        for z in range(et.shape[2]):
            e = et[:, :, z]
            if not e.any():
                continue
            tot += 1
            n, lab_n = ndimage.label(e, structure=ST)[1], ndimage.label(e, structure=ST)[1]
            if n > 1: multi += 1
            if n >= 3: ge3 += 1
            ys, xs = np.where(e)
            area = (ys.max()-ys.min()+1) * (xs.max()-xs.min()+1)
            fills.append(e.sum() / area)
            nz = ncr[:, :, z]
            if nz.any():
                filled = ndimage.binary_fill_holes(e)
                if (nz & ~(filled & ~e)).sum() == 0:
                    encl += 1
    print(f"  {lab:14s} ET dilim {tot:5d}  tam sarma {100*encl/tot:5.2f}%  "
          f"cok bilesenli {100*multi/tot:5.1f}%  >=3 {100*ge3/tot:5.1f}%  "
          f"kutu doluluk medyan {np.median(fills):.3f}  ET/TC medyan {np.median(ratios):.3f}")

run("BraTS2020", "reports/clean_cohort.txt")
run("RHUH-GBM", "data/external/rhuh/patients.txt", D / "data/external/rhuh")
run("BraTS-Africa", "data/external/brats_africa/patients.txt", D / "data/external/brats_africa")
