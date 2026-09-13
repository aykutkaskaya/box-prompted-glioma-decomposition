"""Does the non-enhancing core actually lie inside the enhancing rim?

Section 4.9 explains the enhancing-tumour cap with a ring picture: fill the
enhancing ring and you get tumour core, so the two boxes coincide. A referee
pointed out that the picture is a grade-4 appearance rather than a general one,
and that it can be tested directly on the labels. It can, and it does not hold
everywhere -- which is why the section now separates the anatomy from the
collapse it is offered to explain.

Prints, per cohort, the share of slices carrying both ET and non-enhancing core
on which filling ET covers at least 95% of that core, and the median coverage.

    python scripts/et_ring_enclosure.py
"""
from pathlib import Path

_HERE = Path(__file__).resolve()
import sys, os, io
import numpy as np
from scipy import ndimage
B = str(_HERE.parents[1])   # <root>/src/backend
sys.path.insert(0, B); sys.path.insert(0, B + "/scripts")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import full_validation as FV
from full_validation import load_patient, region_mask
D = _HERE.parents[3]        # <root>

def run(lab, listfile, raw=None):
    FV.RAW = Path(raw) if raw else Path(D / "data/raw")
    pats = [x.strip() for x in io.open(D / listfile, encoding="utf-8") if x.strip()]
    cov, n95, tot = [], 0, 0
    eq_all = eq_diff = diff_tot = 0
    for pid in pats:
        _, seg = load_patient(pid)
        et = region_mask(seg, "ET"); tc = region_mask(seg, "TC")
        ncr = tc & ~et
        for z in range(et.shape[2]):
            e, nz = et[:, :, z], ncr[:, :, z]
            if not (e.any() and nz.any()):
                continue
            tot += 1
            filled = ndimage.binary_fill_holes(e)
            c = (filled & nz).sum() / nz.sum()
            cov.append(c)
            if c >= 0.95: n95 += 1
    print(f"  {lab:14s} ET+NCR dilim {tot:5d}  >=%95 kapsama {n95:5d} ({100*n95/tot:4.1f}%)  medyan {np.median(cov):.3f}")

run("BraTS2020", "reports/clean_cohort.txt")
run("RHUH-GBM", "data/external/rhuh/patients.txt", D / "data/external/rhuh")
run("BraTS-Africa", "data/external/brats_africa/patients.txt", D / "data/external/brats_africa")
