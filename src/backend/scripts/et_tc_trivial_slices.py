from pathlib import Path

_HERE = Path(__file__).resolve()
import sys, os, io
import numpy as np
B = str(_HERE.parents[1])   # <root>/demov2/backend
sys.path.insert(0, B); sys.path.insert(0, B + "/scripts")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import full_validation as FV
from full_validation import load_patient, region_mask
from core.metrics import tight_bbox
D = _HERE.parents[3]        # <root>

def run(lab, lf, raw=None):
    FV.RAW = Path(raw) if raw else Path(D / "data/raw")
    pats = [x.strip() for x in io.open(D / lf, encoding="utf-8") if x.strip()]
    tot = eq = nonecr = eq_ncr = tot_ncr = 0
    for pid in pats:
        _, seg = load_patient(pid)
        et, tc = region_mask(seg, "ET"), region_mask(seg, "TC")
        ncr = tc & ~et
        for z in range(et.shape[2]):
            e, t, n = et[:, :, z], tc[:, :, z], ncr[:, :, z]
            if not (e.any() and t.any()):
                continue
            tot += 1
            same = tight_bbox(e) == tight_bbox(t)
            eq += same
            if not n.any():
                nonecr += 1
            else:
                tot_ncr += 1; eq_ncr += same
    print(f"  {lab:14s} dilim {tot:5d}  esit {100*eq/tot:5.1f}%  |  nekrozsuz {nonecr:4d} ({100*nonecr/tot:4.1f}%)  "
          f"nekrozlu dilimlerde esit {100*eq_ncr/tot_ncr:5.1f}% (n={tot_ncr})")

run("BraTS2020", "reports/clean_cohort.txt")
run("RHUH-GBM", "data/external/rhuh/patients.txt", D / "data/external/rhuh")
run("BraTS-Africa", "data/external/brats_africa/patients.txt", D / "data/external/brats_africa")
