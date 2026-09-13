"""Turn the RHUH-GBM release into something full_validation.py can read.

RHUH-GBM is an external glioblastoma cohort from Rio Hortega University
Hospital (TCIA, CC BY 4.0). Neither model in this study has seen it: the RT-DETR
detector and the MedSAM3 LoRA adapter were both trained on BraTS2020 only. That
is the point -- the mutually held-out BraTS cohort is n=24, strong enough to
call the direction but not the effect size.

Three things have to be reconciled before our loader can touch it:

  layout     RHUH ships one directory per patient with three timepoints in it.
             Only the preoperative study is comparable: the post-op and
             recurrence scans contain resection cavities, and neither model was
             trained on anything like that. Passing them through would measure
             domain shift, not segmentation quality.

  labels     RHUH uses 1=necrosis, 2=peritumoral signal alteration,
             3=enhancing tumour. BraTS uses 1=NCR/NET, 2=ED, 4=ET. Same
             semantics, different value for the enhancing class, so the remap
             is 3 -> 4 and the rest stay put. Without it every ET score would
             read zero and TC would silently lose its enhancing half.

  geometry   RHUH is registered to SRI24 with FLIRT and skull-stripped with
             Synthstrip, so it should already sit on the BraTS grid. Should is
             not is: this script refuses to convert a volume whose shape does
             not match, rather than reporting nonsense later.

Intensity scaling needs no attention. RHUH normalises with CaPTk and BraTS did
something else, but core.brats.normalize_brain_slice rescales each slice by its
own non-zero percentiles, so any volume-level affine cancels out exactly.

    python scripts/prep_rhuh.py --inspect     # report the layout, write nothing
    python scripts/prep_rhuh.py               # convert -> data/external/rhuh
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

DRIVE = Path(__file__).resolve().parents[3]   # <root>
# The faspex package unpacks with two wrapper directories above the cohort.
SRC = (DRIVE / "data" / "external" / "rhuh_raw" /
       "PKG - RHUH-GBM-nii-v1" / "RHUH-GBM_nii_v1")
DST = DRIVE / "data" / "external" / "rhuh"
BRATS_SHAPE = (240, 240, 155)

# RHUH label value -> BraTS label value. 3 (enhancing) is the only one that moves.
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 4}

# Which of our modality names each filename fragment maps to. Order matters
# twice over. t1ce/t1c/gd must be tested before the bare t1, or every contrast
# scan is filed as a plain T1. And flair must be tested before t2, because
# BraTS 2023 renamed FLAIR to "t2f" -- checked in the other order, every FLAIR
# volume would be read as a T2, FLAIR would come back missing, and the whole
# cohort would be skipped as incomplete.
#
# Two naming conventions are in play. RHUH-GBM follows BraTS 2020
# (_flair, _t1ce, _t2, _t1); the BraTS 2023 releases, including BraTS-Africa,
# use -t2f, -t1c, -t2w, -t1n. Both are matched here so one converter serves
# both, and hyphen or underscore is immaterial since only the fragment matters.
MODALITY_PATTERNS = [
    ("flair", (r"flair", r"_fl(?![a-z])", r"t2f(?![a-z])")),
    ("t1ce", (r"t1ce", r"t1c(?![a-z])", r"t1_?gd", r"t1_?post", r"ce_?t1")),
    ("t2", (r"t2(?![a-z*])", r"t2w(?![a-z])")),
    ("t1", (r"t1(?![a-z0-9])", r"t1n(?![a-z])")),
    ("seg", (r"seg", r"label", r"gt(?![a-z])", r"tumou?r")),
    ("adc", (r"adc", r"dwi")),
]
NEEDED = ("flair", "t1ce", "t2", "seg")

# RHUH nests each study under a bare index directory: 0 preoperative, 1 early
# post-op, 2 follow-up at recurrence. Keep the three apart even though only the
# first is used -- collapsing 1 and 2 into one bucket would have them overwrite
# each other in the scan, and the inspect report would then account for 480 of
# the 720 files while looking complete.
INDEX_NAMES = {"0": "preop", "1": "postop", "2": "followup"}
WORD_HINTS = ((r"pre[-_ ]?op|preoperative", "preop"),
              (r"post[-_ ]?op|early", "postop"),
              (r"follow|recurren", "followup"))


def classify_modality(name: str) -> str | None:
    low = name.lower()
    for mod, pats in MODALITY_PATTERNS:
        if any(re.search(p, low) for p in pats):
            return mod
    return None


def timepoint_of(parts: tuple[str, ...]) -> str:
    """Which study a file belongs to, from the path parts below the patient dir.

    A named directory wins over an index, so a re-exported cohort using words
    instead of digits still lands in the right bucket.
    """
    # A single-timepoint release puts the volumes straight in the patient
    # directory with nothing in between. BraTS-Africa is one, and calling that
    # "unknown" would force the caller to ask for a bucket named after an
    # absence.
    if not parts[:-1]:
        return "single"
    for seg in parts[:-1]:
        low = seg.lower()
        for pat, name in WORD_HINTS:
            if re.search(pat, low):
                return name
        if seg in INDEX_NAMES:
            return INDEX_NAMES[seg]
    return "unknown"


def scan(src: Path) -> dict:
    """Group every NIfTI under src by patient, then timepoint, then modality."""
    found: dict[str, dict[str, dict[str, Path]]] = {}
    for p in sorted(src.rglob("*.nii*")):
        rel = p.relative_to(src)
        if len(rel.parts) < 2:
            continue
        patient = rel.parts[0]
        mod = classify_modality(p.name)
        if mod is None:
            continue
        tp = timepoint_of(rel.parts[1:])
        found.setdefault(patient, {}).setdefault(tp, {})[mod] = p
    return found


def inspect(found: dict) -> None:
    print(f"{len(found)} patient directories")
    print(f"timepoints seen: {dict(Counter(tp for pt in found.values() for tp in pt))}")
    print(f"modalities seen: {dict(Counter(m for pt in found.values() for bym in pt.values() for m in bym))}")
    for patient, bytp in list(found.items())[:3]:
        print(f"\n{patient}")
        for tp, bym in sorted(bytp.items()):
            print(f"  [{tp}] " + ", ".join(f"{m}={p.name}" for m, p in sorted(bym.items())))
    missing = {p: sorted(set(NEEDED) - set(bytp.get("preop", {})))
               for p, bytp in found.items() if set(NEEDED) - set(bytp.get("preop", {}))}
    if missing:
        print(f"\n{len(missing)} patients are short of {NEEDED} at the preop timepoint:")
        for p, m in list(missing.items())[:10]:
            print(f"  {p}: missing {m}")


def convert(found: dict, dst: Path, timepoint: str, force: bool) -> dict:
    import nibabel as nib

    dst.mkdir(parents=True, exist_ok=True)
    ok, skipped, report = [], [], []
    for patient in sorted(found):
        bym = found[patient].get(timepoint, {})
        gaps = sorted(set(NEEDED) - set(bym))
        if gaps:
            skipped.append((patient, f"missing {gaps}"))
            continue

        # RHUH's own directory names already carry the cohort ("RHUH-0001"), so
        # only the separator needs normalising -- prefixing the dataset again
        # would produce RHUH_RHUH_0001 in every table downstream.
        pid = re.sub(r"[^A-Za-z0-9]+", "_", patient).strip("_")

        seg_img = nib.load(str(bym["seg"]))
        seg = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.int16)
        present = sorted(int(v) for v in np.unique(seg))
        unknown = [v for v in present if v not in LABEL_REMAP]
        if unknown:
            skipped.append((patient, f"unexpected label values {unknown}"))
            continue
        if seg.shape != BRATS_SHAPE:
            skipped.append((patient, f"seg shape {seg.shape} != {BRATS_SHAPE}"))
            continue

        bad_shape = next((f"{m} shape {nib.load(str(bym[m])).shape[:3]}"
                          for m in ("flair", "t1ce", "t2")
                          if nib.load(str(bym[m])).shape[:3] != BRATS_SHAPE), None)
        if bad_shape:
            skipped.append((patient, f"{bad_shape} != {BRATS_SHAPE}"))
            continue

        remapped = np.zeros_like(seg)
        for src_v, dst_v in LABEL_REMAP.items():
            remapped[seg == src_v] = dst_v

        out = dst / pid
        out.mkdir(exist_ok=True)
        for mod in ("flair", "t1ce", "t2"):
            target = out / f"{pid}_{mod}.nii.gz"
            if force or not target.exists():
                shutil.copyfile(bym[mod], target)
        nib.save(nib.Nifti1Image(remapped, seg_img.affine, seg_img.header),
                 str(out / f"{pid}_seg.nii.gz"))

        ok.append(pid)
        report.append({"patient": pid, "source": patient,
                       "labels_in": present,
                       "wt_voxels": int((remapped > 0).sum()),
                       "tc_voxels": int(np.isin(remapped, (1, 4)).sum()),
                       "et_voxels": int((remapped == 4).sum())})

    return {"converted": ok, "skipped": skipped, "report": report}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--timepoint", default="preop",
                    help="which timepoint to convert; only 'preop' is comparable to BraTS")
    ap.add_argument("--inspect", action="store_true", help="report the layout and stop")
    ap.add_argument("--force", action="store_true", help="re-copy modalities that already exist")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.exists():
        raise SystemExit(f"{src} does not exist -- run scripts/fetch_rhuh.sh first")

    found = scan(src)
    if not found:
        raise SystemExit(f"no NIfTI files found under {src}")
    if a.inspect:
        inspect(found)
        return

    res = convert(found, Path(a.dst), a.timepoint, a.force)
    dst = Path(a.dst)
    (dst / "patients.txt").write_text("\n".join(res["converted"]) + "\n", encoding="utf-8")
    (dst / "prep_report.json").write_text(json.dumps(res["report"], indent=2), encoding="utf-8")

    print(f"converted {len(res['converted'])} patients -> {dst}")
    if res["skipped"]:
        print(f"skipped {len(res['skipped'])}:")
        for p, why in res["skipped"]:
            print(f"  {p}: {why}")
    if res["report"]:
        et = [r["et_voxels"] for r in res["report"]]
        wt = [r["wt_voxels"] for r in res["report"]]
        print(f"median tumour volume: WT {int(np.median(wt))} vox, ET {int(np.median(et))} vox")
        if not any(et):
            print("WARNING: every ET mask is empty -- the label remap is wrong")


if __name__ == "__main__":
    main()
