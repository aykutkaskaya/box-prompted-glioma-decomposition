"""How widely each cohort's reference is drawn, from the stored result files.

Section 4.1 names a rival explanation the design cannot separate: the
oracle-box arm is prompted from the reference it is then scored against, so a
cohort annotated more widely costs that arm nothing, while a detector trained
to one cohort's convention pays for the difference and the whole of the cost
lands in the detector-stage term. The claim needs the extent measured rather
than asserted, which is what this does.

Volume is the reference's own voxel count as recorded per patient; area per
positive slice is that count over the slices the region occupies. Both are
medians over patients, because a handful of very large lesions otherwise set
the mean.

    python scripts/reference_extent.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import corrected  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "reference_extent.json"
SEG = "sam2.1_l"
COHORTS = [("clean", "BraTS2020 (shared held-out set)"),
           ("rhuh", "RHUH-GBM"),
           ("brats_africa", "BraTS-Africa")]


def read(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "error" not in r:
            out[r["patient"]] = r.get("arms", {})
    return out


def extent(cohort: str):
    """Median reference volume and median area per positive slice."""
    box = read(corrected(V / f"{cohort}_box.jsonl", SEG))
    vols, areas = [], []
    for arms in box.values():
        a = arms.get(f"oracle:{SEG}")
        if not a:
            continue
        gt, n = a.get("gt_voxels"), a.get("n_slices")
        if gt:
            vols.append(gt)
            if n:
                areas.append(gt / n)
    return vols, areas


def main() -> None:
    result = {"segmenter": SEG, "cohorts": {}}
    rows = {}
    print(f"{'cohort':<32}{'n':>5}{'median volume':>16}{'area/slice':>13}")
    for c, label in COHORTS:
        vols, areas = extent(c)
        if not vols:
            print(f"{label:<32}   not run")
            continue
        mv, ma = statistics.median(vols), statistics.median(areas)
        rows[c] = (mv, ma)
        result["cohorts"][c] = {
            "n": len(vols),
            "median_volume_voxels": round(mv, 1),
            "median_area_per_positive_slice": round(ma, 1)}
        print(f"{label:<32}{len(vols):>5}{mv:>16,.0f}{ma:>13,.0f}")

    if "clean" in rows:
        base_v, base_a = rows["clean"]
        print(f"\n{'against the held-out set':<32}{'volume':>10}{'area':>10}")
        for c, label in COHORTS[1:]:
            if c not in rows:
                continue
            v, a = rows[c]
            result["cohorts"][c]["volume_ratio_to_held_out"] = round(v / base_v, 2)
            result["cohorts"][c]["area_ratio_to_held_out"] = round(a / base_a, 2)
            print(f"{label:<32}{v / base_v:>10.2f}{a / base_a:>10.2f}")

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
