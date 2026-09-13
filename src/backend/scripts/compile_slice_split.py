"""Split the detector-stage term into missed detection and box geometry.

A hybrid arm keeps the pipeline's own mask wherever the detector fired and
takes the oracle-box mask wherever it did not. Then

    oracle - pipeline  =  (oracle - hybrid)  +  (hybrid - pipeline)
                          box geometry          missed detection

The two sum exactly, because the hybrid is the pipeline on one part of the
slice set and the oracle on the other, and Dice is computed on pooled voxels.

What each half means is worth stating plainly. The second is what a detector
that never missed a slice would recover, which is what lowering the confidence
threshold reaches for. The first is what remains after that: the mask the
detector's own boxes produce where it did fire, against the mask a tight
ground-truth box produces on the same slice. It is not a pure geometry effect
-- a box on the wrong structure lands here too -- so it is an upper bound on
what better box placement and shape could recover, not an estimate of it.

    python scripts/compile_slice_split.py
"""
from __future__ import annotations

import io
import json
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import FIX, corrected  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "slice_split.json"
COHORTS = [("clean", "BraTS2020 (mutually held out)"),
           ("rhuh", "RHUH-GBM"),
           ("brats_africa", "BraTS-Africa")]





def _fix(p):
    from pathlib import Path
    p = Path(p)
    return corrected(p)


def dice(i: int, p: int, g: int) -> float:
    return (2 * i / (p + g)) if (p + g) else 1.0


def patient(rec: dict) -> dict:
    """Three arms over one patient, from the per-slice counts."""
    S = rec["slices"]
    g = sum(x["gt"] for x in S)
    o = (sum(x["oracle_i"] for x in S), sum(x["oracle_p"] for x in S))
    p = (sum(x["pipe_i"] for x in S), sum(x["pipe_p"] for x in S))
    # the hybrid: the pipeline where it fired, the oracle where it did not
    h = (sum(x["pipe_i"] if x["boxes"] else x["oracle_i"] for x in S),
         sum(x["pipe_p"] if x["boxes"] else x["oracle_p"] for x in S))
    missed = [x for x in S if not x["boxes"] and x["gt"]]
    # how well a correct box does on the slices the detector skipped, which is
    # what decides whether filling them in helps at all
    md = [2 * x["oracle_i"] / (x["oracle_p"] + x["gt"])
          for x in missed if (x["oracle_p"] + x["gt"])]
    return {
        "oracle": dice(o[0], o[1], g),
        "hybrid": dice(h[0], h[1], g),
        "pipeline": dice(p[0], p[1], g),
        "n_slices": len(S),
        "n_positive": sum(1 for x in S if x["gt"]),
        "n_missed_positive": len(missed),
        "gt_in_missed": sum(x["gt"] for x in missed),
        "gt_total": g,
        "missed_dice": md,
    }


def main() -> None:
    result = {"cohorts": {}}
    geom_by_patient: dict[str, list[float]] = {}
    print(f"{'cohort':<30}{'n':>4}{'detector':>10}{'missed':>9}{'geometry':>10}"
          f"{'missed %':>10}   slices missed")
    for c, label in COHORTS:
        path = _fix(V / f"{c}_slices.jsonl")
        if not path.exists():
            print(f"{label:<30}   -  (not run yet)")
            continue
        rows = []
        for line in io.open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" not in r and r.get("slices"):
                rows.append(patient(r))
        if not rows:
            print(f"{label:<30}   -  (no usable records)")
            continue

        n = len(rows)
        mean = lambda k: sum(r[k] for r in rows) / n
        # the per-patient box-geometry differences, kept for the contrast below
        geom_by_patient[c] = [r["oracle"] - r["hybrid"] for r in rows]
        det = mean("oracle") - mean("pipeline")
        missed = mean("hybrid") - mean("pipeline")
        geom = mean("oracle") - mean("hybrid")
        share = missed / det if det else float("nan")
        n_missed = sum(r["n_missed_positive"] for r in rows)
        med = sorted(d for r in rows for d in r["missed_dice"])
        med = med[len(med) // 2] if med else float("nan")
        n_pos = sum(r["n_positive"] for r in rows)
        gt_missed = sum(r["gt_in_missed"] for r in rows)
        gt_all = sum(r["gt_total"] for r in rows)

        result["cohorts"][c] = {
            "n": n,
            "oracle": round(mean("oracle"), 4),
            "hybrid": round(mean("hybrid"), 4),
            "pipeline": round(mean("pipeline"), 4),
            "detector_stage": round(det, 4),
            "missed_detection": round(missed, 4),
            "box_geometry": round(geom, 4),
            "missed_share": round(share, 3),
            "positive_slices": n_pos,
            "missed_positive_slices": n_missed,
            "tumour_voxels_on_missed_slices": round(gt_missed / gt_all, 4),
            "oracle_median_dice_on_missed": round(med, 3),
        }
        print(f"{label:<30}{n:>4}{det:>+10.4f}{missed:>+9.4f}{geom:>+10.4f}"
              f"{share * 100:>9.0f}%   {n_missed}/{n_pos} positive")

    if len(result["cohorts"]) == len(COHORTS):
        shares = [r["missed_share"] for r in result["cohorts"].values()]
        result["missed_share_range"] = [round(min(shares), 3), round(max(shares), 3)]
        print(f"\n  missed detection accounts for "
              f"{min(shares) * 100:.0f}-{max(shares) * 100:.0f}% of the "
              f"detector-stage term; the rest is what the detector's own boxes "
              f"produce where it did fire")

    # The contrast of each external cohort against the held-out one. Unpaired,
    # because the cohorts contain different people, so each is resampled on its
    # own; the point estimate is a difference of unrounded means, rounded once.
    if "clean" in geom_by_patient:
        rng = np.random.default_rng(1337)
        base = np.array(geom_by_patient["clean"])
        result["box_geometry_vs_held_out"] = {}
        for c in ("rhuh", "brats_africa"):
            if c not in geom_by_patient:
                continue
            arm = np.array(geom_by_patient[c])
            draws = (rng.choice(arm, (20000, arm.size)).mean(1)
                     - rng.choice(base, (20000, base.size)).mean(1))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            result["box_geometry_vs_held_out"][c] = {
                "point": round(float(arm.mean() - base.mean()), 4),
                "ci": [round(float(lo), 4), round(float(hi), 4)],
            }
            print(f"  box geometry, {c} vs held-out: "
                  f"{arm.mean() - base.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
