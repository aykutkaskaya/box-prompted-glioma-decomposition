"""Is the detector-stage term explained by tumour size rather than by cohort?

The three cohorts differ in more than acquisition, and the most obvious rival
explanation is lesion size: a detector that misses small tumours would produce
a large term on whichever cohort happens to have smaller ones, and the cohort
label would be incidental. This tests that directly, and reports the answer
whichever way it falls.

Two questions, kept separate. Within the pooled patients, does the term vary
with tumour volume at all? And do the cohorts differ in volume in the direction
that would be needed to manufacture the result?

    python scripts/confounder_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import FIX, corrected  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "confounder_check.json"
SEG = "sam2.1_l"
COHORTS = [("clean", "BraTS2020 (shared held-out set)"),
           ("rhuh", "RHUH-GBM"),
           ("brats_africa", "BraTS-Africa")]





def _fix(p):
    from pathlib import Path
    p = Path(p)
    return corrected(p)


def read(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                out[r["patient"]] = r
    return out


def main() -> None:
    ok, pk = f"oracle:{SEG}", f"pipeline:{SEG}"
    per = {}
    for c, _ in COHORTS:
        recs = read(_fix(V / f"{c}_box.jsonl"))
        term, vol = [], []
        for p, r in recs.items():
            a = r.get("arms", {})
            if ok not in a or pk not in a:
                continue
            g = a[ok].get("gt_voxels") or r.get("gt_voxels")
            if not g:
                continue
            term.append(a[ok]["vol_dice"] - a[pk]["vol_dice"])
            vol.append(g)
        per[c] = (np.array(term), np.array(vol, dtype=float))

    result = {"segmenter": SEG, "cohorts": {}, "pooled": {}, "between": {}}

    print(f"{'cohort':<30}{'n':>4}{'median volume':>16}{'term vs volume':>18}")
    for c, label in COHORTS:
        t, v = per[c]
        rho, p = stats.spearmanr(v, t)
        result["cohorts"][c] = {
            "n": len(t), "median_gt_voxels": int(np.median(v)),
            "spearman_term_vs_volume": round(float(rho), 3),
            "p": float(p)}
        print(f"{label:<30}{len(t):>4}{int(np.median(v)):>16,}"
              f"   rho {rho:+.3f}  p {p:.3g}")

    t_all = np.concatenate([per[c][0] for c, _ in COHORTS])
    v_all = np.concatenate([per[c][1] for c, _ in COHORTS])
    rho, p = stats.spearmanr(v_all, t_all)
    result["pooled"] = {"n": len(t_all), "spearman": round(float(rho), 3),
                        "p": float(p)}
    print(f"\n  pooled over all {len(t_all)} patients: "
          f"Spearman {rho:+.3f}, p = {p:.3g}")
    print("  a negative sign means the term is LARGER on smaller tumours"
          if rho < 0 else
          "  a positive sign means the term is larger on larger tumours")

    base = per["clean"][1]
    for c, label in COHORTS[1:]:
        u, p = stats.mannwhitneyu(per[c][1], base, alternative="two-sided")
        bigger = np.median(per[c][1]) > np.median(base)
        result["between"][c] = {
            "median_ratio": round(float(np.median(per[c][1]) / np.median(base)), 2),
            "mannwhitney_p": float(p), "larger_than_held_out": bool(bigger)}
        print(f"  {label} vs held-out: median volume "
              f"{np.median(per[c][1]) / np.median(base):.2f}x, "
              f"Mann-Whitney p = {p:.3g}"
              f"{'  (larger)' if bigger else '  (smaller)'}")

    # the rival explanation needs BOTH: term falls with volume AND the external
    # cohorts have smaller tumours. Report whether that conjunction holds.
    rival = result["pooled"]["spearman"] < 0 and all(
        not v["larger_than_held_out"] for v in result["between"].values())
    result["size_explains_result"] = bool(rival)
    print("\n  size could explain the result" if rival else
          "\n  size does not explain the result: the sign of the correlation and "
          "the direction of the cohort difference do not line up")

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
