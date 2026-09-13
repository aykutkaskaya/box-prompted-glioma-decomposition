"""HD95 and NSD for the two box arms, compiled per cohort.

Overlap and boundary answer different questions and this compiles both so the
manuscript can say where they agree. Patients on which an arm predicts nothing
have no surface and no distance; they are counted rather than scored, because
substituting a number for them would either flatter the arm (dropping them) or
be arbitrary (a fixed penalty).

Distances are heavy-tailed, so the median is reported alongside the mean and the
manuscript should quote whichever it names.

    python scripts/compile_boundary.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import FIX, corrected  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "boundary_summary.json"
COHORTS = [("clean", "BraTS2020 (shared held-out set)"),
           ("rhuh", "RHUH-GBM"),
           ("brats_africa", "BraTS-Africa")]
ARMS = ["oracle", "pipeline"]
DRAWS = 20000





def _fix(p):
    from pathlib import Path
    p = Path(p)
    return corrected(p)


def main() -> None:
    rng = np.random.default_rng(1337)
    result = {"tolerance_mm": 2.0, "draws": DRAWS, "cohorts": {}}

    print(f"{'cohort':<30}{'arm':<10}{'n':>4}{'HD95 med':>10}{'mean':>9}"
          f"{'NSD med':>10}{'empty':>7}")
    for c, label in COHORTS:
        path = _fix(V / f"{c}_boundary.jsonl")
        if not path.exists():
            print(f"{label:<30}  (not run)")
            continue
        recs = [json.loads(l) for l in io.open(path, encoding="utf-8")
                if l.strip()]
        recs = [r for r in recs if "error" not in r]
        row = {"n": len(recs), "arms": {}}
        for arm in ARMS:
            hd = [r[arm]["hd95"] for r in recs if r[arm]["hd95"] is not None]
            ns = [r[arm]["nsd"] for r in recs if r[arm]["nsd"] is not None]
            empty = sum(1 for r in recs if r[arm]["empty"] == "prediction")
            row["arms"][arm] = {
                "scored": len(hd), "empty_prediction": empty,
                "hd95_median": round(float(np.median(hd)), 2),
                "hd95_mean": round(float(np.mean(hd)), 2),
                "nsd_median": round(float(np.median(ns)), 4),
                "nsd_mean": round(float(np.mean(ns)), 4),
            }
            print(f"{label if arm == ARMS[0] else '':<30}{arm:<10}{len(hd):>4}"
                  f"{np.median(hd):>10.2f}{np.mean(hd):>9.2f}"
                  f"{np.median(ns):>10.4f}{empty:>7}")

        # the paired contrast, on the patients where both arms have a surface
        both = [r for r in recs
                if r["oracle"]["hd95"] is not None
                and r["pipeline"]["hd95"] is not None]
        if both:
            d = np.array([r["pipeline"]["hd95"] - r["oracle"]["hd95"]
                          for r in both])
            bs = np.sort([d[rng.integers(0, len(d), len(d))].mean()
                          for _ in range(DRAWS)])
            row["pipeline_minus_oracle_hd95"] = {
                "n": len(d), "mean": round(float(d.mean()), 2),
                "median": round(float(np.median(d)), 2),
                "ci": [round(float(np.percentile(bs, 2.5)), 2),
                       round(float(np.percentile(bs, 97.5)), 2)],
                "pipeline_better": int((d < 0).sum()),
            }
        result["cohorts"][c] = row

    print(f"\n{'cohort':<30}{'HD95 pipe - oracle':>20}{'95% CI':>20}"
          f"{'pipe better':>13}")
    for c, label in COHORTS:
        r = result["cohorts"].get(c, {}).get("pipeline_minus_oracle_hd95")
        if not r:
            continue
        print(f"{label:<30}{r['mean']:>+20.2f}"
              f"   [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]"
              f"{r['pipeline_better']:>10}/{r['n']}")

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
