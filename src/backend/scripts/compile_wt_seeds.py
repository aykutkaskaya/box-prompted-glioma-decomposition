"""Does the detector-stage term keep its ordering at a second detector seed?

The paper's headline decomposition is measured from one detector, and the paper
also argues that out-of-domain comparisons should not be believed from a single
training run. This applies that test to the term the title rests on.

Only the pipeline arm was re-run: the oracle-box arm is prompted with the
ground-truth box and does not depend on the detector, so it is read from the
frozen protocol files and the two seeds are compared against the same reference.

Patients are intersected across the two seeds before anything is averaged. A
seed that failed to load a patient would otherwise be averaged over a different
group, which looks exactly like a seed effect.

    python scripts/compile_wt_seeds.py
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
OUT = DRIVE_ROOT / "reports" / "wt_seed_check.json"
SEG = "sam2.1_l"
COHORTS = [("clean", "BraTS2020 (mutually held out)"),
           ("rhuh", "RHUH-GBM"),
           ("brats_africa", "BraTS-Africa")]
SEEDS = [1337, 42]





def _fix(p):
    from pathlib import Path
    p = Path(p)
    return corrected(p)


def read(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "error" not in r:
            out[r["patient"]] = r.get("arms", {})
    return out


def pipeline_of(cohort: str, seed: int) -> dict:
    """seed 1337 is the frozen protocol run; the rest have their own files."""
    p = (_fix(V / f"{cohort}_box.jsonl") if seed == 1337
         else _fix(V / f"{cohort}_wt_pipeline_seed{seed}.jsonl"))
    arms = read(p)
    key = f"pipeline:{SEG}"
    return {pid: a[key]["vol_dice"] for pid, a in arms.items() if key in a}


def main() -> None:
    box = {c: read(_fix(V / f"{c}_box.jsonl")) for c, _ in COHORTS}
    result = {"seeds": SEEDS, "segmenter": SEG, "cohorts": {}}

    print(f"{'cohort':<30}{'n':>4}  " +
          "  ".join(f"{'seed ' + str(s):>11}" for s in SEEDS) + "     spread")
    for c, label in COHORTS:
        pipes = {s: pipeline_of(c, s) for s in SEEDS}
        okey = f"oracle:{SEG}"
        common = sorted(set.intersection(
            *[set(p) for p in pipes.values()],
            {p for p, a in box[c].items() if okey in a}))
        if not common:
            print(f"{label:<30}   -  (nothing to compare yet)")
            continue
        O = np.array([box[c][p][okey]["vol_dice"] for p in common])
        row = {"n": len(common), "oracle_box": round(float(O.mean()), 4),
               "seeds": {}}
        cells = []
        for s in SEEDS:
            P = np.array([pipes[s][p] for p in common])
            det = float((O - P).mean())
            # the unrounded term is kept because the ratios below divide by a
            # number near 0.006, where rounding the operands first moves the
            # quotient by a tenth (15.41 became 15.5 in an earlier draft)
            row["seeds"][str(s)] = {"pipeline": round(float(P.mean()), 4),
                                    "detector_stage": round(det, 4),
                                    "detector_stage_raw": det}
            cells.append(f"{det:+11.4f}")
        terms = [row["seeds"][str(s)]["detector_stage"] for s in SEEDS]
        row["spread"] = round(max(terms) - min(terms), 4)
        result["cohorts"][c] = row
        print(f"{label:<30}{len(common):>4}  " + "  ".join(cells) +
              f"  {row['spread']:>9.4f}")
    done = [c for c, _ in COHORTS if c in result["cohorts"]]
    if len(done) != len(COHORTS):
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")
        return

    # Two claims can be made from three cohorts and they are not equally
    # strong. The categorical one -- the term is negligible in domain and an
    # order of magnitude larger outside it -- is what the paper argues. The
    # ordinal one, that BraTS-Africa costs more than RHUH-GBM, ranks two
    # external cohorts against each other and is the more fragile of the two.
    print()
    ratios = []
    for s in SEEDS:
        t = [result["cohorts"][c]["seeds"][str(s)]["detector_stage_raw"]
             for c, _ in COHORTS]
        rank = all(b > a for a, b in zip(t, t[1:]))
        r = [x / t[0] for x in t[1:]]
        ratios += r
        result.setdefault("ordering", {})[str(s)] = {
            "terms": [round(x, 4) for x in t],
            "external_over_internal": [round(x, 1) for x in r],
            "rank_order_preserved": bool(rank),
        }
        print(f"  seed {s}: " + " -> ".join(f"{x:+.4f}" for x in t) +
              f"   external/internal {r[0]:.1f}x, {r[1]:.1f}x")

    cat = min(ratios)
    rank = all(v["rank_order_preserved"] for v in result["ordering"].values())
    result["categorical_min_ratio"] = round(cat, 1)
    result["categorical_holds"] = bool(cat >= 5)
    result["rank_order_agrees"] = bool(rank)

    print()
    print(f"  categorical: at every seed, every external cohort costs at least "
          f"{cat:.1f}x the in-domain term")
    print("  rank order between the two external cohorts: "
          + ("preserved" if rank else "NOT preserved, they swap"))

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
