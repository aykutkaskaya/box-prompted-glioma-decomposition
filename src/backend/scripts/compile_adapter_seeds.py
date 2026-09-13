"""The whole-tumour comparison at all three LoRA adapter seeds.

The detector-free arm was run at three adapter seeds on all three cohorts and
the result sat unreported while the manuscript said it did not exist. This
compiles it.

The pipeline and oracle-box arms do not depend on the adapter, so they are read
from the frozen protocol files and every seed is scored against the same
reference. Patients are intersected across the three runs before anything is
averaged.

    python scripts/compile_adapter_seeds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import FIX, corrected  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "adapter_seed_check.json"
SEG = "sam2.1_l"
SEEDS = ["42", "52", "62"]
COHORTS = [("clean", "BraTS2020 (shared held-out set)"),
           ("rhuh", "RHUH-GBM"),
           ("brats_africa", "BraTS-Africa")]
DRAWS = 20000





def _fix(p):
    from pathlib import Path
    p = Path(p)
    return corrected(p)


def read(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r:
                out[r["patient"]] = r.get("arms", {})
    return out


def direct_of(cohort: str, seed: str) -> dict:
    """Seed 42 is the frozen protocol run; the others have their own files."""
    p = (_fix(V / f"{cohort}_direct.jsonl") if seed == "42"
         else _fix(V / f"{cohort}_direct_seed{seed}.jsonl"))
    arms = read(p)
    k = f"direct:seed_{seed}"
    return {pid: a[k]["regions"]["WT"]["vol_dice"]
            for pid, a in arms.items()
            if k in a and "WT" in a[k].get("regions", {})}


def main() -> None:
    rng = np.random.default_rng(1337)
    result = {"seeds": SEEDS, "segmenter": SEG, "draws": DRAWS, "cohorts": {}}

    print(f"{'cohort':<30}{'n':>4}" + "".join(f"{'seed ' + s:>12}" for s in SEEDS)
          + f"{'spread':>9}")
    for c, label in COHORTS:
        box = read(_fix(V / f"{c}_box.jsonl"))
        d = {s: direct_of(c, s) for s in SEEDS}
        ok, pk = f"oracle:{SEG}", f"pipeline:{SEG}"
        ids = sorted(set.intersection(
            *[set(x) for x in d.values()],
            {p for p, a in box.items() if ok in a and pk in a}))
        if not ids:
            print(f"{label:<30}   -  (nothing to compare)")
            continue
        P = np.array([box[p][pk]["vol_dice"] for p in ids])
        O = np.array([box[p][ok]["vol_dice"] for p in ids])
        row = {"n": len(ids), "pipeline": round(float(P.mean()), 4),
               "oracle_box": round(float(O.mean()), 4), "seeds": {}}
        cells = []
        for s in SEEDS:
            D = np.array([d[s][p] for p in ids])
            gap = D - P
            bs = np.sort([gap[rng.integers(0, len(gap), len(gap))].mean()
                          for _ in range(DRAWS)])
            row["seeds"][s] = {
                "detector_free": round(float(D.mean()), 4),
                "gap": round(float(gap.mean()), 4),
                "residual": round(float((D - O).mean()), 4),
                "wins": int((gap > 0).sum()),
                "ci": [round(float(np.percentile(bs, 2.5)), 4),
                       round(float(np.percentile(bs, 97.5)), 4)],
            }
            cells.append(f"{gap.mean():+12.4f}")
        gaps = [row["seeds"][s]["gap"] for s in SEEDS]
        row["gap_spread"] = round(max(gaps) - min(gaps), 4)
        result["cohorts"][c] = row
        print(f"{label:<30}{len(ids):>4}" + "".join(cells)
              + f"{row['gap_spread']:>9.4f}")

    if len(result["cohorts"]) == len(COHORTS):
        print()
        # what survives across seeds and what does not
        for s in SEEDS:
            g = {c: result["cohorts"][c]["seeds"][s]["gap"] for c, _ in COHORTS}
            r = {c: result["cohorts"][c]["seeds"][s]["residual"] for c, _ in COHORTS}
            ratio = g["brats_africa"] / g["rhuh"]
            signs = "".join("+" if r[c] > 0 else "-" for c, _ in COHORTS)
            print(f"  seed {s}: gaps " + " ".join(f"{g[c]:+.4f}" for c, _ in COHORTS)
                  + f"   Africa/RHUH {ratio:.1f}x   residual signs {signs}")
        ratios = [result["cohorts"]["brats_africa"]["seeds"][s]["gap"]
                  / result["cohorts"]["rhuh"]["seeds"][s]["gap"] for s in SEEDS]
        result["africa_over_rhuh"] = [round(x, 1) for x in ratios]
        sign_sets = {"".join("+" if result["cohorts"][c]["seeds"][s]["residual"] > 0
                             else "-" for c, _ in COHORTS) for s in SEEDS}
        result["residual_sign_pattern_stable"] = len(sign_sets) == 1
        result["pipeline_never_ahead"] = all(
            result["cohorts"][c]["seeds"][s]["gap"] > 0
            for c, _ in COHORTS for s in SEEDS)
        print(f"\n  Africa/RHUH ratio across seeds: "
              + ", ".join(f"{x:.1f}x" for x in ratios))
        print(f"  residual sign pattern stable: {result['residual_sign_pattern_stable']}"
              f"   pipeline never ahead: {result['pipeline_never_ahead']}")

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
