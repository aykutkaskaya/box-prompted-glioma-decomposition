"""The decomposition on tumour core, which the paper states only for whole tumour.

Section 3.4 reports the tumour-core pipeline against the headroom of Table 4 and
stops there. The same three arms exist on that region, so the detector-stage
term does too, and it does not repeat what whole tumour shows: in domain it is
an order of magnitude larger, and the two external cohorts fall on opposite
sides of the held-out one rather than both above it. A paper whose title is
about the detector stage should say so where its own data say it.

Paired within a cohort and unpaired between, 20 000 draws at generator seed
1337, matching Section 2.5.

    python scripts/tc_detector_stage_ci.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import corrected  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "tc_detector_stage_ci.json"
SEG = "sam2.1_l"
DRAWS = 20000
SEED = 1337
REGION = "TC"
COHORTS = [("clean", "BraTS2020 (shared held-out set)"),
           ("rhuh", "RHUH-GBM"),
           ("brats_africa", "BraTS-Africa")]
# the tumour-core detector was trained at three seeds; the whole-tumour
# decomposition uses one, so all three are read rather than one chosen
SEEDS = [1337, 42, 62]


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


def pipeline_file(cohort: str, seed: int) -> Path:
    """RHUH-GBM's corrected runs carry the seed in a different position."""
    if cohort == "rhuh":
        return V / f"rhuh_tc_pipeline_s{seed}_normfix.jsonl"
    tail = "" if seed == 1337 else f"_seed{seed}"
    return V / f"{cohort}_tc_pipeline{tail}.jsonl"


def terms(cohort: str, seed: int):
    """Per-patient oracle-box minus pipeline on tumour core."""
    orc = read(corrected(V / f"{cohort}_oracle_regions.jsonl", SEG))
    pipe = read(pipeline_file(cohort, seed))
    ok, pk = f"oracle:{SEG}", f"pipeline:{SEG}"
    ids = []
    for p in sorted(orc):
        if ok not in orc[p] or p not in pipe or pk not in pipe[p]:
            continue
        r = orc[p][ok].get("regions", {}).get(REGION)
        # a patient with no tumour core in the reference is excluded from the
        # region's mean, as Section 2.5 requires
        if not r or not r.get("gt_voxels"):
            continue
        ids.append(p)
    d = np.array([orc[p][ok]["regions"][REGION]["vol_dice"]
                  - pipe[p][pk]["vol_dice"] for p in ids])
    return ids, d


def ci(rng, draws: int, *samples):
    """Percentile interval on the mean, or on a difference of two means."""
    stat = []
    for _ in range(draws):
        means = [s[rng.integers(0, len(s), len(s))].mean() for s in samples]
        stat.append(means[0] if len(means) == 1 else means[0] - means[1])
    stat = np.sort(stat)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def main() -> None:
    rng = np.random.default_rng(SEED)
    result = {"draws": DRAWS, "seed": SEED, "segmenter": SEG, "region": REGION,
              "within": {}, "between": {}}

    print(f"tumour-core detector-stage term, {DRAWS} bootstrap draws\n")
    print(f"{'cohort':<32}{'seed':>6}{'n':>5}{'mean':>10}{'95% CI':>24}")
    data = {}
    for c, label in COHORTS:
        for sd in SEEDS:
            ids, d = terms(c, sd)
            if not len(d):
                print(f"{label:<32}{sd:>6}{'-':>5}   not run")
                continue
            data[(c, sd)] = d
            lo, hi = ci(rng, DRAWS, d)
            result["within"][f"{c}/{sd}"] = {
                "n": len(d), "mean": round(float(d.mean()), 4),
                "ci": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}
            print(f"{label:<32}{sd:>6}{len(d):>5}{d.mean():>+10.4f}"
                  f"   [{lo:+.4f}, {hi:+.4f}]")

    print("\nexternal minus held-out, same detector seed\n")
    print(f"{'contrast':<32}{'seed':>6}{'difference':>12}{'95% CI':>24}")
    for c, label in COHORTS[1:]:
        for sd in SEEDS:
            if (c, sd) not in data or ("clean", sd) not in data:
                continue
            a, b = data[(c, sd)], data[("clean", sd)]
            diff = float(a.mean() - b.mean())
            lo, hi = ci(rng, DRAWS, a, b)
            result["between"][f"{c}-clean/{sd}"] = {
                "difference": round(diff, 4), "ci": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}
            print(f"{label + ' - held out':<32}{sd:>6}{diff:>+12.4f}"
                  f"   [{lo:+.4f}, {hi:+.4f}]")

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
