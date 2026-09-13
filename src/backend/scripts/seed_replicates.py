"""Measure the seed noise floor and test each ablation claim against it.

The 37-run ablation used a single training seed (1337), which was its weakest
point: with one run per configuration there is no way to tell a real effect from
a lucky initialisation. A same-configuration repeat inside the study (runs 30
and 37) differed by 0.0005 F1 and was quoted as a noise floor, but both ran at
seed 1337, so that number measures nondeterminism, not seed variance. The real
figure is roughly twenty times larger, and one claim does not survive it.

This reads the replicate tree and reports, per metric, the spread across seeds,
then divides each claim by that spread. A claim worth stating should clear its
metric's noise by a comfortable factor; one that lands at 1.5x is an
observation, not a finding, and is reported as such.

Layout expected:

    experiments/<slug>/summary/run_summary.json                  seed 1337
    experiments_repeated/seed_<n>/<slug>/summary/run_summary.json  the rest

    python scripts/seed_replicates.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

BASE_SEED = 1337
EXP = DRIVE_ROOT / "experiments"
REP = DRIVE_ROOT / "experiments_repeated"
OUT = DRIVE_ROOT / "reports" / "seed_replicates.json"

# (group, field) pairs worth tracking, and whether more is better
METRICS = [("test", "f1"), ("test", "precision"), ("test", "recall"),
           ("test", "map50"), ("test", "map50_95"),
           ("geometry", "pred_coverage"), ("geometry", "target_coverage"),
           ("geometry", "iou")]

# Claims stated in the report, each as (label, metric, run_a, run_b, direction).
# `direction` is the sign the claim asserts for b - a.
CLAIMS = [
    ("alpha controls predicted-box coverage", ("geometry", "pred_coverage"), 9, 20, -1),
    ("alpha controls target coverage", ("geometry", "target_coverage"), 9, 20, +1),
    ("IC-Arb beats GIoU", ("test", "f1"), 1, 9, +1),
]


def discover() -> dict:
    """{run_id: {seed: summary}} for every configuration with a base run."""
    out: dict[int, dict[int, dict]] = {}
    slugs: dict[int, str] = {}
    for p in sorted(EXP.glob("run_*/summary/run_summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rid = int(d["run_id"])
        out.setdefault(rid, {})[BASE_SEED] = d
        slugs[rid] = d["slug"]
    for seed_dir in sorted(REP.glob("seed_*")):
        m = re.match(r"seed_(\d+)$", seed_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        for p in sorted(seed_dir.glob("run_*/summary/run_summary.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            out.setdefault(int(d["run_id"]), {})[seed] = d
    return {r: v for r, v in out.items() if len(v) > 1}, slugs


def values(runs: dict, rid: int, grp: str, field: str) -> dict:
    return {s: d[grp][field] for s, d in runs.get(rid, {}).items()
            if isinstance(d.get(grp), dict) and field in d[grp]}


def main() -> None:
    runs, slugs = discover()
    if not runs:
        raise SystemExit(f"no replicated configurations found under {REP}")

    print(f"{len(runs)} configurations have more than one seed: "
          + ", ".join(f"run {r} ({sorted(v)})" for r, v in sorted(runs.items())))
    print()

    # ------------------------------------------------------------- noise floor
    print("Seed noise, pooled over replicated configurations")
    print(f"{'metric':<26}{'mean spread':>13}{'max spread':>12}{'n cfg':>7}")
    noise = {}
    for grp, f in METRICS:
        spreads = []
        for rid in runs:
            v = list(values(runs, rid, grp, f).values())
            if len(v) > 1:
                # spread = max - min, which for two seeds is the absolute difference
                # and for three or more is the full observed range
                spreads.append(max(v) - min(v))
        if not spreads:
            continue
        noise[f"{grp}.{f}"] = {"mean": float(np.mean(spreads)),
                               "max": float(np.max(spreads)),
                               "n_configs": len(spreads)}
        print(f"{grp + '.' + f:<26}{np.mean(spreads):>13.4f}{np.max(spreads):>12.4f}{len(spreads):>7}")
    print()

    # ------------------------------------------------------------ per-config
    print("Per configuration (mean +/- sd across seeds, sd only where >2 seeds)")
    per = {}
    for rid in sorted(runs):
        seeds = sorted(runs[rid])
        row = {"slug": slugs.get(rid, ""), "seeds": seeds}
        bits = []
        for grp, f in (("test", "f1"), ("geometry", "pred_coverage"),
                       ("geometry", "target_coverage")):
            v = list(values(runs, rid, grp, f).values())
            if not v:
                continue
            row[f"{grp}.{f}"] = {"mean": float(np.mean(v)),
                                 "sd": float(np.std(v, ddof=1)) if len(v) > 2 else None,
                                 "spread": float(max(v) - min(v)), "n": len(v)}
            sd = row[f"{grp}.{f}"]["sd"]
            bits.append(f"{f} {np.mean(v):.4f}" + (f" +/-{sd:.4f}" if sd is not None else ""))
        per[rid] = row
        print(f"  run {rid:<3} seeds {str(seeds):<18} " + "  ".join(bits))
    print()

    # ---------------------------------------------------------------- claims
    # The denominator is the standard error of the difference between the two
    # configuration means, not the observed range. Range was the first thing I
    # reached for and it is wrong here: its expectation grows with the number of
    # seeds, so running a third replicate made every claim look weaker purely by
    # adding data. sd/sqrt(n) does not have that defect. Where a configuration
    # has only two seeds its sd is estimated from the single difference, which
    # is noisy -- flagged in the output rather than hidden.
    print("Claims against the noise floor")
    print("  denominator = SE of the difference between configuration means")
    out_claims = []
    for label, (grp, f), ra, rb, direction in CLAIMS:
        va = list(values(runs, ra, grp, f).values())
        vb = list(values(runs, rb, grp, f).values())
        if len(va) < 2 or len(vb) < 2:
            print(f"  {label}: needs replicates of runs {ra} and {rb}; have "
                  f"{len(va)} and {len(vb)}")
            continue
        ma, mb = float(np.mean(va)), float(np.mean(vb))
        sa, sb = float(np.std(va, ddof=1)), float(np.std(vb, ddof=1))
        se = float(np.sqrt(sa**2 / len(va) + sb**2 / len(vb)))
        gap = mb - ma
        ratio = abs(gap) / se if se else float("inf")
        agrees = (gap > 0) == (direction > 0)
        thin = min(len(va), len(vb)) < 3
        out_claims.append({"claim": label, "metric": f"{grp}.{f}",
                           "runs": [ra, rb], "gap": round(gap, 4),
                           "mean_a": round(ma, 4), "mean_b": round(mb, 4),
                           "sd_a": round(sa, 4), "sd_b": round(sb, 4),
                           "se_diff": round(se, 5), "ratio": round(ratio, 1),
                           "direction_agrees": agrees, "n_seeds": [len(va), len(vb)]})
        verdict = "holds" if agrees and ratio >= 3 else (
            "marginal" if agrees and ratio >= 2 else "does not clear noise")
        note = "  (one arm has only 2 seeds)" if thin else ""
        print(f"  {label}")
        print(f"    {grp}.{f}: {ma:.4f} -> {mb:.4f}, gap {gap:+.4f}, "
              f"SE {se:.5f}, ratio {ratio:.1f}x -> {verdict}{note}")

    # The classification-loss observation, kept for the record. Runs 16 and 36
    # were never replicated and SAM evaluation was skipped in the replicate tour,
    # so neither half of it can be tested; the F1 half is inside the noise band.
    f1n = noise.get("test.f1", {}).get("mean")
    if f1n:
        print()
        print("Classification loss (focal vs varifocal, runs 37 vs 16)")
        print(f"    claimed F1 gap +0.0158 against a pooled seed noise of {f1n:.4f} "
              f"-> {0.0158 / f1n:.1f}x")
        print("    run 16 has no replicate, and neg_seg_fp_rate has none either "
              "(SAM evaluation was skipped for the replicates), so this is reported "
              "as an observation, not a finding.")

    OUT.write_text(json.dumps({"base_seed": BASE_SEED, "noise": noise,
                               "per_config": per, "claims": out_claims}, indent=1),
                   encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
